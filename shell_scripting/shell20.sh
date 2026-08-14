#!/usr/bin/env bash
# ============================================================
# GPU 작업 큐 슬롯 — PromptMR+ MoE (Option C: 공유 사전학습 → acc별 미세조정) (shell20.sh)
#
# 목적: acc4/acc8을 구분하는 MoE를 Option C로 구축.
#   ① pretrain : 전체 데이터(--acc all)로 공유 모델 1개 학습        -n promptmr_moe_shared
#   ② ft4      : 공유 가중치에서 acc4로 미세조정(--acc 4 --init-from) -n promptmr_moe_acc4
#   ③ ft8      : 공유 가중치에서 acc8로 미세조정(--acc 8 --init-from) -n promptmr_moe_acc8
#   ④ merge    : 두 전문가를 결합 체크포인트로 병합               -n promptmr_moe
#   ⑤ score    : recon_eval(리더보드) 채점 — 추론은 마스크로 자동 라우팅
#
# 추론 라우팅은 test_part.load_model이 결합 ckpt를 감지해 처리(파일명·폴더명 불필요, 마스크 frac로 4/8 판별).
# 슬라이스당 전문가 1개만 실행 → 추론시간 페널티 없음.
#
# 실행(단계별로 나눠 돌릴 수 있음):
#   STAGE=pretrain bash shell_scripting/shell20.sh     # ①만
#   STAGE=ft4      bash shell_scripting/shell20.sh     # ②만 (①의 best_model.pt 필요)
#   STAGE=ft8      bash shell_scripting/shell20.sh     # ③만
#   STAGE=merge    bash shell_scripting/shell20.sh     # ④+⑤ (GPU는 ⑤만 씀)
#   STAGE=all      bash shell_scripting/shell20.sh     # ①→⑤ 순차(실패 시 중단)  ← 기본값
#   SMOKE=1 STAGE=all bash shell_scripting/shell20.sh  # 각 학습 120 iter 스모크(채점 생략)
# ============================================================
set -uo pipefail
export PYTHONUNBUFFERED=1

# --- repo 루트로 이동 ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

STAGE="${STAGE:-all}"
SMOKE="${SMOKE:-0}"

# --- ⚠️ 데이터 경로: 최신 슬롯(shell18/19)과 동일하게 /workspace 기준. 서버가 /root면 여기만 바꾼다. ---
DATA_ROOT="${DATA_ROOT:-/workspace/Data}"
LEADERBOARD="${LEADERBOARD:-/root/Data/leaderboard}"

# --- 로그 ---
SLOT="$(basename "$0" .sh)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="shell_scripting/logs"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SLOT}_${STAGE}_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

# --- net_name / 하이퍼파라미터 ---
NET_SHARED="promptmr_moe_shared"
NET_ACC4="promptmr_moe_acc4"
NET_ACC8="promptmr_moe_acc8"
NET_MOE="promptmr_moe"
SHARED_BEST="../result/${NET_SHARED}/checkpoints/best_model.pt"

# 아키텍처는 3단계 모두 동일해야 한다(--init-from 로드·병합 config 정합). shell18(run5)과 동일 config.
ARCH="--model promptmr --readout-crop --num-cascades 10 --n-feat0 48 --sens-n-feat0 24 --n-history 3 --use-checkpoint"
COMMON="-b 1 -r 100 --num-workers 4 --fg-loss --area-weight --grad-clip 0.1 --accum-steps 2"

PRE_LR="2e-4"; PRE_EPOCHS=10          # 공유 사전학습 (shell18과 동일)
FT_LR="5e-5";  FT_EPOCHS=3            # 미세조정: LR 낮추고 소수 epoch (튜닝 대상)
THRESHOLD="0.1875"                    # 라우터 임계값 frac(<임계 → acc8)

if [ "$SMOKE" = "1" ]; then
    SMOKE_EXTRA="--max-iters 120"; PRE_EPOCHS=1; FT_EPOCHS=1
    NET_SHARED="${NET_SHARED}_smoke"; NET_ACC4="${NET_ACC4}_smoke"; NET_ACC8="${NET_ACC8}_smoke"; NET_MOE="${NET_MOE}_smoke"
    SHARED_BEST="../result/${NET_SHARED}/checkpoints/best_model.pt"
else
    SMOKE_EXTRA=""
fi

echo "=========================================================="
echo "[START] $SLOT  STAGE=$STAGE  SMOKE=$SMOKE  $(date '+%F %T')"
echo "[CWD  ] $(pwd)   [DATA] $DATA_ROOT   [LOG] $LOG"
echo "=========================================================="

# --- 가드: GPU 비었는지(학습·채점 단계에서만) ---
gpu_guard() {
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo "[GPU ] used=${used}MiB"
    if [ -n "$used" ] && [ "$used" -ge 500 ]; then
        echo "[STOP] GPU가 사용 중(${used}MiB)이다. 빈 상태에서 실행할 것."; exit 2
    fi
}
python -c "import cv2" 2>/dev/null || { echo "[STOP] cv2 import 실패 — bash /root/restore_env.sh 후 재실행."; exit 2; }

# ---------- 단계 함수 ----------
do_pretrain() {
    gpu_guard
    echo "[① pretrain] $NET_SHARED (전체 데이터, --acc all)"
    set -x
    python train.py $ARCH $COMMON --acc all -l "$PRE_LR" -e "$PRE_EPOCHS" $SMOKE_EXTRA \
        -n "$NET_SHARED" --run-tag moe_shared \
        -t "$DATA_ROOT/train/" -v "$DATA_ROOT/val/"
    { local st=$?; set +x; } 2>/dev/null
    echo "[① pretrain] exit=$st"; return $st
}
do_ft() {  # $1=acc(4/8)  $2=net_name
    gpu_guard
    if [ ! -f "$SHARED_BEST" ]; then
        echo "[STOP] 공유 사전학습 결과가 없다: $SHARED_BEST (먼저 STAGE=pretrain)."; return 2
    fi
    echo "[② ft$1] $2 (--acc $1 --init-from $SHARED_BEST)"
    set -x
    python train.py $ARCH $COMMON --acc "$1" -l "$FT_LR" -e "$FT_EPOCHS" $SMOKE_EXTRA \
        --init-from "$SHARED_BEST" \
        -n "$2" --run-tag "moe_acc$1" \
        -t "$DATA_ROOT/train/" -v "$DATA_ROOT/val/"
    { local st=$?; set +x; } 2>/dev/null
    echo "[② ft$1] exit=$st"; return $st
}
do_merge() {
    echo "[④ merge] $NET_ACC4 + $NET_ACC8 → $NET_MOE"
    set -x
    python utils/model/merge_moe.py --acc4-net "$NET_ACC4" --acc8-net "$NET_ACC8" \
        --out-net "$NET_MOE" --threshold "$THRESHOLD"
    { local st=$?; set +x; } 2>/dev/null
    [ "$st" -ne 0 ] && { echo "[④ merge] 실패 exit=$st"; return $st; }
    if [ "$SMOKE" = "1" ]; then echo "[⑤ score] SMOKE — 채점 생략"; return 0; fi
    gpu_guard
    echo "[⑤ score] recon_eval -n $NET_MOE"
    mkdir -p "../result/$NET_MOE"
    set -x
    python recon_eval.py -n "$NET_MOE" -p "$LEADERBOARD" 2>&1 | tee "../result/$NET_MOE/recon_eval_moe.txt"
    { st=$?; set +x; } 2>/dev/null
    echo "[⑤ score] exit=$st  점수: ../result/$NET_MOE/recon_eval_moe.txt"; return $st
}

# ---------- STAGE 디스패치 ----------
case "$STAGE" in
    pretrain) do_pretrain; STATUS=$? ;;
    ft4)      do_ft 4 "$NET_ACC4"; STATUS=$? ;;
    ft8)      do_ft 8 "$NET_ACC8"; STATUS=$? ;;
    merge)    do_merge; STATUS=$? ;;
    all)
        do_pretrain && do_ft 4 "$NET_ACC4" && do_ft 8 "$NET_ACC8" && do_merge
        STATUS=$?
        ;;
    *) echo "[STOP] 알 수 없는 STAGE='$STAGE' (pretrain|ft4|ft8|merge|all)"; exit 2 ;;
esac

echo "=========================================================="
echo "[DONE ] $SLOT  STAGE=$STAGE  $(date '+%F %T')  exit=$STATUS"
echo "=========================================================="
exit "$STATUS"
