#!/usr/bin/env bash
# ============================================================
# 목적: MoE(acc4/acc8 분리)가 이득인지 층화 k-fold hold-out에서 판정한다        [shell24.sh]
# run tag : promptmr_moe_f<FOLD>_{shared,acc4,acc8} → 병합본 promptmr_moe_f<FOLD>
#
#   ① pretrain  --acc all            → 공유 모델                (= 단일 모델 baseline 겸용)
#   ② ft4       --acc 4 --init-from ① → acc4 전문가
#   ③ ft8       --acc 8 --init-from ① → acc8 전문가
#   ④ merge     ②+③ → 결합 체크포인트 (추론 시 마스크로 자동 라우팅)
#   ⑤ score     leaderboard 가 있으면 recon_eval, 없으면 건너뜀
#
#   [이 슬롯이 shell20과 다른 점] shell20은 고정 분할(-t train/ -v val/)이었다. 여기는
#     층화 k-fold를 쓴다 — 세 단계가 **같은 hold-out**을 본다는 게 이 실험의 전제다.
#     층화(2026-07-26) 이전 코드였다면 --acc 를 줄 때마다 fold 경계가 움직여서
#     ②③이 ①의 hold-out 볼륨으로 학습했다(실측 40개 중 30개). 지금은 안 그렇다.
#
#   [판정 방법] 셋 다 같은 hold-out 위에 있으므로 val 을 나란히 읽을 수 있다.
#     ①의 val 은 fold 전체(acc4+acc8) 평균이라 ②③(각 acc 절반)과 직접 비교되지 않는다.
#     → **판정의 정본은 채점이다.** recon_eval 이 acc4/acc8 을 따로 찍어주므로
#        ①(단일)과 ④(MoE)를 각각 채점해 SSIM_full·SSIM_bbox·시간 3축으로 비교한다.
#        val 은 학습이 살아있는지 보는 계기판으로만 쓴다.
#
#   [미세조정 하이퍼] lr 은 pretrain 과 같은 2e-4, 5 epoch. shell20의 5e-5×3ep 은
#     3 epoch 내내 소수 4째 자리만 흔들려(0.07694→0.07690→0.07694) 미세조정을 했다고
#     말할 수 없는 상태였다. lr 이 낮았던 건지 이미 수렴인지 가르려고 lr 을 올렸다.
#
#   ⚠️ 실행 전 서버에서 층화 테스트를 먼저 통과시킬 것 (31시간 태우고 분할이 틀리면 손해가 크다):
#        python -m pytest scripts/tests/test_kfold_stratify.py -v
#
# 실행:
#   SMOKE=1 STAGE=all bash shell_scripting/shell24.sh   # 각 단계 120 iter (먼저 이것)
#   STAGE=pretrain bash shell_scripting/shell24.sh      # ① 만 (~21h)
#   STAGE=ft4      bash shell_scripting/shell24.sh      # ② 만 (~5h, ① 필요)
#   STAGE=ft8      bash shell_scripting/shell24.sh      # ③ 만 (~5h, ① 필요)
#   STAGE=merge    bash shell_scripting/shell24.sh      # ④+⑤
#   STAGE=all      bash shell_scripting/shell24.sh      # ①→⑤ 순차 (기본값, ~31h)
#   FOLD=3 STAGE=all bash ...                           # 다른 fold
#   FORCE=1 ...                                         # 기존 결과를 덮어쓸 때만
# ============================================================
set -uo pipefail
export PYTHONUNBUFFERED=1

# --- repo 루트로 이동 ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# ===== [고정 인자] 런마다 안 바뀌는 것 — 여기 바꾸면 이후 모든 런의 baseline이 바뀐다 =====
#   ⚠️ 세 단계가 **완전히 같은 아키텍처**여야 한다. --init-from 이 가중치를 로드하고
#      merge_moe.py 가 두 전문가의 config 일치를 강제하므로(MUST_MATCH), 여기서 갈리면 병합이 거부된다.
BASE_ARGS=( --model promptmr --readout-crop
    --num-cascades 10 --n-feat0 48 --sens-n-feat0 24 --n-history 3
    -b 1 -l 2e-4 -r 100 --num-workers 4 --use-checkpoint
    --fg-loss --area-weight --grad-clip 0.1 --accum-steps 2
    -t /workspace/Data/train/ -v /workspace/Data/val/ )

# ===== [이번 run] 여기만 채운다 =====
FOLD="${FOLD:-2}"
K_FOLD="${K_FOLD:-5}"
KFOLD_SEED="${KFOLD_SEED:-430}"
STAGE="${STAGE:-all}"
SMOKE="${SMOKE:-0}"

PRE_EPOCHS=10        # 공유 사전학습 (shell21 단일 모델과 동일 길이 → baseline 으로도 읽힌다)
FT_EPOCHS=5          # 미세조정 (lr 은 BASE_ARGS 의 2e-4 를 그대로 물려받는다)

LEADERBOARD="${LEADERBOARD:-/root/Data/leaderboard}"

# --- net_name: fold 를 이름에 넣는다. 안 넣으면 옛 promptmr_moe_shared(고정분할)와 섞인다 ---
SUF=""; [ "$SMOKE" = "1" ] && SUF="_smoke"
NET_SHARED="promptmr_moe_f${FOLD}_shared${SUF}"
NET_ACC4="promptmr_moe_f${FOLD}_acc4${SUF}"
NET_ACC8="promptmr_moe_f${FOLD}_acc8${SUF}"
NET_MOE="promptmr_moe_f${FOLD}${SUF}"
SHARED_BEST="../result/${NET_SHARED}/checkpoints/best_model.pt"

if [ "$SMOKE" = "1" ]; then
    EXTRA="--max-iters 120"; PRE_EPOCHS=1; FT_EPOCHS=1
else
    EXTRA=""
fi

# --- 로그 ---
SLOT="$(basename "$0" .sh)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="shell_scripting/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SLOT}_f${FOLD}_${STAGE}_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=========================================================="
echo "[START] $SLOT (fold $FOLD / $K_FOLD)  STAGE=$STAGE  SMOKE=$SMOKE  $(date '+%F %T')"
echo "[CWD  ] $(pwd)"
echo "[LOG  ] $LOG"
echo "[BR   ] $(git branch --show-current 2>/dev/null) ($(git rev-parse --short HEAD 2>/dev/null))"
echo "[CONF ] nc10 · nf48 · sens24 · lr 2e-4 · seed=$KFOLD_SEED · pretrain ${PRE_EPOCHS}ep · ft ${FT_EPOCHS}ep"
echo "=========================================================="

# --- 가드: 층화 코드가 있는가 (이 슬롯의 전제. 없으면 --acc 가 fold 경계를 움직인다) ---
grep -q "_stratified_folds" utils/data/load_data.py 2>/dev/null || {
    echo "[STOP] utils/data/load_data.py 에 _stratified_folds 가 없다 — 층화 이전 코드다."
    echo "       이 상태로 돌리면 ②③이 ①의 hold-out 으로 학습한다. git pull 후 재실행할 것."
    exit 2; }

# --- 가드: cv2 (--fg-loss 의 foreground mask 가 cv2 를 import) ---
python -c "import cv2" 2>/dev/null || { echo "[STOP] cv2 import 실패 — bash /root/restore_env.sh 후 재실행."; exit 2; }

gpu_guard() {
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo "[GPU  ] used=${used}MiB"
    if [ -n "$used" ] && [ "$used" -ge 500 ]; then
        echo "[STOP] GPU가 사용 중(${used}MiB)이다. 빈 상태에서 실행할 것."; return 2
    fi
    return 0
}

# 중단된 학습이면 이어서, 처음이면 새로 — --resume 을 상시로 붙이면 첫 실행이
# FileNotFoundError 로 죽는다(load_resume_checkpoint). 그래서 체크포인트 유무로 정한다.
resume_flag() {   # $1=net_name → "--resume" 또는 ""
    [ "${FORCE:-0}" = "1" ] && return 0   # FORCE=1 은 "처음부터 다시" 라는 뜻이니 resume 을 붙이지 않는다
    if [ -e "../result/$1/checkpoints/model.pt" ] || [ -e "../result/$1/checkpoints/latest.pt" ]; then
        echo "--resume"
    fi
}

# 새로 시작하는데 결과가 이미 있으면(=resume 도 아닌데 덮어쓰기) 멈춘다.
overwrite_guard() {   # $1=net_name
    if [ -n "$(resume_flag "$1")" ] && [ "${FORCE:-0}" != "1" ]; then
        return 0   # resume 경로라 덮어쓰기가 아니다
    fi
    if [ -e "../result/$1/checkpoints/model.pt" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "[STOP] ../result/$1/checkpoints/model.pt 가 이미 있다 — 덮어쓰기: FORCE=1"; return 2
    fi
    return 0
}

# ==========================================================
#  ↓↓↓  단계 함수 — BASE_ARGS 뒤가 각 단계의 델타 전부다  ↓↓↓
# ==========================================================
do_pretrain() {
    gpu_guard || return 2
    overwrite_guard "$NET_SHARED" || return 2
    local R; R="$(resume_flag "$NET_SHARED")"
    echo "[① pretrain] $NET_SHARED  (--acc all, fold $FOLD hold-out) ${R:+[resume]}"
    set -x
    python train.py "${BASE_ARGS[@]}" \
        --k-fold "$K_FOLD" --val-fold "$FOLD" --kfold-seed "$KFOLD_SEED" \
        --acc all -e "$PRE_EPOCHS" $EXTRA $R \
        -n "$NET_SHARED" --run-tag "pmr_moe_f${FOLD}_shared"
    { local st=$?; set +x; } 2>/dev/null
    echo "[① pretrain] exit=$st"
    return $st
}

do_ft() {   # $1=acc(4|8)  $2=net_name
    gpu_guard || return 2
    if [ ! -f "$SHARED_BEST" ]; then
        echo "[STOP] 공유 사전학습 결과가 없다: $SHARED_BEST  (먼저 STAGE=pretrain)"; return 2
    fi
    overwrite_guard "$2" || return 2
    local R; R="$(resume_flag "$2")"
    # --resume 과 --init-from 은 배타적이고 resume 이 우선이다(train.py). 재개면 init-from 은 무시된다.
    echo "[② ft$1] $2  (--acc $1 --init-from shared) ${R:+[resume]}"
    set -x
    python train.py "${BASE_ARGS[@]}" \
        --k-fold "$K_FOLD" --val-fold "$FOLD" --kfold-seed "$KFOLD_SEED" \
        --acc "$1" -e "$FT_EPOCHS" $EXTRA $R \
        --init-from "$SHARED_BEST" \
        -n "$2" --run-tag "pmr_moe_f${FOLD}_acc$1"
    { local st=$?; set +x; } 2>/dev/null
    echo "[② ft$1] exit=$st"
    return $st
}

do_merge() {
    echo "[④ merge] $NET_ACC4 + $NET_ACC8 → $NET_MOE"
    # --threshold 를 안 준다 = merge_moe.py 의 DEFAULT_ACC_THRESHOLD(0.25) 사용.
    # 여기 숫자를 박아두면 나중에 기본값이 바뀔 때 조용히 어긋난다(옛 0.1875 사고 재발).
    set -x
    python scripts/merge_moe.py --acc4-net "$NET_ACC4" --acc8-net "$NET_ACC8" --out-net "$NET_MOE"
    { local st=$?; set +x; } 2>/dev/null
    [ "$st" -ne 0 ] && { echo "[④ merge] 실패 exit=$st"; return $st; }

    if [ "$SMOKE" = "1" ]; then echo "[⑤ score] SMOKE — 채점 생략"; return 0; fi
    if [ ! -d "$LEADERBOARD/acc4" ] || [ ! -d "$LEADERBOARD/acc8" ]; then
        echo "[⑤ score] leaderboard 없음($LEADERBOARD) — 채점 건너뜀."
        echo "          채점은 데이터가 있는 환경에서: 아래 두 net 을 각각 돌려 3축 비교."
        echo "            python recon_eval.py -n $NET_MOE    -p <leaderboard>   # MoE"
        echo "            python recon_eval.py -n $NET_SHARED -p <leaderboard>   # 단일 baseline"
        return 0
    fi
    gpu_guard || return 2
    local OUT_DIR="shell_scripting/results"; mkdir -p "$OUT_DIR"
    echo "[⑤ score] recon_eval — MoE 와 단일 baseline 을 둘 다 채점해야 비교가 된다."
    set -x
    python -u recon_eval.py -n "$NET_MOE" -p "$LEADERBOARD" \
        | tee "$OUT_DIR/recon_eval_${NET_MOE}_${TS}.txt"
    { local st=${PIPESTATUS[0]}; set +x; } 2>/dev/null
    [ "$st" -ne 0 ] && { echo "[⑤ score] MoE 채점 실패 exit=$st"; return $st; }
    set -x
    python -u recon_eval.py -n "$NET_SHARED" -p "$LEADERBOARD" \
        | tee "$OUT_DIR/recon_eval_${NET_SHARED}_${TS}.txt"
    { st=${PIPESTATUS[0]}; set +x; } 2>/dev/null
    echo "[⑤ score] exit=$st  점수 사본: $OUT_DIR/ (git add 해서 공유)"
    return $st
}

# ---------- STAGE 디스패치 ----------
case "$STAGE" in
    pretrain) do_pretrain; STATUS=$? ;;
    ft4)      do_ft 4 "$NET_ACC4"; STATUS=$? ;;
    ft8)      do_ft 8 "$NET_ACC8"; STATUS=$? ;;
    merge)    do_merge; STATUS=$? ;;
    all)      do_pretrain && do_ft 4 "$NET_ACC4" && do_ft 8 "$NET_ACC8" && do_merge
              STATUS=$? ;;
    *) echo "[STOP] 알 수 없는 STAGE='$STAGE' (pretrain|ft4|ft8|merge|all)"; exit 2 ;;
esac

echo "=========================================================="
echo "[DONE ] $SLOT (fold $FOLD)  STAGE=$STAGE  $(date '+%F %T')  exit=$STATUS"
if [ "$SMOKE" = "1" ]; then
    echo "[SMOKE] 확인할 것:"
    echo "        (1) 세 단계 다 OOM 없이 exit 0 인가"
    echo "        (2) [K-Fold] 로그의 val 구성이 **세 단계에서 일관적인가** — 이게 층화의 서명이다."
    echo "            ① --acc all 의 val (acc4 A · acc8 B) 에 대해 ② 는 acc4 A, ③ 은 acc8 B 여야 한다."
    echo "            어긋나면 층화가 안 먹은 것이니 본 학습을 돌리지 말 것."
    echo "        (3) merge 가 config 불일치로 거부되지 않았는가"
else
    echo "[저장] val loss: ../result/promptmr_moe_f${FOLD}_*/val_loss_log_pmr_moe_f${FOLD}_*.npy"
    echo "       병합본  : ../result/${NET_MOE}/checkpoints/best_model.pt"
    echo "[판정] 정본은 채점이다 — ${NET_MOE}(MoE) 와 ${NET_SHARED}(단일) 를 각각 채점해"
    echo "       SSIM_full·SSIM_bbox·시간 3축으로 비교할 것. val 은 계기판일 뿐이다."
fi
echo "=========================================================="
exit "$STATUS"
