#!/usr/bin/env bash
# ============================================================
# GPU 작업 큐 슬롯 — PromptMR+ 실학습 2차 (promptmr_run2): 코일 전부 사용.
# 목적: coil compression 없이 물리코일 15개를 그대로 쓰고, 메모리는 구조적 절감으로 확보한
#       config 로 scratch 학습 → recon_eval 채점. promptmr_run1(코일 8 압축, nc=4)의 대조군.
#
#   run1 대비 바뀐 것:
#     1) --coil-compress 제거        : 가상코일 8 → 물리코일 15 전부.
#     2) --num-cascades 4 → 10       : SPEC(strategy/promptmr_plus.md) 값으로 복귀.
#        cascade 는 실측상 개당 ~0.035GB 로 거의 공짜였다 — run1 의 nc=4 는 메모리 때문에
#        깎을 이유가 별로 없었고, 그만큼 품질을 손해봤을 가능성이 크다.
#     그 외(readout-crop·손실·optimizer·LR·accum)는 run1 과 동일하게 두어 비교 가능하게 유지.
#
#   ⚠️ --num-cascades 는 shell13.sh(탐침) 에서 OK 로 통과한 최대값으로 맞출 것.
#      탐침을 안 돌렸으면 이 슬롯을 먼저 돌리지 말 것 — OOM 으로 몇 시간 날린다.
#   ⚠️ run1 과 손실값을 직접 비교하지 말 것. global_scale 정규화가 코일 축 합으로 계산되는데
#      압축 유무에 따라 그 축이 달라져서 입력 스케일 자체가 다르다. 비교는 recon_eval 점수로.
#   ⚠️ scratch 런이다. run1 체크포인트에서 --resume 은 불가능(코일 config 불일치로 거부됨).
#
# 실행:  bash shell14.sh   (GPU 비었을 때, shell13.sh 확인 후)
# ============================================================
set -uo pipefail
export PYTHONUNBUFFERED=1

# --- repo 루트로 이동 ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# --- 로그 파일 ---
SLOT="$(basename "$0" .sh)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="shell_scripting/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SLOT}_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=========================================================="
echo "[START] $SLOT   $(date '+%F %T')"
echo "[CWD  ] $(pwd)"
echo "[LOG  ] $LOG"
echo "=========================================================="

# --- 가드: GPU 비었는지 (used < 500MiB) ---
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
echo "[GPU ] used=${USED}MiB"
if [ -n "$USED" ] && [ "$USED" -ge 500 ]; then
    echo "[STOP] GPU가 사용 중(${USED}MiB)이다. 실학습은 GPU가 빈 상태에서 실행할 것."
    exit 2
fi

# --- cv2 보장 (fg-loss·recon_eval metrics가 cv2 import) ---
python -c "import cv2" 2>/dev/null || { echo "[STOP] cv2 import 실패 — bash /root/restore_env.sh 로 환경 복구 후 실행할 것."; exit 2; }

# ==========================================================
#  ↓↓↓  1) PromptMR+ 실학습 (코일 전부, scratch 5 epoch)  ↓↓↓
# ==========================================================
set -x
python train.py --model promptmr --readout-crop \
    --num-cascades 10 --n-feat0 48 --sens-n-feat0 24 --n-history 3 --use-checkpoint \
    -b 1 -l 1e-6 -e 5 -r 100 --num-workers 4 \
    --fg-loss --area-weight --grad-clip 0.1 --accum-steps 2 \
    -n promptmr_run2 --run-tag pmr_run2_fullcoil \
    -t /root/Data/train/ -v /root/Data/val/
{ TRAIN_STATUS=$?; set +x; } 2>/dev/null
echo "[TRAIN] promptmr_run2 (코일 전부) 5 epoch 종료   exit=$TRAIN_STATUS"
echo "[MEM  ] epoch 별 peak 은 위 로그의 '[mem] epoch N peak allocated' 줄 참고."

# ==========================================================
#  ↓↓↓  2) recon_eval 채점 (학습 성공 시에만)  ↓↓↓
# ==========================================================
if [ "$TRAIN_STATUS" -eq 0 ]; then
    echo "[RECON] promptmr_run2 채점 (recon_eval → leaderboard SSIM_full+bbox+시간)..."
    set -x
    python recon_eval.py -n promptmr_run2 -p /root/Data/leaderboard 2>&1 \
        | tee ../result/promptmr_run2/recon_eval_run2.txt
    { STATUS=$?; set +x; } 2>/dev/null
else
    echo "[RECON] 학습이 exit=$TRAIN_STATUS 로 끝나 채점 건너뜀."
    STATUS=$TRAIN_STATUS
fi

echo "=========================================================="
echo "[DONE ] $SLOT   $(date '+%F %T')   train=$TRAIN_STATUS recon/exit=$STATUS"
echo "[저장] 점수: ../result/promptmr_run2/recon_eval_run2.txt (+ 이 슬롯 로그)."
echo "[비교] run1(코일 8 압축·nc4) 점수: ../result/promptmr_run1/recon_eval_run1.txt"
echo "       추론시간도 같이 볼 것 — 코일 15개면 sens 계산이 늘어 시간 점수에 불리하다."
echo "=========================================================="
exit "$STATUS"
