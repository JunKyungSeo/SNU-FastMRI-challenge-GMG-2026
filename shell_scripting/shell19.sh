#!/usr/bin/env bash
# ============================================================
# GPU 작업 큐 슬롯 — PromptMR+ K-Fold Cross Validation (shell19.sh)
#
# 목적: 5-Fold Cross Validation (K=5) 지원. Volume (h5 파일) 단위 분할.
#   환경변수 FOLD (0~4) 로 학습/검증 대상을 지정.
#   기본값: FOLD=0, K_FOLD=5.
#
# 실행 방법:
#   [SMOKE 테스트]
#     FOLD=0 SMOKE=1 bash shell_scripting/shell19.sh
#
#   [본 학습]
#     FOLD=0 bash shell_scripting/shell19.sh
#     FOLD=1 bash shell_scripting/shell19.sh
#     ...
#     FOLD=4 bash shell_scripting/shell19.sh
# ============================================================
set -uo pipefail
export PYTHONUNBUFFERED=1

# --- repo 루트로 이동 ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# --- K-Fold 환경변수 설정 ---
FOLD="${FOLD:-0}"
K_FOLD="${K_FOLD:-5}"

# --- 로그 파일 ---
SLOT="$(basename "$0" .sh)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="shell_scripting/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SLOT}_f${FOLD}_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

SMOKE="${SMOKE:-0}"
if [ "$SMOKE" = "1" ]; then
    NET="promptmr_kfold_f${FOLD}_smoke"; TAG="pmr_kf_f${FOLD}_smoke"; EXTRA="--max-iters 120"; EPOCHS=1
else
    NET="promptmr_kfold_f${FOLD}";       TAG="pmr_kf_f${FOLD}";       EXTRA="";                EPOCHS=10
fi

echo "=========================================================="
echo "[START] $SLOT (Fold $FOLD / $K_FOLD)   $(date '+%F %T')   SMOKE=$SMOKE  NET=$NET"
echo "[CWD  ] $(pwd)"
echo "[LOG  ] $LOG"
echo "[CONF ] full coil · nc10 · nf48 · sens24 · lr 2e-4 · K_FOLD=$K_FOLD · FOLD=$FOLD · ${EPOCHS} epoch"
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
#  ↓↓↓  1) PromptMR+ K-Fold 학습  ↓↓↓
# ==========================================================
set -x
python train.py --model promptmr --readout-crop \
    --num-cascades 10 --n-feat0 48 --sens-n-feat0 24 --n-history 3 \
    -b 1 -l 2e-4 -e "$EPOCHS" -r 100 --num-workers 4 --use-checkpoint  \
    --fg-loss --area-weight --grad-clip 0.1 --accum-steps 2 \
    --k-fold "$K_FOLD" --val-fold "$FOLD" $EXTRA \
    -n "$NET" --run-tag "$TAG" \
    -t /workspace/Data/train/ -v /workspace/Data/val/
{ TRAIN_STATUS=$?; set +x; } 2>/dev/null
echo "[TRAIN] $NET (Fold $FOLD) 종료   exit=$TRAIN_STATUS"

if [ "$SMOKE" = "1" ]; then
    echo "=========================================================="
    echo "[SMOKE] 채점 생략 (Fold $FOLD)."
    echo "        (1) OOM 없이 exit 0 인가"
    echo "        (2) iter 0 대비 Loss 가 내려가는가"
    echo "        둘 다 OK면  FOLD=$FOLD bash shell_scripting/shell19.sh  로 본 학습."
    echo "=========================================================="
    exit "$TRAIN_STATUS"
fi

# ==========================================================
#  ↓↓↓  2) 복원 + 채점 (recon_eval)  ↓↓↓
# ==========================================================
if [ "$TRAIN_STATUS" -eq 0 ]; then
    mkdir -p "../result/$NET"
    echo "[RECON] $NET (Fold $FOLD) 복원·채점..."
    set -x
    python recon_eval.py -n "$NET" -p /root/Data/leaderboard 2>&1 \
        | tee "../result/$NET/recon_eval_kfold_f${FOLD}.txt"
    { STATUS=$?; set +x; } 2>/dev/null
else
    echo "[RECON] 학습이 exit=$TRAIN_STATUS 로 끝나 복원·채점 건너뜀."
    STATUS=$TRAIN_STATUS
fi

echo "=========================================================="
echo "[DONE ] $SLOT (Fold $FOLD)   $(date '+%F %T')   train=$TRAIN_STATUS recon/exit=$STATUS"
echo "[저장] 점수: ../result/$NET/recon_eval_kfold_f${FOLD}.txt"
echo "=========================================================="
exit "$STATUS"
