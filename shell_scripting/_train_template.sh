#!/usr/bin/env bash
# ============================================================
# 목적: <이번 run이 뭘 하는지 한 줄>
# run tag : 실행할 <net_name>
# 실행:  bash shell_scripting/shellN.sh
#        (스모크/본 학습 전환은 [이번 run] 블록에서 NET 줄 토글 — 스모크 먼저)
# ============================================================
set -uo pipefail
export PYTHONUNBUFFERED=1

# --- repo 루트로 이동 ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# ===== [고정 인자] 런마다 안 바뀌는 것 — 여기 바꾸면 이후 모든 런의 baseline이 바뀐다 =====
BASE_ARGS=( --model promptmr --readout-crop
    --num-cascades 10 --n-feat0 48 --sens-n-feat0 24 --n-history 3
    -b 1 -l 2e-4 -r 100 --num-workers 4 --use-checkpoint
    --fg-loss --area-weight --grad-clip 0.1 --accum-steps 2
    -t /workspace/Data/train/ -v /workspace/Data/val/ )

# ===== [이번 run] 여기만 채운다 =====
FOLD=5
K_FOLD=5
KFOLD_SEED=430

# 스모크 먼저(기본). 통과하면 smoke 줄을 주석 처리하고 본 학습 줄을 살린다.
NET="<이름>_f${FOLD}_smoke"; TAG="<태그>_f${FOLD}_smoke"; EPOCHS=1;  EXTRA="--max-iters 120"
# NET="<이름>_f${FOLD}";     TAG="<태그>_f${FOLD}";       EPOCHS=10; EXTRA=""

# --- 로그 ---
SLOT="$(basename "$0" .sh)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="shell_scripting/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SLOT}_f${FOLD}_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=========================================================="
echo "[START] $SLOT (fold $FOLD / $K_FOLD)   $(date '+%F %T')   NET=$NET"
echo "[CWD  ] $(pwd)"
echo "[LOG  ] $LOG"
echo "=========================================================="

# --- 가드: 기존 학습 결과를 말없이 덮어쓰지 않는다 (본 학습 1회 ≈ 21시간) ---
if [ -e "../result/$NET/checkpoints/model.pt" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[STOP] ../result/$NET/checkpoints/model.pt 가 이미 있다 — 이어서: --resume / 덮어쓰기: FORCE=1"
    exit 2
fi

# ==========================================================
#  ↓↓↓  실행부 — BASE_ARGS 뒤가 이번 run의 델타 전부다  ↓↓↓
# ==========================================================
set -x
python train.py "${BASE_ARGS[@]}" \
    --k-fold "$K_FOLD" --val-fold "$FOLD" --kfold-seed "$KFOLD_SEED" \
    -e "$EPOCHS" $EXTRA \
    -n "$NET" --run-tag "$TAG"
    # ↑ 이번 run의 델타 인자(예: --aug-on --aug-schedule exp ...)를 -n 줄 위에 추가
{ TRAIN_STATUS=$?; set +x; } 2>/dev/null

echo "=========================================================="
echo "[DONE ] $SLOT (fold $FOLD)   $(date '+%F %T')   exit=$TRAIN_STATUS"
echo "[저장] val loss: ../result/$NET/val_loss_log_${TAG}.npy"
echo "       가중치  : ../result/$NET/checkpoints/best_model.pt"
echo "=========================================================="
exit "$TRAIN_STATUS"
