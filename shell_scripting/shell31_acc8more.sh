#!/usr/bin/env bash
# ============================================================
# 목적: promptmrp_0801 전문가를 상수 LR로 몇 epoch 더 (--init-from, --resume 아님)  [shell31_acc8more.sh]
#
# 실행:  bash shell_scripting/shell31_acc8more.sh                  # acc8 · lr 5e-6 · 4ep
#        LR=1e-5 EPOCHS=6 bash shell_scripting/shell31_acc8more.sh
#        ACC=4 bash shell_scripting/shell31_acc8more.sh
# ============================================================
set -uo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

DATA_ROOT="$([ -d /workspace/Data ] && echo /workspace/Data || echo /root/Data)"
ACC="${ACC:-8}"; LR="${LR:-5e-6}"; EPOCHS="${EPOCHS:-4}"
SRC="../result/promptmrp_0801_acc${ACC}/checkpoints/best_model.pt"
NET="promptmrp_0801_acc${ACC}_ext"

[ -f "$SRC" ] || { echo "[STOP] 원본 체크포인트 없음: $SRC"; exit 2; }
[ -e "../result/$NET/checkpoints/model.pt" ] && [ "${FORCE:-0}" != "1" ] && {
    echo "[STOP] $NET 결과가 이미 있다 — 덮어쓰려면 FORCE=1"; exit 2; }

SLOT="$(basename "$0" .sh)"
LOG_DIR="shell_scripting/logs"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SLOT}_acc${ACC}_$(date +%Y%m%d_%H%M%S).log"

echo "[START] $SLOT   $(date '+%F %T')   NET=$NET  lr=$LR  ${EPOCHS}ep"
echo "[LOG  ] $LOG"

# LR 인자(--warmup-iters / --lr-decay-start-epoch)를 안 주면 build_scheduler 가 None → 상수 LR.
python train.py --model promptmr --readout-crop \
    --num-cascades 10 --n-feat0 48 --sens-n-feat0 24 --n-history 3 \
    -b 1 -r 50 --num-workers 4 --use-checkpoint \
    --fg-loss --area-weight --grad-clip 0.1 --accum-steps 2 \
    --k-fold 5 --val-fold 2 --kfold-seed 430 \
    --aug-on --aug-schedule constant --aug-max-prob 0.5 --aug-delay 0 \
    --l1-weight 0.1 --early-stop-patience 3 \
    -t "$DATA_ROOT/train/" -v "$DATA_ROOT/val/" \
    --acc all --remask "$ACC" --mask-aug offset --max-iters 2179 \
    -l "$LR" -e "$EPOCHS" \
    --init-from "$SRC" \
    -n "$NET" --run-tag "$NET" 2>&1 | tee "$LOG"
ST=${PIPESTATUS[0]}

echo "[DONE ] $SLOT   $(date '+%F %T')   exit=$ST"
bash "$SCRIPT_DIR/notify.sh" "$SLOT acc$ACC 종료 (exit=$ST)" \
    "$NET · lr=$LR · ${EPOCHS}ep · $(TZ=Asia/Seoul date '+%F %T') KST"
exit "$ST"
