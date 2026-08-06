#!/usr/bin/env bash
# ============================================================
# 목적: shell31_acc8more 를 MORE_EPOCH 만큼 이어서 학습 (--resume)          [shell31_acc8more_resume.sh]
#       상수 LR 이라 check_resume_compat 이 -e 연장을 허용한다.
#
# 실행:  bash shell_scripting/shell31_acc8more_resume.sh              # acc8 · 5 epoch 더
#        MORE_EPOCH=6 bash shell_scripting/shell31_acc8more_resume.sh
#        ACC=4 LR=1e-5 bash shell_scripting/shell31_acc8more_resume.sh
# ============================================================
set -uo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

DATA_ROOT="$([ -d /workspace/Data ] && echo /workspace/Data || echo /root/Data)"
ACC="${ACC:-8}"; LR="${LR:-5e-6}"; MORE="${MORE_EPOCH:-5}"
NET="promptmrp_0801_acc${ACC}_ext"

[ -e "../result/$NET/checkpoints/model.pt" ] || [ -e "../result/$NET/checkpoints/latest.pt" ] || {
    echo "[STOP] 이어갈 체크포인트가 없다: ../result/$NET/checkpoints"; exit 2; }

# 끝난 epoch 수 = val_loss_log 행 수. 345MB 체크포인트를 열지 않아도 된다.
DONE=$(python3 -c "import numpy,sys; print(len(numpy.load(sys.argv[1])))" \
       "../result/$NET/val_loss_log_${NET}.npy" 2>/dev/null) || DONE=0
TOTAL=$((DONE + MORE))

SLOT="$(basename "$0" .sh)"
LOG_DIR="shell_scripting/logs"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SLOT}_acc${ACC}_$(date +%Y%m%d_%H%M%S).log"

echo "[START] $SLOT   $(date '+%F %T')   NET=$NET  lr=$LR"
echo "[EPOCH] 완료 $DONE + 추가 $MORE = 총 $TOTAL"
echo "[LOG  ] $LOG"

python train.py --model promptmr --readout-crop \
    --num-cascades 10 --n-feat0 48 --sens-n-feat0 24 --n-history 3 \
    -b 1 -r 50 --num-workers 4 --use-checkpoint \
    --fg-loss --area-weight --grad-clip 0.1 --accum-steps 2 \
    --k-fold 5 --val-fold 2 --kfold-seed 430 \
    --aug-on --aug-schedule constant --aug-max-prob 0.5 --aug-delay 0 \
    --l1-weight 0.1 --early-stop-patience 3 \
    -t "$DATA_ROOT/train/" -v "$DATA_ROOT/val/" \
    --acc all --remask "$ACC" --mask-aug offset --max-iters 2179 \
    -l "$LR" -e "$TOTAL" --resume \
    -n "$NET" --run-tag "$NET" 2>&1 | tee "$LOG"
ST=${PIPESTATUS[0]}

echo "[DONE ] $SLOT   $(date '+%F %T')   총 ${TOTAL}ep  exit=$ST"
bash "$SCRIPT_DIR/notify.sh" "$SLOT acc$ACC 종료 (총 ${TOTAL}ep, exit=$ST)" \
    "$NET · lr=$LR · 완료 $DONE + 추가 $MORE = 총 $TOTAL · $(TZ=Asia/Seoul date '+%F %T') KST"
exit "$ST"
