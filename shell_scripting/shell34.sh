#!/usr/bin/env bash
# 목적: 최종 run과 같은 설정으로 300 iter만 돌려 grad norm 분포를 잰다 (clip 0.1 이 실제로 발동하는지)
set -uo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

DATA_ROOT="$([ -d /workspace/Data ] && echo /workspace/Data || echo /root/Data)"
NET="gnorm_probe"
SLOT="$(basename "$0" .sh)"
LOG_DIR="shell_scripting/logs"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SLOT}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "[START] $SLOT   $(date '+%F %T')   NET=$NET"
echo "[LOG  ] $LOG"

set -x
python train.py --model promptmr --readout-crop \
    --num-cascades 10 --n-feat0 48 --sens-n-feat0 24 --n-history 3 \
    -b 1 -r 50 --num-workers 4 --use-checkpoint \
    --fg-loss --area-weight --grad-clip 0.1 --accum-steps 2 \
    --k-fold 5 --val-fold 2 --kfold-seed 430 \
    --aug-on --aug-schedule constant --aug-max-prob 0.5 --aug-delay 0 \
    --l1-weight 0.1 \
    --acc all -e 1 -l 2e-4 --max-iters 300 \
    -t "$DATA_ROOT/train/" -v "$DATA_ROOT/val/" \
    -n "$NET" --run-tag "$NET"
{ ST=$?; set +x; } 2>/dev/null

echo "[DONE ] $SLOT   $(date '+%F %T')   exit=$ST"
echo "[볼 것] 로그의 [gnorm] 줄 — med / p90 / max / 발동%"
exit "$ST"
