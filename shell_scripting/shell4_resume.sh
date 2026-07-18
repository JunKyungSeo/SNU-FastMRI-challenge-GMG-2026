#!/usr/bin/env bash
# ============================================================
# GPU 작업 큐 슬롯 (자동 재개용) — run13(shell4.sh)을 epoch1부터 이어학습.
# shell4.sh 와 **완전히 동일한 config** + --resume 만 추가.
#   → model.pt(마지막 저장 epoch=1)에서 start_epoch=1 로 재개, epoch 1~9 진행.
# recon(중간 self-scoring) 을 위해 학습을 잠시 죽였다가 다시 붙이는 자동 흐름에서
# AUTO 오케스트레이터가 이 슬롯을 detached 로 실행한다. 수동 실행도 동일하게 안전.
# ============================================================
set -uo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

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

python -c "import cv2" 2>/dev/null || pip install -q opencv-python-headless

# ==========================================================
#  ↓↓↓  실제 GPU 명령 = shell4.sh 와 동일 config + --resume  ↓↓↓
# ==========================================================
set -x
python train.py --model sdum --num-cascades 4 --channels 16 32 \
    -b 1 -l 2.4e-4 -e 10 -r 100 --num-workers 4 \
    --readout-crop --warmup-iters 200 --fg-loss --uc \
    -n sdum_run13 --run-tag run13 -t /root/Data/train/ -v /root/Data/val/ \
    --resume

# ==========================================================
#  ↑↑↑  실제 GPU 명령 끝 (STATUS 는 마지막 명령의 종료코드)  ↑↑↑
# ==========================================================
{ STATUS=$?; set +x; } 2>/dev/null

echo "=========================================================="
echo "[DONE ] $SLOT   $(date '+%F %T')   exit=$STATUS"
echo "=========================================================="
exit "$STATUS"
