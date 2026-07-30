#!/usr/bin/env bash
# ============================================================
# GPU 작업 큐 슬롯 — PromptMR+ 실학습 재개 (promptmr_run1, epoch 6~).
# 목적: shell12.sh(5 epoch + 채점) 결과 확인 후, 같은 config + --resume 로 이어학습.
#   model.pt(마지막 저장 epoch=5) 에서 start_epoch=5 로 재개 → epoch 6부터 -e 값까지 진행.
#   shell12.sh 와 config 완전 동일 + --resume 만 추가(같은 config라야 resume 맞물림).
#   상수 LR(1e-6, --warmup-iters 미사용)이라 스케줄러 fast-forward 이슈 없음 → 깔끔히 이어짐.
#   학습 완주 후 recon_eval 로 재채점.
#   ✎ 총 epoch 수는 아래 -e 값으로 조절(현재 10 = epoch 6~9 진행). 더/덜 돌리려면 -e 만 수정.
#   ⚠️ shell12(5 epoch) 완주 후 실행. GPU 빈 상태에서.
#
# 실행:  bash shell12_resume.sh
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

# --- 가드 ①: GPU 비었는지 ---
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
echo "[GPU ] used=${USED}MiB"
if [ -n "$USED" ] && [ "$USED" -ge 500 ]; then
    echo "[STOP] GPU가 사용 중(${USED}MiB)이다. GPU가 빈 상태에서 실행할 것."
    exit 2
fi
# --- 가드 ②: 재개할 체크포인트 있는지 ---
if [ ! -f ../result/promptmr_run1/checkpoints/model.pt ]; then
    echo "[STOP] ../result/promptmr_run1/checkpoints/model.pt 없음 — shell12(5 epoch) 먼저 완주할 것."
    exit 2
fi

python -c "import cv2" 2>/dev/null || { echo "[STOP] cv2 import 실패 — bash /root/restore_env.sh 로 환경 복구 후 실행할 것."; exit 2; }

# ==========================================================
#  ↓↓↓  1) 이어학습 (shell12 와 동일 config + --resume, -e 10)  ↓↓↓
# ==========================================================
set -x
python train.py --model promptmr --readout-crop --coil-compress --n-virtual-coils 8 \
    --num-cascades 4 --n-feat0 48 --sens-n-feat0 24 --n-history 3 --use-checkpoint \
    -b 1 -l 1e-6 -e 10 -r 100 --num-workers 4 \
    --fg-loss --area-weight --grad-clip 0.1 --accum-steps 2 \
    -n promptmr_run1 --run-tag pmr_run1 \
    -t /root/Data/train/ -v /root/Data/val/ \
    --resume
{ TRAIN_STATUS=$?; set +x; } 2>/dev/null
echo "[TRAIN] promptmr_run1 재개학습 종료   exit=$TRAIN_STATUS"

# ==========================================================
#  ↓↓↓  2) recon_eval 재채점 (학습 성공 시)  ↓↓↓
# ==========================================================
if [ "$TRAIN_STATUS" -eq 0 ]; then
    echo "[RECON] promptmr_run1 재채점..."
    set -x
    python recon_eval.py -n promptmr_run1 -p /root/Data/leaderboard 2>&1 \
        | tee ../result/promptmr_run1/recon_eval_run1.txt
    { STATUS=$?; set +x; } 2>/dev/null
else
    echo "[RECON] 학습이 exit=$TRAIN_STATUS 로 끝나 채점 건너뜀."
    STATUS=$TRAIN_STATUS
fi

echo "=========================================================="
echo "[DONE ] $SLOT   $(date '+%F %T')   train=$TRAIN_STATUS recon/exit=$STATUS"
echo "[저장] 점수: ../result/promptmr_run1/recon_eval_run1.txt (+ 이 슬롯 로그)."
echo "=========================================================="
exit "$STATUS"
