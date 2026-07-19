#!/usr/bin/env bash
# ============================================================
# GPU 작업 큐 슬롯 — PromptMR+ 실학습 1차 (promptmr_run1).
# 목적: jkMRplus 백본(PromptMR+) 첫 실학습. 8GB-적합 config로 scratch 5 epoch → recon_eval 채점.
#   config 근거(2026-07-19 메모리 실측, strategy/promptmr_plus.md 하단):
#     SPEC nc10·full-640 은 8GB OOM. 메모리는 coil/sens 가 지배(cascade·nf 무영향).
#     ⚠️ 학습 k-space 폭(PE)이 가변(368~480). readout-crop은 높이만 자르고 폭은 native → 최악폭 W=480에선
#        15 coil sens 바닥이 nc=1 조차 OOM(7.57GB). 즉 model 축소로는 절대 안 들어감.
#     → 해법 = coil compression(15→8, 에너지 99.999% 보존·recon_slice에 이미 배선) + readout-crop + nc4.
#        최악폭 W=480 에서 실peak 6.06GB, 여유 1.94GB (실검증: --coil-compress로 200 iter 완주, OOM 없음).
#   흐름: 5 epoch 학습 → best_model.pt 로 recon_eval → 리더보드 SSIM_full+bbox+시간.
#   재개: 5 epoch 결과 확인 후, shell12_resume.sh 로 epoch 6부터 이어학습(동일 config + --resume).
#   ⚠️ scratch 런이라 --resume 없음. model.pt 는 매 epoch 저장되어 이후 resume 가능.
#   ⚠️ 실peak near-ceiling(여유 ~0.6GB) → 다른 GPU 작업과 동시 실행 금지(아래 가드).
#
# 실행:  bash shell12.sh   (GPU 비었을 때)
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
python -c "import cv2" 2>/dev/null || pip install -q opencv-python-headless

# ==========================================================
#  ↓↓↓  1) PromptMR+ 실학습 (scratch, 5 epoch)  ↓↓↓
# ==========================================================
set -x
python train.py --model promptmr --readout-crop --coil-compress --n-virtual-coils 8 \
    --num-cascades 4 --n-feat0 48 --sens-n-feat0 24 --n-history 3 --use-checkpoint \
    -b 1 -l 1e-6 -e 5 -r 100 --num-workers 4 \
    --fg-loss --area-weight --grad-clip 0.1 --accum-steps 2 \
    -n promptmr_run1 --run-tag pmr_run1 \
    -t /root/Data/train/ -v /root/Data/val/
{ TRAIN_STATUS=$?; set +x; } 2>/dev/null
echo "[TRAIN] promptmr_run1 5 epoch 종료   exit=$TRAIN_STATUS"

# ==========================================================
#  ↓↓↓  2) recon_eval 채점 (학습 성공 시에만; 학습 끝나 GPU 비어 있음)  ↓↓↓
# ==========================================================
if [ "$TRAIN_STATUS" -eq 0 ]; then
    echo "[RECON] promptmr_run1 채점 (recon_eval → leaderboard SSIM_full+bbox+시간)..."
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
echo "[다음] epoch 6부터 이어학습: bash shell_scripting/shell12_resume.sh"
echo "=========================================================="
exit "$STATUS"
