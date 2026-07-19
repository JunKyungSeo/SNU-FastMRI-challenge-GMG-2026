#!/usr/bin/env bash
# ============================================================
# GPU 작업 큐 슬롯  —  GPU를 쓰는 명령은 이 폴더 스크립트에서만 실행한다.
# 목적: PromptMR+ run1 smoke — tiny config(full "+" ON) 1 epoch 학습.
#       확인: OOM 없음 · train loss 감소 · ../result/promptmr_run1/checkpoints/
#            {model,best_model}.pt 저장. (그 뒤 shell13 로 채점.)
#       resume 확인이 필요하면 이 슬롯을 --resume 붙여 재실행(같은 config).
#       ⚠️ 1 epoch 이 너무 길면(unrolled+checkpoint+FP32) 아래 명령에
#          `--max-iters 300 -r 50` 를 추가해 단축(그 경우에도 run1 유지).
#
# 실행:  bash shell12.sh                         (단독, shell11 probe 통과 후)
# ============================================================
set -uo pipefail
export PYTHONUNBUFFERED=1   # tee 파이프로 가도 python 출력을 실시간으로 (버퍼링 방지)

# --- repo 루트로 이동 (train.py 등은 repo 루트 기준으로 돈다) ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# --- 로그 파일 (이름에 timestamp) ---
SLOT="$(basename "$0" .sh)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="shell_scripting/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SLOT}_${TS}.log"

# stdout + stderr 를 터미널과 로그에 동시에 남긴다
exec > >(tee -a "$LOG") 2>&1

# --- 시작 고지 (어느 슬롯이 어디서 시작됐는지) ---
echo "=========================================================="
echo "[START] $SLOT   $(date '+%F %T')"
echo "[CWD  ] $(pwd)"
echo "[LOG  ] $LOG"
echo "=========================================================="

# ==========================================================
#  ↓↓↓  실제 GPU 명령 (PromptMR+ run1 smoke)  ↓↓↓
#  set -x 로 실행되는 명령을 로그에 그대로 남긴다.
# ==========================================================
set -x
python train.py --model promptmr --num-cascades 6 --n-feat0 16 --channels 24 32 40 \
    --readout-crop -b 1 -l 2e-4 -e 1 -r 100 --num-workers 4 \
    --run-tag run1 -n promptmr_run1 -t /root/Data/train/ -v /root/Data/val/

# ==========================================================
#  ↑↑↑  실제 GPU 명령 끝 (STATUS 는 마지막 명령의 종료코드)  ↑↑↑
# ==========================================================
{ STATUS=$?; set +x; } 2>/dev/null

echo "=========================================================="
echo "[DONE ] $SLOT   $(date '+%F %T')   exit=$STATUS"
echo "[확인] train loss 감소 · best_model.pt 저장."
echo "[다음] shell13 로 promptmr_run1 채점(3축) → training_anal 에 VarNet 대비 기록."
echo "=========================================================="
exit "$STATUS"
