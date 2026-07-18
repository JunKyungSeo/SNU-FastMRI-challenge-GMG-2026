#!/usr/bin/env bash
# ============================================================
# GPU 작업 큐 슬롯  —  GPU를 쓰는 명령은 이 폴더 스크립트에서만 실행한다.
# 목적: run13 = run11 + UC.  run11(--fg-loss) 설정에 --uc 하나만 얹음 (bbox 안 씀).
#       = run11([16,32]×4·readout-crop·warmup200·LR 2.4e-4·--fg-loss) + --uc.
#       → run11 대비 **clean single-lever UC A/B** (fg-loss 유지, UC 효과만 격리).
#       run12(shell3)는 bbox+UC 묶음이라 UC 단독 효과를 못 봄 → 이 run13이 UC 순효과 측정.
#       ⚠️ run13도 UC를 쓰므로 shell2 smoke(UC 경로) 통과 후에 돌릴 것. UC 전용이라 더는 "smoke 실패 시
#          fallback"이 아니다(UC가 깨졌으면 이것도 깨짐). smoke 통과 시 run12와 별개로 돌리는 UC 격리 arm.
#
# 실행:  bash shell4.sh                          (단독, shell2 smoke 통과 후)
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

# --- cv2 보장: fg-loss 경로가 metrics(cv2) 를 import. headless 서버. ---
python -c "import cv2" 2>/dev/null || pip install -q opencv-python-headless

# ==========================================================
#  ↓↓↓  실제 GPU 명령 = run11 + UC  (run11 config + --uc)  ↓↓↓
# ==========================================================
set -x
python train.py --model sdum --num-cascades 4 --channels 16 32 \
    -b 1 -l 2.4e-4 -e 10 -r 100 --num-workers 4 \
    --readout-crop --warmup-iters 200 --fg-loss --uc \
    -n sdum_run13 --run-tag run13 -t /root/Data/train/ -v /root/Data/val/

# ==========================================================
#  ↑↑↑  실제 GPU 명령 끝 (STATUS 는 마지막 명령의 종료코드)  ↑↑↑
# ==========================================================
{ STATUS=$?; set +x; } 2>/dev/null

echo "=========================================================="
echo "[DONE ] $SLOT   $(date '+%F %T')   exit=$STATUS"
echo "=========================================================="
exit "$STATUS"
