#!/usr/bin/env bash
# ============================================================
# GPU 작업 큐 슬롯  —  GPU를 쓰는 명령은 이 폴더 스크립트에서만 실행한다.
# 목적: PromptMR+ run1 공식 채점 — recon_eval.py 로 leaderboard 에서
#       SSIM_full + SSIM_bbox + 추론시간 3축 산출.
#       ⚠️ load_model 이 best_model.pt 의 저장 args(saved.model=='promptmr')로
#          PromptMR config 자동 복원 → --model/--num-cascades 등 불필요.
#       ⚠️ metrics.py 가 cv2 import → libGL 없는 서버는 opencv-python-headless 필요(아래 보장).
#       ⚠️ volume forward 가 60s 넘으면 recon_eval 가 경고(시간도 점수 축) — 로그 확인.
#
# 실행:  bash shell13.sh                         (단독, shell12 학습 후)
# ============================================================
set -uo pipefail
export PYTHONUNBUFFERED=1   # tee 파이프로 가도 python 출력을 실시간으로 (버퍼링 방지)

# --- repo 루트로 이동 (recon_eval.py 등은 repo 루트 기준으로 돈다) ---
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

echo "=========================================================="
echo "[START] $SLOT   $(date '+%F %T')"
echo "[CWD  ] $(pwd)"
echo "[LOG  ] $LOG"
echo "=========================================================="

# --- cv2 보장 (metrics.py import용, libGL 없는 서버 → headless). 이미 되면 no-op ---
python -c "import cv2" 2>/dev/null || pip install -q opencv-python-headless

# ==========================================================
#  ↓↓↓  실제 GPU 명령 (recon_eval)  ↓↓↓
# ==========================================================
set -x
if [ -f "../result/promptmr_run1/checkpoints/best_model.pt" ]; then
    python recon_eval.py -n hspromptmr_run1 -p /root/Data/leaderboard
else
    echo "[skip] hspromptmr_run1 checkpoint(../result/promptmr_run1/checkpoints/best_model.pt) 없음 — shell12 먼저."
fi

# ==========================================================
#  ↑↑↑  실제 GPU 명령 끝 (STATUS 는 마지막 명령의 종료코드)  ↑↑↑
# ==========================================================
{ STATUS=$?; set +x; } 2>/dev/null

echo "=========================================================="
echo "[DONE ] $SLOT   $(date '+%F %T')   exit=$STATUS"
echo "[다음] 위 3축(SSIM_full/bbox/time)을 training_anal 에 VarNet(run11 등) 대비 기록."
echo "=========================================================="
exit "$STATUS"
