#!/usr/bin/env bash
# ============================================================
# GPU 작업 큐 슬롯  —  GPU를 쓰는 명령은 이 폴더 스크립트에서만 실행한다.
# 목적: run7·run8 공식 채점 — recon_eval.py로 leaderboard에서 SSIM_full + SSIM_bbox + 추론시간 3축 산출.
#       run7 checkpoint는 태그충돌로 -n sdum_run5 에 저장됨(주의). run8은 -n sdum_run8.
#       recon_eval는 checkpoint의 저장 args로 SDUM config 자동 복원 → --cascade/--chans 불필요.
#       ⚠️ metrics.py가 cv2 import → 이 서버는 opencv-python-headless 필요(libGL 없음). 아래서 보장.
#
# 실행:  bash shellN.sh                          (단독)
#        bash shell1.sh ; bash shell2.sh ; ...   (큐로 이어서, run8 뒤에)
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

DATA=/root/Data/leaderboard

# --- run7 (checkpoint: sdum_run5, 태그충돌 잔재) ---
if [ -f "../result/sdum_run5/checkpoints/best_model.pt" ]; then
    echo "===== recon_eval: run7 (sdum_run5) ====="
    python recon_eval.py -n sdum_run10 -p "$DATA"
else
    echo "[skip] run7 checkpoint(../result/sdum_run5/checkpoints/best_model.pt) 없음"
fi

# ==========================================================
#  ↑↑↑  실제 GPU 명령 끝 (STATUS 는 마지막 명령의 종료코드)  ↑↑↑
# ==========================================================
{ STATUS=$?; set +x; } 2>/dev/null

echo "=========================================================="
echo "[DONE ] $SLOT   $(date '+%F %T')   exit=$STATUS"
echo "=========================================================="
exit "$STATUS"
