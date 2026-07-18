#!/usr/bin/env bash
# ============================================================
# GPU 작업 큐 슬롯  —  GPU를 쓰는 명령은 이 폴더 스크립트에서만 실행한다.
# 목적: run11(--fg-loss) 공식 채점 — recon_eval.py로 leaderboard에서 SSIM_full+SSIM_bbox+추론시간 3축 산출.
#       목적 = shell3(run12)의 BASELINE(run10 vs run11) 결정.
#       ⚠️ load_model 이 best_model.pt 를 읽음(test_part.py) → run11의 best-val 체크포인트가 채점된다.
#       recon_eval는 checkpoint 저장 args로 SDUM config 자동 복원 → --cascade/--chans 불필요.
#
#   판정 규칙 (repo 컨벤션 = 직전 best 위에 하나만 얹는 A/B):
#       비교 대상: run10 = full 0.9118 / bbox 0.9021 (현재 best).
#       run11 의 0.5·full+0.5·bbox 가 run10 보다 높으면 → shell3 BASELINE=run11 (fg-loss 유지)
#       낮으면                                        → shell3 BASELINE=run10 (fg-loss 뺌)
#
# 실행:  bash shell1.sh                          (단독, GPU 비었을 때)
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

# --- 시작 고지 (어느 슬롯이 어디서 시작됐는지) ---
echo "=========================================================="
echo "[START] $SLOT   $(date '+%F %T')"
echo "[CWD  ] $(pwd)"
echo "[LOG  ] $LOG"
echo "=========================================================="

# --- cv2 보장: metrics.py 가 cv2 import (headless 서버 = opencv-python-headless). ---
python -c "import cv2" 2>/dev/null || pip install -q opencv-python-headless

# ==========================================================
#  ↓↓↓  실제 GPU 명령 (recon_eval)  ↓↓↓
# ==========================================================
set -x
python recon_eval.py -n sdum_run11 -p /root/Data/leaderboard

# ==========================================================
#  ↑↑↑  실제 GPU 명령 끝 (STATUS 는 마지막 명령의 종료코드)  ↑↑↑
# ==========================================================
{ STATUS=$?; set +x; } 2>/dev/null

echo "=========================================================="
echo "[DONE ] $SLOT   $(date '+%F %T')   exit=$STATUS"
echo "[다음] 위 run11 3축을 run10(full 0.9118/bbox 0.9021)과 비교 →"
echo "       shell3(run12) 의 BASELINE= 을 run10/run11 로 채운 뒤 → shell2(smoke) → shell3(run12)."
echo "=========================================================="
exit "$STATUS"
