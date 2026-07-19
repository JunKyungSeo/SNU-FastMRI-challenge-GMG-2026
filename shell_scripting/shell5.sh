#!/usr/bin/env bash
# ============================================================
# GPU 작업 큐 슬롯  —  GPU를 쓰는 명령은 이 폴더 스크립트에서만 실행한다.
# 목적: run13(run11+UC, shell4) 공식 채점 — recon_eval.py로 leaderboard SSIM_full+bbox+시간 3축.
#       = UC 순효과 판별(run12는 bbox+UC 묶음이라 UC 단독 못 봄):
#         run13 ≈ run11(full 0.9161 / bbox 0.9095 / 1514ms) → UC 중립 → run12 퇴행 원흉 = bbox(C1)
#         run13 <  run11                                    → UC 유해 → SPEC "UC 제거" 회귀 근거
#       recon_eval는 best_model.pt 를 채점(test_part.load_model). UC config는 저장 args로 자동 복원.
#       ⚠️ shell4/run13 완주 후 실행(아래 가드가 학습 진행 중이면 거부).
#       ⚠️ 점수는 슬롯 로그 + ../result/sdum_run13/recon_eval_run13.txt 에 이중 저장(run12 유실 재발 방지).
#
# 실행:  bash shell5.sh                          (단독, shell4/run13 완주 후)
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

# --- 가드 ①: run13 학습이 아직 진행 중이면 거부 (GPU 충돌·미완 모델 채점 방지) ---
if pgrep -f "train.py.*sdum_run13" >/dev/null 2>&1; then
    echo "[STOP] run13 학습(train.py)이 아직 진행 중이다. shell4/run13 완주 후 실행할 것."
    exit 2
fi
# --- 가드 ②: 체크포인트 없으면 거부 ---
if [ ! -f ../result/sdum_run13/checkpoints/best_model.pt ]; then
    echo "[STOP] ../result/sdum_run13/checkpoints/best_model.pt 없음 — run13 학습 결과가 없다."
    exit 2
fi

# --- cv2 보장: metrics.py 가 cv2 import (headless 서버 = opencv-python-headless). ---
python -c "import cv2" 2>/dev/null || pip install -q opencv-python-headless

# ==========================================================
#  ↓↓↓  실제 GPU 명령 (recon_eval) — 점수를 결과 dir 파일에도 저장  ↓↓↓
# ==========================================================
set -x
python recon_eval.py -n sdum_run13 -p /root/Data/leaderboard 2>&1 | tee ../result/sdum_run13/recon_eval_run13.txt

# ==========================================================
#  ↑↑↑  실제 GPU 명령 끝 (STATUS 는 마지막 명령의 종료코드; pipefail 로 recon_eval 실패 반영)  ↑↑↑
# ==========================================================
{ STATUS=$?; set +x; } 2>/dev/null

echo "=========================================================="
echo "[DONE ] $SLOT   $(date '+%F %T')   exit=$STATUS"
echo "[비교] run13 vs run11(best): full 0.9161 / bbox 0.9095 / 1514ms."
echo "       run13 ≈ run11 → UC 중립(bbox가 run12 원흉) · run13 < run11 → UC 유해(UC 제거 근거)."
echo "[저장] 점수: ../result/sdum_run13/recon_eval_run13.txt (+ 이 슬롯 로그)."
echo "=========================================================="
exit "$STATUS"
