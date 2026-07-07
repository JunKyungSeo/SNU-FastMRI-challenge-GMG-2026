#!/usr/bin/env bash
# ============================================================
# GPU 작업 큐 슬롯  —  GPU를 쓰는 명령은 이 폴더 스크립트에서만 실행한다.
# 목적: <이 스크립트가 무엇을 하는지 한 줄로 작성>
#
# 실행:  bash shellN.sh                          (단독)
#        bash shell1.sh ; bash shell2.sh ; ...   (큐로 이어서)
# ============================================================
set -uo pipefail

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
#  ↓↓↓  실제 GPU 명령을 여기에 작성 (이 아래만 고치면 됨)  ↓↓↓
# ==========================================================

# 예시 — 지우고 실제 명령으로 교체:
# python train.py --model sdum --num-cascades 4 --channels 16 32 \
#     -b 1 -l 2e-4 -e 1 --max-iters 200 -r 10 --num-workers 4 \
#     -n sdum_q_${SLOT} -t /root/Data/train/ -v /root/Data/val/

# ↓ 빈 슬롯 표시: 명령을 채우면 이 echo 와 false 두 줄을 지우세요
echo "[TODO] $SLOT: 실제 명령이 비어 있습니다. 이 두 줄을 지우고 명령을 채우세요."
false

# ==========================================================
#  ↑↑↑  실제 GPU 명령 끝 (STATUS 는 마지막 명령의 종료코드)  ↑↑↑
# ==========================================================
STATUS=$?

echo "=========================================================="
echo "[DONE ] $SLOT   $(date '+%F %T')   exit=$STATUS"
echo "=========================================================="
exit "$STATUS"
