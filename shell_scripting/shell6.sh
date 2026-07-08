#!/usr/bin/env bash
# ============================================================
# GPU 작업 큐 슬롯  —  GPU를 쓰는 명령은 이 폴더 스크립트에서만 실행한다.
# 목적: run6 = 대안 5b (깨끗한 귀속) — readout-crop 빼고 -e 1.
#       run4(640 해상도)와 해상도를 동일하게 두고 "cosine+warmup 만으로 정체가 풀리나"를
#       readout-crop confound 없이 확인. (근거: run4-5-extra.md §대안 5b)
#       ⚠️ SSIM_full/bbox 로깅(item 1·2) 아직 안 붙임 → val 로그는 whole-image SSIMLoss만.
#
# 실행:  bash shellN.sh                          (단독)
#        bash shell1.sh ; bash shell2.sh ; ...   (큐로 이어서)
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
#  ↓↓↓  실제 GPU 명령을 여기에 작성 (이 아래만 고치면 됨)  ↓↓↓
#  set -x 로 실행되는 명령을 로그에 그대로 남긴다
# ==========================================================
set -x

# run6 (5b): readout-crop 없음(=run4와 640 동일), warmup 200→cosine, 1 epoch.
python train.py --model sdum --num-cascades 4 --channels 16 32 \
    -b 1 -l 2e-4 -e 1 -r 500 --num-workers 4 \
    --warmup-iters 200 \
    -n sdum_run6 --run-tag run6 -t /root/Data/train/ -v /root/Data/val/

# ==========================================================
#  ↑↑↑  실제 GPU 명령 끝 (STATUS 는 마지막 명령의 종료코드)  ↑↑↑
# ==========================================================
{ STATUS=$?; set +x; } 2>/dev/null

echo "=========================================================="
echo "[DONE ] $SLOT   $(date '+%F %T')   exit=$STATUS"
echo "=========================================================="
exit "$STATUS"
