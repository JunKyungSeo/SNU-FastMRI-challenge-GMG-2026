#!/usr/bin/env bash
# ============================================================
# GPU 작업 큐 슬롯  —  GPU를 쓰는 명령은 이 폴더 스크립트에서만 실행한다.
# 목적: run8 = 용량(capacity) 실험 — run7(readout-crop+cosine, loss 0.037) 위에서 readout-crop이 비운
#       VRAM 헤드룸(~1.8GB)을 활용. run7 대비 2개 레버: --num-cascades 4→6, --channels [16,32]→[24,48].
#       cascade 깊이는 grad-checkpointing 덕에 VRAM 거의 안 씀(activation-flat), channels 폭이 헤드룸을 실제 사용.
#       ⚠️ OOM 위험 borderline~medium — [24,48]×6은 추정 ~7.1–8.1GB(8192 근접). run7이 GPU 비운 뒤
#          선택적 선점검: python scripts/probe_oom.py --cascades 6 --channels 24 48  (peak≤~7.0GB면 안전).
#          OOM 나면 run9에서 [20,40] 또는 cascades 5로 후퇴(시간 낭비 아님). ⚠️ 예상 ~18–20h(느림).
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

# run8 (capacity): run7 + cascades 4→6 + channels [16,32]→[24,48], readout-crop 유지, cosine warmup 200, 2 epoch.
python train.py --model sdum --num-cascades 6 --channels 24 48 \
    -b 1 -l 2.4e-4 -e 2 -r 100 --num-workers 4 \
    --readout-crop --warmup-iters 200 \
    -n sdum_run8 --run-tag run8 -t /root/Data/train/ -v /root/Data/val/

# ==========================================================
#  ↑↑↑  실제 GPU 명령 끝 (STATUS 는 마지막 명령의 종료코드)  ↑↑↑
# ==========================================================
{ STATUS=$?; set +x; } 2>/dev/null

echo "=========================================================="
echo "[DONE ] $SLOT   $(date '+%F %T')   exit=$STATUS"
echo "=========================================================="
exit "$STATUS"
