#!/usr/bin/env bash
# ============================================================
# GPU 작업 큐 슬롯  —  GPU를 쓰는 명령은 이 폴더 스크립트에서만 실행한다.
# 목적: run9 = 장기학습(under-training 해소). run7 진단이 "capacity가 아니라 덜 훈련됨"이라,
#       capacity는 run7 그대로([16,32]×4, 빠름 ~5.2h/ep) 두고 EPOCH를 2→4로 늘린다.
#       cosine total_iters=epoch×len 자동 확장 → LR이 곡선 하강 중에 0으로 꺾이지 않게 됨(run7의 조기감쇠 해소).
#       readout-crop 유지(속도+추론시간 점수 이득). run8 capacity 결과와 대비해 "시간 vs 용량" 판단.
#       ⚠️ resume 미구현이라 scratch ~21h — 도중 죽으면 처음부터. (resume 구현은 별도 플랜 진행 중)
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

# run9: run7과 동일 capacity([16,32]×4), epoch만 2→4로 늘려 under-training 해소 + cosine 완만화.
python train.py --model sdum --num-cascades 4 --channels 16 32 \
    -b 1 -l 2.4e-4 -e 4 -r 100 --num-workers 4 \
    --readout-crop --warmup-iters 200 \
    -n sdum_run9 --run-tag run9 -t /root/Data/train/ -v /root/Data/val/

# ==========================================================
#  ↑↑↑  실제 GPU 명령 끝 (STATUS 는 마지막 명령의 종료코드)  ↑↑↑
# ==========================================================
{ STATUS=$?; set +x; } 2>/dev/null

echo "=========================================================="
echo "[DONE ] $SLOT   $(date '+%F %T')   exit=$STATUS"
echo "=========================================================="
exit "$STATUS"
