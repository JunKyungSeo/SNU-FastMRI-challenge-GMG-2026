#!/usr/bin/env bash
# ============================================================
# GPU 작업 큐 슬롯  —  GPU를 쓰는 명령은 이 폴더 스크립트에서만 실행한다.
# 목적: 스모크 테스트 — 새 코드(--readout-crop + cosine/warmup 스케줄러) 경로가 안 깨지나 몇 분 확인.
#       체크: ① cropped 경로 loss 유한? ② "[scheduler] warmup=... → cosine" 프린트 뜨나?
#             ③ 실제 s/iter로 readout-crop 속도이득 실측(run4 대비). 9h run(shell5) 전 게이트.
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

# 스모크: run5(shell5)와 같은 config, 단 --max-iters 30 -e 1 로 몇 분만.
# warmup 10 → 30 iters 안에 warmup(1~10)+cosine(11~) 둘 다 관찰 가능.
python train.py --model sdum --num-cascades 4 --channels 16 32 \
    -b 1 -l 2e-4 -e 1 -r 5 --max-iters 30 --num-workers 4 \
    --readout-crop --warmup-iters 10 \
    -n sdum_smoke --run-tag smoke -t /root/Data/train/ -v /root/Data/val/

# ==========================================================
#  ↑↑↑  실제 GPU 명령 끝 (STATUS 는 마지막 명령의 종료코드)  ↑↑↑
# ==========================================================
{ STATUS=$?; set +x; } 2>/dev/null

echo "=========================================================="
echo "[DONE ] $SLOT   $(date '+%F %T')   exit=$STATUS"
echo "=========================================================="
exit "$STATUS"
