#!/usr/bin/env bash
# ============================================================
# GPU 작업 큐 슬롯  —  GPU를 쓰는 명령은 이 폴더 스크립트에서만 실행한다.
# 목적: run10 = run9와 전부 동일한 조건([16,32]×4, readout-crop, cosine warmup200, LR 2.4e-4)에서
#       EPOCH만 4→10으로 늘림. under-training 가설을 끝까지 밀어 "얼마까지 오르나" plateau 확인.
#       cosine total_iters=10×len 자동 확장 → LR이 10 epoch 내내 완만히 감쇠(조기 0 없음).
#       ⚠️ 예상 ~55h(약 2.3일, ~5.5h/ep × 10). 중간에 죽으면 --resume 가능(이제 구현됨):
#          같은 명령에 --resume만 붙여 shell 슬롯에 넣으면 마지막 완료 epoch부터 재개.
#          (단 resume end-to-end 스모크 shellB는 아직 미검증 — 첫 사용 전 한 번 돌려보길 권장.)
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

# run10: run9와 동일, epoch만 4→10.
python train.py --model sdum --num-cascades 4 --channels 16 32 \
    -b 1 -l 2.4e-4 -e 10 -r 100 --num-workers 4 \
    --readout-crop --warmup-iters 200 \
    -n sdum_run10 --run-tag run10 -t /root/Data/train/ -v /root/Data/val/

# ==========================================================
#  ↑↑↑  실제 GPU 명령 끝 (STATUS 는 마지막 명령의 종료코드)  ↑↑↑
# ==========================================================
{ STATUS=$?; set +x; } 2>/dev/null

echo "=========================================================="
echo "[DONE ] $SLOT   $(date '+%F %T')   exit=$STATUS"
echo "=========================================================="
exit "$STATUS"
