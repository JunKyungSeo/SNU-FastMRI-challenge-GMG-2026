#!/usr/bin/env bash
# ============================================================
# GPU 작업 큐 슬롯  —  GPU를 쓰는 명령은 이 폴더 스크립트에서만 실행한다.
# 목적: --resume 스모크(Task 5) — epoch0만 돌고 죽은 SDUM run을 --resume으로 이어붙여
#       start_epoch=1부터 재개되는지, scheduler fast-forward lr / val_loss_log 이어쓰기가
#       맞는지 확인. ⚠️ 1번 명령 완료(epoch0 "loss file saved!" 로그) 후 수동 kill 필요 —
#       완전 무인 큐 잡이 아님. 아래 절차대로 "수동 2단계"로 실행할 것.
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
#  set -x 로 실행되는 명령을 로그에 그대로 남긴다 (로그가 명령을 안 찍던 문제 해결)
# ==========================================================
set -x

# ⚠️ 수동 2단계 스모크 (자동 큐잡 아님) — Task 5 브리프 Step 4:
#
# [1단계] 아래 명령을 먼저 실행. 2-epoch 계획(-e 2)이지만 epoch0가 끝나
#   ("loss file saved! ..." 로그 + val_loss_log_resume_smoke.npy 저장) 되는 즉시
#   Ctrl-C로 죽인다 (epoch0만 돌고 죽은 상황을 흉내).
#
# python train.py --model sdum --num-cascades 4 --channels 16 32 \
#     -b 1 -l 2.4e-4 -e 2 -r 50 --max-iters 60 --readout-crop --warmup-iters 20 \
#     -n sdum_resume_smoke --run-tag resume_smoke -t /root/Data/train/ -v /root/Data/val/
#
# [2단계] 위에서 kill 확인 후, --resume을 붙여 재개(같은 -n/--run-tag/config):
#   기대 로그: "[resume] ... start_epoch=1 best_val=..." / "[resume] scheduler fast-forward 1ep, lr=..."
#   / 루프가 "Epoch # 1"부터 시작(epoch0 재실행 안 함) / val_loss_log_resume_smoke.npy가 2행(epoch 0,1).
#
# python train.py --model sdum --num-cascades 4 --channels 16 32 \
#     -b 1 -l 2.4e-4 -e 2 -r 50 --max-iters 60 --readout-crop --warmup-iters 20 --resume \
#     -n sdum_resume_smoke --run-tag resume_smoke -t /root/Data/train/ -v /root/Data/val/
#
# ↓ 빈 슬롯 표시(PENDING): 위 두 블록의 주석을 풀어 수동으로 단계별 실행하기 전까지는
#   이 슬롯을 자동 큐(`bash shell1.sh ; bash shell2.sh ; ...`)에 넣지 말 것 — 중간에
#   사람이 개입(kill)해야 하는 잡이라 무인 실행 시 1단계가 -e 2 끝까지 돌아버려
#   "epoch0에서 죽음" 상황을 재현하지 못한다.
echo "[TODO] $SLOT: GPU 스모크는 PENDING — 현재 GPU를 다른 run(run8)이 점유 중이라 실행 보류."
false

# ==========================================================
#  ↑↑↑  실제 GPU 명령 끝 (STATUS 는 마지막 명령의 종료코드)  ↑↑↑
# ==========================================================
{ STATUS=$?; set +x; } 2>/dev/null

echo "=========================================================="
echo "[DONE ] $SLOT   $(date '+%F %T')   exit=$STATUS"
echo "=========================================================="
exit "$STATUS"
