#!/usr/bin/env bash
# ============================================================
# GPU 작업 큐 슬롯 — run13 강제 재시작.
# 목적: 진행 중인 학습을 모두 종료하고, run13을 --resume 으로 "이번 epoch"부터 다시 시작,
#       학습 완주 후 recon_eval 로 leaderboard 채점까지 한 번에.
#   현재 model.pt = epoch5 종료본 → --resume 시 start_epoch=6 → 이번 epoch(6)부터 재개.
#   진행 중이던 epoch6 iter 진척은 아직 저장 전이라 버려지고 epoch6을 처음부터 다시 돈다(정상).
#   epoch5 상태는 checkpoints/model.pt + result/sdum_run13_5ep 백업에 안전하게 보존됨.
#   재개 명령은 shell4.sh(원본 run13 학습) 와 완전히 동일 config + --resume (같은 config라야 resume 맞물림).
#   채점은 학습 성공(exit 0) 시에만 실행 → best_model.pt 를 recon_eval 이 채점(shell5 와 동일 로직).
#   ⚠️ 학습 프로세스를 kill 하는 파괴적 작업.
#   ⚠️ run13 터미널에 '; bash shell5.sh' 채점 체인이 걸려 있으면, kill 직후 shell4_resume 이
#      정상 종료로 넘어가 shell5 가 딸려 실행될 수 있다. 이 스크립트를 돌리기 전에 그 터미널의
#      체인을 Ctrl-C 로 끊어 둘 것 (안 그러면 shell5 가 GPU 를 선점해 아래 재개가 OOM 난다).
#
# 실행:  bash shell4_restart.sh
# ============================================================
set -uo pipefail
export PYTHONUNBUFFERED=1

# --- repo 루트로 이동 (train.py 는 repo 루트 기준으로 돈다) ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# --- 로그 파일 (이름에 timestamp) ---
SLOT="$(basename "$0" .sh)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="shell_scripting/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SLOT}_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=========================================================="
echo "[START] $SLOT   $(date '+%F %T')"
echo "[CWD  ] $(pwd)"
echo "[LOG  ] $LOG"
echo "=========================================================="

# --- 1) 진행 중인 학습 모두 종료 (SIGTERM → 대기 → 남으면 SIGKILL) ---
PAT="train.py"
if pgrep -f "$PAT" >/dev/null 2>&1; then
    echo "[KILL] 진행 중 train.py 종료 (SIGTERM)..."
    pkill -TERM -f "$PAT"
    for _ in $(seq 1 30); do
        pgrep -f "$PAT" >/dev/null 2>&1 || break
        sleep 2
    done
    if pgrep -f "$PAT" >/dev/null 2>&1; then
        echo "[KILL] 아직 살아있음 → SIGKILL"
        pkill -KILL -f "$PAT"
        sleep 3
    fi
    echo "[KILL] 종료 완료."
else
    echo "[KILL] 실행 중인 train.py 없음 (건너뜀)."
fi

# --- 2) GPU VRAM 반환 대기 (이전 프로세스가 메모리 놓을 때까지 최대 ~60s) ---
echo "[GPU ] VRAM 반환 대기..."
for _ in $(seq 1 30); do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo "  used=${USED}MiB"
    [ -n "$USED" ] && [ "$USED" -lt 1000 ] && break
    sleep 2
done

# --- 3) 체크포인트 확인 후 run13 재개 (이번 epoch=6 부터) ---
if [ ! -f ../result/sdum_run13/checkpoints/model.pt ]; then
    echo "[STOP] ../result/sdum_run13/checkpoints/model.pt 없음 — resume 불가."
    exit 2
fi

# cv2 보장(재개 자체엔 불필요하나 파이프라인 일관성 위해 유지)
python -c "import cv2" 2>/dev/null || pip install -q opencv-python-headless

set -x
python train.py --model sdum --num-cascades 4 --channels 16 32 \
    -b 1 -l 2.4e-4 -e 10 -r 100 --num-workers 4 \
    --readout-crop --warmup-iters 200 --fg-loss --uc \
    -n sdum_run13 --run-tag run13 -t /root/Data/train/ -v /root/Data/val/ \
    --resume
{ TRAIN_STATUS=$?; set +x; } 2>/dev/null
echo "[TRAIN] run13 학습 종료   exit=$TRAIN_STATUS"

# --- 4) reconstruct + 채점 (학습 성공 시에만; 학습 끝나 GPU 는 이미 비어 있음) ---
if [ "$TRAIN_STATUS" -eq 0 ]; then
    echo "[RECON] run13 채점 시작 (recon_eval → leaderboard SSIM_full+bbox+시간)..."
    set -x
    python recon_eval.py -n sdum_run13 -p /root/Data/leaderboard 2>&1 \
        | tee ../result/sdum_run13/recon_eval_run13.txt
    { STATUS=$?; set +x; } 2>/dev/null
else
    echo "[RECON] 학습이 exit=$TRAIN_STATUS 로 끝나 채점 건너뜀."
    STATUS=$TRAIN_STATUS
fi

echo "=========================================================="
echo "[DONE ] $SLOT   $(date '+%F %T')   train=$TRAIN_STATUS recon/exit=$STATUS"
echo "[저장] 점수: ../result/sdum_run13/recon_eval_run13.txt (+ 이 슬롯 로그)."
echo "=========================================================="
exit "$STATUS"
