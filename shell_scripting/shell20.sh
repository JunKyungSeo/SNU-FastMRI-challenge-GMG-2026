#!/usr/bin/env bash
set -uo pipefail
export PYTHONUNBUFFERED=1

# --- repo 루트로 이동 ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

STAGE="${STAGE:-all}"

DATA_ROOT="${DATA_ROOT:-/root/Data}"
LEADERBOARD="${LEADERBOARD:-/root/Data/leaderboard}"

# --- 로그 ---
SLOT="$(basename "$0" .sh)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="shell_scripting/logs"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SLOT}_${STAGE}_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

# --- net_name / 하이퍼파라미터 ---
NET_SHARED="promptmr_moe_shared"
NET_ACC4="promptmr_moe_acc4"
NET_ACC8="promptmr_moe_acc8"
NET_MOE="promptmr_moe"
SHARED_BEST="../result/${NET_SHARED}/checkpoints/best_model.pt"

ARCH="--model promptmr --readout-crop --num-cascades 10 --n-feat0 48 --sens-n-feat0 24 --n-history 3 --use-checkpoint"
COMMON="-b 1 -r 100 --num-workers 4 --fg-loss --area-weight --grad-clip 0.1 --accum-steps 2"

PRE_LR="2e-4"; PRE_EPOCHS=10      
FT_LR="5e-5";  FT_EPOCHS=3            
THRESHOLD="0.25"                    

echo "=========================================================="
echo "[START] $SLOT  STAGE=$STAGE  SMOKE=$SMOKE  $(date '+%F %T')"
echo "[CWD  ] $(pwd)   [DATA] $DATA_ROOT   [LOG] $LOG"
echo "=========================================================="

python -c "import cv2" 2>/dev/null || { echo "[STOP] cv2 import 실패 — bash /root/restore_env.sh 후 재실행."; exit 2; }

# ---------- 단계 함수 ----------
do_pretrain() {
    echo "[① pretrain] $NET_SHARED (전체 데이터, --acc all)"
    set -x
    python train.py $ARCH $COMMON --acc all -l "$PRE_LR" -e "$PRE_EPOCHS" $SMOKE_EXTRA \
        -n "$NET_SHARED" --run-tag moe_shared \
        -t "$DATA_ROOT/train/" -v "$DATA_ROOT/val/"
    { local st=$?; set +x; } 2>/dev/null
    echo "[① pretrain] exit=$st"; return $st
}
do_ft() {  # $1=acc(4/8)  $2=net_name
    if [ ! -f "$SHARED_BEST" ]; then
        echo "[STOP] 공유 사전학습 결과가 없다: $SHARED_BEST (먼저 STAGE=pretrain)."; return 2
    fi
    echo "[② ft$1] $2 (--acc $1 --init-from $SHARED_BEST)"
    set -x
    python train.py $ARCH $COMMON --acc "$1" -l "$FT_LR" -e "$FT_EPOCHS" $SMOKE_EXTRA \
        --init-from "$SHARED_BEST" \
        -n "$2" --run-tag "moe_acc$1" \
        -t "$DATA_ROOT/train/" -v "$DATA_ROOT/val/"
    { local st=$?; set +x; } 2>/dev/null
    echo "[② ft$1] exit=$st"; return $st
}
do_merge() {
    echo "[④ merge] $NET_ACC4 + $NET_ACC8 → $NET_MOE"
    set -x
    python scripts/merge_moe.py --acc4-net "$NET_ACC4" --acc8-net "$NET_ACC8" \
        --out-net "$NET_MOE" --threshold "$THRESHOLD"
    { local st=$?; set +x; } 2>/dev/null
    [ "$st" -ne 0 ] && { echo "[④ merge] 실패 exit=$st"; return $st; }
    echo "[⑤ score] recon_eval -n $NET_MOE"
    mkdir -p "../result/$NET_MOE"
    set -x
    python recon_eval.py -n "$NET_MOE" -p "$LEADERBOARD" 2>&1 | tee "../result/$NET_MOE/recon_eval_moe.txt"
    { st=$?; set +x; } 2>/dev/null
    echo "[⑤ score] exit=$st  점수: ../result/$NET_MOE/recon_eval_moe.txt"; return $st
}

# ---------- STAGE 디스패치 ----------
case "$STAGE" in
    pretrain) do_pretrain; STATUS=$? ;;
    ft4)      do_ft 4 "$NET_ACC4"; STATUS=$? ;;
    ft8)      do_ft 8 "$NET_ACC8"; STATUS=$? ;;
    merge)    do_merge; STATUS=$? ;;
    all)
        do_pretrain && do_ft 4 "$NET_ACC4" && do_ft 8 "$NET_ACC8" && do_merge
        STATUS=$?
        ;;
    *) echo "[STOP] 알 수 없는 STAGE='$STAGE' (pretrain|ft4|ft8|merge|all)"; exit 2 ;;
esac

exit "$STATUS"
