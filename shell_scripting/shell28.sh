#!/usr/bin/env bash
# ============================================================
# 목적: 본실험 전 값싼 사전확인 2건 — (A) offset 민감도 (B) L1 혼합 안정성   [shell28.sh]
#
#   A offset : 기존 acc8 전문가로 hold-out 을 offset 0~7 강제 채점. 학습 0.
#              canonical(학습에서 본 offset) 과 나머지의 격차 = --remask 회수 상한.
#   B l1     : --l1-weight on/off 짧은 학습. 손실 곡선이 튀는지만 본다.
#
#   shell26/27 과 net 이름이 안 겹친다. B 는 --l1-weight 기본 0(비트 동일)이라 켜기 전엔 무영향.
#
# 실행:  bash shell_scripting/shell28.sh                 # A+B
#        STAGE=offset bash shell_scripting/shell28.sh    # A 만
#        STAGE=l1     bash shell_scripting/shell28.sh    # B 만
# ============================================================
set -uo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# ===== [이번 run] =====
STAGE="${STAGE:-all}"
DATA_ROOT="${DATA_ROOT:-/root/Data}"
# 탐침 대상은 **remask 없이 학습된** 모델이어야 한다(격차가 남아 있어야 잰다).
#   PROBE_KFOLD=0 이면 고정분할 → hold-out 은 $DATA_ROOT/val.
#   PROBE_KFOLD>0 이면 그 fold 가 hold-out (학습 때와 **같은** fold/seed 를 줄 것).
PROBE_NET="${PROBE_NET:-promptmr_moe_f2_shared_aug}"
PROBE_ACC="${PROBE_ACC:-8}"
PROBE_KFOLD="${PROBE_KFOLD:-5}"
L1_W="${L1_W:-0.1}"
L1_ITERS="${L1_ITERS:-300}"
FOLD="${FOLD:-2}"

BASE_ARGS=( --model promptmr --readout-crop
    --num-cascades 10 --n-feat0 48 --sens-n-feat0 24 --n-history 3
    -b 1 -l 2e-4 -r 50 --num-workers 4 --use-checkpoint
    --fg-loss --area-weight --grad-clip 0.1 --accum-steps 2
    -t "$DATA_ROOT/train/" -v "$DATA_ROOT/val/"
    --k-fold 5 --val-fold "$FOLD" --kfold-seed 430 --acc "$PROBE_ACC" )

SLOT="$(basename "$0" .sh)"
TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p shell_scripting/logs
LOG="shell_scripting/logs/${SLOT}_${STAGE}_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=========================================================="
echo "[START] $SLOT  STAGE=$STAGE  $(date '+%F %T')"
echo "[LOG  ] $LOG"
echo "=========================================================="

# --- 가드: GPU 비었는지 (used < 500MiB) ---
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
echo "[GPU  ] used=${USED}MiB"
if [ -n "$USED" ] && [ "$USED" -ge 500 ]; then
    echo "[STOP] GPU가 사용 중(${USED}MiB)이다. shell26이 끝난 뒤 실행할 것."
    exit 2
fi

# --- 가드: 탐침 체크포인트 존재 ---
if { [ "$STAGE" = "all" ] || [ "$STAGE" = "offset" ]; } \
   && [ ! -e "../result/$PROBE_NET/checkpoints/best_model.pt" ]; then
    echo "[STOP] ../result/$PROBE_NET/checkpoints/best_model.pt 없음. PROBE_NET= 으로 지정하라. 후보:"
    ls -1 ../result 2>/dev/null | grep -i acc8
    exit 2
fi

STATUS=0

if [ "$STAGE" = "all" ] || [ "$STAGE" = "offset" ]; then
    if [ "$PROBE_KFOLD" -gt 0 ]; then
        HOLDOUT=( --k-fold "$PROBE_KFOLD" --val-fold "$FOLD" --kfold-seed 430
                  --data-path-train "$DATA_ROOT/train/" --data-path-val "$DATA_ROOT/val/" )
    else
        HOLDOUT=( -p "$DATA_ROOT/val" )
    fi
    echo "[A offset] $PROBE_NET  acc$PROBE_ACC"
    set -x
    python scripts/probe_offset.py -n "$PROBE_NET" --acc "$PROBE_ACC" "${HOLDOUT[@]}"
    { A=$?; set +x; } 2>/dev/null
    [ "$A" -ne 0 ] && STATUS="$A"
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "l1" ]; then
    ARM_DIR="$(mktemp -d)"
    for W in 0.0 "$L1_W"; do
        NET="probe_l1_${W}"
        rm -rf "../result/$NET"
        echo "[B l1] --l1-weight $W  ($L1_ITERS iter)"
        set -x
        python train.py "${BASE_ARGS[@]}" \
            --l1-weight "$W" -e 1 --max-iters "$L1_ITERS" \
            -n "$NET" --run-tag "probe_l1_${W}" 2>&1 | tee "$ARM_DIR/$W.txt"
        { B=${PIPESTATUS[0]}; set +x; } 2>/dev/null
        [ "$B" -ne 0 ] && STATUS="$B"
    done

    echo
    echo "===== B 결과: 손실 궤적 ====="
    paste <(printf 'iter  l1=0.0\n'; grep -o 'Iter = \[ *[0-9]*/.*Loss = [0-9.e+-]*' "$ARM_DIR/0.0.txt" \
                | sed -E 's/Iter = \[ *([0-9]+).*Loss = /\1  /') \
          <(printf "l1=$L1_W\n"; grep -o 'Loss = [0-9.e+-]*' "$ARM_DIR/$L1_W.txt" | sed 's/Loss = //') \
        | column -t
    echo
    echo "===== B 판정 ====="
    echo "ValLoss (1-masked SSIM, 낮을수록 좋음) — 두 arm 모두 순수 SSIM 이라 이것만 비교 가능:"
    for W in 0.0 "$L1_W"; do
        printf '  l1=%-5s %s\n' "$W" "$(grep -o 'ValLoss *= *[0-9.]*' "$ARM_DIR/$W.txt" | tail -1)"
    done
    echo "⚠️ TrainLoss 는 arm 마다 손실 **함수가 다르다**((1-w)·SSIM + w·L1) — 나란히 놓지 말 것."
    echo "⚠️ ${L1_ITERS} iter 는 안정성 확인용이다. 이득 판정은 L1_ITERS 를 키워서 다시."
    rm -rf "$ARM_DIR"
fi

echo "=========================================================="
echo "[DONE ] $SLOT  STAGE=$STAGE  $(date '+%F %T')  exit=$STATUS"
echo "=========================================================="
exit "$STATUS"
