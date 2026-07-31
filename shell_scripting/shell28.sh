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
# [왜] 둘 다 본실험이 60h 급이다. 2h 로 "걸지 말지"만 먼저 가른다.
#      A 는 학습·채점의 offset 분포가 다른 게 실제로 점수를 깎는지(280볼륨 전수조사로
#      분포차 자체는 이미 확인됨, augmentation.md), B 는 손실을 바꿨을 때 학습이 튀는지.
#
# [해석]
#   A 격차 <0.002 → 모델이 offset 에 둔감. remask 의 offset 몫은 0 (볼륨 2배 몫은 남으니
#                   --remask-offset canonical 로 전환). 그리고 shell26 의 remask 이득은
#                   "마스크 다양성"이 아니라 "스텝 2배"로 읽어야 한다.
#     격차 >=0.005 → 사각지대 실재. remask 확정 + acc4 에도 적용(offset 3 미노출 17.5%).
#                    val 도 학습풀이라 이 격차를 구조적으로 못 본다는 뜻이기도 하다.
#   B 궤적이 l1=0.0 과 비슷 → 안정. L1_ITERS 를 키워 이득 판정으로.
#     튀거나 발산      → α(L1_W) 를 낮추거나 폐기. run12 처럼 "안정적으로 나쁜" 것과 구분할 것.
#   ⚠️ TrainLoss 는 arm 마다 손실 함수가 달라 비교 불가. ValLoss(순수 SSIM)만 같은 자다.
#   ⚠️ A 는 회수 "상한"이지 remask 학습이 그걸 회수한다는 뜻이 아니다. shell27 채점과 대조할 것.
#
# 실행:  bash shell_scripting/shell28.sh                 # A+B
#        STAGE=offset bash shell_scripting/shell28.sh    # A 만
#        STAGE=l1     bash shell_scripting/shell28.sh    # B 만
#        L1_ITERS=2000 bash shell_scripting/shell28.sh   # B 를 길게(이득 신호까지)
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
