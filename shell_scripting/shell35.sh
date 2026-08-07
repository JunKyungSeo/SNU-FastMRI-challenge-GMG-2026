#!/usr/bin/env bash
# ============================================================
# 목적: SWDC A/B의 ON arm — 기준값은 shell32 실측(fwdbwd 4.15s · ValLoss 0.135@600iter)  [shell35.sh]
#
# 실행:  bash shell_scripting/shell35.sh              # probe (시간·peak, ~5분)
#        STAGE=ab bash shell_scripting/shell35.sh     # 1 epoch, 같은 벽시계 iter로 학습
#        STAGE=ab ITERS=NNN ...                       # probe 안 돌렸으면 iter 직접 지정
# ============================================================
set -uo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

STAGE="${STAGE:-probe}"
FOLD="${FOLD:-2}"
PROBE_ACC="${PROBE_ACC:-8}"
B0="${B0:-4.15}"          # shell32 probe 실측 fwdbwd(스칼라 DC, 10:3:48 @384x480)
BASE_ITERS=600            # shell32 lever_ab 기준 iter (ValLoss 0.135)
DATA_ROOT="${DATA_ROOT:-$([ -d /workspace/Data ] && echo /workspace/Data || echo /root/Data)}"

SLOT="$(basename "$0" .sh)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="shell_scripting/logs"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SLOT}_${STAGE}_${TS}.log"
exec > >(tee -a "$LOG") 2>&1
OUT="shell_scripting/results/swdc_${TS}"; mkdir -p "$OUT"

echo "[START] $SLOT $STAGE   $(date '+%F %T')   $(git rev-parse --short HEAD)"

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
[ -n "$USED" ] && [ "$USED" -ge 500 ] && { echo "[STOP] GPU 사용 중(${USED}MiB)"; exit 2; }

STATUS=0

if [ "$STAGE" = probe ]; then
    python scripts/probe_oom_promptmr.py --crop-h 384 --width 480 --coils 15 \
        --with-optimizer --reps 3 --num-cascades 10 --n-history 3 \
        --n-feat0 48 --sens-n-feat0 24 2>&1 | tee "$OUT/probe.txt"
    echo "[비교] 스칼라 DC 기준 fwdbwd=${B0}s (shell32 실측) — 위 fwdbwd와 비 = ab 의 iter 스케일"
else
    T1=$(awk '/^OK/ { for(i=1;i<=NF;i++){ split($i,kv,"=")
        if (kv[1]=="fwdbwd") { sub("s$","",kv[2]); print kv[2] } } }' \
        shell_scripting/results/swdc_*/probe.txt 2>/dev/null | tail -1)
    if [ -n "${ITERS:-}" ]; then I="$ITERS"
    elif [ -n "$T1" ]; then I=$(awk -v i="$BASE_ITERS" -v b="$B0" -v t="$T1" 'BEGIN{printf "%d", i*b/t}')
    else echo "[STOP] probe 결과가 없다 — 먼저 probe를 돌리거나 ITERS= 지정"; exit 2; fi
    NET="swdc_ab_f${FOLD}"
    rm -rf "../result/$NET"
    echo "---------- swdc ON (iter=$I, 기준 ${BASE_ITERS}·${B0}s 대비 벽시계 동일) ----------"
    set -x
    python train.py --model promptmr --readout-crop \
        --num-cascades 10 --n-feat0 48 --sens-n-feat0 24 --n-history 3 \
        -b 1 -l 2e-4 -r 20 --num-workers 4 --use-checkpoint \
        --fg-loss --area-weight --grad-clip 0.1 --accum-steps 2 --l1-weight 0.1 \
        -t "$DATA_ROOT/train/" -v "$DATA_ROOT/val/" \
        --k-fold 5 --val-fold "$FOLD" --kfold-seed 430 --acc "$PROBE_ACC" \
        -e 1 --max-iters "$I" \
        -n "$NET" --run-tag "$NET" 2>&1 | tee "$OUT/train_on.txt"
    { A=${PIPESTATUS[0]}; set +x; } 2>/dev/null
    [ "$A" -ne 0 ] && STATUS="$A"
    echo
    echo "===== ValLoss (기준: 스칼라 DC 0.135 @600iter, shell32 lever_ab) ====="
    grep -o 'ValLoss = [0-9.e+-]*' "$OUT/train_on.txt" 2>/dev/null | tail -1
fi

echo
echo "[DONE ] $SLOT $STAGE   $(date '+%F %T')   exit=$STATUS"
echo "[저장] $OUT/"
echo "[판정] probe: peak>7.0GB면 폐기·롤백. ab: ValLoss가 0.135를 같은 벽시계에서 못 이기면 롤백."
echo "       이겨도 추론시간이 점수항 — fwdbwd 비(B0 대비)만큼 recon도 느려진다고 보고 손익 계산."
exit "$STATUS"
