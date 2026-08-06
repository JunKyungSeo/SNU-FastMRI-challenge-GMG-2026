#!/usr/bin/env bash
# ============================================================
# 목적: --sens-chunk 천장 — 감도맵 코일을 몇 개씩 묶어야 빠른가 (레버 C)   [shell33.sh]
#   값은 안 바뀐다(pytest 로 강제). 바뀌는 건 s/iter 와 VRAM 뿐이다.
#
# 실행:  bash shell_scripting/shell33.sh                    # 탐침 (~10분)
#        STAGE=train bash shell_scripting/shell33.sh        # chunk 1 vs 5 실학습 A/B
# ============================================================
set -uo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# ===== [이번 run] =====
STAGE="${STAGE:-probe}"
CHUNKS="${CHUNKS:-1 3 5 8 15}"          # 탐침할 코일 묶음 크기. 1 = 현재 baseline
TRAIN_CHUNKS="${TRAIN_CHUNKS:-1 5}"     # 실학습 A/B 할 값
ITERS="${ITERS:-60}"
FOLD="${FOLD:-2}"
PROBE_ACC="${PROBE_ACC:-8}"
DATA_ROOT="${DATA_ROOT:-$([ -d /workspace/Data ] && echo /workspace/Data || echo /root/Data)}"

SLOT="$(basename "$0" .sh)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="shell_scripting/logs"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SLOT}_${STAGE}_${TS}.log"
exec > >(tee -a "$LOG") 2>&1
OUT="shell_scripting/results/schunk_${TS}"; mkdir -p "$OUT"

echo "=========================================================="
echo "[START] $SLOT $STAGE   $(date '+%F %T')   $(git rev-parse --short HEAD)"
echo "[LOG  ] $LOG"
echo "=========================================================="

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -n "$USED" ] && [ "$USED" -ge 500 ]; then
    echo "[STOP] GPU가 사용 중(${USED}MiB) — peak 이 의미 없어진다."; exit 2
fi

STATUS=0

if [ "$STAGE" = probe ]; then
    # 최악폭 W=480. chunk=1 이 shell32 기준선(peak 3.3GB)과 맞아야 나머지가 의미 있다.
    for K in $CHUNKS; do
        python scripts/probe_oom_promptmr.py --crop-h 384 --width 480 --coils 15 \
            --with-optimizer --reps 3 --num-cascades 10 --n-history 3 \
            --n-feat0 48 --sens-n-feat0 24 --sens-chunk "$K" 2>&1 | tee -a "$OUT/probe.txt"
    done
    echo
    echo "===== chunk 별 fwdbwd · peak (W=480) ====="
    grep -E '^(OK|OOM)' "$OUT/probe.txt" | column -t
else
    for K in $TRAIN_CHUNKS; do
        NET="schunk${K}_f${FOLD}"
        rm -rf "../result/$NET"
        echo; echo "---------- --sens-chunk $K ----------"
        set -x
        python train.py --model promptmr --readout-crop \
            --num-cascades 10 --n-feat0 48 --sens-n-feat0 24 --n-history 3 \
            -b 1 -l 2e-4 -r 20 --num-workers 4 --use-checkpoint \
            --fg-loss --area-weight --grad-clip 0.1 --accum-steps 2 --l1-weight 0.1 \
            -t "$DATA_ROOT/train/" -v "$DATA_ROOT/val/" \
            --k-fold 5 --val-fold "$FOLD" --kfold-seed 430 --acc "$PROBE_ACC" \
            --sens-chunk "$K" -e 1 --max-iters "$ITERS" \
            -n "$NET" --run-tag "$NET" 2>&1 | tee "$OUT/train_k${K}.txt"
        { A=${PIPESTATUS[0]}; set +x; } 2>/dev/null
        [ "$A" -ne 0 ] && STATUS="$A"
    done
    echo
    echo "===== chunk 별 s/iter · peak · 마지막 Loss (로그 Time 은 -r 20 간격의 합) ====="
    echo "  chunk>1 은 비트 동일이 아니다(누산 순서). 같은 seed 라 Loss 는 앞자리가 같아야 정상."
    for K in $TRAIN_CHUNKS; do
        awk -v k="$K" '/Iter = \[/ { t=$0; sub(/.*Time = /,"",t); sub(/s.*/,"",t); spi=t/20
                                     l=$0; sub(/.*Loss = /,"",l); sub(/ .*/,"",l); loss=l }
                       /\[mem\]/   { m=$0 }
                       END { printf "chunk=%-3s %.2f s/iter · Loss %s · %s\n", k, spi, loss, m }' \
            "$OUT/train_k${K}.txt" 2>/dev/null
    done
fi

echo
echo "=========================================================="
echo "[DONE ] $SLOT $STAGE   $(date '+%F %T')   exit=$STATUS"
echo "[저장] $OUT/"
echo "[판정] OK 로 끝나는 가장 큰 chunk 가 천장. peak > 7.0GB 면 실학습엔 쓰지 말 것(폭 변동 여유)."
echo "       chunk=1 대비 시간이 안 줄면 레버 C 는 폐기 — 값은 어차피 같으니 잃을 게 없다."
echo "       상한은 감도맵 U-Net 몫(전체 U-Net FLOP 의 약 1/4)이다. 그 이상은 안 나온다."
echo "=========================================================="
exit "$STATUS"
