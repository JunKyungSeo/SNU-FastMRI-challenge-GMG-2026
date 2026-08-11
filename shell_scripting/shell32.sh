#!/usr/bin/env bash
# ============================================================
# 목적: 용량 레버 천장 — n-history / num-cascades / feature 폭을 어디까지 올릴 수 있나  [shell32.sh]
#   ⚠️ OOM 은 이 슬롯의 정상 출력이다. config 당 프로세스 1개(VRAM 초기화). 최악 shape 384×480×15코일.
#
# 실행:  bash shell_scripting/shell32.sh                       # 천장 탐침만 (~15분, 학습 없음)
#        STAGE=time bash shell_scripting/shell32.sh            # 실측 s/iter·peak (~30분)
#        STAGE=ab AB_CAND="10:3:48 12:11:48" bash shell_scripting/shell32.sh   # 성능 A/B (수 시간)
#        STAGE=all bash shell_scripting/shell32.sh             # 셋 다 순서대로
# ============================================================
set -uo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# ===== [이번 run] =====
STAGE="${STAGE:-ceiling}"                        # ceiling | time | ab | all
DATA_ROOT="${DATA_ROOT:-$([ -d /workspace/Data ] && echo /workspace/Data || echo /root/Data)}"
FOLD="${FOLD:-2}"
PROBE_ACC="${PROBE_ACC:-8}"                      # 약한 팔(acc8) 기준. shell30 과 동일
TIME_CAND="${TIME_CAND:-10:3:48 12:11:48}"       # nc:nh:nf (sens=nf/2)
TIME_ITERS="${TIME_ITERS:-60}"                   # REPORT 의 3배 — 쓸 수 있는 보고 줄이 2개는 나오게
REPORT="${REPORT:-20}"                           # -r. 로그의 Time 은 이 간격 동안의 합이다
AB_CAND="${AB_CAND:-10:3:48 12:11:48}"
AB_ITERS="${AB_ITERS:-600}"                      # 기준선 기준 iter. 느린 arm 은 같은 시간이 되게 줄인다
PROBE_TXT="${PROBE_TXT:-}"                       # STAGE=ab 단독 실행 시 이전 ceiling 결과 경로

SLOT="$(basename "$0" .sh)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="shell_scripting/logs"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SLOT}_${STAGE}_${TS}.log"
exec > >(tee -a "$LOG") 2>&1
OUT="shell_scripting/results/lever_${TS}"; mkdir -p "$OUT"
[ -z "$PROBE_TXT" ] && PROBE_TXT="$OUT/probe.txt"

echo "=========================================================="
echo "[START] $SLOT   $(date '+%F %T')   STAGE=$STAGE"
echo "[BR   ] $(git rev-parse --abbrev-ref HEAD) ($(git rev-parse --short HEAD))"
echo "[CONF ] 기준선 nc10 nh3 nf48/sens24 · fold=$FOLD acc=$PROBE_ACC · data=$DATA_ROOT"
echo "[LOG  ] $LOG"
echo "=========================================================="

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
echo "[GPU  ] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)  used=${USED}MiB"
if [ -n "$USED" ] && [ "$USED" -ge 500 ]; then
    echo "[STOP] GPU가 사용 중(${USED}MiB)이다. 빈 상태라야 peak 이 의미가 있다."; exit 2
fi

STATUS=0
BASE_ARGS=( --model promptmr --readout-crop
    -b 1 -l 2e-4 -r "$REPORT" --num-workers 4 --use-checkpoint
    --fg-loss --area-weight --grad-clip 0.1 --accum-steps 2
    --l1-weight "${L1_W:-0.1}"
    -t "$DATA_ROOT/train/" -v "$DATA_ROOT/val/"
    --k-fold 5 --val-fold "$FOLD" --kfold-seed 430 --acc "$PROBE_ACC" )

# nc nh nf → 탐침 1회. sens 는 폭의 절반(기존 48:24 관례 유지)
W() { python scripts/probe_oom_promptmr.py --crop-h 384 --width 480 --coils 15 \
        --with-optimizer --reps 3 --num-cascades "$1" --n-history "$2" \
        --n-feat0 "$3" --sens-n-feat0 "$(($3 / 2))" 2>&1 | tee -a "$PROBE_TXT"; }

# nc nh nf → probe.txt 에서 fwd+bwd 초 (없으면 빈 문자열)
fwdbwd() { awk -v a="cascades=$1" -v b="n_hist=$2" -v c="n_feat0=$3" \
    '$1=="OK" && $2==a && $3==c && $5==b && $6=="ckpt=True" {
        t=$11; sub("fwdbwd=","",t); sub("s$","",t); print t; exit }' "$PROBE_TXT" 2>/dev/null; }

# 학습 1 arm. 결과 폴더는 이 슬롯 전용 이름(lever_*)이라 재실행하면 지우고 다시 쓴다(shell28 과 동일)
train_arm() {   # $1=nc $2=nh $3=nf $4=iters $5=net
    rm -rf "../result/$5"
    set -x
    python train.py "${BASE_ARGS[@]}" --num-cascades "$1" --n-history "$2" \
        --n-feat0 "$3" --sens-n-feat0 "$(($3 / 2))" -e 1 --max-iters "$4" \
        -n "$5" --run-tag "$5" 2>&1 | tee "$OUT/$5.txt"
    { local st=${PIPESTATUS[0]}; set +x; } 2>/dev/null
    [ "$st" -ne 0 ] && STATUS="$st"
    return 0
}

if [ "$STAGE" = ceiling ] || [ "$STAGE" = all ]; then
    echo; echo "=== A) 기준선 — 실측 3.35GB 근처에 안 떨어지면 아래는 다 무의미하다 ==="
    W 10 3 48
    python scripts/probe_oom_promptmr.py --crop-h 384 --width 480 --coils 15 --with-optimizer \
        --reps 3 --num-cascades 10 --n-history 3 --n-feat0 48 --sens-n-feat0 24 \
        --no-use-checkpoint 2>&1 | tee -a "$PROBE_TXT"

    echo; echo "=== B) n-history 축 (nc10 nf48) — 원본값 11 을 되찾을 수 있나 ==="
    for NH in 5 7 9 11 16; do W 10 "$NH" 48; done

    echo; echo "=== C) num-cascades 축 (nh3 nf48) ==="
    for NC in 12 16 20 24 32; do W "$NC" 3 48; done

    echo; echo "=== D) feature 폭 축 (nc10 nh3, 피라미드 동반 확대) ==="
    for NF in 56 64 80 96; do W 10 3 "$NF"; done

    echo; echo "=== E) 조합 후보 (10:3:64 는 D 에서 이미 쟀다) ==="
    for C in 12:11:48 16:5:48 12:11:64; do
        IFS=: read -r NC NH NF <<<"$C"; W "$NC" "$NH" "$NF"
    done

    B0="$(fwdbwd 10 3 48)"
    echo; echo "===== 통과 (peak 오름차순) · h/epoch ≈ 4.95h × 시간비 ====="
    grep '^OK' "$PROBE_TXT" | awk -v b="${B0:-0}" '
      { nc=nh=nf=ck=pa=fb=pk="?"
        for(i=1;i<=NF;i++){ n=split($i,kv,"="); if(n<2) continue; k=kv[1]; v=kv[2]
          if(k=="cascades")nc=v; else if(k=="n_hist")nh=v; else if(k=="n_feat0")nf=v
          else if(k=="ckpt")ck=v; else if(k=="param")pa=v
          else if(k=="fwdbwd"){sub("s$","",v); fb=v}
          else if(k=="peak"){sub("GB$","",v); pk=v} }
        he=(b>0 ? sprintf("%.2f", 4.95*fb/b) : 0)
        printf "%s  nc=%-3s nh=%-3s nf=%-3s ckpt=%-5s param=%-7s %5.2fs %5.2fGB %6.2fh/ep\n",
               pk, nc,nh,nf,ck,pa, fb+0, pk+0, he+0 }' \
      | sort -g | sed 's/^[0-9.][0-9.]* *//'
    echo; echo "===== OOM ====="
    if grep -q '^OOM' "$PROBE_TXT"; then
        awk '$1=="OOM"{ nc=nh=nf=ck=pa="?"
            for(i=1;i<=NF;i++){ n=split($i,kv,"="); if(n<2) continue
              if(kv[1]=="cascades")nc=kv[2]; else if(kv[1]=="n_hist")nh=kv[2]
              else if(kv[1]=="n_feat0")nf=kv[2]; else if(kv[1]=="ckpt")ck=kv[2]
              else if(kv[1]=="param")pa=kv[2] }
            printf "  nc=%-3s nh=%-3s nf=%-3s ckpt=%-5s param=%s\n", nc,nh,nf,ck,pa }' "$PROBE_TXT"
    else echo "  (없음 — 이 격자는 8GB 안에서 안 터졌다. 천장은 메모리가 아니라 시계다)"; fi
fi

if [ "$STAGE" = time ] || [ "$STAGE" = all ]; then
    echo; echo "=== 실측: 탐침의 시간비가 진짜 학습에 그대로 옮겨지는지 ==="
    for C in $TIME_CAND; do
        IFS=: read -r NC NH NF <<<"$C"; N="lever_t_${NC}_${NH}_${NF}"
        echo; echo "---------- arm $C ----------"
        train_arm "$NC" "$NH" "$NF" "$TIME_ITERS" "$N"
    done
    echo; echo "===== arm 별 s/iter · [mem] 실측 peak ====="
    echo "  (로그의 Time 은 iter 당이 아니라 -r $REPORT 간격 동안의 합이다 → 나눠서 쓴다)"
    for C in $TIME_CAND; do
        IFS=: read -r NC NH NF <<<"$C"; N="lever_t_${NC}_${NH}_${NF}"
        awk -v n="$C" -v R="$REPORT" '
          /Iter = \[/ { it=$0; sub(/.*Iter = \[ */,"",it); sub(/\/.*/,"",it)
                        tot=$0; sub(/.*Iter = \[[^\/]*\//,"",tot); sub(/\].*/,"",tot)
                        t=$0;  sub(/.*Time = /,"",t);  sub(/s.*/,"",t)
                        if (it+0 > 0) { spi=t/R; N2=tot+0 } }
          /\[mem\]/ { m=$0 }
          END { if (spi>0) printf "%-12s %.2f s/iter · %d iter/ep → %.1f h/ep · %s\n", n, spi, N2, spi*N2/3600, m
                else       printf "%-12s (보고 줄이 부족하다 — TIME_ITERS 를 -r 보다 크게)\n", n }' \
          "$OUT/$N.txt" 2>/dev/null
    done
fi

if [ "$STAGE" = ab ] || [ "$STAGE" = all ]; then
    echo; echo "=== 성능 A/B — 같은 벽시계 시간을 주고 ValLoss 비교 ==="
    echo "    (iter 를 맞추면 느린 모델에 조용히 더 많은 연산을 주는 셈이라 결과가 부풀려진다)"
    B0="$(fwdbwd 10 3 48)"
    for C in $AB_CAND; do
        IFS=: read -r NC NH NF <<<"$C"; N="lever_ab_${NC}_${NH}_${NF}"
        T="$(fwdbwd "$NC" "$NH" "$NF")"
        if [ -n "$B0" ] && [ -n "$T" ]; then
            I=$(awk -v i="$AB_ITERS" -v b="$B0" -v t="$T" 'BEGIN{printf "%d", i*b/t}')
        else
            I="$AB_ITERS"; echo "  [주의] $C 의 탐침 시간을 못 찾았다 → iter 균등($I). 시간 맞춤 아님."
        fi
        echo; echo "---------- arm $C  (iter=$I) ----------"
        train_arm "$NC" "$NH" "$NF" "$I" "$N"
    done
    echo; echo "===== arm 별 ValLoss (낮을수록 좋다) ====="
    for C in $AB_CAND; do
        IFS=: read -r NC NH NF <<<"$C"; N="lever_ab_${NC}_${NH}_${NF}"
        printf "%-12s %s\n" "$C" "$(grep -o 'ValLoss = [0-9.e+-]*' "$OUT/$N.txt" 2>/dev/null | tail -1)"
    done
fi

echo
echo "=========================================================="
echo "[DONE ] $SLOT   $(date '+%F %T')   exit=$STATUS"
echo "[저장] $PROBE_TXT · arm 로그 $OUT/"
echo "[판정] 천장 = OK 로 끝나는 가장 큰 값. 단 peak > 7.0GB 면 실학습엔 쓰지 말 것(폭 변동·단편화 여유)."
echo "       메모리보다 시계가 먼저 막는다 — h/epoch 열을 보고 남은 대회 시간으로 자를 것."
echo "       파라미터가 크면 추론시간(점수항)도 손해다. SSIM 이 비슷하면 작은 쪽이 이긴다."
echo "[다음] 고른 config 는 resume 호환 필드라 처음부터 학습이다(--init-from 도 shape 로 죽는다)."
echo "=========================================================="
exit "$STATUS"
