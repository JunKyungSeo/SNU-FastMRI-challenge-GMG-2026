#!/usr/bin/env bash
# ============================================================
# 목적: 2.5D 인접 슬라이스(--num-adj-slices 3) 사전확인 — 짧은 학습으로 A/B   [shell30.sh]
#
#   shell28·shell29 와 같은 성격의 **값싼 사전확인** 슬롯이다. 본실험이 60h 급이라
#   1~2h 로 "걸지 말지"만 먼저 가른다. 변경점은 --num-adj-slices 하나뿐(A=1, B=3).
#   (번호 주의: shell29 는 팀원의 180° 방향 민감도 탐침이다. 다른 실험이니 헷갈리지 말 것.)
#
# [왜 지금] strategy/model_strategy.md §3 재측정으로 **옛 기각 근거가 반증**됐다.
#   옛 근거: "병변 bbox 인접 상관 0.08~0.59 로 낮다 → 병변 blur → SSIM_bbox 악화".
#   재측정(170파일·병변박스 7394개·대조군 36761개, scripts/adjacent_slice_corr.py):
#     · gap1 병변 bbox r 이 **같은 크기 대조군보다 +0.12 높다**(acc4 0.627 vs 0.510).
#     · gap1 SSIM 은 대조군과 동일(0.680 vs 0.682).
#     → 낮아 보였던 절대값은 '작은 패치'의 분산 정규화 아티팩트였다. 병변 특이 blur 근거 없음.
#   단 gap2 이상은 병변 SSIM 이 대조군 아래로 뒤집힌다(0.396<0.487) → **a=1(3슬라이스)만** 쓴다.
#   ⚠️ 이 재측정이 보인 것은 "이웃이 쓸 정보를 갖고 있다"까지다. "네트가 뭉개지 않고 쓴다"는
#      미증명 — blur 는 데이터가 아니라 모델·손실의 성질이다. 그래서 이 슬롯이 필요하다.
#
# [무엇을 보나]
#   ① 안 터지는가        : 코일축이 15→45 로 3배가 된다. VRAM peak 와 conv in_chans 정합.
#   ② 손실 궤적          : adj=3 이 adj=1 대비 튀거나 느려지지 않는가.
#   ③ 1 iter 시간        : h5 읽기 3배 + 코일축 3배. 본실험이 몇 배가 되는지 여기서 잰다.
#   ④ ValLoss            : 같은 iter 예산에서 더 낮은가(신호가 있으면 본실험 근거).
#
# [해석]
#   ② 발산·NaN           → 배선 버그다. 스케일(감도맵 RSS 그룹별 정규화)부터 의심할 것.
#   ③ 3배 이상 느려짐    → 본실험 60h→180h. --num-workers 상향으로 회수되는지 먼저 볼 것.
#   ④ ValLoss 개선 없음  → iter 예산이 짧아서일 수 있다(정규화 효과는 늦게 나온다). 판정 보류하고
#                          ADJ_ITERS 를 키워 재시도. 반대로 **악화면** 병변 blur 가 실재한다는
#                          첫 실측 신호다 → model_strategy §3.1 에 기록하고 기각 유지.
#   ⚠️ TrainLoss 는 두 arm 의 손실 함수가 같으므로(둘 다 fg masked SSIM) 나란히 비교 가능하다.
#      shell28 의 B(ℓ1) arm 과 달리 여기선 TrainLoss 도 같은 자다.
#
# [주의] --coil-compress 와 배타적이다(load_data 가 fail-fast). 코일축을 SVD 로 섞으면
#        서로 다른 슬라이스가 한 가상코일로 뭉개진다.
#        --remask 와는 충돌하지 않는다 — 마스크는 샘플당 하나를 3슬라이스에 동일 적용한다.
#        ⚠️ 이 배선은 아직 GPU 에서 한 번도 안 돌았다(로컬에 torch 없음). 이 슬롯이 첫 실측이다.
#
# 실행:  bash shell_scripting/shell30.sh                  # A(adj=1) + B(adj=3)
#        ARMS=3 bash shell_scripting/shell30.sh           # adj=3 만
#        ADJ_ITERS=120 bash shell_scripting/shell30.sh    # 스모크(터지는지만, ~20분)
#        ADJ_ITERS=2000 bash shell_scripting/shell30.sh   # 길게(ValLoss 신호까지)
# ============================================================
set -uo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# ===== [이번 run] =====
DATA_ROOT="${DATA_ROOT:-$([ -d /workspace/Data ] && echo /workspace/Data || echo /root/Data)}"
FOLD="${FOLD:-2}"
PROBE_ACC="${PROBE_ACC:-8}"     # acc8 우선: 약한 팔(bbox 0.9068)이고 gap1 상관은 acc4와 동등(0.624)
ADJ_ITERS="${ADJ_ITERS:-600}"   # arm 당 iter. 600 이면 arm 당 ~45분(4.6 s/iter 기준)
ARMS="${ARMS:-1 3}"             # 돌릴 --num-adj-slices 값들

BASE_ARGS=( --model promptmr --readout-crop
    --num-cascades 10 --n-feat0 48 --sens-n-feat0 24 --n-history 3
    -b 1 -l 2e-4 -r 50 --num-workers 4 --use-checkpoint
    --fg-loss --area-weight --grad-clip 0.1 --accum-steps 2
    -t "$DATA_ROOT/train/" -v "$DATA_ROOT/val/"
    --k-fold 5 --val-fold "$FOLD" --kfold-seed 430 --acc "$PROBE_ACC" )

SLOT="$(basename "$0" .sh)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="shell_scripting/logs"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SLOT}_f${FOLD}_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

ARM_DIR="shell_scripting/results/adj_probe_${TS}"; mkdir -p "$ARM_DIR"

echo "=========================================================="
echo "[START] $SLOT   $(date '+%F %T')"
echo "[BR   ] $(git rev-parse --abbrev-ref HEAD) ($(git rev-parse --short HEAD))"
echo "[CONF ] fold=$FOLD acc=$PROBE_ACC iters=$ADJ_ITERS arms='$ARMS'  data=$DATA_ROOT"
echo "[근거 ] model_strategy §3 재측정 — 옛 기각 근거(bbox blur) 반증. a=1 한정."
echo "=========================================================="

python -c "import cv2" 2>/dev/null || { echo "[STOP] cv2 import 실패 — bash /root/restore_env.sh 후 재실행."; exit 2; }

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
echo "[GPU  ] used=${USED}MiB"
if [ -n "$USED" ] && [ "$USED" -ge 500 ]; then
    echo "[STOP] GPU가 사용 중(${USED}MiB)이다. 빈 상태에서 실행할 것."; exit 2
fi

STATUS=0
for ADJ in $ARMS; do
    NET="promptmr_probe_adj${ADJ}_f${FOLD}"
    echo
    echo "---------- arm: --num-adj-slices $ADJ  (net=$NET) ----------"
    set -x
    python train.py "${BASE_ARGS[@]}" \
        --num-adj-slices "$ADJ" -e 1 --max-iters "$ADJ_ITERS" \
        -n "$NET" --run-tag "probe_adj${ADJ}" 2>&1 | tee "$ARM_DIR/adj${ADJ}.txt"
    { A=${PIPESTATUS[0]}; set +x; } 2>/dev/null
    echo "[arm adj=$ADJ] exit=$A"
    [ "$A" -ne 0 ] && STATUS="$A"
done

echo
echo "=========================================================="
echo "===== 결과: 손실 궤적 (같은 손실 함수라 나란히 비교 가능) ====="
for ADJ in $ARMS; do
    f="$ARM_DIR/adj${ADJ}.txt"
    [ -f "$f" ] || continue
    echo "--- adj=$ADJ ---"
    grep -o 'Iter = \[ *[0-9]*/.*Loss = [0-9.e+-]*' "$f" \
        | sed -E 's/Iter = \[ *([0-9]+).*Loss = /  iter \1  loss /' | head -14
    echo "  [시간] $(grep -o 'Time = [0-9.]*s' "$f" | tail -3 | tr '\n' ' ')"
    echo "  [VRAM] $(grep -o '\[mem\].*' "$f" | tail -1)"
    echo "  [Val ] $(grep -o 'ValLoss = [0-9.e+-]*' "$f" | tail -1)"
done
echo "=========================================================="
echo "[DONE ] $SLOT   $(date '+%F %T')   exit=$STATUS"
echo "[저장] arm 로그: $ARM_DIR/adj*.txt"
echo "[판정] ① 안 터짐 ② 궤적 안 튐 ③ iter 시간 배수 ④ ValLoss 비교."
echo "       ③이 3배 이상이면 본실험 60h→180h — --num-workers 상향으로 회수되는지 먼저 볼 것."
echo "       ④가 악화면 병변 blur 실재의 첫 실측 신호다 → model_strategy §3.1 에 기록하고 기각 유지."
echo "=========================================================="
exit "$STATUS"
