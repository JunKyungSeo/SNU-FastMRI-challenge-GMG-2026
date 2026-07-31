#!/usr/bin/env bash
# ============================================================
# 목적: 180° 회전 증강을 넣을지 값싸게 가른다 — 방향 민감도 탐침. 학습 0.   [shell29.sh]
#
#   기존 체크포인트로 hold-out 을 네 방향(원본 / 좌우flip / 상하flip / 180°)으로 채점한다.
#   k-space·마스크·target·bbox 를 함께 돌리므로 "같은 무릎을 뒤집어 찍은 획득"이 되고,
#   원본과의 격차가 그 방향 증강이 **새로 가르칠 수 있는 양의 상한**이다.
#
# [왜] 180° 는 90° 와 달리 축을 안 바꾼다(readout 은 readout, PE 는 PE). 마스크 변환도
#      지금 _flip_lr 이 쓰는 roll(flip(M),1) 그대로라 새 위험이 없고, 닫힌 형태라 보간도 없다
#      (샘플 실측 relmaxerr 2.2e-07). 즉 **기계적으로는 지금 당장 넣을 수 있다.**
#      남는 건 "그게 좋은 데이터냐" 하나인데, 상하 성분은 관절선을 위쪽 1/3 → 아래쪽 1/3 로
#      옮긴다(=병변 박스가 채점에 없는 자리로 간다). A/B 는 60h 급이라 먼저 이걸로 가른다.
#
# [해석]
#   대조군 좌우 격차 : flip 증강으로 학습된 체크포인트면 0 근처여야 정상.
#                      크게 나오면 증강이 실제로 안 걸린 것이니 그것부터 확인.
#   신규분 180° 격차 : <0.002  → 모델이 이미 둔감. 배울 게 없으니 접는다.
#                      <0.010  → 회수 상한이 그 값. 60h A/B 를 태울 값은 아니다.
#                      >=0.010 → 여지는 실재. 단 채점셋엔 뒤집힌 무릎이 없으므로 이 값이
#                                리더보드 이득으로 그대로 바뀌지는 않는다. A/B 근거로만.
#   180° 격차 ≈ 상하 격차면 새 정보는 전부 상하 성분이다(좌우는 이미 쓰고 있으니).
#   ⚠️ 이건 상한이지 점수 예측이 아니다. shell28 의 offset 탐침과 같은 성격이다.
#
# [비용] 10 볼륨 × 약 27 슬라이스 × 4 방향 × 0.73s ≈ 13 분/acc. 학습 0, 디스크 쓰기 0.
#        PROBE_VOLS=0 이면 hold-out 전수(40 볼륨, 약 55 분/acc).
#
# 실행:
#   bash shell_scripting/shell29.sh                    # acc8, 10 볼륨 (기본)
#   ACCS="4 8" bash shell_scripting/shell29.sh         # 두 가속률 다
#   PROBE_VOLS=0 bash shell_scripting/shell29.sh       # hold-out 전수
#   PROBE_VOLS=2 bash shell_scripting/shell29.sh       # 스모크(경로만 확인, 3분)
#   PROBE_NET=promptmr_kfold_f2 bash shell_scripting/shell29.sh   # 증강 없이 학습된 모델로
# ============================================================
set -uo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# ===== [이번 run] =====
DATA_ROOT="${DATA_ROOT:-$([ -d /workspace/Data ] && echo /workspace/Data || echo /root/Data)}"
# 탐침 대상. 증강 유무에 따라 읽는 법이 달라진다:
#   flip 으로 학습된 모델(_aug) → 좌우 격차가 대조군(0 이어야 정상), 180° 격차 ≈ 순수 상하 몫.
#   증강 없이 학습된 모델        → 좌우 격차도 정보다(flip 이 회수했을 양의 상한).
#   ⚠️ shell28 의 기본값(promptmr_moe_f2_shared_aug)은 서버에 없다 — 그걸 복사하지 말 것.
#      증강 없이 학습된 f2(promptmr_kfold_f2)도 체크포인트가 안 남아 있어서, 좌우 격차는
#      flip 학습본을 대조군으로 쓰는 수밖에 없다.
PROBE_NET="${PROBE_NET:-promptmr_moe_f2_aug_rm}"
ACCS="${ACCS:-8}"
PROBE_VOLS="${PROBE_VOLS:-10}"      # 0 = hold-out 전수
# PROBE_KFOLD=0 이면 고정분할 → hold-out 은 $DATA_ROOT/val.
# >0 이면 그 fold 가 hold-out (학습 때와 **같은** fold/seed 를 줄 것 — 어긋나면 학습 볼륨으로 재게 된다).
PROBE_KFOLD="${PROBE_KFOLD:-5}"
FOLD="${FOLD:-2}"
KFOLD_SEED="${KFOLD_SEED:-430}"

SLOT="$(basename "$0" .sh)"
TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p shell_scripting/logs
LOG="shell_scripting/logs/${SLOT}_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=========================================================="
echo "[START] $SLOT  방향 민감도 탐침  $(date '+%F %T')"
echo "[LOG  ] $LOG"
echo "[CFG  ] net=$PROBE_NET  acc=[$ACCS]  volumes=$PROBE_VOLS  fold=$FOLD"
echo "=========================================================="

# --- 가드: GPU 비었는지 (used < 500MiB) ---
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
echo "[GPU  ] used=${USED}MiB"
if [ -n "$USED" ] && [ "$USED" -ge 500 ]; then
    echo "[STOP] GPU가 사용 중(${USED}MiB)이다. 앞 슬롯이 끝난 뒤 실행할 것."
    exit 2
fi

# --- 가드: 탐침 체크포인트 존재 ---
if [ ! -e "../result/$PROBE_NET/checkpoints/best_model.pt" ]; then
    echo "[STOP] ../result/$PROBE_NET/checkpoints/best_model.pt 없음. PROBE_NET= 으로 지정하라."
    echo "       쓸 수 있는 체크포인트:"
    for n in $(ls -1 ../result 2>/dev/null); do
        [ -e "../result/$n/checkpoints/best_model.pt" ] || continue
        echo "            $n"
    done
    exit 2
fi

if [ "$PROBE_KFOLD" -gt 0 ]; then
    HOLDOUT=( --k-fold "$PROBE_KFOLD" --val-fold "$FOLD" --kfold-seed "$KFOLD_SEED"
              --data-path-train "$DATA_ROOT/train/" --data-path-val "$DATA_ROOT/val/" )
else
    HOLDOUT=( -p "$DATA_ROOT/val" )
fi

STATUS=0
for ACC in $ACCS; do
    echo
    echo "===== acc$ACC ====="
    set -x
    python scripts/probe_orient.py -n "$PROBE_NET" --acc "$ACC" \
        --max-volumes "$PROBE_VOLS" "${HOLDOUT[@]}"
    { RC=$?; set +x; } 2>/dev/null
    [ "$RC" -ne 0 ] && STATUS="$RC"
done

echo
echo "===== 다음 행동 ====="
echo "  180° 격차 <0.002  → 접는다. transforms.py 에 손대지 않는다."
echo "  180° 격차 >=0.010 → --aug-rot180 구현 후 shell22 형식으로 A/B (기본 off = 비트 동일)."
echo "  ⚠️ 격차는 '증강이 새로 가르칠 수 있는 양의 상한'이지 리더보드 이득 예측이 아니다."
echo "=========================================================="
echo "[DONE ] $SLOT  $(date '+%F %T')  exit=$STATUS"
echo "=========================================================="
exit "$STATUS"
