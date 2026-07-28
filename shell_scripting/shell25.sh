#!/usr/bin/env bash
# ============================================================
# GPU 작업 큐 슬롯 — shell22(MRAugment ON arm) 결과 채점: recon_eval        [shell25.sh]
#
# 목적: shell22 가 학습한 promptmr_kfold_f<FOLD>_aug 를 leaderboard 로 채점한다.
#   shell22 는 증강 A/B 의 ON arm 이라 점수는 OFF arm(promptmr_kfold_f<FOLD>, baseline)과
#   나란히 봐야 뜻이 있다 → 기본 NETS 에 둘 다 넣되, 체크포인트가 없는 net 은 건너뛴다
#   (지금 이 서버엔 baseline 체크포인트가 없어 ON arm 만 채점되고 OFF 는 SKIP 된다).
#
#   ⚠️ 실행 순서: shell22 학습이 끝난 뒤(=GPU 가 빈 뒤)에 돌린다. best_model.pt 는 학습이
#      끝나야 최종본이 되고, 아래 GPU 가드가 사용 중이면 멈추므로 학습 중엔 실행되지 않는다.
#   ※ recon_eval 은 acc4·acc8 을 다 재구성한 뒤 점수를 한 번에 출력한다(중간이 조용한 건 정상).
#   ※ num_cascades/n_feat0 등 아키텍처는 recon_eval 이 체크포인트에 저장된 args 에서 읽는다
#      (test_part.load_model) — 여기서 넘길 필요 없다.
#
# 실행:  bash shell_scripting/shell25.sh                    # fold 2, ON(+있으면 OFF) 채점
#        FOLD=3 bash shell_scripting/shell25.sh             # 다른 fold
#        NETS="promptmr_kfold_f2_aug" bash shell_scripting/shell25.sh   # 특정 net 만
# ============================================================
set -uo pipefail
export PYTHONUNBUFFERED=1

# --- repo 루트로 이동 (recon_eval.py 는 repo 루트, '../result' 기준으로 돈다) ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# ===== [고정 인자] =====
LEADERBOARD="${LEADERBOARD:-/root/Data/leaderboard}"

# ===== [이번 run] 여기만 채운다 =====
FOLD="${FOLD:-2}"
NETS="${NETS:-promptmr_kfold_f${FOLD}_aug promptmr_kfold_f${FOLD}}"

# --- 로그 ---
SLOT="$(basename "$0" .sh)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="shell_scripting/logs"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SLOT}_f${FOLD}_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

# 점수 사본 — repo 안(results/)에 떨어뜨려 커밋·푸시로 모든 환경에서 공유한다(점수의 정본)
OUT_DIR="shell_scripting/results"; mkdir -p "$OUT_DIR"

echo "=========================================================="
echo "[START] $SLOT (fold $FOLD)   $(date '+%F %T')"
echo "[CWD  ] $(pwd)   [LOG] $LOG"
echo "[BR   ] $(git branch --show-current 2>/dev/null) ($(git rev-parse --short HEAD 2>/dev/null))"
echo "[대상 ] $NETS"
echo "[채점 ] $LEADERBOARD   (점수 = 0.5·SSIM_full + 0.5·SSIM_bbox + 추론시간)"
echo "=========================================================="

# --- 가드: leaderboard 데이터 (recon_eval 이 acc4/acc8 둘 다 요구, 없으면 즉시 assert 로 죽는다) ---
if [ ! -d "$LEADERBOARD/acc4" ] || [ ! -d "$LEADERBOARD/acc8" ]; then
    echo "[STOP] leaderboard 데이터 없음: $LEADERBOARD/{acc4,acc8}. LEADERBOARD=<경로> 로 지정할 것."
    exit 2
fi

# --- 가드: GPU 비었는지 (used < 500MiB) — shell22 가 GPU 를 쥐고 있으면 여기서 멈춘다 ---
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
echo "[GPU  ] used=${USED}MiB"
if [ -n "$USED" ] && [ "$USED" -ge 500 ]; then
    echo "[STOP] GPU가 사용 중(${USED}MiB)이다. 학습이 끝나 GPU가 빈 뒤 실행할 것."
    exit 2
fi

# ==========================================================
#  ↓↓↓  실행부 — net 마다 reconstruct + 채점 (각각 별도 프로세스 = VRAM 초기화)  ↓↓↓
# ==========================================================
STATUS=0
for NET in $NETS; do
    echo ""
    echo "---------- $NET ----------"
    if [ ! -f "../result/$NET/checkpoints/best_model.pt" ]; then
        echo "[SKIP ] $NET — ../result/$NET/checkpoints/best_model.pt 없음 (미학습이거나 이 서버에 없음)"
        continue
    fi
    OUT="$OUT_DIR/recon_eval_${NET}_${TS}.txt"
    set -x
    python -u recon_eval.py -n "$NET" -p "$LEADERBOARD" | tee "$OUT"
    { st=${PIPESTATUS[0]}; set +x; } 2>/dev/null   # tee 가 아니라 python 의 종료코드를 잡는다
    echo "[$NET] exit=$st   점수: $OUT"
    [ "$st" -ne 0 ] && STATUS=$st
done

echo ""
echo "=========================================================="
echo "[DONE ] $SLOT (fold $FOLD)   $(date '+%F %T')   exit=$STATUS"
echo "[저장] 점수 사본: $OUT_DIR/recon_eval_*_${TS}.txt   (+ 재구성 h5: ../result/<net>/reconstructions_leaderboard/)"
echo "[공유] git add $OUT_DIR/recon_eval_*_${TS}.txt 커밋·푸시하면 전 환경에서 점수가 보인다."
echo "[비교] ON(promptmr_kfold_f${FOLD}_aug) vs OFF(promptmr_kfold_f${FOLD}) 를 full·bbox·시간 3축으로 나란히 볼 것."
echo "       ⚠️ 두 arm 의 hold-out 분할이 같은지 먼저 확인 — 다르면(층화 교체 전후) A/B 비교가 무효다."
echo "=========================================================="
exit "$STATUS"
