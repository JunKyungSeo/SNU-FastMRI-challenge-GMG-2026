#!/usr/bin/env bash
# ============================================================
# GPU 작업 큐 슬롯 — shell25(MoE) 결과 채점: recon_eval                     [shell26.sh]
#
# 목적: shell25 가 만든 promptmr_moe_f<FOLD>(MoE 병합본)와 promptmr_moe_f<FOLD>_shared
#   (공유 사전학습 = 단일 모델 baseline)를 leaderboard 로 채점한다. shell25 헤더대로
#   **판정의 정본은 채점**이다 — val 은 세 단계 구성이 달라 직접 비교가 안 되므로,
#   MoE 와 단일 baseline 을 각각 채점해 SSIM_full·SSIM_bbox·시간 3축으로 비교한다.
#   (체크포인트가 없는 net 은 건너뛴다 — shell25 미완이면 둘 다 SKIP 된다.)
#
#   ⚠️ 선행: shell25(STAGE=all 또는 merge)가 끝나 promptmr_moe_f<FOLD>/checkpoints/best_model.pt
#      가 생긴 뒤에 돌린다. 없으면 SKIP 만 하고 끝난다.
#   ⚠️ 실행 순서: GPU 가 빈 뒤에 돌린다(아래 GPU 가드가 사용 중이면 멈춘다).
#   ※ recon_eval 은 acc4·acc8 을 다 재구성한 뒤 점수를 한 번에 출력한다(중간이 조용한 건 정상).
#   ※ MoE 는 추론 시 마스크로 acc4/acc8 전문가를 자동 라우팅한다(merge_moe 가 심은 threshold).
#      아키텍처·threshold 는 체크포인트에서 읽으므로 여기서 넘길 인자 없다.
#
# 실행:  bash shell_scripting/shell26.sh                    # fold 2, MoE + 단일 baseline 채점
#        FOLD=3 bash shell_scripting/shell26.sh             # 다른 fold
#        NETS="promptmr_moe_f2" bash shell_scripting/shell26.sh   # MoE 만
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
# shell25 가 AUG=1(기본)이면 net 이름에 _aug 를 붙인다 — 채점 대상도 같은 규칙으로 맞춘다.
AUG="${AUG:-1}"; ASUF=""; [ "$AUG" = "1" ] && ASUF="_aug"
# MoE 먼저, 단일 baseline 다음 — 둘 다 채점해야 MoE 가 이득인지 판정된다(shell25 ⑤ 와 동일).
NETS="${NETS:-promptmr_moe_f${FOLD}${ASUF} promptmr_moe_f${FOLD}_shared${ASUF}}"

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

# --- 가드: GPU 비었는지 (used < 500MiB) ---
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
        echo "[SKIP ] $NET — ../result/$NET/checkpoints/best_model.pt 없음 (shell25 미완이거나 이 서버에 없음)"
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
echo "[비교] MoE(promptmr_moe_f${FOLD}) vs 단일(promptmr_moe_f${FOLD}_shared) 를 full·bbox·시간 3축으로."
echo "       MoE 는 모델 2개라 디스크·로딩이 크다 — SSIM 이득이 시간/크기 손해를 넘는지까지 봐야 순위 판단이 된다."
echo "=========================================================="

# ==========================================================
#  ↓↓↓  이메일 알림 (notify.sh 재사용) — shell25 종료 + 채점 결과  ↓↓↓
#     이번 run 의 점수 사본(results/recon_eval_*_$TS.txt)에서 SSIM 요약만 뽑아 본문에 넣는다:
#     Leaderboard SSIM_full·SSIM_bbox + acc4/acc8 각각의 SSIM_full·SSIM_bbox.
# ==========================================================
NOTIFY="$SCRIPT_DIR/notify.sh"
MAIL_BODY="shell25 종료 — recon_eval 채점 결과"$'\n'"서버 $(date '+%F %T') · 한국 $(date -u -d '+9 hours' '+%F %T') KST · host $(hostname) · fold $FOLD"
FOUND=0
shopt -s nullglob
for f in "$OUT_DIR"/recon_eval_*_"${TS}".txt; do
    [ -s "$f" ] || continue                       # 빈 파일(SKIP·중단)은 건너뜀
    net=$(basename "$f" .txt); net=${net#recon_eval_}; net=${net%_"$TS"}
    MAIL_BODY+=$'\n\n'"■ $net"$'\n'"$(grep -E 'SSIM_(full|bbox)' "$f")"
    FOUND=1
done
shopt -u nullglob
[ "$FOUND" = 0 ] && MAIL_BODY+=$'\n\n'"(채점된 net 없음 — 체크포인트 미존재로 전부 SKIP)"
if [ -f "$NOTIFY" ]; then
    bash "$NOTIFY" "shell25 채점 완료 (fold $FOLD)" "$MAIL_BODY" || echo "[NOTIFY] 메일 발송 실패 — 무시하고 종료(exit=$STATUS)"
else
    echo "[NOTIFY] $NOTIFY 없음 — 이메일 생략"
fi

exit "$STATUS"
