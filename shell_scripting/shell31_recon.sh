#!/usr/bin/env bash
# ============================================================
# 목적: promptmrp_0801 채점 — acc4+acc8 병합(MoE) 또는 shared 단일             [shell31_recon.sh]
#
# 실행:  bash shell_scripting/shell31_recon.sh                 # merged (기본, 없으면 병합부터)
#        TARGET=shared bash shell_scripting/shell31_recon.sh   # shared 만
# ============================================================
set -uo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

LEADERBOARD="${LEADERBOARD:-/root/Data/leaderboard}"
RUN="promptmrp_0801"
TARGET="${TARGET:-merged}"
case "$TARGET" in
    merged) NET="${RUN}_moe" ;;
    shared) NET="${RUN}_shared" ;;
    *) echo "[STOP] TARGET 은 merged|shared"; exit 2 ;;
esac

SLOT="$(basename "$0" .sh)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="shell_scripting/logs"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SLOT}_${TS}.log"
exec > >(tee -a "$LOG") 2>&1
OUT_DIR="shell_scripting/results"; mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/recon_eval_${NET}_${TS}.txt"

echo "=========================================================="
echo "[START] $SLOT   $(date '+%F %T')   TARGET=$TARGET  NET=$NET"
echo "[LOG  ] $LOG   [채점] $LEADERBOARD"
echo "=========================================================="

CKPT="../result/$NET/checkpoints/best_model.pt"
if [ "$TARGET" = merged ] && [ ! -f "$CKPT" ]; then
    python scripts/merge_moe.py --acc4-net "${RUN}_acc4" --acc8-net "${RUN}_acc8" --out-net "$NET" || exit $?
fi
[ -f "$CKPT" ] || { echo "[STOP] 체크포인트 없음: $CKPT"; exit 2; }

set -x
python -u recon_eval.py -n "$NET" -p "$LEADERBOARD" | tee "$OUT"
{ STATUS=${PIPESTATUS[0]}; set +x; } 2>/dev/null

echo "=========================================================="
echo "[DONE ] $SLOT   $(date '+%F %T')   exit=$STATUS"
echo "[저장] 점수: $OUT   (git add 해서 공유)"
echo "=========================================================="

bash "$SCRIPT_DIR/notify.sh" "$NET 채점 완료 (exit=$STATUS)" \
    "$(TZ=Asia/Seoul date '+%F %T') KST"$'\n\n'"$(grep -E 'SSIM_(full|bbox)|Recon Time' "$OUT")"
exit "$STATUS"
