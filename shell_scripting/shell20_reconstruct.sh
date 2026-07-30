#!/usr/bin/env bash
set -uo pipefail
export PYTHONUNBUFFERED=1   

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# --- 로그 파일 (이름에 timestamp) ---
SLOT="$(basename "$0" .sh)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="shell_scripting/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SLOT}_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

NET="promptmr_moe"
LEADERBOARD="${LEADERBOARD:-/root/Data/leaderboard}"
OUT="../result/${NET}/recon_eval_moe_${TS}.txt"   # 점수 사본(timestamp 로 덮어쓰기 방지)

echo "=========================================================="
echo "[START] $SLOT   $(date '+%F %T')   NET=$NET"
echo "[CWD  ] $(pwd)"
echo "[LOG  ] $LOG"
echo "[BR   ] $(git branch --show-current 2>/dev/null) ($(git rev-parse --short HEAD 2>/dev/null))"
echo "=========================================================="

set -x
python -u recon_eval.py -n "$NET" -p "$LEADERBOARD" | tee "$OUT"
{ STATUS=${PIPESTATUS[0]}; set +x; } 2>/dev/null   # tee 가 아니라 python 의 종료코드를 잡는다

echo "=========================================================="
echo "[DONE ] $SLOT   $(date '+%F %T')   recon_eval exit=$STATUS"
echo "[저장] 점수: $OUT   (+ 로그: $LOG)"
echo "=========================================================="
exit "$STATUS"
