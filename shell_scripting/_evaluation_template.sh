#!/usr/bin/env bash
# ============================================================
# 목적: recon_eval
# run tag : <채점할 net_name>
# 실행:  bash shell_scripting/shellN.sh
#   ⚠️ recon_eval 은 acc4·acc8 을 다 처리한 뒤 점수를 한 번에 출력한다
#      (진행 중 화면이 조용한 건 정상, 멈춘 게 아니다).
# ============================================================
set -uo pipefail
export PYTHONUNBUFFERED=1

# --- repo 루트로 이동 (recon_eval.py 는 repo 루트, '../result' 기준으로 돈다) ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# ===== [고정 인자] =====
LEADERBOARD="${LEADERBOARD:-/root/Data/leaderboard}"

# ===== [이번 run] 여기만 채운다 =====
NET="<채점할 net_name>"

# --- 로그 ---
SLOT="$(basename "$0" .sh)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="shell_scripting/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SLOT}_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

OUT="../result/${NET}/recon_eval_${TS}.txt"   # 점수 사본 (timestamp 로 덮어쓰기 방지)

echo "=========================================================="
echo "[START] $SLOT   $(date '+%F %T')   NET=$NET"
echo "[CWD  ] $(pwd)"
echo "[LOG  ] $LOG"
echo "[BR   ] $(git branch --show-current 2>/dev/null) ($(git rev-parse --short HEAD 2>/dev/null))"
echo "=========================================================="

# ==========================================================
#  ↓↓↓  실행부 — reconstruct + 채점  ↓↓↓
# ==========================================================
set -x
python -u recon_eval.py -n "$NET" -p "$LEADERBOARD" | tee "$OUT"
{ STATUS=${PIPESTATUS[0]}; set +x; } 2>/dev/null   # tee 가 아니라 python 의 종료코드를 잡는다

echo "=========================================================="
echo "[DONE ] $SLOT   $(date '+%F %T')   exit=$STATUS"
echo "[저장] 점수: $OUT   (+ 재구성 h5: ../result/${NET}/reconstructions_leaderboard/)"
echo "=========================================================="
exit "$STATUS"
