cd "$(cd "$(dirname "$0")" && pwd)" || exit 1
source shell_scripting/_final_base.sh
python recon_eval.py \
  -n "${RUN}_moe" \
  -p "${LEADERBOARD_DATA_DIR:-$DATA_ROOT/leaderboard}"
