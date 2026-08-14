#!/usr/bin/env bash
# 목적: 최종 run 2~7단계 연속 — acc8 18 → tail → acc4 7 → tail → merge → 채점 → 재현성 1 epoch
set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$(cd "$(dirname "$0")" && pwd)/.." || exit 1
source shell_scripting/_final_base.sh

LOG="shell_scripting/logs/final_pipeline_$(date +%Y%m%d_%H%M%S).log"
mkdir -p shell_scripting/logs shell_scripting/results
exec > >(tee -a "$LOG") 2>&1
echo "[PIPE ] 시작 $(date '+%F %T')  RUN=$RUN  LOG=$LOG"

step() {   # $1=단계명  $2...=명령. 성공·실패 모두 메일. KILLOK=1 이면 실패해도 다음 단계로.
    local n="$1"; shift
    "$@"; local st=$?
    bash shell_scripting/notify.sh "$n $([ $st -eq 0 ] && echo 완료 || echo "종료 exit=$st")" "$(date '+%F %T')  로그: $LOG"
    [ $st -eq 0 ] || [ "${KILLOK:-0}" = 1 ] || exit $st
}

step "1/6 acc8 18ep" bash shell_scripting/train_final_acc8.sh
step "2/6 acc8 tail" bash shell_scripting/train_final_tail_acc8.sh
KILLOK=1 \
step "3/6 acc4 6ep"  bash shell_scripting/train_final_acc4.sh   # 손으로 끊어도 tail 로 진행
step "4/6 acc4 tail" bash shell_scripting/train_final_tail_acc4.sh

NET="${RUN}_moe"
OUT="shell_scripting/results/recon_eval_${NET}_$(date +%Y%m%d_%H%M%S).txt"
python scripts/merge_moe.py --acc4-net "${RUN}_acc4_tail" --acc8-net "${RUN}_acc8_tail" --out-net "$NET" \
    && python -u recon_eval.py -n "$NET" -p "$DATA_ROOT/leaderboard" | tee "$OUT"
ST=$?
bash shell_scripting/notify.sh "5/6 채점 $([ $ST -eq 0 ] && echo 완료 || echo "실패 exit=$ST")" "$(cat "$OUT" 2>/dev/null)"
[ $ST -eq 0 ] || exit $ST

step "6/6 재현성 1ep" bash shell_scripting/train_final_shared_repro.sh
echo "[PIPE ] 끝 $(date '+%F %T')  점수=$OUT"
