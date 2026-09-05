#!/usr/bin/env bash
# 목적: acc8 ep17 재개 → ext 4ep → tail → acc4 8+2 → merge → 채점
# 실행 전: 기존 파이프라인·train.py 프로세스 kill (사람이)
set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$(cd "$(dirname "$0")" && pwd)/.." || exit 1
source shell_scripting/_final_base.sh

pgrep -f "train.py" >/dev/null && { echo "[STOP] train.py 가 아직 돈다 — 먼저 정지"; exit 1; }

LOG="shell_scripting/logs/final_acc8_ext_add_$(date +%Y%m%d_%H%M%S).log"
mkdir -p shell_scripting/logs shell_scripting/results
exec > >(tee -a "$LOG") 2>&1
echo "[PIPE ] 시작 $(date '+%F %T')  RUN=$RUN  LOG=$LOG"

step() {   # $1=단계명  $2...=명령. 성공·실패 모두 메일. KILLOK=1 이면 실패해도 다음 단계로.
    local n="$1"; shift
    "$@"; local st=$?
    bash shell_scripting/notify.sh "$n $([ $st -eq 0 ] && echo 완료 || echo "종료 exit=$st")" "$(date '+%F %T')  로그: $LOG"
    [ $st -eq 0 ] || [ "${KILLOK:-0}" = 1 ] || exit $st
}

# ep17 중간부터 재개. latest.pt(accum 경계 저장)에서 skip_iters 만큼 흘려보낸 뒤 이어 밟는다.
ESP=0 RESUME=1 FORCE=1 \
step "1/6 acc8 ep17 재개" bash shell_scripting/train_final_acc8.sh

step "2/6 acc8 ext 4ep"  bash shell_scripting/train_final_acc8_ext.sh

# tail 은 ext 의 best 에서 출발해야 한다 — MAIN 을 넘겨 기본값(_acc8)을 덮는다.
MAIN="../result/${RUN}_acc8_ext/checkpoints/best_model.pt" \
step "3/6 acc8 tail"     bash shell_scripting/train_final_tail_acc8.sh

KILLOK=1 \
step "4/6 acc4 8ep"      bash shell_scripting/train_final_acc4.sh   # 손으로 끊어도 tail 로 진행
step "5/6 acc4 tail"     bash shell_scripting/train_final_tail_acc4.sh

NET="${RUN}_moe"
OUT="shell_scripting/results/recon_eval_${NET}_$(date +%Y%m%d_%H%M%S).txt"
python scripts/merge_moe.py --acc4-net "${RUN}_acc4_tail" --acc8-net "${RUN}_acc8_tail" --out-net "$NET" \
    && python -u recon_eval.py -n "$NET" -p "$DATA_ROOT/leaderboard" | tee "$OUT"
ST=$?
bash shell_scripting/notify.sh "6/6 채점 $([ $ST -eq 0 ] && echo 완료 || echo "실패 exit=$ST")" "$(cat "$OUT" 2>/dev/null)"
[ $ST -eq 0 ] || exit $ST
echo "[PIPE ] 끝 $(date '+%F %T')  점수=$OUT"
