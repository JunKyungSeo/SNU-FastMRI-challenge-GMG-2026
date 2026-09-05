#!/usr/bin/env bash
# 목적: acc8 early-stop 취소 — ep7 재개 → tail → acc4 6+2 → merge → 채점 → 재현성 1 epoch
# 실행 전: 파이프라인·tail 프로세스 kill + ../result/<RUN>_acc8_tail 삭제 (사람이)
set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$(cd "$(dirname "$0")" && pwd)/.." || exit 1
source shell_scripting/_final_base.sh

pgrep -f "train.py" >/dev/null && { echo "[STOP] train.py 가 아직 돈다 — 먼저 정지"; exit 1; }
[ -e "../result/${RUN}_acc8_tail/checkpoints/model.pt" ] && { echo "[STOP] acc8_tail 결과가 남아 있다 — rm 후 다시"; exit 1; }

LOG="shell_scripting/logs/final_earlycancel_$(date +%Y%m%d_%H%M%S).log"
mkdir -p shell_scripting/logs shell_scripting/results
exec > >(tee -a "$LOG") 2>&1
echo "[PIPE ] 시작 $(date '+%F %T')  RUN=$RUN  LOG=$LOG"

step() {   # $1=단계명  $2...=명령. 성공·실패 모두 메일. KILLOK=1 이면 실패해도 다음 단계로.
    local n="$1"; shift
    "$@"; local st=$?
    bash shell_scripting/notify.sh "$n $([ $st -eq 0 ] && echo 완료 || echo "종료 exit=$st")" "$(date '+%F %T')  로그: $LOG"
    [ $st -eq 0 ] || [ "${KILLOK:-0}" = 1 ] || exit $st
}

# ep7 재개. ESP=0 으로 early-stop 을 끈다(resume 일치 필드에서 제외돼 있어 조정 가능).
# FORCE=1 은 덮어쓰기가 아니라 run_final 의 model.pt 가드를 통과시키는 용도 — --resume 이 붙어
# model.pt(epoch=7)의 optimizer·EMA 까지 복원하고 이어 돈다.
ESP=0 RESUME=1 FORCE=1 \
step "1/6 acc8 ep7~17 재개" bash shell_scripting/train_final_acc8.sh

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
