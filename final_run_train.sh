#!/usr/bin/env bash
# 최종 run 재현 학습 — shared 25 → acc8 18+2 → acc4 6+2 → merge
# 실행:  bash final_run_train.sh
set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$(cd "$(dirname "$0")" && pwd)" || exit 1   # 
source shell_scripting/_final_base.sh
export RUN   # 하위 슬롯도 같은 RUN 을 쓴다. 우리 환경에서 시험만 할 땐 RUN=... 로 갈라 준다

LOG="shell_scripting/logs/final_run_train_$(date +%Y%m%d_%H%M%S).log"
mkdir -p shell_scripting/logs
exec > >(tee -a "$LOG") 2>&1
echo "[RUN  ] $RUN   LOG=$LOG"

# 산출물 경로. train.py·recon_eval.py 가 ../result 를 고정으로 쓰므로(recon_eval.py 는 수정 금지)
# 인자를 주면 ../result 를 그리로 심볼릭 링크한다. 안 주면 기존대로 ../result 에 쌓는다.
OUT="${1:-}"
if [ -n "$OUT" ]; then
    mkdir -p "$OUT" || exit 1
    OUT="$(cd "$OUT" && pwd)"
    [ -e ../result ] && [ ! -L ../result ] && { echo "[STOP] ../result 가 실디렉토리다 — 옮기고 다시"; exit 1; }
    [ -L ../result ] && rm ../result
    ln -s "$OUT" ../result
    echo "[out  ] ../result → $OUT"
fi

# shared 는 -e 30 그대로 돌려 30-epoch cosine 을 밟게 하고 25 에서 세운다 — -e 25 로 바꾸면
# horizon 이 다시 그려져 원본과 다른 곡선이 된다(ep25 지점 3.538e-5 를 재현 못 함).
NET="${RUN}_shared"
NPY="../result/$NET/val_loss_log_$NET.npy"
PT="../result/$NET/checkpoints/model.pt"
bash shell_scripting/train_final_shared.sh &
SH=$!
STOPPED=0
while kill -0 "$SH" 2>/dev/null; do
    n=$(python3 -c "import numpy,sys;print(len(numpy.load(sys.argv[1])))" "$NPY" 2>/dev/null || echo 0)
    if [ "$n" -ge 25 ] && [ "$PT" -nt "$NPY" ]; then
        echo "[stop ] shared $n epoch · 체크포인트 저장 확인 → 정지"
        sleep 10
        pkill -f -- "--run-tag $NET\$"
        STOPPED=1
        break
    fi
    sleep 60
done
wait "$SH" 2>/dev/null; ST=$?
# 25 에 못 닿고 끝난 경우(가드 [STOP]·early-stop·크래시) 여기서 세운다 — 안 그러면 기존 run 의
# best_model.pt 를 초기값으로 삼아 acc8 이 그냥 시작된다.
[ "$STOPPED" = 1 ] || { echo "[STOP] shared 가 25 epoch 에 못 닿고 끝났다 (exit=$ST)"; exit 1; }

bash shell_scripting/train_final_acc8.sh      || exit $?
bash shell_scripting/train_final_tail_acc8.sh || exit $?
bash shell_scripting/train_final_acc4.sh      || exit $?
bash shell_scripting/train_final_tail_acc4.sh || exit $?

python scripts/merge_moe.py --acc4-net "${RUN}_acc4_tail" --acc8-net "${RUN}_acc8_tail" --out-net "${RUN}_moe" || exit $?
echo "[DONE ] $(date '+%F %T')  → ../result/${RUN}_moe/checkpoints/best_model.pt"
