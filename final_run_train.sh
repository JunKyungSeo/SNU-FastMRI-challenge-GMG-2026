#!/usr/bin/env bash
# 최종 run 재현 학습 — shared 25 → acc8 18(+재개) → ext 4 → tail 2 → acc4 8 → tail 2 → merge
# 실행:  bash final_run_train.sh [RESULT_DIR] [DATA_DIR]
set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$(cd "$(dirname "$0")" && pwd)" || exit 1

# 데이터 경로. 안 주면 /workspace/Data 또는 /root/Data 를 자동으로 고른다.
# _final_base.sh 가 이 값으로 -t/-v 를 만들므로 source 보다 먼저 정한다.
[ -n "${2:-}" ] && export DATA_ROOT="$2"
source shell_scripting/_final_base.sh
export RUN   # 하위 슬롯도 같은 RUN 을 쓴다. 시험만 할 땐 RUN=... 로 갈라 준다

LOG="shell_scripting/logs/final_run_train_$(date +%Y%m%d_%H%M%S).log"
mkdir -p shell_scripting/logs
exec > >(tee -a "$LOG") 2>&1
echo "[RUN  ] $RUN   DATA=$DATA_ROOT   LOG=$LOG"

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

# acc8 은 두 번에 나눠 돈다. 첫 호출은 --early-stop-patience 6 이라 val 이 6 epoch 연속
# 개선되지 않으면 도중에 선다(정상 종료라 exit 0). 두 번째 호출이 ESP=0 으로 그 지점부터
# 이어받아 18 epoch 을 채운다. -e 는 resume 일치 필드라 바꿀 수 없어 이 방식을 쓴다.
bash shell_scripting/train_final_acc8.sh                          || exit $?
ESP=0 RESUME=1 FORCE=1 bash shell_scripting/train_final_acc8.sh   || exit $?

# 연장 4 epoch. learning rate 는 본 런 cosine 이 찍는 하한 3.019e-5 를 상수로 유지한다.
bash shell_scripting/train_final_acc8_ext.sh || exit $?

# tail 은 연장분의 best 에서 출발한다.
MAIN="../result/${RUN}_acc8_ext/checkpoints/best_model.pt" \
bash shell_scripting/train_final_tail_acc8.sh || exit $?

bash shell_scripting/train_final_acc4.sh      || exit $?
bash shell_scripting/train_final_tail_acc4.sh || exit $?

python scripts/merge_moe.py --acc4-net "${RUN}_acc4_tail" --acc8-net "${RUN}_acc8_tail" --out-net "${RUN}_moe" || exit $?
echo "[DONE ] $(date '+%F %T')  → ../result/${RUN}_moe/checkpoints/best_model.pt"
