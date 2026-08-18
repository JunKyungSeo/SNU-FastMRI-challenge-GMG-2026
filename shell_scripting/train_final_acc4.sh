#!/usr/bin/env bash
# 목적: 최종 run 3단계 — acc4 전문가 8 epoch (shared best 에서 출발). tail 2 를 더해 총 10.
# 실행:  bash shell_scripting/train_final_acc4.sh
set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$(cd "$(dirname "$0")" && pwd)/.." || exit 1
source shell_scripting/_final_base.sh
source shell_scripting/_aug_acc4.sh

[ -f "$SHARED_BEST" ] || { echo "[STOP] shared 가 없다: $SHARED_BEST"; exit 2; }

# ⚠️ --lr-min-factor 0.545 는 곡선의 **끝값**만 정한다 — 하한을 tail 시작 learning rate
# 1.090e-4 에 맞추는 값이라 -e 를 바꿔도 그대로 둔다.
# decay 를 warmup 직후(2)로 앞당긴 건 acc8 실측 때문이다: 2e-4 상수 구간에서 val 이 악화되고
# 감쇠 구간에서야 회복되는데 회복이 훨씬 느리다(acc8 은 감쇠 5 epoch 을 쓰고도 미회복).
# early-stop 0 — -e 8 이면 (행수−1−best) 가 7 까지 올라가 patience 6 에 닿는다. -e 6 일 때
# 안 걸렸던 건 최대치가 5 라 구조적으로 못 닿은 것뿐이다(acc8 은 이걸로 ep6 에서 멈췄다).
# affine 은 _aug_acc4.sh 의 delay 3·peak 5 를 여기서 덮는다 — delay 0·peak 3 이라 ep0 부터
# 올라가 ep3 에 0.5 로 닿는다. _aug_acc4.sh 를 안 고치는 건 tail 이 그 파일을 공유하기
# 때문이다: 거기 delay 3 이 남아 있어야 tail 2 epoch 에 affine 이 안 걸린다.
run_final "${RUN}_acc4" \
    --acc all -e 8 --max-iters 2179 \
    --warmup-iters 2180 --lr-decay-start-epoch 2 --lr-min-factor 0.545 \
    --affine-delay 0 --affine-peak-epoch 3 \
    --early-stop-patience 0 \
    --init-from "$SHARED_BEST"
