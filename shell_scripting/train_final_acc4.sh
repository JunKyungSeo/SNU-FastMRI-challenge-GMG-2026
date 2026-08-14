#!/usr/bin/env bash
# 목적: 최종 run 3단계 — acc4 전문가 7 epoch (shared best 에서 출발). tail 2 를 더해 총 9.
# 실행:  bash shell_scripting/train_final_acc4.sh
set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$(cd "$(dirname "$0")" && pwd)/.." || exit 1
source shell_scripting/_final_base.sh
source shell_scripting/_aug_acc4.sh

[ -f "$SHARED_BEST" ] || { echo "[STOP] shared 가 없다: $SHARED_BEST"; exit 2; }

# ⚠️ acc8 과 같은 이유로 --lr-min-factor 가 0.09 가 아니다. 총 9 epoch 중 마지막 2 는 tail 이
# 가져가므로 본 런은 7 에서 멈추고, 하한을 tail 의 시작 learning rate 1.090e-4 (= 원래
# 10-epoch 곡선의 로컬 7 지점) 에 맞춘다.  0.545 = 0.09 + 0.91·0.5·(1+cos(π·3/6)).
run_final "${RUN}_acc4" \
    --acc all -e 7 --max-iters 2179 \
    --warmup-iters 2180 --lr-decay-start-epoch 4 --lr-min-factor 0.545 \
    --early-stop-patience 6 \
    --init-from "$SHARED_BEST"
