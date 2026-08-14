#!/usr/bin/env bash
# 목적: 최종 run 3단계 — acc4 전문가 6 epoch (shared best 에서 출발). tail 2 를 더해 총 8.
# 실행:  bash shell_scripting/train_final_acc4.sh
set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$(cd "$(dirname "$0")" && pwd)/.." || exit 1
source shell_scripting/_final_base.sh
source shell_scripting/_aug_acc4.sh

[ -f "$SHARED_BEST" ] || { echo "[STOP] shared 가 없다: $SHARED_BEST"; exit 2; }

# ⚠️ acc8 과 같은 이유로 --lr-min-factor 가 0.09 가 아니다. 총 8 epoch 중 마지막 2 는 tail 이
# 가져가므로 본 런은 6 에서 멈추고, 하한을 tail 의 시작 learning rate 1.090e-4 에 맞춘다.
# min-factor 는 곡선의 **끝값**만 정하므로 -e 를 7→6 으로 줄여도 0.545 그대로다 — 대신 감쇠가
# ep4→6 (2 epoch) 으로 짧아져 ep5 를 1.7725e-4 대신 1.5450e-4 로 12.8% 낮게 지나간다.
run_final "${RUN}_acc4" \
    --acc all -e 6 --max-iters 2179 \
    --warmup-iters 2180 --lr-decay-start-epoch 4 --lr-min-factor 0.545 \
    --early-stop-patience 6 \
    --init-from "$SHARED_BEST"
