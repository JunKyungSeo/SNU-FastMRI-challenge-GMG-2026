#!/usr/bin/env bash
# 목적: acc4 전문가 tail 2 epoch — hold-out(val fold)까지 합쳐 학습 (전역 38~40)
# 실행:  bash shell_scripting/train_final_tail_acc4.sh
set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$(cd "$(dirname "$0")" && pwd)/.." || exit 1
source shell_scripting/_final_base.sh
source shell_scripting/_aug_acc4.sh

MAIN="../result/${RUN}_acc4/checkpoints/best_model.pt"
[ -f "$MAIN" ] || { echo "[STOP] acc4 본 런이 없다: $MAIN"; exit 2; }

# learning rate 는 본 런 끝(1.090e-4)을 이어받아 원래 하한 1.8e-5 까지 (min-factor = 1.8/10.90).
# ⚠️ 8+2 와 달리 원래 10-epoch 곡선에서 3 epoch(ep7→10) 걸릴 하강을 tail 2 epoch 에 압축한다
# — hold-out 이 25% 를 차지하는 분포에서 그만큼 높은 learning rate 로 돈다.
run_final "${RUN}_acc4_tail" \
    --holdout-tail --acc all -e 2 --max-iters 2179 \
    -l 1.090e-4 --lr-decay-start-epoch 0 --lr-min-factor 0.165 \
    --early-stop-patience 0 \
    --init-from "$MAIN" \
    && promote_last_epoch "${RUN}_acc4_tail"
