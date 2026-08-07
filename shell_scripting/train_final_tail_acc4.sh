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

# LR 은 본 런 로컬 ep8 지점(6.350e-5)을 이어받아 원래 하한 1.8e-5 까지 (min-factor = 1.8/6.35).
run_final "${RUN}_acc4_tail" \
    --holdout-tail --acc all -e 2 --max-iters 2179 \
    -l 6.350e-5 --lr-decay-start-epoch 0 --lr-min-factor 0.283 \
    --early-stop-patience 0 \
    --init-from "$MAIN" \
    && promote_last_epoch "${RUN}_acc4_tail"
