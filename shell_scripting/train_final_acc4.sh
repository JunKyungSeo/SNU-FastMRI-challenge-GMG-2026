#!/usr/bin/env bash
# 목적: 최종 run 3단계 — acc4 전문가 10 epoch (shared best 에서 출발)
# 실행:  bash shell_scripting/train_final_acc4.sh
set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$(cd "$(dirname "$0")" && pwd)/.." || exit 1
source shell_scripting/_final_base.sh
source shell_scripting/_aug_acc4.sh

[ -f "$SHARED_BEST" ] || { echo "[STOP] shared 가 없다: $SHARED_BEST"; exit 2; }

run_final "${RUN}_acc4" \
    --acc all -e 10 --max-iters 2179 \
    --warmup-iters 2180 --lr-decay-start-epoch 4 --lr-min-factor 0.09 \
    --early-stop-patience 6 \
    --init-from "$SHARED_BEST"
