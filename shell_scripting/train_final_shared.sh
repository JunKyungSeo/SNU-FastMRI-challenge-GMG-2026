#!/usr/bin/env bash
# 목적: 최종 run 1단계 — shared 30 epoch (acc4·acc8 공용 초기값)
# 실행:  bash shell_scripting/train_final_shared.sh
set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$(cd "$(dirname "$0")" && pwd)/.." || exit 1
source shell_scripting/_final_base.sh
source shell_scripting/_aug_shared.sh

run_final "${RUN}_shared" \
    --acc all -e 30 \
    --lr-decay-start-epoch 5 --lr-min-factor 0.09 \
    --early-stop-patience 8
