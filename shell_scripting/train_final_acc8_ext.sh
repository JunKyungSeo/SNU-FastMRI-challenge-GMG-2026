#!/usr/bin/env bash
# 목적: acc8 연장 4 epoch — 본 런 ep17 과 같은 learning rate·증강으로 이어 학습
# 실행:  bash shell_scripting/train_final_acc8_ext.sh
set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$(cd "$(dirname "$0")" && pwd)/.." || exit 1
source shell_scripting/_final_base.sh
source shell_scripting/_aug_acc8.sh

MAIN="../result/${RUN}_acc8/checkpoints/best_model.pt"
[ -f "$MAIN" ] || { echo "[STOP] acc8 본 런이 없다: $MAIN"; exit 2; }

# learning rate 3.019e-5 **상수** — 본 런 cosine 이 ep18 에서 찍는 하한이자 tail 의 시작값이다.
# --warmup-iters·--lr-decay-start-epoch 를 안 주면 build_scheduler 가 None 을 돌려 상수가 된다.
# affine: 본 런 ep17 은 peak 0.5 였다. ext 는 로컬 epoch 이 0 부터라 _aug_acc8.sh 의 delay 5 를
# 그대로 두면 4 epoch 내내 0 이 된다 → delay·peak 를 0 으로 눌러 처음부터 0.5 로 맞춘다.
run_final "${RUN}_acc8_ext" \
    --acc all -e 4 --max-iters 2179 \
    -l 3.019e-5 --affine-delay 0 --affine-peak-epoch 0 \
    --early-stop-patience 0 \
    --init-from "$MAIN"
