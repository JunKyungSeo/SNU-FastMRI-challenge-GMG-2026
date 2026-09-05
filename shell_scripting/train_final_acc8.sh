#!/usr/bin/env bash
# 목적: 최종 run 2단계 — acc8 전문가 18 epoch (shared best 에서 출발). tail 2 를 더해 총 20.
# 실행:  bash shell_scripting/train_final_acc8.sh
set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$(cd "$(dirname "$0")" && pwd)/.." || exit 1
source shell_scripting/_final_base.sh
source shell_scripting/_aug_acc8.sh

[ -f "$SHARED_BEST" ] || { echo "[STOP] shared 가 없다: $SHARED_BEST"; exit 2; }

# ⚠️ --lr-min-factor 가 다른 단계(0.09)와 다른 건 오타가 아니다. 총 20 epoch 중 마지막 2 는
# tail 슬롯이 hold-out 을 합쳐 돌므로 본 런은 18 에서 멈춘다. build_scheduler 의 horizon 은
# --num-epochs 에서만 오므로(train_part.py:288) -e 18 이면 cosine 이 18 에서 하한을 찍는다
# → 하한을 tail 의 시작 LR 3.019e-5 (= 원래 20-epoch 곡선의 로컬 18 지점) 에 맞춘다.
#    0.151 = 0.09 + 0.91·0.5·(1+cos(π·10/12)).  -e 를 바꾸면 이 값도 다시 계산할 것.
run_final "${RUN}_acc8" \
    --acc all -e 18 --max-iters 2179 \
    --warmup-iters 3270 --lr-decay-start-epoch 8 --lr-min-factor 0.151 \
    --early-stop-patience "${ESP:-6}" \
    --init-from "$SHARED_BEST"
