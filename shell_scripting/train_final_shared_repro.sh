#!/usr/bin/env bash
# 목적: 재현성 검증용 — shared 의 epoch 0 만 한 번 더 돌려 로그를 남긴다 (대회측 요구).
# 실행:  bash shell_scripting/train_final_shared_repro.sh
#
# ⚠️ 본 런과 **같은 GPU 를 동시에 못 쓴다**(본 런이 6.0GB allocated / 7.5GB reserved 를 잡는다).
#    train_final_shared.sh 가 끝난 뒤 큐에 넣을 것. 1 epoch = 학습 6.9h + val 0.25h ≈ 7.1h.
#
# 비교 기준: logs/train_final_shared_20260807_130330.log.recovered 의 epoch 0
#   TrainLoss = 0.06902  ValLoss = 0.09072
#   [gnorm] epoch 0 n=2179 med=0.0951 p90=0.262 max=2.57 clip(0.1) 발동 46.9%
#   [mem]   epoch 0 peak allocated = 6.00GB / reserved = 7.52GB
#
# 인자는 _final_base.sh + _aug_shared.sh 를 그대로 써서 본 런과 한 글자도 다르지 않게 하고,
# 여기서 바꾸는 건 딱 둘이다 — -e 30 → 1, NET 뒤에 _repro.
#   · -e 1 이어도 epoch 0 의 LR 은 본 런과 동일하다. shared 는 warmup 이 없고
#     --lr-decay-start-epoch 5 → decay 시작이 5450 step 인데 epoch 0 은 0~1089 step 이라
#     통째로 상수 구간(=base 2e-4)에 들어간다. total_iters 가 65370 이든 2180 이든 안 닿는다.
#   · NET 을 바꾸는 건 필수다. 같은 이름이면 본 런의 체크포인트를 덮어쓴다.
#
# 로그에서 본 런과 **다르게 나오는 게 정상**인 두 줄(둘 다 -e 때문이고 학습에는 영향 없음):
#   [scheduler] ... total_iters=2180   (본 런 65370)
#   Epoch = [  0/  1] ...              (본 런 [  0/ 30])
#
# ⚠️ 비트 단위 일치까지는 보장 못 한다. seed_fix 가 cudnn.deterministic=True·benchmark=False 는
#    켜지만 torch.use_deterministic_algorithms(True) 는 안 켠다(utils/common/utils.py:76).
#    일부 CUDA backward 커널의 atomicAdd 누적 순서가 run 마다 달라질 수 있어 끝자리가 흔들릴 수 있다.
#    그렇다고 지금 그 플래그를 켜면 **본 런과 다른 코드 경로**로 도는 것이라 재현성 검증이 아니게 된다.
set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$(cd "$(dirname "$0")" && pwd)/.." || exit 1
source shell_scripting/_final_base.sh
source shell_scripting/_aug_shared.sh

run_final "${RUN}_shared_repro" \
    --acc all -e 1 --lr-decay-start-epoch 5 --lr-min-factor 0.09 \
    --early-stop-patience 8
