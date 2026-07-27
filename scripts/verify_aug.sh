#!/usr/bin/env bash
# ============================================================
# 증강(MRIAugmentor) 배선 검증 — GPU 불필요, 큐에 넣지 말고 그냥 실행한다.
#
#   목적: 로컬(torch 없음)에서 작성한 증강 배선이 서버에서 실제로 동작하는지 한 번에 확인.
#   실행: bash scripts/verify_aug.sh          # repo 어디서 실행해도 됨
#
#   검사 3종:
#     [1] 단위 테스트 — flip 정확성·ACS 보존·bbox 동기·결정성·스케줄·가중치·배선 불변식
#     [2] CLI 도달성  — train.py의 --aug-* 인자가 AugmentConfig까지 전달되는지
#     [3] off 동등성  — --aug-on 없이 만든 DataTransform이 기존과 같은 6-튜플인지
#
#   전부 통과하면 마지막에 "ALL CHECKS PASSED"가 찍히고 exit 0.
#   실패하면 어느 검사인지 출력하고 exit 1 → 그 출력을 그대로 Claude에게 붙여주면 된다.
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1
echo "[verify_aug] repo root = $(pwd)"
echo "[verify_aug] python     = $(python --version 2>&1)"
echo

FAILED=0

echo "=== [1/3] 단위 테스트 (scripts/tests/test_augment.py) ==="
python -m pytest scripts/tests/test_augment.py -v
if [ $? -ne 0 ]; then echo "[FAIL] 단위 테스트"; FAILED=1; else echo "[ok] 단위 테스트"; fi
echo

echo "=== [2/3] CLI 인자가 AugmentConfig까지 도달하는지 ==="
python - <<'PY'
import sys
sys.argv = ["train.py", "--aug-on", "--aug-delay", "2", "--no-aug-flip", "-e", "10"]
from train import parse
from utils.data.transforms import AugmentConfig, MRIAugmentor

a = parse()
cfg = AugmentConfig(
    on=a.aug_on, schedule=a.aug_schedule, delay=a.aug_delay,
    total_epochs=a.num_epochs, exp_decay=a.aug_exp_decay,
    max_prob=a.aug_max_prob, flip=a.aug_flip, mask=a.aug_mask,
    weight_flip=a.aug_weight_flip, weight_mask=a.aug_weight_mask,
    orig_mask_prob=a.aug_orig_mask_prob,
)
m = MRIAugmentor(cfg)
m.set_epoch(9)
p, p_flip, p_mask = m.probs()
print(f"  flip={cfg.flip} (기대 False) | p={p:.4f} p_flip={p_flip:.4f} p_mask={p_mask:.4f}")
assert cfg.flip is False, "--no-aug-flip이 반영되지 않았다"
assert p > 0.5, f"10ep 런의 마지막 epoch인데 p가 너무 낮다: {p}"
assert p_flip == 0.0, f"flip이 꺼졌는데 p_flip이 0이 아니다: {p_flip}"
print("  OK")
PY
if [ $? -ne 0 ]; then echo "[FAIL] CLI 도달성"; FAILED=1; else echo "[ok] CLI 도달성"; fi
echo

echo "=== [3/3] 증강 off일 때 기존과 동일한 6-튜플인지 ==="
python - <<'PY'
import numpy as np
from utils.data.transforms import DataTransform

mask = np.zeros(368, dtype=np.float32); mask[::4] = 1.0; mask[170:199] = 1.0
rng = np.random.default_rng(0)
img = rng.standard_normal((4, 64, 368)) + 1j * rng.standard_normal((4, 64, 368))
kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img, axes=(-2, -1)),
                                     axes=(-2, -1), norm="ortho"), axes=(-2, -1)).astype(np.complex64)
target = rng.random((384, 384)).astype(np.float32)

out = DataTransform(isforward=False, max_key="max")(mask, kspace, target, {"max": 1.0}, "knee_acc4_1.h5", 0)
print(f"  반환 길이 = {len(out)} (기대 6)")
assert len(out) == 6, f"플래그 전부 off인데 {len(out)}-튜플"
print("  OK")
PY
if [ $? -ne 0 ]; then echo "[FAIL] off 동등성"; FAILED=1; else echo "[ok] off 동등성"; fi
echo

if [ "$FAILED" -eq 0 ]; then
  echo "=========================================="
  echo " ALL CHECKS PASSED — 증강 배선 검증 완료"
  echo "=========================================="
  exit 0
else
  echo "=========================================="
  echo " 실패한 검사가 있다 — 위 출력을 그대로 전달할 것"
  echo "=========================================="
  exit 1
fi
