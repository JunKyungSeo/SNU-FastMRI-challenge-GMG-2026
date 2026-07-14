"""C1 (--bbox-loss-lambda) 검증 — 전부 CPU, GPU/모델 불필요.

핵심 두 가지:
  1. 학습 손실이 **채점식과 동일**한가 (utils/common/metrics.py::ssim_bbox 는 고정 파일).
  2. 플래그를 끄면(λ=0) 기존 학습과 **비트 동일**한가.
"""
import json
import types

import h5py
import numpy as np
import pytest
import torch

from utils.common.loss_function import SSIMLoss, bbox_ssim_loss
from utils.common.metrics import SSIM, ssim_bbox
from utils.data.transforms import DataTransform

TRAIN_IMG = "/root/Data/train/image/knee_acc4_1.h5"


def _real_slice_with_boxes():
    """실제 데이터에서 박스가 있는 슬라이스 하나를 가져온다."""
    with h5py.File(TRAIN_IMG, "r") as hf:
        ann = json.loads(hf.attrs["annotations"])
        maximum = float(hf.attrs["max"])
        s = int(sorted(ann, key=lambda k: -len(ann[k]))[0])   # 박스 제일 많은 슬라이스
        target = hf["image_label"][s].astype(np.float32)
    return target, ann[str(s)], maximum


def test_bbox_loss_matches_scoring_function():
    """bbox_ssim_loss = 1 - mean(metrics.ssim_bbox) 를 소수점 8자리까지 재현해야 한다.
    (학습 신호와 채점식이 어긋나면 bbox를 미는 의미가 없다.)"""
    target, boxes, maximum = _real_slice_with_boxes()
    rng = np.random.default_rng(0)
    recon = target + rng.normal(0, target.std() * 0.05, target.shape).astype(np.float32)

    t = torch.from_numpy(target)[None]          # (1,H,W)
    r = torch.from_numpy(recon)[None]
    dr = torch.tensor([maximum], dtype=torch.float32)

    ours = bbox_ssim_loss(SSIMLoss(), r, t, dr, [boxes]).item()

    scorer = SSIM()
    vals = [ssim_bbox(scorer, r[0], t[0], b, maximum) for b in boxes]
    vals = [v for v in vals if v is not None]
    expected = 1.0 - float(np.mean(vals))

    assert len(vals) > 0, "테스트 슬라이스에 유효 박스가 없음"
    assert ours == pytest.approx(expected, abs=1e-8), f"학습손실 {ours} != 채점식 {expected}"


def test_win_size_filter_matches_scoring():
    """클리핑 후 w·h < win_size(7) 인 박스는 채점이 None → 학습에서도 제외돼야 한다."""
    ssim = SSIMLoss()
    t = torch.rand(1, 384, 384)
    r = t.clone()
    dr = torch.tensor([1.0])

    too_small = [{"x": 10, "y": 10, "width": 6, "height": 50}]     # w=6 < 7
    clipped_out = [{"x": 380, "y": 10, "width": 50, "height": 50}]  # 클리핑하면 w=4 < 7
    assert ssim_bbox(SSIM(), r[0], t[0], too_small[0], 1.0) is None
    assert ssim_bbox(SSIM(), r[0], t[0], clipped_out[0], 1.0) is None
    # 채점이 버리는 박스만 있으면 유효 박스 0개 → 손실 정확히 0
    assert bbox_ssim_loss(ssim, r, t, dr, [too_small + clipped_out]).item() == 0.0

    ok = [{"x": 10, "y": 10, "width": 7, "height": 7}]              # 경계값 7 = 유효
    assert ssim_bbox(SSIM(), r[0], t[0], ok[0], 1.0) is not None
    assert bbox_ssim_loss(ssim, r, t, dr, [ok]).item() == pytest.approx(0.0, abs=1e-6)  # 완벽복원 → 1-1=0


def test_no_box_slice_is_exactly_zero():
    """박스 없는 슬라이스 → λ항이 정확히 0 → 기존 손실과 동일(비트)."""
    ssim = SSIMLoss()
    r, t = torch.rand(1, 384, 384), torch.rand(1, 384, 384)
    dr = torch.tensor([1.0])
    term = bbox_ssim_loss(ssim, r, t, dr, [[]])
    assert term.item() == 0.0
    base = ssim(r, t, dr)
    assert (base + 0.5 * term).item() == base.item()      # 손실이 한 비트도 안 바뀜


def test_bbox_loss_has_gradient():
    """λ항이 실제로 학습 신호를 만드는지(=grad가 흐르는지)."""
    target, boxes, maximum = _real_slice_with_boxes()
    t = torch.from_numpy(target)[None]
    r = torch.zeros_like(t, requires_grad=True)
    term = bbox_ssim_loss(SSIMLoss(), r, t, torch.tensor([maximum]), [boxes])
    term.backward()
    assert r.grad is not None and r.grad.abs().sum() > 0


def _tuple_from_transform(return_bbox):
    """DataTransform을 run10/11 설정(readout-crop)으로 돌려 반환 튜플을 얻는다."""
    with h5py.File(TRAIN_IMG, "r") as hf:
        attrs = dict(hf.attrs)
        target = hf["image_label"][13]
    with h5py.File(TRAIN_IMG.replace("/image/", "/kspace/"), "r") as hf:
        kspace = hf["kspace"][13]
        mask = np.asarray(hf["mask"]) if "mask" in hf else np.ones(kspace.shape[-1])
    tf = DataTransform(isforward=False, max_key="max", readout_crop=True,
                       return_bbox=return_bbox)
    return tf(mask, kspace, target, attrs, "knee_acc4_1.h5", 13)


def test_off_is_bit_identical():
    """λ=0(=return_bbox off) → 튜플 길이·내용이 기존과 완전히 동일해야 한다."""
    old = _tuple_from_transform(return_bbox=False)   # 기존 코드가 받던 것
    assert len(old) == 6, f"off인데 6-튜플이 아님: len={len(old)}"

    new = _tuple_from_transform(return_bbox=True)    # C1 on
    assert len(new) == 7, "on이면 bbox가 맨 끝에 하나 더 붙어야 함"
    assert isinstance(new[-1], list)

    # 앞 6개(=모델이 실제로 먹는 것)는 on/off에서 비트 동일
    for i, (a, b) in enumerate(zip(old, new[:6])):
        if torch.is_tensor(a):
            assert torch.equal(a, b), f"튜플 원소 {i} 가 on/off에서 다름"
        else:
            assert a == b, f"튜플 원소 {i} 가 on/off에서 다름"


def test_val_loader_never_gets_bbox():
    """val/leaderboard 로더에는 bbox가 붙으면 안 된다(자연 분포·기존 언패킹 유지)."""
    from utils.data.load_data import create_data_loaders
    args = types.SimpleNamespace(
        max_key="max", target_key="image_label", input_key="kspace",
        batch_size=1, num_workers=0, seed=430,
        bbox_loss_lambda=0.5, return_bbox=True,      # 학습에선 on
        readout_crop=True,
    )
    val = create_data_loaders(data_path=__import__("pathlib").Path("/root/Data/val/"),
                              args=args, shuffle=False)          # shuffle=False → is_train=False
    batch = next(iter(val))
    assert len(batch) == 6, f"val 로더에 bbox가 붙었다: len={len(batch)}"
