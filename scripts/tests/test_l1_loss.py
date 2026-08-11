"""C11 ℓ1 혼합 손실 — 스케일·off 동일성·gradient. 데이터·GPU 불필요."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.common.loss_function import SSIMLoss, masked_l1_loss


def _fixture(seed=0, B=2, H=64, W=64):
    torch.manual_seed(seed)
    return (torch.rand(B, H, W), torch.rand(B, H, W),
            torch.tensor([0.8, 1.3]), (torch.rand(B, H, W) > 0.3).float())


def test_same_order_of_magnitude_as_ssim():
    """data_range 나눗셈의 존재 이유 — 안 나누면 L1이 SSIM을 압도한다."""
    out, tgt, dr, fg = _fixture()
    s = SSIMLoss()(out, tgt, dr, mask=fg)
    l1 = masked_l1_loss(out, tgt, dr, mask=fg)
    assert 0.1 < (l1 / s).item() < 10.0


def test_scales_with_data_range():
    """입력을 a배 하면 (data_range도 a배) L1항이 그대로여야 한다 = 스케일 불변."""
    out, tgt, dr, fg = _fixture()
    base = masked_l1_loss(out, tgt, dr, mask=fg)
    scaled = masked_l1_loss(out * 3.0, tgt * 3.0, dr * 3.0, mask=fg)
    assert torch.allclose(base, scaled, atol=1e-6)


def test_mask_restricts_region():
    """마스크 밖 픽셀을 흔들어도 값이 안 변해야 한다."""
    out, tgt, dr, fg = _fixture()
    before = masked_l1_loss(out, tgt, dr, mask=fg)
    perturbed = out + (1 - fg) * 100.0
    assert torch.allclose(before, masked_l1_loss(perturbed, tgt, dr, mask=fg), atol=1e-5)


def test_zero_when_identical():
    out, _, dr, fg = _fixture()
    assert masked_l1_loss(out, out, dr, mask=fg).item() == pytest.approx(0.0, abs=1e-7)


def test_mix_at_w0_is_exactly_ssim():
    """--l1-weight 0.0 = 기존 손실과 비트 동일 (train_part는 아예 분기를 안 탄다)."""
    out, tgt, dr, fg = _fixture()
    s = SSIMLoss()(out, tgt, dr, mask=fg)
    assert torch.equal((1 - 0.0) * s + 0.0 * masked_l1_loss(out, tgt, dr, mask=fg), s)


def test_gradient_flows_and_is_finite():
    out, tgt, dr, fg = _fixture()
    o = out.clone().requires_grad_(True)
    ((1 - 0.1) * SSIMLoss()(o, tgt, dr, mask=fg)
     + 0.1 * masked_l1_loss(o, tgt, dr, mask=fg)).backward()
    assert torch.isfinite(o.grad).all() and o.grad.abs().sum() > 0


def test_mask_none_path():
    out, tgt, dr, _ = _fixture()
    v = masked_l1_loss(out, tgt, dr)
    assert torch.isfinite(v) and v.item() > 0
