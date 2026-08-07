"""SWDC(공간가변 DC 가중치) 정합성 테스트 — CPU + tiny net, GPU 불필요.

핵심 계약 셋:
1) 기본 생성(레버 없음)에서 SWDC가 켜져 있어야 한다(dc_weight_map 존재).
2) swdc on + 초기값(W=1) 출력은 스칼라 dc_weight=1 출력과 수학적으로 동일해야 한다
   (soft_dc = A^H(1⊙(MAx−k0)) = ffx − img_zf, 선형성). 연산 순서가 달라 비트 동일은 아님.
3) swdc=False(테스트 전용 kwarg)는 기존 구조 그대로여야 한다(위 등가성의 비교 기준).
"""
import pytest
import torch

from utils.model.promptmr.promptmr_plus import PromptMR

# test_checkpoint_equivalence.py의 TINY와 같은 값(자체 보유 — 테스트 간 import 안 함)
TINY = dict(
    num_cascades=2,
    n_feat0=6,
    feature_dim=[8, 10, 12],
    prompt_dim=[4, 6, 8],
    sens_n_feat0=6,
    sens_feature_dim=[8, 10, 12],
    sens_prompt_dim=[4, 6, 8],
    prompt_size=[8, 4, 2],
    n_history=2,
    n_buffer=4,
    adaptive_input=True,
)
H = W = 32
N_COILS = 15


def _make_net(swdc, seed=0):
    # dc_weight(ones)·dc_weight_map(ones) 둘 다 RNG를 안 쓰므로, 같은 seed면
    # swdc on/off 넷의 나머지 가중치가 전부 같다 → 초기값 등가성 비교가 성립한다.
    torch.manual_seed(seed)
    return PromptMR(**TINY, swdc=swdc).double()


def _make_batch(seed=1):
    g = torch.Generator().manual_seed(seed)
    kspace = torch.randn(1, N_COILS, H, W, 2, generator=g, dtype=torch.float64)
    mask1d = torch.zeros(W, dtype=torch.float64)
    mask1d[::3] = 1.0
    mask1d[W // 2 - 2: W // 2 + 2] = 1.0
    mask = mask1d.view(1, 1, 1, W, 1).expand(1, 1, H, W, 1).contiguous()
    return kspace * mask, mask


def _forward(net, kspace, mask, use_checkpoint=False):
    return net(kspace, mask, num_low_frequencies=None, mask_type=("cartesian",),
               use_checkpoint=use_checkpoint, compute_sens_per_coil=True)


def test_swdc_is_default():
    """레버가 없으므로 기본 생성 = SWDC. config/CLI 배선 없이 모든 신규 학습에 실려야 한다."""
    torch.manual_seed(0)
    net = PromptMR(**TINY)
    for blk in net.cascades:
        assert blk.dc_weight_map.shape == (1, 1, 640, 480, 1)
        assert not hasattr(blk, "dc_weight")


def test_swdc_off_keeps_scalar_dc():
    """swdc=False는 테스트 전용 비교 기준 — 기존 구조(스칼라)가 그대로여야 한다."""
    net = _make_net(swdc=False)
    for blk in net.cascades:
        assert hasattr(blk, "dc_weight") and blk.dc_weight.shape == (1,)
        assert not hasattr(blk, "dc_weight_map")


def test_swdc_init_equals_scalar_baseline():
    """W=1 초기값에서 swdc 경로는 스칼라 η=1 경로와 수학적으로 같아야 한다."""
    kspace, mask = _make_batch()
    with torch.no_grad():
        out_off = _forward(_make_net(swdc=False).eval(), kspace, mask)
        out_on = _forward(_make_net(swdc=True).eval(), kspace, mask)
    torch.testing.assert_close(out_on, out_off, rtol=1e-6, atol=1e-8)


def test_swdc_map_receives_grad():
    net = _make_net(swdc=True).train()
    kspace, mask = _make_batch()
    _forward(net, kspace, mask).square().mean().backward()
    for i, blk in enumerate(net.cascades):
        g = blk.dc_weight_map.grad
        assert g is not None, f"cascade {i}: dc_weight_map grad 없음"
        # 샘플된 열에만 잔차가 있으므로 grad는 그 열에서 0이 아니어야 한다
        assert g.abs().sum() > 0, f"cascade {i}: dc_weight_map grad 전부 0"
