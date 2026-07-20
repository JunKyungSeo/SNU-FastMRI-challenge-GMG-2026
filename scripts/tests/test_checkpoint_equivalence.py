"""gradient checkpointing 이 수치적으로 무해함을 강제하는 테스트.

checkpointing 은 순전파 activation 을 버리고 역전파 때 다시 계산하는 것이므로,
출력과 모든 파라미터 gradient 가 checkpointing 없는 경우와 **같아야** 한다.
다르면 그건 메모리 절감이 아니라 버그다.

CPU + 아주 작은 net 으로 돌아가므로 GPU 없이 실행된다.
"""
import pytest
import torch

from utils.model.promptmr.promptmr_plus import PromptMR


# 실제 학습 config(cascade 10 / feature 48)는 CPU 테스트로 못 돌린다.
# checkpointing 의 정합성은 규모와 무관하므로 최소 구조로 확인한다.
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
N_COILS = 15        # 실제 데이터셋의 물리 코일 수(압축 없이 전부 사용)


def _make_net(seed=0):
    torch.manual_seed(seed)
    return PromptMR(**TINY).double()      # float64: 재계산 오차가 아니라 로직 차이만 잡도록


def _make_batch(n_coils=N_COILS, seed=1):
    g = torch.Generator().manual_seed(seed)
    kspace = torch.randn(1, n_coils, H, W, 2, generator=g, dtype=torch.float64)
    # 폭축 마스크: 중앙 ACS + 언더샘플링. adapter 와 같은 (B,1,H,W,1) 레이아웃.
    mask1d = torch.zeros(W, dtype=torch.float64)
    mask1d[::3] = 1.0
    acs = slice(W // 2 - 2, W // 2 + 2)
    mask1d[acs] = 1.0
    mask = mask1d.view(1, 1, 1, W, 1).expand(1, 1, H, W, 1).contiguous()
    return kspace * mask, mask


def _forward_backward(net, kspace, mask, use_checkpoint, compute_sens_per_coil):
    net.zero_grad(set_to_none=True)
    out = net(
        kspace, mask,
        num_low_frequencies=None,
        mask_type=("cartesian",),
        use_checkpoint=use_checkpoint,
        compute_sens_per_coil=compute_sens_per_coil,
    )
    out.square().mean().backward()      # 스칼라 손실이면 무엇이든 무방
    grads = {n: p.grad.detach().clone() for n, p in net.named_parameters() if p.grad is not None}
    return out.detach().clone(), grads


@pytest.mark.parametrize("compute_sens_per_coil", [True, False])
def test_checkpointing_matches_no_checkpointing(compute_sens_per_coil):
    """checkpoint on/off 가 출력과 모든 param grad 에서 일치해야 한다."""
    kspace, mask = _make_batch()

    net = _make_net()
    net.train()
    out_ckp, grad_ckp = _forward_backward(net, kspace, mask, True, compute_sens_per_coil)
    out_raw, grad_raw = _forward_backward(net, kspace, mask, False, compute_sens_per_coil)

    torch.testing.assert_close(out_ckp, out_raw, rtol=1e-9, atol=1e-9)

    assert grad_ckp.keys() == grad_raw.keys()
    assert len(grad_ckp) > 0, "grad 가 하나도 안 잡혔다 = 테스트가 아무것도 검증하지 않음"
    for name in grad_ckp:
        torch.testing.assert_close(
            grad_ckp[name], grad_raw[name], rtol=1e-9, atol=1e-9,
            msg=lambda m, n=name: f"param '{n}' 의 grad 가 checkpointing 여부에 따라 다름:\n{m}",
        )


def test_every_parameter_receives_grad():
    """checkpointing 이 일부 파라미터의 grad 를 끊어먹지 않는지 확인.

    reentrant checkpoint 는 입력에 requires_grad 가 없으면 조용히 grad 를 끊는다.
    cascade 를 checkpoint 로 감쌀 때 이 실수가 나기 쉬워 명시적으로 막는다.
    """
    kspace, mask = _make_batch()
    net = _make_net()
    net.train()
    _forward_backward(net, kspace, mask, True, True)

    # _dummy_tensor 는 ReentrantWrapper 가 reentrant checkpoint 의 "입력 중 하나는
    # requires_grad 여야 한다" 제약을 만족시키려고 만든 인공 파라미터로, _DummyWrapper.forward
    # 가 그대로 버린다 → 원래 grad 가 없는 게 정상. 실제 연산 파라미터만 본다.
    missing = [n for n, p in net.named_parameters()
               if p.requires_grad and p.grad is None and not n.endswith("_dummy_tensor")]
    assert not missing, f"checkpointing 경로에서 grad 가 안 흐른 파라미터: {missing}"


def test_sens_per_coil_matches_batched():
    """코일별 감도 추정(메모리 절감 경로)이 한 번에 계산한 것과 같아야 한다.

    코일 압축을 없애면 이 경로가 상시 사용되므로, 등가성이 깨지면 곧바로 품질 문제가 된다.
    """
    kspace, mask = _make_batch()
    net = _make_net()
    net.train()
    out_per_coil, grad_per_coil = _forward_backward(net, kspace, mask, False, True)
    out_batched, grad_batched = _forward_backward(net, kspace, mask, False, False)

    torch.testing.assert_close(out_per_coil, out_batched, rtol=1e-9, atol=1e-9)
    for name in grad_per_coil:
        torch.testing.assert_close(
            grad_per_coil[name], grad_batched[name], rtol=1e-9, atol=1e-9,
            msg=lambda m, n=name: f"param '{n}' grad 가 per-coil 여부에 따라 다름:\n{m}",
        )


@pytest.mark.parametrize("n_coils", [4, 8, 15])
def test_coil_count_is_free(n_coils):
    """모델이 코일 수를 구조에 못박지 않는지 확인(압축 8 → 전체 15 전환의 근거).

    같은 가중치가 코일 수만 바꿔도 동작해야 압축 유무가 아키텍처 변경이 아니게 된다.
    """
    net = _make_net()
    net.eval()
    kspace, mask = _make_batch(n_coils=n_coils)
    with torch.no_grad():
        out = net(kspace, mask, num_low_frequencies=None, mask_type=("cartesian",),
                  use_checkpoint=False, compute_sens_per_coil=True)
    assert out.shape == (1, 384, 384)
    assert torch.isfinite(out).all()
