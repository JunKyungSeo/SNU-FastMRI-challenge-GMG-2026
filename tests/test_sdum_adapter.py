import torch
from utils.model.sdum.config import tiny_8gb_args
from utils.model.sdum.adapter import build_sdum

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _fake_batch(B=1, C=15, H=384, W=372):
    # 우리 파이프라인 계약: masked_kspace (B,C,H,W,2) float32, mask (B,1,1,W,1) byte
    ksp = torch.randn(B, C, H, W, 2, dtype=torch.float32)
    mask = torch.zeros(B, 1, 1, W, 1, dtype=torch.uint8)
    mask[..., ::4, :] = 1                          # 대략 acc4 uniform
    mask[..., W // 2 - 15 : W // 2 + 15, :] = 1    # ACS 중앙
    ksp = ksp * mask.permute(0, 1, 4, 2, 3).reshape(B, 1, 1, W, 1).to(ksp.dtype)  # 대략 언더샘플
    return ksp.to(DEV), mask.to(DEV)


def test_adapter_forward_shape_and_finite():
    args = tiny_8gb_args(num_cascades=2, channels=[16, 32])
    model = build_sdum(args).eval().to(DEV)
    ksp, mask = _fake_batch()
    with torch.no_grad():
        out = model(ksp, mask)
    assert out.shape == (1, 384, 384), out.shape
    assert torch.isfinite(out).all()


def test_one_training_step():
    args = tiny_8gb_args(num_cascades=2, channels=[16, 32])
    model = build_sdum(args).train().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    ksp, mask = _fake_batch()
    tgt = torch.randn(1, 384, 384, device=DEV)
    out = model(ksp, mask)
    loss = torch.nn.functional.l1_loss(out, tgt)
    opt.zero_grad()
    loss.backward()
    opt.step()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0 and all(torch.isfinite(g).all() for g in grads)
