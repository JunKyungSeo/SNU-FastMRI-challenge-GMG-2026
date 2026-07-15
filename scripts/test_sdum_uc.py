import torch
from utils.model.sdum.config import tiny_8gb_args
from utils.model.sdum.adapter import build_sdum

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_uc_flag_sets_both_conds():
    off = tiny_8gb_args(num_cascades=2, channels=[16, 32])
    assert off.time_cond is False and off.label_cond is False
    on = tiny_8gb_args(num_cascades=2, channels=[16, 32], uc=True)
    assert on.time_cond is True and on.label_cond is True


def test_uc_explicit_override_wins():
    # uc=True지만 time_cond를 명시로 끄면 명시가 우선(회귀 방지용 계약)
    a = tiny_8gb_args(uc=True, time_cond=False)
    assert a.time_cond is False and a.label_cond is True


def _fake_batch(B=1, C=15, H=384, W=372):
    ksp = torch.randn(B, C, H, W, 2, dtype=torch.float32)
    mask = torch.zeros(B, 1, 1, W, 1, dtype=torch.uint8)
    mask[..., ::4, :] = 1
    mask[..., W // 2 - 15 : W // 2 + 15, :] = 1
    ksp = ksp * mask.permute(0, 1, 4, 2, 3).reshape(B, 1, 1, W, 1).to(ksp.dtype)
    return ksp.to(DEV), mask.to(DEV)


def test_uc_forward_shape_and_finite():
    model = build_sdum(tiny_8gb_args(num_cascades=2, channels=[16, 32], uc=True)).eval().to(DEV)
    ksp, mask = _fake_batch()
    with torch.no_grad():
        out = model(ksp, mask)
    assert out.shape == (1, 384, 384), out.shape
    assert torch.isfinite(out).all()


def test_uc_off_still_forwards():
    # baseline 경로 보존: uc off도 정상 (B,384,384)
    model = build_sdum(tiny_8gb_args(num_cascades=2, channels=[16, 32])).eval().to(DEV)
    ksp, mask = _fake_batch()
    with torch.no_grad():
        out = model(ksp, mask)
    assert out.shape == (1, 384, 384) and torch.isfinite(out).all()


def test_uc_conditioning_is_actually_used():
    """silent no-op 방지: UC 파라미터(time_embed/label_emb/emb_layers)에 grad가 흘러야 한다.
    grad가 전부 0/None이면 조건화가 forward에서 안 쓰인 것(과거 함정)."""
    model = build_sdum(tiny_8gb_args(num_cascades=2, channels=[16, 32], uc=True)).train().to(DEV)
    ksp, mask = _fake_batch()
    tgt = torch.randn(1, 384, 384, device=DEV)
    out = model(ksp, mask)
    torch.nn.functional.l1_loss(out, tgt).backward()
    cond_named = [(n, p) for n, p in model.named_parameters()
                  if ("time_embed" in n or "label_emb" in n or "emb_layers" in n)]
    assert cond_named, "UC 파라미터가 생성되지 않음"
    nonzero = [n for n, p in cond_named
               if p.grad is not None and torch.any(p.grad != 0)]
    assert nonzero, "UC 파라미터에 grad가 안 흐름 → 조건화가 forward에서 미사용(silent no-op)"
