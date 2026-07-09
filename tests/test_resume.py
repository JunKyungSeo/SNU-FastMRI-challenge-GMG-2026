import math
from types import SimpleNamespace
import torch
from utils.learning.train_part import build_scheduler, steps_per_epoch


def _dummy_opt(lr):
    p = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.AdamW([p], lr=lr)


def test_build_scheduler_none_for_varnet():
    args = SimpleNamespace(model="varnet", warmup_iters=200, num_epochs=2)
    assert build_scheduler(_dummy_opt(1e-3), args, 100) is None


def test_build_scheduler_none_when_warmup_zero():
    args = SimpleNamespace(model="sdum", warmup_iters=0, num_epochs=2)
    assert build_scheduler(_dummy_opt(1e-3), args, 100) is None


def test_build_scheduler_warmup_then_cosine():
    args = SimpleNamespace(model="sdum", warmup_iters=10, num_epochs=1)
    opt = _dummy_opt(2e-4)
    sch = build_scheduler(opt, args, 100)          # total_iters = 100
    assert sch is not None
    # 생성 직후(step 0): lr = base * (0+1)/warmup
    assert abs(opt.param_groups[0]["lr"] - 2e-4 * (1 / 10)) < 1e-12
    # warmup 끝(step 10)까지: cosine progress 0 → lr = base
    for _ in range(10):
        sch.step()
    assert abs(opt.param_groups[0]["lr"] - 2e-4) < 1e-9


def test_steps_per_epoch():
    assert steps_per_epoch(SimpleNamespace(max_iters=0), 100) == 100
    assert steps_per_epoch(SimpleNamespace(max_iters=30), 100) == 30
    assert steps_per_epoch(SimpleNamespace(max_iters=200), 100) == 100


def test_fast_forward_lands_at_correct_lr():
    args = SimpleNamespace(model="sdum", warmup_iters=10, num_epochs=2, max_iters=0)
    len_loader = 50
    k = 1 * steps_per_epoch(args, len_loader)       # 1 epoch fast-forward = 50 step
    opt = _dummy_opt(2e-4)
    sch = build_scheduler(opt, args, len_loader)    # total_iters = 100
    for _ in range(k):
        sch.step()
    total = args.num_epochs * len_loader
    progress = (k - args.warmup_iters) / max(1, total - args.warmup_iters)
    expected = 2e-4 * 0.5 * (1.0 + math.cos(math.pi * progress))
    assert abs(opt.param_groups[0]["lr"] - expected) < 1e-9
