import math
import pytest
from types import SimpleNamespace
import torch
import numpy as np
from pathlib import Path
from utils.learning.train_part import build_scheduler, steps_per_epoch, check_resume_compat, load_resume_checkpoint, load_val_log, save_latest


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


def _args(**kw):
    base = dict(model="sdum", num_cascades=4, channels=[16, 32], num_epochs=4, warmup_iters=200, lr=2.4e-4)
    base.update(kw)
    return SimpleNamespace(**base)


def test_check_resume_compat_ok():
    check_resume_compat(_args(), _args())          # no raise


def test_check_resume_compat_arch_mismatch():
    with pytest.raises(ValueError):
        check_resume_compat(_args(num_cascades=4), _args(num_cascades=6))


def test_check_resume_compat_horizon_mismatch():
    with pytest.raises(ValueError):
        check_resume_compat(_args(num_epochs=4), _args(num_epochs=2))


def test_check_resume_compat_lr_mismatch():
    with pytest.raises(ValueError):
        check_resume_compat(_args(lr=2.4e-4), _args(lr=1e-4))


def test_load_resume_checkpoint_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_resume_checkpoint(tmp_path, _args())


def test_load_resume_checkpoint_roundtrip(tmp_path):
    saved = _args()
    torch.save(
        {"epoch": 2, "args": saved, "model": {}, "optimizer": {},
         "best_val_loss": 0.11, "exp_dir": str(tmp_path)},
        tmp_path / "model.pt",
    )
    ckpt = load_resume_checkpoint(tmp_path, _args())     # compat OK
    assert ckpt["epoch"] == 2 and abs(ckpt["best_val_loss"] - 0.11) < 1e-9


def test_load_resume_checkpoint_compat_fail(tmp_path):
    torch.save({"epoch": 2, "args": _args(num_cascades=4), "model": {},
                "optimizer": {}, "best_val_loss": 0.11, "exp_dir": str(tmp_path)},
               tmp_path / "model.pt")
    with pytest.raises(ValueError):
        load_resume_checkpoint(tmp_path, _args(num_cascades=6))


def _save_ckpt(path, epoch, iter_=None, **kw):
    d = {"epoch": epoch, "args": _args(**kw), "model": {}, "optimizer": {},
         "best_val_loss": 0.11, "exp_dir": str(path.parent)}
    if iter_ is not None:
        d["iter"] = iter_
    torch.save(d, path)


def test_latest_wins_when_mid_epoch(tmp_path):
    """epoch 5를 완료(model.pt=(5,0))하고 epoch 5 중간까지 진행(latest.pt=(5,2500))이면 latest."""
    _save_ckpt(tmp_path / "model.pt", 5)
    _save_ckpt(tmp_path / "latest.pt", 5, 2500)
    ckpt = load_resume_checkpoint(tmp_path, _args())
    assert ckpt["epoch"] == 5 and ckpt["iter"] == 2500


def test_model_wins_when_epoch_completed(tmp_path):
    """epoch가 정상 완료되면 model.pt=(6,0)이 같은 epoch의 latest.pt=(5,4500)를 이긴다.
    → 낡은 중간 체크포인트를 지우지 않아도 재개가 뒤로 가지 않는다."""
    _save_ckpt(tmp_path / "model.pt", 6)
    _save_ckpt(tmp_path / "latest.pt", 5, 4500)
    ckpt = load_resume_checkpoint(tmp_path, _args())
    assert ckpt["epoch"] == 6 and ckpt.get("iter", 0) == 0


def test_latest_only(tmp_path):
    """첫 epoch도 못 끝내고 죽으면 model.pt가 아예 없다 — latest.pt만으로 재개돼야 한다."""
    _save_ckpt(tmp_path / "latest.pt", 0, 1500)
    ckpt = load_resume_checkpoint(tmp_path, _args())
    assert ckpt["epoch"] == 0 and ckpt["iter"] == 1500


def test_latest_compat_checked(tmp_path):
    """latest.pt로 재개할 때도 config 일치 검사가 걸려야 한다."""
    _save_ckpt(tmp_path / "latest.pt", 3, 500, num_cascades=4)
    with pytest.raises(ValueError):
        load_resume_checkpoint(tmp_path, _args(num_cascades=6))


def test_save_latest_is_atomic(tmp_path):
    """save_latest는 tmp에 쓰고 rename — 끝나면 tmp가 남지 않고 latest.pt만 있어야 한다."""
    save_latest(_args(), tmp_path, epoch=2, iter_done=1000,
                model=torch.nn.Linear(2, 2), optimizer=_dummy_opt(1e-3), best_val_loss=0.2)
    assert (tmp_path / "latest.pt").exists()
    assert not (tmp_path / "latest.pt.tmp").exists()
    ck = torch.load(tmp_path / "latest.pt", map_location="cpu", weights_only=False)
    assert ck["epoch"] == 2 and ck["iter"] == 1000


def test_load_val_log(tmp_path):
    base = str(tmp_path / "vl")
    assert load_val_log(base).shape == (0, 2)
    np.save(base, np.array([[0, 0.12]]))                 # base + '.npy'
    out = load_val_log(base)
    assert out.shape == (1, 2) and abs(out[0, 1] - 0.12) < 1e-9
