import math
import pytest
from types import SimpleNamespace
import torch
import numpy as np
from pathlib import Path
from utils.learning.train_part import build_scheduler, steps_per_epoch, check_resume_compat, load_resume_checkpoint, load_val_log, save_latest, should_early_stop


def _dummy_opt(lr):
    p = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.AdamW([p], lr=lr)


def test_build_scheduler_none_for_varnet():
    args = SimpleNamespace(model="varnet", warmup_iters=200, num_epochs=2)
    assert build_scheduler(_dummy_opt(1e-3), args, 100) is None


def test_build_scheduler_none_when_warmup_zero():
    # warmup·decay 둘 다 off면 promptmr에서도 상수 LR(스케줄러 없음)이어야 한다.
    args = SimpleNamespace(model="promptmr", warmup_iters=0, num_epochs=2)
    assert build_scheduler(_dummy_opt(1e-3), args, 100) is None


def test_build_scheduler_warmup_then_cosine():
    args = SimpleNamespace(model="promptmr", warmup_iters=10, num_epochs=1)
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
    args = SimpleNamespace(model="promptmr", warmup_iters=10, num_epochs=2, max_iters=0)
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


# ---- LR 스케줄: 상수 유지 구간(--lr-decay-start-epoch) + 하한(--lr-min-factor) -------------

def _lr_seq(sch, opt, n):
    """step 0..n-1 에서 **실제로 쓰이는** LR 수열과, n번 step 뒤의 LR을 함께 돌려준다.
    (scheduler.step()은 optimizer step **뒤에** 불리므로 i번째 step이 쓰는 값은 lr_lambda(i)다.)"""
    seq = []
    for _ in range(n):
        seq.append(opt.param_groups[0]["lr"])
        sch.step()
    return seq, opt.param_groups[0]["lr"]


def test_hold_then_cosine_with_floor():
    """warmup 없이 앞 5 epoch은 base 유지 → 그 뒤 cosine → 끝에서 하한(base×0.05)."""
    args = SimpleNamespace(model="promptmr", warmup_iters=0, lr_decay_start_epoch=5,
                           num_epochs=12, lr_min_factor=0.05, max_iters=0)
    opt = _dummy_opt(2e-4)
    sch = build_scheduler(opt, args, 10)            # spe=10 → decay_start=50, total=120
    seq, end = _lr_seq(sch, opt, 120)
    # epoch 0~4(=step 0~49)는 감쇠 없음. step 50은 cosine의 progress=0이라 역시 base.
    assert all(abs(x - 2e-4) < 1e-12 for x in seq[:51])
    assert seq[51] < 2e-4                           # 그 다음부터 실제로 내려간다
    assert abs(end - 2e-4 * 0.05) < 1e-12           # horizon 끝 = 하한에 정확히 착지


def test_floor_reached_with_max_iters():
    """--max-iters로 epoch이 짧아져도 마지막 step이 하한에 닿아야 한다.
    build_scheduler가 len_loader를 쓰면 horizon이 5배로 늘어 여기서 아직 감쇠 중이 된다."""
    args = SimpleNamespace(model="promptmr", warmup_iters=0, lr_decay_start_epoch=0,
                           num_epochs=3, lr_min_factor=0.1, max_iters=20)
    opt = _dummy_opt(1e-3)
    sch = build_scheduler(opt, args, 100)           # spe=min(20,100)=20 → total=60
    _, end = _lr_seq(sch, opt, 60)
    assert abs(end - 1e-3 * 0.1) < 1e-12


def test_defaults_match_legacy_cosine():
    """기본값(-1, 0.0)이면 예전 곡선 그대로 — warmup 직후부터 전 구간 cosine, 끝에서 0."""
    args = SimpleNamespace(model="promptmr", warmup_iters=200, num_epochs=2, max_iters=0,
                           lr_decay_start_epoch=-1, lr_min_factor=0.0)
    base, len_loader = 2e-4, 500
    opt = _dummy_opt(base)
    sch = build_scheduler(opt, args, len_loader)
    total = args.num_epochs * len_loader            # accum=1
    seq, end = _lr_seq(sch, opt, total)
    for i in (0, 199, 200, 700, total - 1):
        want = base * ((i + 1) / 200 if i < 200
                       else 0.5 * (1.0 + math.cos(math.pi * (i - 200) / (total - 200))))
        assert abs(seq[i] - want) < 1e-12, i
    assert abs(end) < 1e-15


# ---- early stopping (val_loss_log에서 파생 → resume 시 카운터가 공짜로 이어진다) ----------

def test_should_early_stop():
    log = np.array([[0, 0.5], [1, 0.4], [2, 0.41], [3, 0.42]])   # 최저=epoch 1, 이후 2 epoch 경과
    assert not should_early_stop(log, 0)                          # 0 = off
    assert not should_early_stop(log, 3)
    assert should_early_stop(log, 2)
    assert not should_early_stop(np.empty((0, 2)), 1)             # 첫 epoch 전
    assert not should_early_stop(np.array([[0, 0.5]]), 1)         # 방금이 best면 안 멈춘다


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


# ---- 마스크 재생성(--remask) resume 가드 -----------------------------------
#   remask 는 **학습셋 자체**를 바꾸는 플래그다(전문가당 80→160 volume). 값이 다른데 재개하면
#   조용히 다른 데이터로 이어학습하게 되므로 coil_compress 와 같은 등급으로 막는다.
#   ⚠️ 필드명은 argparse dest 와 글자 그대로 같아야 한다 — 어긋나면 getattr 이 양쪽 다
#      ABSENT_DEFAULT 를 집어 항상 일치하고, 가드가 영구 no-op 이 된다. 이 테스트가 그걸 잡는다.

def _rm_args(**kw):
    base = dict(model="promptmr", num_cascades=10, n_feat0=48, sens_n_feat0=24,
                n_history=3, warmup_iters=0, lr=2e-4, accum_steps=2,
                coil_compress=False, k_fold=5, val_fold=2, kfold_seed=430,
                seed=430, remask="off", remask_offset="uniform")
    base.update(kw)
    return SimpleNamespace(**base)


def test_resume_compat_allows_same_remask():
    check_resume_compat(_rm_args(), _rm_args())
    check_resume_compat(_rm_args(remask="8"), _rm_args(remask="8"))


def test_resume_compat_rejects_remask_change():
    with pytest.raises(ValueError, match="remask"):
        check_resume_compat(_rm_args(remask="off"), _rm_args(remask="8"))
    with pytest.raises(ValueError, match="remask"):
        check_resume_compat(_rm_args(remask="4"), _rm_args(remask="8"))


def test_resume_compat_checks_offset_and_seed_only_when_remask_on():
    # remask off 끼리면 offset·seed 가 달라도 입력이 안 바뀌므로 막지 않는다
    check_resume_compat(_rm_args(remask_offset="uniform", seed=430),
                        _rm_args(remask_offset="canonical", seed=1))
    with pytest.raises(ValueError, match="remask_offset"):
        check_resume_compat(_rm_args(remask="8", remask_offset="uniform"),
                            _rm_args(remask="8", remask_offset="canonical"))
    with pytest.raises(ValueError, match="seed"):
        check_resume_compat(_rm_args(remask="8", seed=430), _rm_args(remask="8", seed=1))


def test_resume_compat_old_ckpt_defaults_to_off():
    """remask 필드가 없는 옛 체크포인트는 '기능 부재 = off' 로 간주한다."""
    old = SimpleNamespace(model="promptmr", num_cascades=10, n_feat0=48, sens_n_feat0=24,
                          n_history=3, warmup_iters=0, lr=2e-4, accum_steps=2,
                          coil_compress=False, k_fold=5, val_fold=2, kfold_seed=430)
    check_resume_compat(old, _rm_args())                      # off 로 재개 → 통과
    with pytest.raises(ValueError, match="remask"):
        check_resume_compat(old, _rm_args(remask="4"))        # 켜서 재개 → 거부
