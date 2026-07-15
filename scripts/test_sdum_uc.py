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
