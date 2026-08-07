# EMA(train_part.EMA) 단위 검증 — CPU만 사용. GPU 서버에서 `pytest scripts/tests/test_ema.py`.
import torch
import torch.nn as nn

from utils.learning.train_part import EMA


def _model(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))


def test_update_matches_manual():
    m = _model()
    ema = EMA(m, decay=0.5)
    w0 = {k: v.clone() for k, v in m.state_dict().items()}
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    ema.update(m)
    d = min(0.5, (1 + 1) / (10 + 1))   # 첫 update는 cold-start 보정 d=2/11이 이겨야 한다
    for k, v in m.state_dict().items():
        assert torch.allclose(ema.shadow[k], w0[k] * d + v * (1 - d))


def test_swap_is_involution():
    m = _model()
    ema = EMA(m)
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(2.0)   # θ와 shadow를 다르게 만든다
    theta = {k: v.clone() for k, v in m.state_dict().items()}
    shadow = {k: v.clone() for k, v in ema.shadow.items()}
    ema.swap(m)
    for k, v in m.state_dict().items():
        assert torch.equal(v, shadow[k])            # 검증 시점: 모델 = shadow
    ema.swap(m)
    for k, v in m.state_dict().items():
        assert torch.equal(v, theta[k])             # 복귀: 학습 가중치 그대로
        assert torch.equal(ema.shadow[k], shadow[k])


def test_state_dict_roundtrip():
    m = _model()
    ema = EMA(m)
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(3.0)
    for _ in range(5):
        ema.update(m)
    ema2 = EMA(_model(seed=1))
    ema2.load_state_dict(ema.state_dict())
    assert ema2.updates == 5
    for k in ema.shadow:
        assert torch.equal(ema2.shadow[k], ema.shadow[k])


def test_int_buffer_copied_not_averaged():
    m = nn.BatchNorm1d(3)   # num_batches_tracked(int64) 버퍼 보유
    ema = EMA(m)
    m(torch.randn(8, 3))
    ema.update(m)
    assert ema.shadow["num_batches_tracked"].item() == 1
