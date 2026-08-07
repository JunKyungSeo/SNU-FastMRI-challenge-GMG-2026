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


class _Aliased(nn.Module):
    """운영 모델의 별칭 두 형태를 그대로 재현한다 — state_dict()에 중복 키가 생기는 조건.

    위의 _model()은 nn.Sequential이라 별칭이 없어 이 버그를 **원리적으로** 못 잡는다.
    실제 모델에서는 promptmr_plus.py:570(sens_net_ckp = wrapper(sens_net))과
    :277(act=nn.PReLU() mutable default)이 각각 이 두 모양을 만든다.
    """

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.inner = nn.Linear(3, 3)
        self.inner_ckp = self.inner   # 같은 모듈을 두 이름으로 등록
        act = nn.PReLU()
        self.act1 = act
        self.act2 = act               # 한 인스턴스를 두 곳이 공유


def test_alias_model_really_has_duplicate_keys():
    """이 테스트가 실제로 별칭을 재고 있는지부터 확인 — 아니면 아래 둘이 조용히 무의미해진다."""
    sd = _Aliased().state_dict()
    assert len(sd) == 6, sorted(sd)
    assert len({v.data_ptr() for v in sd.values()}) == 3


def test_swap_is_involution_with_aliases():
    """별칭이 있으면 옛 구현은 같은 텐서를 m번 교환해 위수 3 순열이 됐다 —
    swap 두 번 뒤 모델이 θ가 아니라 θ_EMA로 돌아와 학습이 거기서 재개됐다."""
    m = _Aliased()
    ema = EMA(m)
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(2.0)
    theta = {k: v.clone() for k, v in m.state_dict().items()}
    shadow = {k: v.clone() for k, v in ema.shadow.items()}

    ema.swap(m)
    for k, v in m.state_dict().items():
        assert torch.equal(v, shadow[k]), k       # 검증 시점: 모델 = shadow
    ema.swap(m)
    for k, v in m.state_dict().items():
        assert torch.equal(v, theta[k]), k        # 복귀: 학습 가중치 그대로
        assert torch.equal(ema.shadow[k], shadow[k]), k


def test_alias_shadow_entries_stay_in_sync():
    """별칭 키들의 shadow가 벌어지면 save_model이 desync된 값을 저장하고
    load_state_dict는 등록 순서상 뒤쪽 키가 이겨 잡탕 체크포인트가 된다."""
    m = _Aliased()
    ema = EMA(m)
    for i in range(3):
        with torch.no_grad():
            for p in m.parameters():
                p.add_(1.0)
        ema.update(m)
        ema.swap(m); ema.swap(m)   # 매 epoch 검증 전후 교환을 흉내
        assert torch.equal(ema.shadow["inner.weight"], ema.shadow["inner_ckp.weight"]), i
        assert torch.equal(ema.shadow["act1.weight"], ema.shadow["act2.weight"]), i


def test_alias_state_dict_keys_unchanged():
    """저장 키 집합이 그대로여야 기존 체크포인트·strict=True 로드가 안 깨진다."""
    m = _Aliased()
    ema = EMA(m)
    assert ema.state_dict()["shadow"].keys() == m.state_dict().keys()
    ema2 = EMA(_Aliased())
    ema2.load_state_dict(ema.state_dict())
    for k in ema.shadow:
        assert torch.equal(ema2.shadow[k], ema.shadow[k]), k


def test_int_buffer_copied_not_averaged():
    m = nn.BatchNorm1d(3)   # num_batches_tracked(int64) 버퍼 보유
    ema = EMA(m)
    m(torch.randn(8, 3))
    ema.update(m)
    assert ema.shadow["num_batches_tracked"].item() == 1
