"""MoE 전문가 둘이 파라미터를 공유하지 않는지 — CPU만, 데이터 불필요.

promptmr_plus.py:277 의 `act=nn.PReLU()` 는 mutable default argument라 import 시점에 만들어진
PReLU **하나**를 모든 CAB이, 심지어 **다른 모델 인스턴스끼리도** 공유한다. test_part.load_model 이
한 프로세스에서 전문가 둘을 만들어 각각 로드하므로, 막지 않으면 나중에 로드한 acc8 의 slope 가
acc4 전문가까지 덮어써 acc4 가 acc8 의 활성화로 돈다 — 점수에만 나타나고 로그엔 안 보인다.

근본 수정(act=None)은 학습 의미를 바꾸므로(전역 slope 1개 → CAB마다 1개) 대회 후로 미루고,
지금은 test_part 가 전문가 하나를 deepcopy 해서 인스턴스 바깥 공유만 끊는다.
"""
import copy

import pytest
import torch

torch.manual_seed(0)
promptmr = pytest.importorskip("utils.model.promptmr.promptmr_plus")

KW = dict(num_cascades=2, n_feat0=8, feature_dim=[12, 16, 20], prompt_dim=[4, 8, 12],
          sens_n_feat0=4, sens_feature_dim=[6, 8, 10], sens_prompt_dim=[2, 4, 6],
          n_history=1, num_adj_slices=1)


def _shared_keys():
    """서로 다른 두 인스턴스가 저장소를 공유하는 키 — 이게 비면 테스트가 무의미해진다."""
    a, b = promptmr.PromptMR(**KW), promptmr.PromptMR(**KW)
    ptrs = {v.data_ptr() for v in a.state_dict().values()}
    return [k for k, v in b.state_dict().items() if v.data_ptr() in ptrs]


def test_cross_instance_sharing_still_exists():
    """전제 확인. 언젠가 act=None 근본 수정이 들어가면 여기가 먼저 깨져서 알려준다."""
    assert _shared_keys(), "인스턴스 간 공유가 사라졌다 — deepcopy 가 이제 불필요할 수 있다"


def test_deepcopy_isolates_experts():
    k = _shared_keys()[0]
    e4 = promptmr.PromptMR(**KW)
    with torch.no_grad():
        e4.state_dict()[k].fill_(0.11)
    e4 = copy.deepcopy(e4)                      # ← test_part.load_model 이 하는 일
    e8 = promptmr.PromptMR(**KW)
    with torch.no_grad():
        e8.state_dict()[k].fill_(0.88)          # 나중에 로드되는 쪽
    assert e4.state_dict()[k].item() == pytest.approx(0.11)
    assert e8.state_dict()[k].item() == pytest.approx(0.88)


def test_deepcopy_keeps_internal_aliasing():
    """모델 **내부** 공유(sens_net ↔ sens_net_ckp 이중 등록)까지 끊으면 감도맵이 두 벌이 되어
    학습·추론 의미가 달라진다. deepcopy 는 내부는 보존하고 바깥만 끊는 게 핵심이다."""
    m = copy.deepcopy(promptmr.PromptMR(**KW))
    sd = m.state_dict()
    assert len({v.data_ptr() for v in sd.values()}) < len(sd), "내부 별칭이 사라졌다"
