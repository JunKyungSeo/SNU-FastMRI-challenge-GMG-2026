"""_pyramid 파생이 PromptMR+ 원본 default 표를 그대로 재현하는지 고정한다 — 전부 CPU, GPU 불필요.

이 테스트가 지키는 계약: `n_feat0=48 · sens_n_feat0=24` 에서 네 리스트가 하드코딩 시절과 **한 글자도
다르지 않아야** 한다. 어긋나면 48/24 로 학습한 지금까지의 체크포인트가 load_state_dict 에서 죽고,
과거 run 과의 점수 비교도 성립하지 않는다. 파생으로 바꾼 뒤 이게 유일한 안전망이다.
"""
import pytest

from utils.model.promptmr.config import _pyramid, promptmr_kwargs


def test_default_kwargs_match_original_table():
    """48/24 기본값에서 하드코딩 시절과 동일한 네 리스트가 나온다."""
    kw = promptmr_kwargs()
    assert kw["feature_dim"] == [72, 96, 120]
    assert kw["prompt_dim"] == [24, 48, 72]
    assert kw["sens_feature_dim"] == [36, 48, 60]
    assert kw["sens_prompt_dim"] == [12, 24, 36]


def test_pyramid_scales_with_base():
    """폭을 올리면 피라미드도 같은 비율로 따라온다(스윕이 성립하는 근거).
    16 → [24,32,40] 은 옛 hspromptmr 계열이 --channels 로 직접 주던 값과 같다."""
    assert _pyramid(96) == ([144, 192, 240], [48, 96, 144])
    assert _pyramid(64) == ([96, 128, 160], [32, 64, 96])
    assert _pyramid(16) == ([24, 32, 40], [8, 16, 24])


def test_pyramid_keeps_three_levels():
    """폭만 넓히고 깊이는 건드리지 않는다 — SSIM_bbox 근거로 3레벨 고정(config.py docstring)."""
    feat, prompt = _pyramid(80)
    assert len(feat) == 3 and len(prompt) == 3


def test_pyramid_rejects_non_multiple_of_4():
    """0.5·1.5 배가 정수로 안 떨어지는 폭은 조용히 내림하지 말고 막는다."""
    with pytest.raises(ValueError):
        _pyramid(50)
