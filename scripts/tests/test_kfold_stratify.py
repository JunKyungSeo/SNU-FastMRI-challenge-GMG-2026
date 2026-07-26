"""kfold_split acc 층화 분할 테스트 (2026-07-26 설계).

핵심 불변식: --acc all/4/8 어느 값으로 분할해도 같은 볼륨은 항상 같은 fold.
kfold_split은 (root/"kspace").glob("*.h5")의 **파일명만** 보고 내용은 안 열므로
tmp_path에 빈 파일 touch로 실데이터 없이 검증한다.

⚠️ 로컬에 torch가 없으면 import에서 죽는다 — 서버에서 실행:
  python -m pytest scripts/tests/test_kfold_stratify.py -v
"""
from types import SimpleNamespace

import numpy as np
import pytest

from utils.data.load_data import _stratified_folds, kfold_split


# ---------- _stratified_folds 단위 테스트 (파일시스템 불필요) ----------

def _ids(acc, n, root="train"):
    return [f"{root}__knee_acc{acc}_{i}.h5" for i in range(n)]


def test_group_assignment_independent_of_other_groups():
    """한 그룹의 fold 배정은 다른 그룹의 존재와 무관 — 층화의 존재 이유."""
    both = _stratified_folds({"4": _ids(4, 30), "8": _ids(8, 30)}, k=5, seed=430)
    only4 = _stratified_folds({"4": _ids(4, 30)}, k=5, seed=430)
    for i in range(5):
        assert {v for v in both[i] if "acc4" in v} == only4[i]


def test_folds_balanced_per_group():
    """100+100을 k=5로 → fold마다 정확히 acc4 20 + acc8 20."""
    folds = _stratified_folds({"4": _ids(4, 100), "8": _ids(8, 100)}, k=5, seed=430)
    for f in folds:
        assert sum(1 for v in f if "acc4" in v) == 20
        assert sum(1 for v in f if "acc8" in v) == 20


def test_deterministic_same_seed():
    a = _stratified_folds({"4": _ids(4, 40), "8": _ids(8, 40)}, k=5, seed=430)
    b = _stratified_folds({"4": _ids(4, 40), "8": _ids(8, 40)}, k=5, seed=430)
    assert a == b


def test_group_smaller_than_k_raises():
    with pytest.raises(ValueError, match="k="):
        _stratified_folds({"4": _ids(4, 3)}, k=5, seed=430)
