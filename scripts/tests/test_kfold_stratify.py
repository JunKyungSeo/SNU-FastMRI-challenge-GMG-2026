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
    """한 그룹의 fold 배정은 다른 그룹의 존재와 무관 — 층화의 존재 이유.

    sorted(ids_by_group) 순회라 "4"는 항상 첫 그룹, "8"은 항상 그다음 그룹이다.
    RandomState(seed)를 매 반복 새로 만들기 때문에 성립하는 성질이라, 첫 그룹("4")만
    확인하면 이걸 루프 밖으로 빼는 회귀(그룹끼리 RNG를 공유)를 놓친다 — 항상 첫 번째라
    상태 오염의 영향을 안 받기 때문. 비-첫 그룹인 "8"도 함께 확인해야 그 회귀를 잡는다.
    """
    both = _stratified_folds({"4": _ids(4, 30), "8": _ids(8, 30)}, k=5, seed=430)
    only4 = _stratified_folds({"4": _ids(4, 30)}, k=5, seed=430)
    only8 = _stratified_folds({"8": _ids(8, 30)}, k=5, seed=430)
    for i in range(5):
        assert {v for v in both[i] if "acc4" in v} == only4[i]
        assert {v for v in both[i] if "acc8" in v} == only8[i]


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


# ---------- kfold_split 통합 테스트 (tmp_path에 빈 h5 touch) ----------

def _make_data(tmp_path, n4=50, n8=50):
    """train/·val/ 두 root에 같은 이름의 acc4/acc8 파일을 touch.
    실데이터처럼 두 root에 동명 볼륨이 존재 → volume_id가 구분해 총 2*(n4+n8)개."""
    for root in ("train", "val"):
        d = tmp_path / root / "kspace"
        d.mkdir(parents=True)
        for i in range(n4):
            (d / f"knee_acc4_{i}.h5").touch()
        for i in range(n8):
            (d / f"knee_acc8_{i}.h5").touch()
    return tmp_path


def _args(tmp_path, acc="all", k=5, fold=0, seed=430):
    return SimpleNamespace(
        data_path_train=tmp_path / "train", data_path_val=tmp_path / "val",
        k_fold=k, val_fold=fold, kfold_seed=seed, acc=acc)


def test_acc_filter_keeps_fold_boundaries(tmp_path):
    """핵심 불변식: --acc 4/8의 분할 == --acc all의 분할에서 그 acc만 남긴 것."""
    _make_data(tmp_path)
    for fold in range(5):
        tr_all, va_all = kfold_split(_args(tmp_path, "all", fold=fold))
        tr_4, va_4 = kfold_split(_args(tmp_path, "4", fold=fold))
        tr_8, va_8 = kfold_split(_args(tmp_path, "8", fold=fold))
        assert va_4 == {v for v in va_all if "acc4" in v}
        assert tr_4 == {v for v in tr_all if "acc4" in v}
        assert va_8 == {v for v in va_all if "acc8" in v}
        assert tr_8 == {v for v in tr_all if "acc8" in v}


def test_partition_complete_disjoint_balanced(tmp_path):
    """5개 val fold가 전체 200 볼륨을 정확히 1회씩 커버 + train/val 서로소 + fold마다 acc 20/20."""
    _make_data(tmp_path)   # 총 200 = acc4 100 + acc8 100
    seen = set()
    for fold in range(5):
        tr, va = kfold_split(_args(tmp_path, fold=fold))
        assert tr.isdisjoint(va)
        assert len(tr) + len(va) == 200
        assert len(va) == 40
        assert sum(1 for v in va if "acc4" in v) == 20
        assert sum(1 for v in va if "acc8" in v) == 20
        seen |= va
    assert len(seen) == 200


def test_kfold_split_deterministic(tmp_path):
    _make_data(tmp_path)
    assert kfold_split(_args(tmp_path)) == kfold_split(_args(tmp_path))


def test_file_without_acc_token_raises(tmp_path):
    """acc 토큰 없는 h5가 섞이면 fail-fast (이 데이터셋은 acc4/acc8 전수 규약)."""
    _make_data(tmp_path)
    (tmp_path / "train" / "kspace" / "weird_noacc.h5").touch()
    with pytest.raises(ValueError, match="accN"):
        kfold_split(_args(tmp_path))


def test_acc_filter_no_match_raises(tmp_path):
    _make_data(tmp_path, n8=0)   # acc8 파일 없음
    with pytest.raises(ValueError):
        kfold_split(_args(tmp_path, acc="8"))
