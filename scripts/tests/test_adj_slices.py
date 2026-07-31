"""2.5D 인접 슬라이스 적층 규약 테스트 — 전부 CPU, 합성 h5(임시파일), GPU 불필요.

여기서 고정하는 계약은 모델이 전제하는 것과 **정확히 같아야** 한다:
  · 코일축은 **슬라이스-major** — (adj*C, H, W) 에서 [s-1의 C코일, s의 C코일, s+1의 C코일] 순서.
    promptmr_plus 의 sens_reduce/sens_expand·감도 RSS 그룹 정규화가 이 순서를 전제한다.
    순서가 뒤집히면 에러 없이 조용히 잘못된 감도맵으로 학습된다 → 그래서 테스트가 필요하다.
  · 볼륨 경계는 **clamp(자기 복제)**. 0-padding 이면 그 코일 그룹이 전부 0이 되어 감도맵
    RSS 정규화에서 0 나눗셈이 난다.
  · num_adj_slices=1 이면 기존과 완전히 동일(단일 슬라이스, 추가 I/O 0).
"""
import numpy as np
import pytest

from utils.data.load_data import SliceData

COILS, H, W, NSL = 4, 8, 12, 5


class _RawTransform:
    """SliceData 가 넘겨주는 것을 그대로 돌려준다(전처리 없이 적층만 검사)."""
    return_grappa = False

    def __call__(self, mask, input, target, attrs, fname, slice, grappa=None):
        return input


def _make_volume(tmp_path, name="knee_acc4_1.h5"):
    """슬라이스마다 값이 다른 합성 k-space 볼륨. 슬라이스 j 는 전부 j 로 채운다 →
    적층 순서를 값으로 바로 식별할 수 있다."""
    import h5py
    d = tmp_path / "kspace"
    d.mkdir(parents=True, exist_ok=True)
    ks = np.zeros((NSL, COILS, H, W), dtype=np.complex64)
    for j in range(NSL):
        ks[j] = j + 1                      # 1..NSL (0 은 clamp 검사에서 헷갈리므로 피한다)
    with h5py.File(d / name, "w") as hf:
        hf.create_dataset("kspace", data=ks)
        hf.create_dataset("mask", data=np.ones(W, dtype=np.float32))
    return tmp_path


def _dataset(root, adj):
    return SliceData(root=root, transform=_RawTransform(), input_key="kspace",
                     target_key="image_label", forward=True, num_adj_slices=adj)


def _groups(arr, adj):
    """(adj*C,H,W) → 슬라이스 그룹별 대표값 리스트. 그룹 내부가 균일한지도 함께 확인."""
    assert arr.shape[0] == adj * COILS, f"코일축이 adj*C 가 아니다: {arr.shape}"
    out = []
    for g in range(adj):
        blk = arr[g * COILS:(g + 1) * COILS]
        vals = np.unique(blk.real)
        assert vals.size == 1, f"그룹 {g} 내부가 균일하지 않다 — 적층이 섞였다: {vals}"
        out.append(float(vals[0]))
    return out


def test_adj1_is_single_slice(tmp_path):
    """num_adj_slices=1 은 기존 동작 — 코일축 그대로, 해당 슬라이스만."""
    root = _make_volume(tmp_path)
    ds = _dataset(root, 1)
    assert len(ds) == NSL
    for s in range(NSL):
        arr = ds[s]
        assert arr.shape == (COILS, H, W)
        assert _groups(arr, 1) == [s + 1]


def test_adj3_stacks_slice_major(tmp_path):
    """중앙 슬라이스: [s-1, s, s+1] 순서로 코일축에 쌓인다."""
    root = _make_volume(tmp_path)
    ds = _dataset(root, 3)
    for s in (1, 2, 3):                    # 경계 아닌 슬라이스
        arr = ds[s]
        assert arr.shape == (3 * COILS, H, W)
        assert _groups(arr, 3) == [s, s + 1, s + 2], f"slice {s} 적층 순서가 틀렸다"


def test_adj3_clamps_at_volume_edges(tmp_path):
    """첫/마지막 슬라이스는 자기 복제(clamp) — 0-padding 이면 안 된다."""
    root = _make_volume(tmp_path)
    ds = _dataset(root, 3)
    assert _groups(ds[0], 3) == [1, 1, 2], "첫 슬라이스가 clamp 되지 않았다"
    assert _groups(ds[NSL - 1], 3) == [NSL - 1, NSL, NSL], "마지막 슬라이스가 clamp 되지 않았다"
    # 0 이 섞이면 감도맵 RSS 정규화에서 0 나눗셈 → 명시적으로 배제
    for s in (0, NSL - 1):
        assert 0.0 not in _groups(ds[s], 3), "clamp 대신 0-padding 이 들어갔다"


def test_even_num_adj_rejected(tmp_path):
    """짝수는 중앙 슬라이스가 없어 center_slice 규약이 깨진다 → 생성 시점에 막는다."""
    root = _make_volume(tmp_path)
    with pytest.raises(ValueError, match="홀수"):
        _dataset(root, 2)


def test_adj_conflicts_with_coil_compress():
    """코일축을 SVD 로 섞는 coil compression 과는 배타 — create_data_loaders 가 fail-fast."""
    import types
    from utils.data.load_data import create_data_loaders
    args = types.SimpleNamespace(
        num_adj_slices=3, coil_compress=True,
        max_key="max", target_key="image_label", input_key="kspace",
        batch_size=1, num_workers=0, seed=430,
    )
    with pytest.raises(ValueError, match="coil-compress"):
        create_data_loaders("/nonexistent", args, shuffle=True)
