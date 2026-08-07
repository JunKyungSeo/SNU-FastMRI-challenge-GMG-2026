"""hold-out tail(--holdout-tail) 샘플러 불변식 — 전부 CPU, GPU·데이터 파일 불필요.

tail 런은 hold-out(val fold) 볼륨까지 학습에 넣되 epoch당 슬라이스 수는 본 런과 같게 둔다.
이 파일이 지키는 것:
  1. epoch당 개수가 n 고정이고, 그 안에 hold-out이 **전량** 들어간다(중복 없이).
  2. 나머지 추출분은 epoch마다 **다시** 뽑힌다 — 두 epoch이 같은 부분집합을 보면 tail이 무의미.
  3. hold-out이 앞뒤로 몰리지 않고 섞인다.
  4. 같은 (seed, epoch)면 순서까지 재현된다(epoch 중간 재개가 성립하는 근거).
  5. 기본 off면 기존 경로(RandomSampler) 그대로. 그리고 sampler와 DataLoader가 **같은**
     generator를 공유한다 — 안 그러면 train()의 manual_seed(seed+epoch) 재추첨 훅이
     크래시 없이 no-op이 되어 2번만 조용히 사라진다.
"""
import tempfile
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from utils.data.load_data import HoldoutTailSampler, create_data_loaders

HOLD, POOL, N = 548, 4358, 2179   # acc4 전문가 실측(hold-out 548 · train 풀 4358 · epoch당 2179)
SEED = 430


def make_loader(hold=HOLD, pool=POOL, n=N):
    """샘플러를 진짜 DataLoader에 물려서 낸다 — DataLoader가 generator에서 base_seed를 먼저
    한 번 뽑으므로, 샘플러만 따로 돌리면 실제 학습과 다른 순열을 재게 된다."""
    g = torch.Generator()
    ds = TensorDataset(torch.arange(hold + pool))
    sampler = HoldoutTailSampler(list(range(hold)), list(range(hold, hold + pool)), n, g)
    return DataLoader(ds, batch_size=1, sampler=sampler, generator=g)


def epoch_idx(dl, epoch, seed=SEED):
    """train_part.train()이 epoch마다 하는 재시드를 그대로 흉내낸 뒤 인덱스 수집."""
    dl.generator.manual_seed(seed + epoch)
    return [int(b[0]) for b in dl]


def test_epoch_composition():
    dl = make_loader()
    assert len(dl) == N
    idx = epoch_idx(dl, 0)
    assert len(idx) == N
    assert len(set(idx)) == N                                # 중복 없음
    assert set(range(HOLD)) <= set(idx)                       # hold-out 전량
    assert sum(1 for i in idx if i >= HOLD) == N - HOLD       # 나머지는 풀에서 정확히 1631


def test_redrawn_every_epoch():
    dl = make_loader()
    k = N - HOLD
    draws = [{i for i in epoch_idx(dl, e) if i >= HOLD} for e in range(3)]
    for a, b in ((0, 1), (0, 2), (1, 2)):
        overlap = len(draws[a] & draws[b])
        assert overlap != k                                   # 같은 부분집합이면 tail이 무의미
        assert 0.25 * k < overlap < 0.55 * k                  # 독립 추출 기대치 k²/POOL = 0.374k


def test_holdout_is_mixed_not_bunched():
    dl = make_loader()
    for e in range(3):
        front = sum(1 for i in epoch_idx(dl, e)[:N // 2] if i < HOLD)
        assert 0 < front < HOLD                               # 앞뒤로 몰리면 이 범위를 벗어난다
        assert abs(front - HOLD / 2) < HOLD / 4               # 기대 274


def test_same_seed_same_order():
    dl = make_loader()
    assert epoch_idx(dl, 1) == epoch_idx(dl, 1)
    assert epoch_idx(dl, 1) != epoch_idx(dl, 2)


def test_n_smaller_than_holdout():
    """퇴화 경계: __len__과 실제 개수가 어긋나면 스케줄러 horizon이 거짓말을 한다."""
    dl = make_loader(hold=10, pool=100, n=4)
    idx = epoch_idx(dl, 0)
    assert len(dl) == 4 and len(idx) == 4
    assert all(i < 10 for i in idx)


def test_default_off_and_generator_shared():
    """3슬라이스짜리 h5 하나만 두고 create_data_loaders의 배선을 본다(GPU·본 데이터 불필요).
    로더는 만들기만 하고 돌리지 않는다 — 여기서 재는 건 sampler 종류와 generator 공유뿐."""
    def args(**kw):
        base = dict(batch_size=1, input_key="kspace", target_key="image_label",
                    max_key="max", seed=SEED, num_workers=0, acc="all", max_iters=N)
        base.update(kw)
        return SimpleNamespace(**base)

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "kspace").mkdir()
        (root / "image").mkdir()
        with h5py.File(root / "kspace" / "knee_acc4_1.h5", "w") as hf:
            hf.create_dataset("kspace", data=np.zeros((3, 1, 2, 4), dtype=np.complex64))
        h5py.File(root / "image" / "knee_acc4_1.h5", "w").close()

        off = create_data_loaders(root, args(), shuffle=True, is_train=True)
        assert isinstance(off.sampler, RandomSampler)         # 기본 off = 기존 경로 그대로
        assert off.sampler.generator is off.generator

        on = create_data_loaders(root, args(), shuffle=True, is_train=True,
                                 holdout={"knee_acc4_1.h5"})
        assert isinstance(on.sampler, HoldoutTailSampler)
        assert on.sampler.generator is on.generator           # 공유가 깨지면 재추첨이 no-op
        assert len(on) == 3 and len(list(on.sampler)) == 3    # 풀이 비어도 hold-out 전량
