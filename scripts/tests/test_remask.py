"""마스크 재생성(--remask) 불변식 회귀 테스트 — 전부 CPU, GPU·데이터 파일 불필요(합성 배열).

학습풀 h5의 kspace는 마스크 곱 이전 full k-space라, 딸려온 마스크를 버리고 대회 생성기로
만든 마스크를 곱하면 같은 볼륨을 반대 가속률로 쓸 수 있다(정답은 마스크와 무관).
→ MoE 전문가 하나가 보는 볼륨이 80 → 160.

이 파일이 지키는 것:
  1. 생성기가 주최측 마스크와 같은 물건인가 (골든 벡터 + 마지막 열 규약 + round tie).
  2. 라우터(moe.extract_acc_from_mask)가 재생성 마스크를 옳은 전문가로 보내는가.
  3. 재생성이 **full k-space를 실제로 쓰는가** (옛 마스크가 0이던 열에서 에너지를 살려내는가).
  4. off면 기존 경로와 비트 동일하고, val/forward에는 절대 안 걸리는가.
  5. epoch이 transform까지 도달하는가 (안 오면 매 epoch 같은 마스크 = 이득만 조용히 증발).
  6. offset RNG 스트림이 flip과 분리돼 있는가 (A/B가 같은 flip 추출 위에서 비교돼야 함).
"""
import numpy as np
import pytest

from utils.data.transforms import (
    AugmentConfig, MRIAugmentor, DataTransform,
    CENTER_FRACTION, equispaced_mask, canonical_offset, remask_offset, infer_acc, _fftc,
)
from utils.model.moe import extract_acc_from_mask, DEFAULT_ACC_THRESHOLD

TARGET_W = 384


def make_kspace(coils=4, H=64, W=368, seed=0):
    """full k-space 모방 — 어느 열에도 에너지가 있다(마스크 곱 이전)."""
    rng = np.random.default_rng(seed)
    img = (rng.standard_normal((coils, H, W))
           + 1j * rng.standard_normal((coils, H, W)))
    return _fftc(img).astype(np.complex64)


def make_target(seed=1):
    return np.random.default_rng(seed).random((TARGET_W, TARGET_W)).astype(np.float32)


def make_attrs():
    return {"max": np.float32(1.0)}


def make_dt(remask_acc=None, is_train=True, isforward=False, augmentor=None, **kw):
    return DataTransform(isforward, "max", augmentor=augmentor, is_train=is_train,
                         remask_acc=remask_acc, **kw)


# ---------------------------------------------------------------- 생성기 정확성

def test_generator_golden_vectors():
    """로컬 샘플 실측값을 고정 — acc4 W=368(off 0)은 114줄·ACS[170,199), acc8 W=372(off 2)는 74줄·ACS[171,201)."""
    m4 = equispaced_mask(368, 4, 0)
    assert m4.sum() == 114
    assert abs(float(m4.astype(np.float64).mean()) - 0.30978260869565216) < 1e-12
    assert int(round(CENTER_FRACTION * 368)) == 29 and (368 - 29 + 1) // 2 == 170
    assert m4[170:199].all() and m4[169] == 0            # ACS는 [170,199), 그 왼쪽은 격자점 아님

    m8 = equispaced_mask(372, 8, 2)
    assert m8.sum() == 74
    assert abs(float(m8.astype(np.float64).mean()) - 0.1989247311827957) < 1e-12
    assert int(round(CENTER_FRACTION * 372)) == 30 and (372 - 30 + 1) // 2 == 171
    assert m8[171:201].all()


def test_canonical_offset_matches_local_samples():
    """학습풀 규칙 offset == (W//2)%acc — 로컬 두 볼륨 실측(acc4 W368→0, acc8 W372→2)."""
    assert canonical_offset(368, 4) == 0
    assert canonical_offset(372, 8) == 2


def test_last_column_convention():
    """(W-1)%acc == offset 이면 마지막 열이 샘플돼야 한다.
    np.arange(offset, W-1, acc) 로 격자를 만들면 그 열이 조용히 빠진다 — 이 테스트가 그 회귀를 잡는다."""
    for W in (320, 368, 372, 386, 400, 480):
        for acc in (4, 8):
            for off in range(acc):
                m = equispaced_mask(W, acc, off)
                assert bool(m[W - 1] >= 1) == ((W - 1 - off) % acc == 0), (W, acc, off)


def test_round_has_no_tie():
    """round(0.08·W)가 .5 경계에 걸리는 W가 없어야 int(round())가 안전하다."""
    for W in range(200, 801):
        assert abs(CENTER_FRACTION * W - round(CENTER_FRACTION * W)) < 0.499


def test_acs_is_identical_across_accelerations():
    """ACS는 가속률과 무관하게 round(0.08·W) — acc4/acc8 합성본의 ACS가 같아야 한다."""
    for W in (320, 368, 372, 400, 480):
        a4 = equispaced_mask(W, 4, canonical_offset(W, 4))
        a8 = equispaced_mask(W, 8, canonical_offset(W, 8))
        nlf = int(round(CENTER_FRACTION * W))
        pad = (W - nlf + 1) // 2
        assert a4[pad:pad + nlf].all() and a8[pad:pad + nlf].all()


# ---------------------------------------------------------------- 라우터 정합

def test_router_agrees_with_generated_acc():
    """생성한 마스크를 추론 라우터에 그대로 물려도 학습 때와 같은 전문가로 가는가.
    학습셋 폭 범위(320~480) × 전 offset 전수. 2026-07-28에 mask thinning을 지운 '결정타'가
    바로 이 불일치였으므로, 여기서 마진까지 못 박는다."""
    import torch
    lo4, hi8 = 1.0, 0.0
    for W in range(320, 481, 2):
        for acc in (4, 8):
            for off in range(acc):
                m = torch.from_numpy(equispaced_mask(W, acc, off))
                assert extract_acc_from_mask(m) == acc, (W, acc, off)
                frac = float(np.asarray(m, dtype=np.float64).mean())
                if acc == 4:
                    lo4 = min(lo4, frac)
                else:
                    hi8 = max(hi8, frac)
    # 전수 실측 마진 — acc4 하단 0.30663(+0.0566), acc8 상단 0.20000(−0.0500).
    # 임계가 0.25에서 움직이거나 생성기가 바뀌면 여기서 먼저 걸린다.
    assert lo4 > DEFAULT_ACC_THRESHOLD + 0.05, lo4
    assert hi8 <= DEFAULT_ACC_THRESHOLD - 0.05, hi8


# ---------------------------------------------------------------- 파이프라인 배선

def test_remask_uses_full_kspace():
    """재생성의 존재 이유 — 옛 마스크가 0이던 열의 에너지를 실제로 살려내는가.
    acc8 마스크가 달린 볼륨에 acc4를 재생성하면 새로 켜진 열에 값이 있어야 한다."""
    W = 368
    K, T = make_kspace(W=W), make_target()
    old = equispaced_mask(W, 8, canonical_offset(W, 8))    # 딸려온 acc8 마스크

    dt = make_dt(remask_acc=4, remask_offset="canonical")
    mask_t, kspace, *_ = dt(old, K, T, make_attrs(), "knee_acc8_1.h5", 0)

    new = mask_t.numpy().reshape(-1).astype(bool)
    ks = kspace.numpy()                                     # (coils, H, W, 2)
    energy = np.abs(ks[..., 0] + 1j * ks[..., 1]).sum(axis=(0, 1))

    assert new.sum() > old.sum(), "acc4 재생성이면 라인 수가 늘어야 한다"
    assert np.allclose(energy[~new], 0.0), "마스크 밖 열은 정확히 0이어야 한다"
    fresh = new & (old < 1)
    assert fresh.any() and (energy[fresh] > 0).all(), "새로 켠 열에 full k-space 에너지가 실려야 한다"


def test_remask_off_matches_hand_computed_pipeline():
    """off면 "딸려온 마스크를 그대로 곱한 것"과 바이트까지 같아야 한다.

    ⚠️ 기대값을 **구현 밖에서** 손으로 만들어 비교한다. off를 off와 비교하면(둘 다 같은 코드를
    타므로) 결정적 파이프라인을 자기 자신과 대조하는 동어반복이 되어, 게이트가 망가져도 통과한다."""
    W = 372
    K, T = make_kspace(W=W), make_target()
    m = equispaced_mask(W, 8, canonical_offset(W, 8))

    out = make_dt(remask_acc=None)(m, K, T, make_attrs(), "knee_acc8_1.h5", 3)
    assert len(out) == 6, "off면 정확히 6-튜플 (train_part의 하드코딩 언패킹)"
    mask_t, kspace, target_out, maximum, fname, sl = out

    exp_mask = (m >= 1).reshape(1, 1, W, 1)
    exp_ks = (K * (m >= 1).astype(m.dtype)).astype(K.dtype)
    got = kspace.numpy()
    got_ks = (got[..., 0] + 1j * got[..., 1]).astype(K.dtype)

    assert np.array_equal(mask_t.numpy().astype(bool), exp_mask)
    assert np.array_equal(got_ks, exp_ks)
    assert np.array_equal(target_out.numpy(), T)
    assert (fname, sl) == ("knee_acc8_1.h5", 3)

    # negative — 켜면 반드시 달라야 한다. 게이트가 죽어 off에서도 재생성되면 위 단언이,
    #            켜도 재생성이 안 되면 이 단언이 걸린다.
    on = make_dt(remask_acc=4, remask_offset="canonical")(m, K, T, make_attrs(), "knee_acc8_1.h5", 3)
    assert not np.array_equal(on[0].numpy().astype(bool), exp_mask)

    # "off" 문자열도 None과 같게 정규화되는가(load_data는 None을 넘기지만 계약은 계약이다)
    s = make_dt(remask_acc="off")(m, K, T, make_attrs(), "knee_acc8_1.h5", 3)
    assert np.array_equal(s[0].numpy(), mask_t.numpy())


@pytest.mark.parametrize("is_train,isforward", [(False, False), (False, True)])
def test_val_and_forward_never_remask(is_train, isforward):
    """val/forward 로더에는 재생성이 걸리면 안 된다(검증 분포·과거 런 비교 가능성 보존)."""
    W = 368
    K = make_kspace(W=W)
    T = None if isforward else make_target()
    attrs = -1 if isforward else make_attrs()
    m = equispaced_mask(W, 4, canonical_offset(W, 4))

    dt = make_dt(remask_acc=8, is_train=is_train, isforward=isforward)
    out = dt(m, K, T, attrs, "knee_acc4_1.h5", 0)
    got = out[0].numpy().reshape(-1)
    assert np.array_equal(got.astype(bool), m.astype(bool))


def test_set_epoch_reaches_transform():
    """epoch이 transform까지 안 오면 uniform offset이 매 epoch 같아져 이득만 조용히 사라진다."""
    dt = make_dt(remask_acc=8)
    assert dt.epoch == 0
    dt.set_epoch(7)
    assert dt.epoch == 7


def test_uniform_offset_varies_and_is_deterministic():
    W = 368
    K, T = make_kspace(W=W), make_target()
    m = equispaced_mask(W, 4, canonical_offset(W, 4))

    def run(epoch, sl):
        dt = make_dt(remask_acc=8, remask_offset="uniform", remask_seed=430)
        dt.set_epoch(epoch)
        return dt(m, K, T, make_attrs(), "knee_acc4_1.h5", sl)[0].numpy().reshape(-1).astype(bool)

    assert np.array_equal(run(0, 0), run(0, 0))                       # 같은 키 → 같은 마스크
    seen = {equispaced_mask(W, 8, o).astype(bool).tobytes() for o in range(8)}
    drawn = {run(e, s).tobytes() for e in range(6) for s in range(6)}
    assert drawn <= seen, "생성된 마스크가 전부 acc8 마스크 가족 안에 있어야 한다"
    assert len(drawn) >= 4, "epoch·slice를 바꿔도 offset이 안 흩어지면 RNG 배선이 죽은 것"


def test_flip_decisions_unchanged_by_remask():
    """remask를 켜도 **어느 슬라이스가 뒤집히는지**가 그대로여야 한다.
    달라지면 A/B 두 arm이 서로 다른 증강 추출 위에서 비교돼, 차이가 remask 탓인지
    flip 배치가 달라진 탓인지 못 가린다. (offset RNG의 salt 분리가 지키는 성질)"""
    W = 368
    K, T = make_kspace(W=W), make_target()
    m = equispaced_mask(W, 4, canonical_offset(W, 4))
    flipped_T = T[:, ::-1]

    def flip_seq(remask_acc):
        aug = MRIAugmentor(AugmentConfig(on=True, schedule="constant", max_prob=0.5, affine=False,  # flip 스트림 순수 탐침 — affine 추첨 제외
                                         seed=430))
        dt = make_dt(remask_acc=remask_acc, augmentor=aug)
        dt.set_epoch(2)
        return [bool(np.array_equal(dt(m, K, T, make_attrs(), "knee_acc4_1.h5", i)[2].numpy(),
                                    flipped_T))
                for i in range(30)]

    off, on = flip_seq(None), flip_seq(8)
    assert any(off) and not all(off), "p=0.5인데 전부 같은 결정이면 이 테스트가 아무것도 안 잰다"
    assert off == on


def test_remask_composes_with_flip():
    """flip과 함께 켜도 마스크·k-space 정렬이 유지되고 라인 수가 보존되는가."""
    W = 368
    K, T = make_kspace(W=W), make_target()
    m = equispaced_mask(W, 4, canonical_offset(W, 4))
    aug = MRIAugmentor(AugmentConfig(on=True, schedule="constant", max_prob=1.0, seed=430))

    dt = make_dt(remask_acc=8, remask_offset="canonical", augmentor=aug)
    mask_t, kspace, *_ = dt(m, K, T, make_attrs(), "knee_acc4_1.h5", 0)

    got = mask_t.numpy().reshape(-1).astype(bool)
    ref = equispaced_mask(W, 8, canonical_offset(W, 8)).astype(bool)
    assert got.sum() == ref.sum(), "flip은 샘플 라인 수를 보존해야 한다"
    ks = kspace.numpy()
    energy = np.abs(ks[..., 0] + 1j * ks[..., 1]).sum(axis=(0, 1))
    assert np.allclose(energy[~got], 0.0), "flip 후에도 마스크 밖은 0"


# ---------------------------------------------------------------- 마스크 증강(--mask-aug)

def _family(W, acc):
    """그 폭·가속률에서 나올 수 있는 마스크 전부(offset 0..acc-1)."""
    return {equispaced_mask(W, acc, o).astype(bool).tobytes() for o in range(acc)}


def test_infer_acc_reads_supplied_mask():
    """딸려온 마스크에서 가속률을 읽는다 — ACS 밖 격자 간격. 학습셋 폭 범위 × 전 offset 전수."""
    for W in (320, 368, 372, 400, 480):
        for acc in (4, 8):
            for off in range(acc):
                assert infer_acc(equispaced_mask(W, acc, off)) == acc, (W, acc, off)


@pytest.mark.parametrize("acc,W", [(4, 368), (8, 372)])
def test_mask_aug_offset_keeps_acceleration(acc, W):
    """offset 모드는 **위상만** 흔든다 — 가속률은 파일 원래 값 그대로여야 한다.
    --remask 없이 도는 경로라, 가속률을 딸려온 마스크에서 못 읽으면 여기서 걸린다."""
    import torch
    K, T = make_kspace(W=W), make_target()
    m = equispaced_mask(W, acc, canonical_offset(W, acc))
    fam = _family(W, acc)

    drawn = set()
    for e in range(6):
        dt = make_dt(remask_acc=None, mask_aug="offset")
        dt.set_epoch(e)
        got = dt(m, K, T, make_attrs(), f"knee_acc{acc}_1.h5", 0)[0].numpy().reshape(-1).astype(bool)
        assert got.tobytes() in fam, "가속률이 바뀌었다 — offset 모드가 아니다"
        assert extract_acc_from_mask(torch.from_numpy(got.astype(np.float32))) == acc
        drawn.add(got.tobytes())
    assert len(drawn) >= 2, "epoch을 바꿔도 offset이 안 흩어지면 RNG 배선이 죽은 것"


def test_mask_aug_full_draws_only_4_or_8():
    """full 모드는 가속률까지 흔들되 대회에 존재하는 4|8 밖으로 나가면 안 된다(둘 다 나와야 한다)."""
    W = 368
    K, T = make_kspace(W=W), make_target()
    m = equispaced_mask(W, 4, canonical_offset(W, 4))
    fam = {equispaced_mask(W, a, o).astype(bool).tobytes(): a for a in (4, 8) for o in range(a)}

    accs = set()
    for e in range(4):
        dt = make_dt(remask_acc=None, mask_aug="full")
        dt.set_epoch(e)
        for sl in range(6):
            got = dt(m, K, T, make_attrs(), "knee_acc4_1.h5", sl)[0].numpy().reshape(-1).astype(bool)
            assert got.tobytes() in fam, "acc4|acc8 마스크 가족 밖의 물건이 나왔다"
            accs.add(fam[got.tobytes()])
    assert accs == {4, 8}


def test_mask_aug_off_keeps_supplied_mask():
    """off면 딸려온 마스크를 그대로 곱한 것과 바이트까지 같아야 한다(기대값은 구현 밖에서 손계산)."""
    W = 372
    K, T = make_kspace(W=W), make_target()
    m = equispaced_mask(W, 8, canonical_offset(W, 8))

    mask_t, kspace, *_ = make_dt(remask_acc=None, mask_aug="off")(
        m, K, T, make_attrs(), "knee_acc8_1.h5", 3)
    got = kspace.numpy()
    assert np.array_equal(mask_t.numpy().astype(bool), (m >= 1).reshape(1, 1, W, 1))
    assert np.array_equal((got[..., 0] + 1j * got[..., 1]).astype(K.dtype),
                          (K * (m >= 1).astype(m.dtype)).astype(K.dtype))


@pytest.mark.parametrize("is_train,isforward", [(False, False), (False, True)])
def test_val_and_forward_never_mask_aug(is_train, isforward):
    """--remask 와 같은 규칙 — 마스크 증강도 학습 로더 전용이다."""
    W = 368
    K = make_kspace(W=W)
    T = None if isforward else make_target()
    attrs = -1 if isforward else make_attrs()
    m = equispaced_mask(W, 4, canonical_offset(W, 4))

    dt = make_dt(remask_acc=None, mask_aug="full", is_train=is_train, isforward=isforward)
    got = dt(m, K, T, attrs, "knee_acc4_1.h5", 0)[0].numpy().reshape(-1)
    assert np.array_equal(got.astype(bool), m.astype(bool))


# ---------------------------------------------------------------- 가드

def test_bad_combos_raise():
    """오조합은 데이터 없이도 거부돼야 한다(로더 구성 시점 = 체크포인트 로드 전)."""
    from types import SimpleNamespace
    from utils.data.load_data import create_data_loaders

    def args(**kw):
        base = dict(remask="4", acc="8", batch_size=1, input_key="kspace",
                    target_key="image_label", max_key="max", seed=430, num_workers=0)
        base.update(kw)
        return SimpleNamespace(**base)

    with pytest.raises(ValueError, match="--acc all"):
        create_data_loaders("/nonexistent", args(), shuffle=True, is_train=True)
    with pytest.raises(ValueError, match="GRAPPA"):
        create_data_loaders("/nonexistent", args(acc="all", return_grappa=True),
                            shuffle=True, is_train=True)


# resume 가드(check_resume_compat)의 remask 테스트는 scripts/tests/test_resume.py 에 있다 —
# 그 파일이 이미 train_part 를 import 하고, 여기로 끌어오면 이 파일이 GPU 스택 의존이 된다.
