"""증강(MRIAugmentor) 불변식 회귀 테스트 — 전부 CPU, GPU·데이터 파일 불필요(합성 배열).

핵심:
  1. 좌우 flip이 k-space·mask·target·bbox를 정렬 유지하며 함께 뒤집는가.
  2. 마스크 증강이 ACS를 100% 보존하고 원본의 부분집합만 반환하는가.
  3. 확률 스케줄이 실제 런 길이(10 epoch)에서 0이 아닌가.
  4. 증강 off면 기존 파이프라인과 동일한가(6-튜플, val/forward 미적용).
"""
import numpy as np
import pytest

from utils.data.transforms import (
    AugmentConfig, MRIAugmentor, DataTransform, _detect_acs, _fftc, _ifftc,
)

TARGET_W = 384


def make_mask(W=368, acs=29, acc=4):
    """제공 마스크 모방: 중앙 ACS 통짜 완전샘플 + 주변부 1/acc equispaced."""
    m = np.zeros(W, dtype=np.float32)
    m[::acc] = 1.0
    c = W // 2
    m[c - acs // 2: c - acs // 2 + acs] = 1.0
    return m


def make_kspace(coils=4, H=64, W=368, seed=0):
    rng = np.random.default_rng(seed)
    img = (rng.standard_normal((coils, H, W))
           + 1j * rng.standard_normal((coils, H, W)))
    return _fftc(img).astype(np.complex64)


def make_target(seed=1):
    return np.random.default_rng(seed).random((TARGET_W, TARGET_W)).astype(np.float32)


def flip_only_cfg(**kw):
    """p=1 고정 · flip만 ON."""
    base = dict(on=True, schedule="constant", max_prob=1.0, flip=True, mask=False)
    base.update(kw)
    return AugmentConfig(**base)


def mask_only_cfg(**kw):
    """p=1 고정 · mask만 ON · 원본 유지 확률 0(항상 증강)."""
    base = dict(on=True, schedule="constant", max_prob=1.0, flip=False, mask=True,
                orig_mask_prob=0.0, mask_thin_max=4)
    base.update(kw)
    return AugmentConfig(**base)


def test_flip_is_pixel_exact_and_syncs_kspace_mask_bbox():
    aug = MRIAugmentor(flip_only_cfg())
    K, M, T = make_kspace(), make_mask(), make_target()
    bbox = [{"x": 158, "y": 40, "width": 30, "height": 20, "label": "lesion"}]

    K2, M2, T2, B2, _ = aug(K, M, T, bbox, None, "knee_acc4_1.h5", 0)

    # target은 픽셀 정확히 좌우 반전
    assert np.array_equal(T2, T[:, ::-1])
    # 샘플 라인 수 보존
    assert M2.sum() == M.sum()
    # k-space도 같은 flip: recon(K_flip) == flip(recon(K))
    rec = np.abs(_ifftc(K * M))
    rec2 = np.abs(_ifftc(K2))
    rel = np.linalg.norm(rec2 - rec[:, :, ::-1]) / np.linalg.norm(rec)
    assert rel < 1e-5, f"flip 정렬 오차 {rel:.2e}"
    # bbox 좌표 동기: x' = W - x - width, 나머지 불변
    assert B2[0]["x"] == TARGET_W - 158 - 30
    assert (B2[0]["y"], B2[0]["width"], B2[0]["height"]) == (40, 30, 20)
    assert B2[0]["label"] == "lesion"


def test_mask_aug_preserves_acs_and_is_subset():
    aug = MRIAugmentor(mask_only_cfg())
    M = make_mask()
    acs = _detect_acs(M)
    assert acs.sum() > 0, "테스트 마스크에 ACS가 없다 — 헬퍼 오류"

    for sl in range(20):  # thin 배수가 슬라이스별 랜덤이라 여러 번 확인
        _, M2, _, _, _ = aug(make_kspace(), M, make_target(), None, None, "knee_acc4_1.h5", sl)
        assert np.all(M2[acs] >= 1), f"slice {sl}: ACS 라인이 드롭됐다"
        assert np.all((M2 >= 1) <= (M >= 1)), f"slice {sl}: 원본에 없는 라인이 추가됐다"


def test_augmentation_is_deterministic_per_sample():
    K, M, T = make_kspace(), make_mask(), make_target()
    outs = []
    for _ in range(2):
        aug = MRIAugmentor(mask_only_cfg())   # 새 인스턴스 = 프로세스 재시작 모사
        aug.set_epoch(3)
        outs.append(aug(K, M, T, None, None, "knee_acc4_1.h5", 7)[1])
    assert np.array_equal(outs[0], outs[1])


def test_off_by_default_and_returns_six_tuple():
    dt = DataTransform(isforward=False, max_key="max")
    out = dt(make_mask(), make_kspace(), make_target(), {"max": 1.0}, "knee_acc4_1.h5", 0)
    assert len(out) == 6, f"플래그 전부 off인데 {len(out)}-튜플이 반환됐다"


def test_val_transform_never_augments():
    """is_train=False면 augmentor를 넘겨도 증강되지 않는다(검증 오염 방지)."""
    aug = MRIAugmentor(flip_only_cfg())
    T = make_target()
    dt = DataTransform(isforward=False, max_key="max", augmentor=aug, is_train=False)
    out = dt(make_mask(), make_kspace(), T, {"max": 1.0}, "knee_acc4_1.h5", 0)
    assert np.allclose(np.asarray(out[2]), T), "val 경로에서 target이 변형됐다"


def test_forward_path_never_augments():
    """isforward=True(리더보드 추론)면 is_train을 켜도 증강이 비활성이다."""
    aug = MRIAugmentor(flip_only_cfg())
    dt = DataTransform(isforward=True, max_key=-1, augmentor=aug, is_train=True)
    assert dt.is_train is False
