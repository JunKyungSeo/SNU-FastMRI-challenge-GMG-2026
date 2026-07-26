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


def test_schedule_covers_a_ten_epoch_run():
    """10ep 런에서 p가 실제로 올라간다 — 기존 기본값(전 구간 0) 재발 방지."""
    cfg = AugmentConfig(on=True, schedule="exp", delay=2, total_epochs=10,
                        exp_decay=5.0, max_prob=0.55)
    aug = MRIAugmentor(cfg)
    ps = []
    for e in range(10):
        aug.set_epoch(e)
        ps.append(aug._prob())

    assert ps[0] == 0.0 and ps[1] == 0.0        # delay 구간
    assert ps[2] == 0.0                          # t == D 는 지수식이 정확히 0
    assert all(p > 0.0 for p in ps[3:]), f"delay 이후에도 0이 있다: {ps}"
    assert all(b > a for a, b in zip(ps[3:-1], ps[4:])), f"단조 증가가 아니다: {ps}"
    assert ps[3] == pytest.approx(0.2573, abs=1e-4)
    assert ps[9] == pytest.approx(0.5468, abs=1e-4)


def test_schedule_scales_with_run_length():
    """같은 플래그로 런 길이만 바꿔도 '마지막 epoch에서 p_max 근처'가 유지된다."""
    last = []
    for T in (10, 25):
        aug = MRIAugmentor(AugmentConfig(on=True, schedule="exp", delay=2,
                                         total_epochs=T, exp_decay=5.0, max_prob=0.55))
        aug.set_epoch(T - 1)
        last.append(aug._prob())
    assert all(0.5 < p < 0.55 for p in last), f"런 길이에 따라 증강량이 달라진다: {last}"


def test_schedule_modes_and_guards():
    common = dict(on=True, delay=2, total_epochs=10, max_prob=0.55)
    c = MRIAugmentor(AugmentConfig(schedule="constant", **common))
    c.set_epoch(0); assert c._prob() == 0.0      # delay는 constant에도 적용
    c.set_epoch(5); assert c._prob() == 0.55

    r = MRIAugmentor(AugmentConfig(schedule="ramp", **common))
    r.set_epoch(6); assert r._prob() == pytest.approx(0.275)

    # T <= D 가드(1ep 스모크 런 등)에서 ZeroDivisionError가 나지 않는다
    g = MRIAugmentor(AugmentConfig(on=True, schedule="exp", delay=2,
                                   total_epochs=1, exp_decay=5.0, max_prob=0.55))
    g.set_epoch(0); assert g._prob() == 0.0
    g.set_epoch(3); assert g._prob() == 0.55

    off = MRIAugmentor(AugmentConfig(on=False, schedule="constant", max_prob=1.0))
    off.set_epoch(9); assert off._prob() == 0.0


def test_default_delay_is_zero_so_aug_starts_immediately():
    aug = MRIAugmentor(AugmentConfig(on=True, total_epochs=10, max_prob=0.55))
    aug.set_epoch(0); assert aug._prob() == 0.0
    aug.set_epoch(1); assert aug._prob() > 0.0


def test_probs_applies_weights_and_orig_mask_mix():
    aug = MRIAugmentor(AugmentConfig(on=True, schedule="constant", max_prob=0.5,
                                     weight_flip=1.0, weight_mask=1.0, orig_mask_prob=0.5))
    p, p_flip, p_mask = aug.probs()
    assert p == 0.5
    assert p_flip == pytest.approx(0.5)
    assert p_mask == pytest.approx(0.25)      # p · w · (1 - orig_mask_prob)


def test_probs_are_clipped_to_one():
    aug = MRIAugmentor(AugmentConfig(on=True, schedule="constant", max_prob=0.9,
                                     weight_flip=4.0, orig_mask_prob=0.0))
    _, p_flip, _ = aug.probs()
    assert p_flip == 1.0


def test_weight_zero_disables_one_transform_only():
    """weight_flip=0이면 flip은 안 걸리고 mask만 걸린다(분리 A/B의 근거).
    슬라이스별 RNG가 달라 _aug_mask 내부 thin이 1로 뽑히는 슬라이스는(그대로 반환, 의도된 동작)
    라인 수가 안 줄어들 수 있다 → 여러 슬라이스를 돌며 '적어도 하나는 줄었다'로 확인한다."""
    aug = MRIAugmentor(AugmentConfig(on=True, schedule="constant", max_prob=1.0,
                                     flip=True, mask=True, weight_flip=0.0,
                                     weight_mask=1.0, orig_mask_prob=0.0))
    M, T = make_mask(), make_target()
    shrunk = []
    for sl in range(20):
        _, M2, T2, _, _ = aug(make_kspace(), M, T, None, None, "knee_acc4_1.h5", sl)
        assert np.array_equal(T2, T), f"slice {sl}: weight_flip=0인데 flip이 걸렸다"
        assert M2.sum() <= M.sum(), f"slice {sl}: 원본보다 라인이 늘었다"
        shrunk.append(M2.sum() < M.sum())
    # thin==1인 슬라이스는 마스크가 그대로라 전체가 아니라 '적어도 하나'로 확인한다
    assert any(shrunk), "20개 슬라이스 전부에서 mask 증강이 걸리지 않았다"


def test_set_epoch_reaches_augmentor_through_transform():
    """train_part가 잡는 경로(dataset.transform → augmentor)로 epoch이 전달된다."""
    aug = MRIAugmentor(AugmentConfig(on=True, schedule="exp", delay=2, total_epochs=10,
                                     exp_decay=5.0, max_prob=0.55))
    dt = DataTransform(isforward=False, max_key="max", augmentor=aug, is_train=True)
    dt.set_epoch(0)
    assert aug.probs()[0] == 0.0
    dt.set_epoch(9)
    assert aug.probs()[0] > 0.5

    # augmentor가 없는 val transform에서도 set_epoch은 안전한 no-op
    DataTransform(isforward=False, max_key="max").set_epoch(5)


def test_datatransform_mask_and_kspace_stay_consistent():
    """VarNet의 data-consistency layer는 반환된 mask_t로 kspace를 마스킹한다는 전제로 동작한다.
    mask_t는 mask1d에서, kspace는 augmentor가 반환한 K에서 각각 만들어지는 서로 다른 코드
    경로라 __call__ 내부 순서가 바뀌면 둘이 어긋날 수 있는데, 그래도 크래시도 안 나고 조용히
    복원 품질만 나빠진다. 그래서 '마스크가 0인 PE 컬럼은 k-space도 정확히 0'을 직접 검증한다.
    (증강이 실제로 라인을 지우도록 mask aug를 p=1로 강제해 검증이 vacuous하지 않게 만든다.)"""
    aug = MRIAugmentor(mask_only_cfg())
    dt = DataTransform(isforward=False, max_key="max", augmentor=aug, is_train=True)

    out = dt(make_mask(), make_kspace(), make_target(), {"max": 1.0}, "knee_acc4_1.h5", 0)
    mask_t, kspace_t, target_out, maximum, fname, sl = out[:6]

    mask_np = np.asarray(mask_t).reshape(-1)                 # (1,1,W,1) → (W,)
    kspace_np = np.asarray(kspace_t)                          # (coils, H, W, 2)

    assert mask_np.sum() > 0, "마스크가 전부 0이라 아래 검증이 vacuous하다"
    assert mask_np.sum() < mask_np.shape[0], "mask aug가 라인을 하나도 안 지웠다 — 검증이 vacuous하다"

    zero_cols = np.where(mask_np == 0)[0]
    assert np.all(kspace_np[:, :, zero_cols, :] == 0), (
        "mask==0인 PE 컬럼에 0이 아닌 k-space가 남아있다 — mask_t와 kspace가 어긋났다")


def test_probs_are_gated_by_on_off_switch():
    """probs()가 반환하는 p_flip·p_mask는 '실효 확률'이어야 한다 — cfg.flip/cfg.mask가
    꺼져 있으면 실제로 그 증강은 절대 걸리지 않으므로(__call__의 if cfg.flip/if cfg.mask 가드),
    로그에도 0으로 찍혀야 한다. 그렇지 않으면 [aug] 로그가 '꺼진 증강이 0.5 확률로 걸린다'는
    거짓말을 하게 된다(서버 검증 스크립트 verify_aug.sh가 잡아낸 결함)."""
    flip_off = MRIAugmentor(AugmentConfig(on=True, schedule="constant", max_prob=0.5,
                                          flip=False, mask=True, orig_mask_prob=0.0))
    p, p_flip, p_mask = flip_off.probs()
    assert p > 0.0
    assert p_flip == 0.0
    assert p_mask > 0.0

    mask_off = MRIAugmentor(AugmentConfig(on=True, schedule="constant", max_prob=0.5,
                                          flip=True, mask=False, orig_mask_prob=0.0))
    _, p_flip2, p_mask2 = mask_off.probs()
    assert p_flip2 > 0.0
    assert p_mask2 == 0.0
