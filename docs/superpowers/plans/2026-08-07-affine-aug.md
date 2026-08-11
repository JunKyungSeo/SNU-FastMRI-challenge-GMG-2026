# Affine(회전+확대) 증강 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** MRAugment 방식의 affine 증강(회전 U(−5°,+5°) + 확대 U(1.0,1.1), shear 없음)을 `--aug-on` 아래 기본 동작으로 추가한다 (사용자 확정 2026-08-07: A/B 없이 바로 채택, bbox 변환 불필요, 보간 블러는 대책 필수).

**Architecture:** 기존 flip과 달리 label을 재사용할 수 없다(384² 고정 크롭이라 기하 변환 시 프레임 밖 정보가 없다). 대신 **full k-space**(remask 작업에서 train 170/170 실측 확인)에서 코일별 IFFT → 복소 영상 affine 워프 → FFT로 "변형된 환자의 획득"을 만들고, **target을 워프된 코일 영상의 RSS에서 재생성**한다 — 입력·정답이 같은 변형에서 유도되어 구성상 정합한다(제공 label ↔ full recon RSS corr 1.0000000000, remask 검증). 워프는 **mask 곱 전** `kspace_c`에서 수행하고, 이후 mask 곱·flip은 기존 경로 그대로다(affine은 mask를 건드리지 않는다 — 샘플 격자는 불변). 보간 블러 대책 = **INTER_LANCZOS4**(bilinear 대비 고주파 보존, 테스트로 강제). 감도맵은 모델(sens_net)이 슬라이스별 ACS에서 재추정하므로 "코일이 환자와 함께 도는" 유효 획득으로 정합 문제 없음.

**Tech Stack:** numpy, cv2(warpAffine — utils/common에서 이미 사용 중인 의존성), 기존 `_fftc/_ifftc`(centered FFT), pytest.

## Global Constraints

- **이 로컬(노트북)에 torch가 없다** — pytest·검증 스크립트 실행은 **서버 분할 터미널에서 사람이** 한다(aug 플랜 선례 ecbf6b7 "서버 검증 게이트"). 로컬에서는 `py -3.10 -m py_compile` 문법 확인까지만. GPU 학습·recon은 Claude가 절대 직접 실행하지 않는다(CLAUDE.md §6).
- `--aug-on` 기본 off 계약 유지: **끄면 기존 학습과 비트 동일**. 켜면 affine이 기본 포함되고(`--aug-affine` 기본 True), `--no-aug-affine`이면 flip 추첨 스트림까지 구버전과 동일하게 재현된다(affine 분기가 rng를 소비하지 않도록 short-circuit — 테스트로 강제).
- **shell31 계열(현행 본학습)은 이미 `--aug-on`이라 재실행부터 affine이 자동 적용된다** — 사용자 지시(바로 채택)에 부합. 되돌리려면 `--no-aug-affine` 또는 revert.
- bbox·GRAPPA는 affine 변환 미구현 — 현행 셸에서 미사용임을 확인했으므로(λ=0 → bbox는 None으로 들어옴) 변환 대신 **fail-fast assert**로 잠재 조합을 시끄럽게 막는다.
- fg_mask는 DataTransform이 증강 **후** target에서 재계산(transforms.py:499-501)하므로 수정 불필요. `maximum`(attrs max)은 flip 선례대로 유지 — 회전·확대는 내부 최대값을 근사 보존한다.
- complex128 승격 걱정 불필요: `_affine`의 FFT 왕복이 complex128을 만들어도 DataTransform 말미(transforms.py:526)가 `orig_cdtype`로 캐스트해 되돌린다 — 옛 FFT 증강 대비용으로 이미 존재하는 메커니즘(실사용처 확인됨).
- 재현성: 기존 `_rng(seed, epoch, fname, slice)` 메커니즘 재사용 — 같은 seed면 재학습 시 같은 추첨.
- 커밋: user.name=kindofwater / user.email=bedso@snu.ac.kr (repo local 확인됨). **AI attribution 금지.**

## File Structure

- Modify `utils/data/transforms.py` — 상단 import·헤더 주석, `AugmentConfig`(:187-200), 모듈 함수 `_rss_crop` 신설, `MRIAugmentor._affine` 신설, `__call__`(:273-290) 재구성
- Modify `scripts/tests/test_augment.py` — import 확장, `flip_cfg`에 affine=False, 신규 테스트 8건 + 팬텀 헬퍼
- Modify `train.py` — `--aug-exp-decay`(:115) 다음에 인자 3개
- Modify `utils/data/load_data.py:271-279` — AugmentConfig에 3필드 전달
- Modify `utils/learning/train_part.py:520` — [aug] 로그 라인에 p_affine 표기
- Create `scripts/verify_affine_target.py` — 서버용 재생성 규약 정합 + 워프 시간 실측 (비GPU)

---

### Task 1: transforms.py 본체 — _affine · target 재생성 · __call__ 재구성 (+테스트)

**Files:**
- Modify: `utils/data/transforms.py:1-4`(import), `:21-37`(헤더 주석), `:187-200`(AugmentConfig), `:273-290`(__call__), 그리고 `_flip_lr` 뒤에 `_affine`·모듈 레벨 `_rss_crop` 추가
- Test: `scripts/tests/test_augment.py`

**Interfaces:**
- Consumes: `_fftc/_ifftc(x, axes=(-2,-1))`(기존), `_flip_lr(K, target, bbox, mask, grappa=None) → (K2, tgt, M2, bbox, g)`(기존), `_rng(fname, sl)`(기존)
- Produces: `AugmentConfig(affine: bool=True, rot_max: float=5.0, scale_max: float=1.1)` 필드, `MRIAugmentor._affine(kspace_c, target_hw, rng) → (warped_kspace_c, new_target)`, `_rss_crop(imgs, hw=(384,384)) → np.float32 (384,384)`. Task 2가 이 필드명을 그대로 배선하고, Task 3 스크립트가 `_rss_crop`·`_affine`을 import한다.

- [ ] **Step 1: 실패하는 테스트 작성** — `scripts/tests/test_augment.py` 수정.

먼저 import(:13-15)를 확장하고 `flip_cfg`(:40-44)에 `affine=False`를 추가한다(기존 flip 테스트들은 flip 단독 계약이므로). 참고: 스케줄 테스트들(:114-191)은 `AugmentConfig`를 직접 만들지만 `__call__`을 실행하지 않아 affine 기본 True의 영향이 없고, `__call__`을 실제로 태우는 기존 테스트는 전부 `flip_cfg` 경유라 이 한 줄로 전부 보호된다(2026-08-07 확인):

```python
import cv2
import numpy as np
import pytest

from utils.data.transforms import (
    AugmentConfig, MRIAugmentor, DataTransform, _detect_acs, _fftc, _ifftc, _rss_crop,
)
```

```python
def flip_cfg(**kw):
    """p=1 고정(항상 flip). affine=False — 이 헬퍼는 flip 단독 계약 테스트용."""
    base = dict(on=True, schedule="constant", max_prob=1.0, affine=False)
    base.update(kw)
    return AugmentConfig(**base)
```

파일 끝에 팬텀 헬퍼와 신규 테스트 8건을 추가한다:

```python
# ---------------------------------------------------------------------------
# affine(회전+확대) 증강 — 2026-08-07 채택. full k-space 워프 + target 재생성 방식.
# ---------------------------------------------------------------------------

def make_phantom_kspace(coils=3, H=128, W=368, seed=2):
    """복원이 의미 있는 팬텀 k-space: 저주파 blob 영상 × 부드러운 코일 감도.
    make_kspace(백색잡음)는 워프 정합을 눈으로도 수치로도 판별할 수 없어 별도로 둔다."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:H, 0:W]
    img = np.zeros((H, W), np.float32)
    for _ in range(6):
        cy, cx = rng.uniform(H * 0.3, H * 0.7), rng.uniform(W * 0.3, W * 0.7)
        r = rng.uniform(6, 18)
        img += np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * r * r)).astype(np.float32)
    profiles = [(0.2, 0.3), (0.8, 0.5), (0.4, 0.8)][:coils]
    sig = 0.5 * max(H, W)
    coil_imgs = np.stack([
        img * (0.5 + 0.5 * np.exp(-((y - H * ky) ** 2 + (x - W * kx) ** 2) / (2 * sig ** 2)))
        for ky, kx in profiles
    ]).astype(np.complex64)
    return _fftc(coil_imgs).astype(np.complex64), img


class _FixedRng:
    """_affine의 추첨(각도, 배율)을 고정하는 스텁 — uniform 호출 순서에 의존."""
    def __init__(self, deg, s):
        self.vals = [deg, s]
    def uniform(self, a, b):
        return self.vals.pop(0)


def affine_cfg(**kw):
    base = dict(on=True, schedule="constant", max_prob=1.0, delay=0,
                total_epochs=10, affine=True, rot_max=5.0, scale_max=1.1, seed=430)
    base.update(kw)
    return AugmentConfig(**base)


def test_affine_pair_roundtrip_consistency():
    """반환 target == 워프된 k-space를 다시 복원한 RSS crop (FFT 왕복·crop 규약 정합)."""
    K, _ = make_phantom_kspace()
    aug = MRIAugmentor(affine_cfg())
    Kw, tgt = aug._affine(K, (384, 384), _FixedRng(3.0, 1.05))
    assert Kw.shape == K.shape and tgt.shape == (384, 384) and tgt.dtype == np.float32
    ref = _rss_crop(_ifftc(Kw), (384, 384))
    assert np.corrcoef(ref.ravel(), tgt.ravel())[0, 1] > 0.9999


def test_affine_matches_independent_magnitude_warp():
    """복소 코일 워프→RSS ≈ 원본 RSS를 같은 행렬로 워프 — 행렬·중심·축순서 검증.
    (RSS가 비선형이라 완전 동일은 아님 — 팬텀에서 corr>0.99면 기하가 맞는 것)"""
    K, _ = make_phantom_kspace()
    imgs = _ifftc(K)
    rss0 = np.sqrt((np.abs(imgs) ** 2).sum(0)).astype(np.float32)
    H, W = rss0.shape
    aug = MRIAugmentor(affine_cfg())
    _, tgt = aug._affine(K, (384, 384), _FixedRng(3.0, 1.05))
    mat = cv2.getRotationMatrix2D(((W - 1) / 2.0, (H - 1) / 2.0), 3.0, 1.05)
    ref_full = cv2.warpAffine(rss0, mat, (W, H), flags=cv2.INTER_LANCZOS4,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    ref = _rss_crop(ref_full[np.newaxis, ...], (384, 384))
    assert np.corrcoef(ref.ravel(), tgt.ravel())[0, 1] > 0.99


def test_affine_kspace_support_stays_in_mask():
    """affine은 mask를 건드리지 않고, 반환 K의 에너지는 mask 열 안에만 있어야 한다."""
    K, _ = make_phantom_kspace()
    M = make_mask(W=368)
    aug = MRIAugmentor(affine_cfg())
    out_K, out_M, tgt, bbox, g = aug(K, M, make_target(), None, None, "p.h5", 0)
    assert np.array_equal((out_M >= 1), np.roll(np.flip(M >= 1), 1)) or np.array_equal(out_M, M)
    cols = (out_M >= 1)
    assert np.abs(out_K[..., ~cols]).max() == 0.0


def test_affine_deterministic_and_epoch_varies():
    K, _ = make_phantom_kspace()
    M = make_mask(W=368)
    a1, a2 = MRIAugmentor(affine_cfg()), MRIAugmentor(affine_cfg())
    o1 = a1(K, M, make_target(), None, None, "p.h5", 5)
    o2 = a2(K, M, make_target(), None, None, "p.h5", 5)
    assert np.array_equal(o1[2], o2[2])          # 같은 (seed,epoch,fname,sl) → 같은 target
    a2.set_epoch(1)
    o3 = a2(K, M, make_target(), None, None, "p.h5", 5)
    assert not np.array_equal(o1[2], o3[2])      # epoch가 다르면 다른 추첨


def test_no_affine_reproduces_legacy_flip_stream():
    """--no-aug-affine이면 affine 분기가 rng를 소비하지 않아 구버전 flip 추첨과 동일해야 한다."""
    K, M, T = make_kspace(), make_mask(), make_target()
    aug = MRIAugmentor(affine_cfg(affine=False, max_prob=0.5))
    out = aug(K, M, T, None, None, "a.h5", 3)
    ref_rng = aug._rng("a.h5", 3)
    flipped = ref_rng.random() < 0.5             # 구버전과 동일한 첫 추첨이 flip 게이트
    Mb = (M >= 1).astype(M.dtype)
    exp_tgt = aug._flip_lr(K * Mb, T, None, Mb)[1] if flipped else T
    assert np.array_equal(out[2], np.ascontiguousarray(exp_tgt))


def test_affine_guards_bbox_and_grappa():
    """bbox 손실·GRAPPA는 affine 변환 미구현 — 켠 채 조합하면 조용히 어긋나는 대신 죽어야 한다."""
    K, _ = make_phantom_kspace()
    M = make_mask(W=368)
    aug = MRIAugmentor(affine_cfg())
    with pytest.raises(AssertionError):
        aug(K, M, make_target(), [{"x": 1, "y": 1, "width": 5, "height": 5}], None, "p.h5", 0)
    with pytest.raises(AssertionError):
        aug(K, M, make_target(), None, np.ones((384, 384), np.float32), "p.h5", 0)


def test_affine_width_variants_and_dtype():
    """W=368(<384, pad 경로)·W=480(>384, crop 경로) 모두 동작, complex64 유지."""
    for W in (368, 480):
        K, _ = make_phantom_kspace(W=W)
        aug = MRIAugmentor(affine_cfg())
        Kw, tgt = aug._affine(K.astype(np.complex64), (384, 384), _FixedRng(-4.0, 1.1))
        assert tgt.shape == (384, 384) and np.isfinite(tgt).all()
        assert Kw.shape == K.shape


def test_lanczos_retains_more_highfreq_than_bilinear():
    """보간 블러 대책의 근거 고정: LANCZOS4가 bilinear보다 고주파를 더 보존한다."""
    _, img = make_phantom_kspace()
    img = img + 0.2 * np.random.default_rng(3).standard_normal(img.shape).astype(np.float32)
    H, W = img.shape
    mat = cv2.getRotationMatrix2D(((W - 1) / 2.0, (H - 1) / 2.0), 5.0, 1.0)

    def hf_energy(x):
        F = np.fft.fftshift(np.fft.fft2(x))
        cy, cx = H // 2, W // 2
        Y, X = np.ogrid[:H, :W]
        ring = ((Y - cy) ** 2 + (X - cx) ** 2) > (min(H, W) // 4) ** 2
        return float(np.abs(F[ring]).sum())

    wl = cv2.warpAffine(img, mat, (W, H), flags=cv2.INTER_LANCZOS4)
    wb = cv2.warpAffine(img, mat, (W, H), flags=cv2.INTER_LINEAR)
    assert hf_energy(wl) > hf_energy(wb)
```

- [ ] **Step 2: (로컬) 문법 확인 — RED 확인은 서버 게이트로 이월**

Run: `py -3.10 -m py_compile scripts/tests/test_augment.py`
Expected: exit 0. (pytest RED 실행은 로컬 torch 부재로 불가 — Task 4에서 사람이 서버에서 확인한다. 이 시점 기대 결과: `ImportError: cannot import name '_rss_crop'` 계열로 신규 8건 전부 FAIL.)

- [ ] **Step 3: transforms.py 구현.**

(a) `:1-4` import에 cv2 추가:

```python
import json
import zlib
import cv2
import numpy as np
import torch
```

(b) `:33-34` 헤더 주석의 미채택 기록을 갱신 — 기존 두 줄을 다음으로 교체:

```python
#   미채택(기록): 위-아래(readout) 플립 = 무릎 뼈 위치가 뒤바뀌는 비현실 형태 (shell29 실측 상한 +0.003).
#   affine(회전±5°+확대≤1.1, 2026-08-07 채택): translation 미채택 사유였던 "384 crop과 정렬 불가"를
#     target 재생성(full k-space 워프 → RSS)으로 해소했다 — 아래 MRIAugmentor._affine 참고.
```

(c) `AugmentConfig.__init__`(:191-200)의 `max_prob` 줄과 `seed` 줄 사이에 3필드 추가:

```python
        self.max_prob = kw.get("max_prob", 0.55)           # 논문 기본 p_max=0.55
        # affine(회전+확대) — 2026-08-07 채택(사용자 확정: shear 없음·확대만·±5°).
        self.affine = kw.get("affine", True)               # --aug-on 아래 기본 on. False면 flip 추첨 스트림까지 구버전 동일
        self.rot_max = kw.get("rot_max", 5.0)              # 회전 최대각(도) — U(-rot_max, +rot_max)
        self.scale_max = kw.get("scale_max", 1.1)          # 확대 상한 — U(1.0, scale_max). 축소는 빈 띠가 생겨 안 쓴다
        self.seed = kw.get("seed", 430)
```

(d) `MRIAugmentor` 클래스 정의 **앞**(모듈 레벨, `_detect_acs` 근처)에 `_rss_crop` 추가:

```python
def _rss_crop(imgs, hw=(384, 384)):
    """코일 복소 영상 → RSS → 중앙 crop(부족한 축은 0-pad) → float32.
    affine 증강의 target 재생성 규약 — 제공 label과의 정합은
    scripts/verify_affine_target.py가 실데이터 전수로 확인한다(corr≥0.9999)."""
    rss = np.sqrt((np.abs(imgs) ** 2).sum(axis=0)).astype(np.float32)
    H, W = rss.shape
    th, tw = hw
    out = np.zeros(hw, dtype=np.float32)
    sh, dh = max((H - th) // 2, 0), max((th - H) // 2, 0)
    sw, dw = max((W - tw) // 2, 0), max((tw - W) // 2, 0)
    ch, cw = min(H, th), min(W, tw)
    out[dh:dh + ch, dw:dw + cw] = rss[sh:sh + ch, sw:sw + cw]
    return out
```

(e) `_flip_lr` 메서드 뒤에 `_affine` 추가:

```python
    def _affine(self, kspace_c, target_hw, rng):
        """회전 U(-rot_max,+rot_max)° + 확대 U(1,scale_max)를 '환자 통째로' 적용하고
        target을 재생성한다 (MRAugment 방식). 반환: (warped_kspace_c, new_target).

        왜 label 재사용이 아니라 재생성인가: label은 384² 고정 크롭이라 기하 변환 시
        프레임 밖 정보가 없다. 학습 h5의 kspace는 full(remask에서 170/170 실측)이므로
        코일별 IFFT→워프→FFT로 '변형된 환자의 획득'을 만들고 정답도 같은 워프 영상의
        RSS에서 뽑으면 입력·정답이 구성상 정합한다(label↔full recon RSS corr 1.0 — remask 검증).

        보간: INTER_LANCZOS4 — bilinear 대비 고주파 보존(테스트로 강제. 채점이 SSIM이라
        블러가 곧 손해다). 회전 모서리 공백은 0 채움 — 입력·정답이 같은 공백을 공유하므로
        오지도(誤指導)가 없다. 코일 감도는 환자와 '함께 도는' 셈이 되는데, 이는 다른 코일
        배치의 유효한 획득이고 감도는 sens_net이 슬라이스별로 재추정하므로 문제없다.
        ⚠️ 2.5D(인접 슬라이스 입력)를 켜는 날에는 '한 입력 안의 슬라이스들은 같은 변형'
        제약이 새로 생긴다 — 지금은 슬라이스 1장 입력이라 해당 없음."""
        deg = float(rng.uniform(-self.cfg.rot_max, self.cfg.rot_max))
        s = float(rng.uniform(1.0, self.cfg.scale_max))
        C, H, W = kspace_c.shape
        mat = cv2.getRotationMatrix2D(((W - 1) / 2.0, (H - 1) / 2.0), deg, s)
        imgs = _ifftc(kspace_c)
        out = np.empty_like(imgs)
        for c in range(C):
            ch2 = np.stack([imgs[c].real, imgs[c].imag], axis=-1).astype(np.float32, copy=False)
            w = cv2.warpAffine(ch2, mat, (W, H), flags=cv2.INTER_LANCZOS4,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            out[c] = w[..., 0] + 1j * w[..., 1]
        return _fftc(out), _rss_crop(out, target_hw)
```

(f) `__call__`(:273-290)을 다음으로 교체:

```python
    def __call__(self, kspace_c, mask, target, bbox, grappa, fname, sl):
        """kspace_c:(coils,H,W) complex(raw full) · mask:(W,) · target:(384,384) or None
        · grappa:(384,384) magnitude or None.
        반환: (kspace_masked_complex, mask_1d, target, bbox, grappa).
        순서: affine(mask 곱 전, full k-space + target 재생성) → mask 곱 → flip(기존).
        affine이 안 걸리면 target은 제공 label 그대로다(재생성은 걸린 샘플만)."""
        M = (mask >= 1).astype(mask.dtype)
        if (not self.cfg.on) or (target is None):
            return kspace_c * M, M, target, bbox, grappa
        p = self.prob()
        if p <= 0.0:
            return kspace_c * M, M, target, bbox, grappa
        rng = self._rng(fname, sl)
        tgt = target

        # 1) affine(회전+확대) — full k-space에서. cfg.affine=False면 rng를 소비하지
        #    않아(and 단락) flip 추첨 스트림이 구버전과 비트 동일하게 유지된다.
        if self.cfg.affine and (rng.random() < p):
            assert not bbox, "affine 증강은 bbox 변환 미구현 — --bbox-loss-lambda와 같이 켤 수 없다"
            assert grappa is None, "affine 증강은 GRAPPA 워프 미구현 — GRAPPA 입력과 같이 켤 수 없다"
            kspace_c, tgt = self._affine(kspace_c, (384, 384), rng)

        K = kspace_c * M
        # 2) 좌-우(width) flip — target·bbox·mask·GRAPPA 동기 (기존 그대로, 독립 추첨)
        if rng.random() < p:
            K, tgt, M, bbox, grappa = self._flip_lr(K, tgt, bbox, M, grappa)
        return K, M.astype(mask.dtype), np.ascontiguousarray(tgt), bbox, grappa
```

- [ ] **Step 4: (로컬) 문법 확인**

Run: `py -3.10 -m py_compile utils/data/transforms.py scripts/tests/test_augment.py`
Expected: exit 0

- [ ] **Step 5: Commit**

```bash
git add utils/data/transforms.py scripts/tests/test_augment.py
git commit -m "feat(aug): affine(회전 ±5°+확대 ≤1.1) 증강 — full k-space 워프 + target 재생성"
```

---

### Task 2: 배선 — train.py 인자 · load_data 전달 · 학습 로그

**Files:**
- Modify: `train.py:115` 다음(--aug-exp-decay 뒤)
- Modify: `utils/data/load_data.py:271-279`
- Modify: `utils/learning/train_part.py:520`

**Interfaces:**
- Consumes: Task 1의 `AugmentConfig(affine, rot_max, scale_max)` 필드명
- Produces: CLI `--aug-affine/--no-aug-affine`(기본 True), `--aug-rot-max`(5.0), `--aug-scale-max`(1.1). 학습 로그 `[aug] epoch=N p_flip=… p_affine=…`

- [ ] **Step 1: train.py — `--aug-exp-decay`(:115) 바로 다음에 추가**

```python
    parser.add_argument('--aug-affine', action=argparse.BooleanOptionalAction, default=True,
                        help='회전+확대 증강(--aug-on 아래 기본 on). --no-aug-affine이면 flip만(구버전 재현)')
    parser.add_argument('--aug-rot-max', type=float, default=5.0, help='affine 회전 최대각(도) — U(-x,+x)')
    parser.add_argument('--aug-scale-max', type=float, default=1.1, help='affine 확대 상한 — U(1.0,x). 축소 미지원(빈 띠)')
```

- [ ] **Step 2: load_data.py:271-279 — AugmentConfig 호출에 3필드 추가** (`max_prob` 줄과 `seed` 줄 사이):

```python
        max_prob=getattr(args, "aug_max_prob", 0.55),
        affine=getattr(args, "aug_affine", True),
        rot_max=getattr(args, "aug_rot_max", 5.0),
        scale_max=getattr(args, "aug_scale_max", 1.1),
        seed=getattr(args, "seed", 430),
```

- [ ] **Step 3: train_part.py:520 로그 라인 교체**

기존:
```python
            print(f"[aug] epoch={epoch} p_flip={augmentor.prob():.4f}")
```
교체:
```python
            p = augmentor.prob()
            p_aff = p if augmentor.cfg.affine else 0.0
            print(f"[aug] epoch={epoch} p_flip={p:.4f} p_affine={p_aff:.4f}")
```

- [ ] **Step 4: (로컬) 문법 확인**

Run: `py -3.10 -m py_compile train.py utils/data/load_data.py utils/learning/train_part.py`
Expected: exit 0

- [ ] **Step 5: Commit**

```bash
git add train.py utils/data/load_data.py utils/learning/train_part.py
git commit -m "feat(aug): affine 배선 — CLI 3인자(기본 on)·AugmentConfig 전달·p_affine 로그"
```

---

### Task 3: scripts/verify_affine_target.py — 재생성 규약 정합 + 시간 실측 (서버용, 비GPU)

**Files:**
- Create: `scripts/verify_affine_target.py`

**Interfaces:**
- Consumes: Task 1의 `_rss_crop`, `MRIAugmentor._affine`, `AugmentConfig`
- Produces: 사람이 서버에서 실행하는 판정 스크립트. corr<0.9999 볼륨이 있으면 exit 1.

- [ ] **Step 1: 스크립트 작성**

```python
"""affine 증강의 target 재생성 규약(_rss_crop)이 제공 label과 정합하는지 실데이터로 확인.

비GPU — 서버 분할 터미널에서 바로 실행한다 (remask의 280볼륨 검증 선례):
    python scripts/verify_affine_target.py --data /root/Data/train
검증 1 (규약): 워프 없이 full k-space → _rss_crop == image_label (corr ≥ 0.9999).
    실패하면 crop/pad 규약이 주최측 label 생성과 다른 것 — 학습 전에 반드시 잡아야 한다.
검증 2 (비용): _affine 1회 시간(ms/slice) 실측 — p=0.5면 평균 부하는 그 절반.
"""
import argparse
import os
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data.transforms import AugmentConfig, MRIAugmentor, _ifftc, _rss_crop  # noqa: E402


class _FixedRng:
    def __init__(self, deg, s):
        self.vals = [deg, s]
    def uniform(self, a, b):
        return self.vals.pop(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/root/Data/train", help=".../train (아래 image/·kspace/ 전제)")
    ap.add_argument("--max-volumes", type=int, default=0, help="0=전수")
    args = ap.parse_args()

    kdir, idir = os.path.join(args.data, "kspace"), os.path.join(args.data, "image")
    files = sorted(f for f in os.listdir(kdir) if f.endswith(".h5"))
    if args.max_volumes:
        files = files[: args.max_volumes]

    aug = MRIAugmentor(AugmentConfig(on=True))
    bad, times = [], []
    for i, fn in enumerate(files):
        with h5py.File(os.path.join(kdir, fn), "r") as fk, \
             h5py.File(os.path.join(idir, fn), "r") as fi:
            sl = fk["kspace"].shape[0] // 2
            kspace = np.asarray(fk["kspace"][sl])          # (coils, H, W) complex, full
            label = np.asarray(fi["image_label"][sl], dtype=np.float32)
        regen = _rss_crop(_ifftc(kspace), label.shape)
        corr = float(np.corrcoef(regen.ravel(), label.ravel())[0, 1])
        t0 = time.perf_counter()
        aug._affine(kspace, label.shape, _FixedRng(5.0, 1.1))
        times.append((time.perf_counter() - t0) * 1e3)
        mark = "OK " if corr >= 0.9999 else "BAD"
        if corr < 0.9999:
            bad.append((fn, corr))
        print(f"[{i+1}/{len(files)}] {mark} {fn}  corr={corr:.10f}")

    print(f"\n_affine 시간: median {np.median(times):.1f} ms/slice (p=0.5면 평균 부하는 절반)")
    if bad:
        print(f"[FAIL] corr<0.9999 볼륨 {len(bad)}개 — _rss_crop 규약이 label 생성과 다르다. 학습 금지.")
        for fn, c in bad:
            print(f"    {fn}  corr={c:.10f}")
        sys.exit(1)
    print("[PASS] 재생성 규약이 전 볼륨에서 label과 정합.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: (로컬) 문법 확인**

Run: `py -3.10 -m py_compile scripts/verify_affine_target.py`
Expected: exit 0

- [ ] **Step 3: Commit**

```bash
git add scripts/verify_affine_target.py
git commit -m "feat(aug): verify_affine_target — 재생성 규약 label 정합·워프 시간 서버 검증"
```

---

### Task 4: [사람] 서버 검증 게이트 — pytest + verify + 판정

**Files:** 없음 (실행만. Claude는 아래 명령을 안내할 뿐 직접 실행하지 않는다 — pytest는 비GPU지만 torch가 서버에만 있다)

- [ ] **Step 1: 서버 분할 터미널에서 pytest**

```bash
python -m pytest scripts/tests/test_augment.py -v
```
Expected: 기존 12건 + 신규 8건 전부 PASS. 실패 시 결과를 공유하면 수정 라운드를 돈다.

- [ ] **Step 2: 재생성 규약 + 시간 검증**

```bash
python scripts/verify_affine_target.py --data /root/Data/train
```
판정:
- `[FAIL]`(corr<0.9999)이 하나라도 있으면 → **학습 금지**, `_rss_crop`의 crop/pad 규약을 label 생성 방식에 맞춰 수정해야 한다(수정 라운드).
- `_affine` median이 **200 ms/slice를 넘으면** num_workers=4 로더가 GPU를 굶길 위험 — 본학습 첫 epoch에서 iter 시간이 기존 대비 늘었는지 확인하고, 늘었으면 보고(대응: 워프를 코일 루프 대신 (C·2)채널 일괄 warpAffine로 바꾸는 후속 커밋).

- [ ] **Step 3: 채택 확인**

두 게이트가 green이면 완료 — **다음 본학습(shell31 계열 재실행)부터 affine이 자동 포함**된다(`--aug-on`이 이미 켜져 있으므로). 학습 로그 첫 epoch의 `[aug] epoch=0 p_flip=0.5000 p_affine=0.5000` 라인으로 실적용을 확인한다. 문제가 생기면 `--no-aug-affine`(구버전 추첨 스트림까지 동일) 또는 Task 1-3 revert로 되돌린다.

---

## 계획에서 뺀 것 (ponytail — 이유와 복귀 조건)

- **shear·translation·축소(scale<1)**: 사용자 제외 확정(2026-08-07). affine 행렬 구조라 필요해지면 범위 파라미터 추가만으로 복귀 가능.
- **bbox·GRAPPA 워프**: 현행 셸에서 둘 다 미사용(λ=0 → bbox는 None, GRAPPA 미배선) — fail-fast assert로 대체. `--bbox-loss-lambda`나 GRAPPA를 켤 계획이 생기면 그때 `_flip_lr`처럼 동기 변환을 구현.
- **affine 전용 스케줄(램프)**: 보간 대책을 스케줄이 아니라 LANCZOS4로 잡았다(±5°·≤1.1은 온건). 품질 우려가 실측되면 기존 `--aug-schedule exp`가 flip·affine에 함께 적용되므로 슬롯 인자만 바꾸면 된다 — 코드 복귀 불필요.
- **A/B 슬롯**: 사용자 지시로 생략(바로 채택). 효과 판정이 필요해지면 `--no-aug-affine` OFF arm으로 shell22 형식 A/B를 그때 구성.
- **maximum(attrs max) 재계산**: flip 선례대로 유지. 확대로 밝은 조직이 프레임 밖으로 나가 최대값이 줄 수 있으나 data_range가 보수적으로 커지는 방향이라 무해.
