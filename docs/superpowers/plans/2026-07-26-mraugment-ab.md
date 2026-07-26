# 증강 활성화 배선 + 증강 ON/OFF A/B 실행 플랜

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 죽어 있는 `MRIAugmentor`를 실제로 켤 수 있게 배선하고, 증강 ON/OFF를 동일 fold·seed에서 비교해 과적합 완화 여부를 판정한다.

**Architecture:** 증강 로직(`transforms.py`)은 그대로 두고, ① 스케줄 함수를 MRAugment 공식 형태(delay `D` · 총 epoch `T` 정규화)로 교체 ② 변환별 가중치를 분리 ③ `train.py` 인자와 `train_part.py`의 `set_epoch` 통로를 연결 ④ 매 epoch `p`를 로깅. 그 위에서 shell 슬롯 하나(≈21h)로 ON arm만 새로 학습하고, 과적합이 관측된 기존 런을 OFF arm으로 재사용한다.

**Tech Stack:** Python 3.10.12 · numpy · PyTorch 2.3.1+cu121 · pytest · argparse

**설계 문서:** [`docs/superpowers/specs/2026-07-26-mraugment-ab-design.md`](../specs/2026-07-26-mraugment-ab-design.md)

## Global Constraints

- **GPU를 쓰는 명령은 Claude가 절대 직접 실행하지 않는다** (`CLAUDE.md` §6). 슬롯 스크립트 작성까지만 하고 `bash shellN.sh` 실행은 사람이 한다. `pytest`·`git`은 GPU를 안 쓰므로 즉시 실행한다.
- **기본값은 전부 off** — `--aug-on` 없이 돌린 학습은 변경 전과 동일해야 한다.
- 커밋 메시지·PR에 **AI attribution을 넣지 않는다** (`CLAUDE.md` §9). 커밋 전 **작성자 identity를 사람에게 확인**한다.
- 증강은 **train 로더에만** 붙는다. val/forward에는 절대 붙지 않는다.
- `argparse`가 `ArgumentDefaultsHelpFormatter`를 쓰므로 **help 문자열의 `%`는 `%%`로 이스케이프**해야 한다(`train.py:43` 선례).
- 작업 브랜치는 현재 `k-fold`. 새 브랜치를 팔지는 첫 커밋 전에 사람에게 확인한다.
- 비목표(건드리지 않는다): phase/noise/라인 지터 활성화 · translation · TTA · consistency 손실 · GRAPPA prior · bbox 가중 손실 · 합성 k-space.

---

### Task 1: 기존 증강 동작을 회귀 테스트로 고정

증강 코드는 삭제된 노트북에서 1회 눈으로만 검증됐다. Task 2·3이 이 코드를 고치므로, **먼저 현재 동작을 테스트로 못 박는다.** 이 테스트들은 첫 실행에 **바로 통과한다** — TDD의 red 단계가 아니라 리팩터링 안전망(characterization test)이므로 의도된 것이다.

**Files:**
- Create: `tests/test_augment.py`

**Interfaces:**
- Consumes: `utils.data.transforms`의 `AugmentConfig`, `MRIAugmentor`, `DataTransform`, `_detect_acs`, `_fftc`, `_ifftc`
- Produces: 헬퍼 `make_mask(W, acs, acc)`, `make_kspace(coils, H, W, seed)` — Task 2·3의 테스트가 재사용한다.

- [ ] **Step 1: 테스트 파일 작성**

```python
"""증강(MRIAugmentor) 불변식 회귀 테스트. GPU·데이터 파일 불필요(합성 배열)."""
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
```

- [ ] **Step 2: 실행해서 전부 통과 확인 (안전망 확보)**

Run: `python -m pytest tests/test_augment.py -v`
Expected: 6 passed. **하나라도 실패하면 Task 2로 넘어가지 말고 원인을 먼저 규명한다** — 기존 증강 구현 자체에 버그가 있다는 뜻이다.

- [ ] **Step 3: 커밋**

```bash
git add tests/test_augment.py
git commit -m "test(aug): MRIAugmentor 불변식 회귀 테스트 추가 (flip 정확성·ACS 보존·bbox 동기·결정성·off 기본값)"
```

---

### Task 2: 스케줄을 MRAugment 공식 형태로 교체

현재 `_prob()`는 `warmup_epochs=10` 미만이면 0을 반환하고 램프가 `ramp_epochs=15`(25ep 런 가정)에 고정돼 있다. 실제 런은 `-e 10`(epoch 0~9)이라 **전 구간 `p=0`** → 증강이 0회 발생한다. 공식 repo(`MathFLDS/MRAugment`, `mraugment/data_augment.py`)의 `schedule_p()`와 동일하게 **총 epoch `T`로 정규화**한다.

**Files:**
- Modify: `utils/data/transforms.py:178-200` (`AugmentConfig` 필드), `utils/data/transforms.py:215-225` (`_prob`)
- Modify: `tests/test_augment.py` (테스트 추가)

**Interfaces:**
- Consumes: Task 1의 헬퍼
- Produces: `AugmentConfig(delay, total_epochs, exp_decay, schedule, max_prob, ...)` · `MRIAugmentor._prob() -> float` · `MRIAugmentor.probs() -> (p, p_flip, p_mask)` (Task 3에서 `p_flip`/`p_mask`를 채우고 Task 5가 로깅에 쓴다)

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_augment.py` 맨 아래에 추가

```python
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
    assert 0.5 < ps[-1] < 0.55                   # 마지막 epoch에 p_max 근처(상류와 동일하게 미달)


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
    r.set_epoch(6); assert r._prob() == pytest.approx(0.55 * 4 / 8)

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
```

- [ ] **Step 2: 실행해서 실패 확인**

Run: `python -m pytest tests/test_augment.py -v`
Expected: 새 테스트 4개 FAIL(기존 6개는 통과). `AugmentConfig`가 `delay`/`total_epochs`/`exp_decay`를 무시하고(`**kw`를 `get`으로만 읽음) `warmup_epochs=10`이 남아 있어 `_prob()`가 계속 0을 반환한다.

- [ ] **Step 3: `AugmentConfig` 필드 교체** — `utils/data/transforms.py:183-187`

```python
        # 스케줄 — MRAugment 공식 repo(mraugment/data_augment.py: schedule_p)와 동일 형태.
        #   p(t<D)=0, p(T)=p_max. T로 정규화하므로 런 길이가 바뀌어도 증강량이 스케일된다.
        self.schedule = kw.get("schedule", "exp")          # 'exp' | 'ramp' | 'constant'
        self.delay = kw.get("delay", 0)                    # D: 증강 없이 보내는 앞쪽 epoch 수
        self.total_epochs = kw.get("total_epochs", 1)      # T: args.num_epochs (load_data가 채움)
        self.exp_decay = kw.get("exp_decay", 5.0)          # 논문/공식 repo 기본 c=5.0
        self.max_prob = kw.get("max_prob", 0.55)           # 논문 기본 p_max=0.55
```

- [ ] **Step 4: `_prob()` 교체** — `utils/data/transforms.py:215-225`

```python
    def _prob(self):
        """MRAugment 공식 schedule_p와 동일: D=delay, T=total_epochs, c=exp_decay/(T-D).
        p(t<D)=0이고 p(T)=p_max로 정규화되므로 런 길이가 달라도 증강량이 일관된다.
        (epoch은 0..T-1이라 마지막 epoch의 p는 p_max에 살짝 못 미친다 — 상류와 동일.)"""
        c = self.cfg
        if not c.on:
            return 0.0
        D, T, t, p_max = int(c.delay), int(c.total_epochs), self.epoch, float(c.max_prob)
        if t < D:
            return 0.0
        if c.schedule == "constant":
            return p_max
        if T - D <= 0:                       # 가드: 1 epoch 스모크 런 등
            return p_max
        if c.schedule == "ramp":
            return p_max * (t - D) / (T - D)
        k = c.exp_decay / (T - D)
        return float(p_max / (1.0 - np.exp(-(T - D) * k)) * (1.0 - np.exp(-(t - D) * k)))
```

- [ ] **Step 5: 실행해서 통과 확인**

Run: `python -m pytest tests/test_augment.py -v`
Expected: 10 passed (Task 1의 6개 + 새 4개). Task 1 테스트가 깨지면 필드 교체가 기존 동작을 건드린 것이므로 되돌린다.

- [ ] **Step 6: 커밋**

```bash
git add utils/data/transforms.py tests/test_augment.py
git commit -m "fix(aug): 증강 확률 스케줄을 총 epoch 정규화 형태로 교체 (10ep 런에서 p가 끝까지 0이던 문제)"
```

---

### Task 3: 변환별 가중치 분리 + 중복 draw 정리

지금은 flip·mask가 같은 `p`를 공유하고, mask만 `rng.random()`을 2회 뽑아 실효 확률이 `p·(1-orig_mask_prob)`다. 공식 repo의 `--aug_weight_*`(= `p_i = p·w_i`)에 대응하는 가중치를 넣고 draw를 하나로 합친다. **확률값은 기존과 동일**(순수 정리 + 가중치 추가)하며, 증강이 실행된 적이 없어 RNG 소비 변화로 인한 재현성 부채는 없다.

**Files:**
- Modify: `utils/data/transforms.py:178-200` (`AugmentConfig`에 가중치 2개), `utils/data/transforms.py:267-303` (`__call__`)
- Modify: `tests/test_augment.py`

**Interfaces:**
- Consumes: Task 2의 `_prob()`
- Produces: `MRIAugmentor.probs() -> (p, p_flip, p_mask)` — Task 5가 로깅에 쓴다.

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_augment.py` 맨 아래에 추가

```python
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
    """weight_flip=0이면 flip은 안 걸리고 mask만 걸린다(분리 A/B의 근거)."""
    aug = MRIAugmentor(AugmentConfig(on=True, schedule="constant", max_prob=1.0,
                                     flip=True, mask=True, weight_flip=0.0,
                                     weight_mask=1.0, orig_mask_prob=0.0))
    M, T = make_mask(), make_target()
    _, M2, T2, _, _ = aug(make_kspace(), M, T, None, None, "knee_acc4_1.h5", 0)
    assert np.array_equal(T2, T), "weight_flip=0인데 flip이 걸렸다"
    assert M2.sum() < M.sum(), "mask 증강이 걸리지 않았다"
```

- [ ] **Step 2: 실행해서 실패 확인**

Run: `python -m pytest tests/test_augment.py -v`
Expected: 새 3개 FAIL — `AttributeError: 'MRIAugmentor' object has no attribute 'probs'`.

- [ ] **Step 3: `AugmentConfig`에 가중치 추가** — `self.flip`/`self.mask` 정의 바로 아래

```python
        # 변환별 가중치(공식 repo --aug_weight_*): 실제 확률 = p · w_i. 한쪽만 0으로 두면 분리 A/B가 된다.
        self.weight_flip = kw.get("weight_flip", 1.0)
        self.weight_mask = kw.get("weight_mask", 1.0)
```

- [ ] **Step 4: `probs()` 추가 + `__call__` 정리**

`set_epoch` 아래에 추가:

```python
    def probs(self):
        """(p, p_flip, p_mask). 학습 로그와 __call__이 같은 값을 쓰도록 한 곳에서 계산."""
        c = self.cfg
        p = self._prob()
        p_flip = min(1.0, max(0.0, p * float(c.weight_flip)))
        p_mask = min(1.0, max(0.0, p * float(c.weight_mask) * (1.0 - float(c.orig_mask_prob))))
        return p, p_flip, p_mask
```

`__call__`에서 `p = self._prob()` 이후 mask·flip 블록을 다음으로 교체:

```python
        p, p_flip, p_mask = self.probs()
        if p <= 0.0:
            return K, M, target, bbox, grappa
        rng = self._rng(fname, sl)
        tgt = target

        # 1) mask augmentation — 원본(고정)마스크 혼합을 p_mask에 이미 반영(test 분포 정합). GRAPPA엔 미적용.
        if cfg.mask and (rng.random() < p_mask):
            M = self._aug_mask(rng, M)
            K = kspace_c * M

        # 2) 좌-우(width) flip — target·bbox·mask·GRAPPA 동기(정렬 유지)
        if cfg.flip and (rng.random() < p_flip):
            K, tgt, M, bbox, grappa = self._flip_lr(K, tgt, bbox, M, grappa)
```

기존 `p = self._prob()` 한 줄과 그 아래 `if p <= 0.0` 블록은 위 코드로 대체되므로 중복을 남기지 않는다. phase/noise 블록(3·4번)은 `p`를 그대로 쓰므로 손대지 않는다.

- [ ] **Step 5: 실행해서 통과 확인**

Run: `python -m pytest tests/test_augment.py -v`
Expected: 13 passed. Task 1의 `test_flip_is_pixel_exact...`·`test_mask_aug_preserves_acs...`가 여전히 통과해야 한다(가중치 기본 1.0 + `orig_mask_prob` 명시 설정이라 확률이 그대로).

- [ ] **Step 6: 커밋**

```bash
git add utils/data/transforms.py tests/test_augment.py
git commit -m "refactor(aug): 변환별 가중치(p_i = p·w_i) 도입 + mask 중복 draw 정리"
```

---

### Task 4: `train.py` 증강 인자 + `load_data.py` 전달

**Files:**
- Modify: `train.py:62` 뒤(`--init-from` 다음)에 인자 블록 추가
- Modify: `utils/data/load_data.py:194-208` (`AugmentConfig(...)` 인자)

**Interfaces:**
- Consumes: Task 2·3의 `AugmentConfig` 필드명(`delay`/`total_epochs`/`exp_decay`/`weight_flip`/`weight_mask`)
- Produces: CLI 플래그 `--aug-on --aug-schedule --aug-max-prob --aug-delay --aug-exp-decay --aug-flip/--no-aug-flip --aug-mask/--no-aug-mask --aug-weight-flip --aug-weight-mask --aug-mask-thin-max --aug-orig-mask-prob` — Task 6의 shell 슬롯이 쓴다.

- [ ] **Step 1: `train.py`에 인자 추가**

```python
    # ---- 증강 (MRAugment 부분집합: 좌우 flip + ACS 보존 mask thinning). 기본 off = 기존 학습과 동일 ----
    #   스케줄은 공식 repo(MathFLDS/MRAugment) schedule_p와 동일 형태 — delay D + 총 epoch T 정규화.
    parser.add_argument('--aug-on', action='store_true', help='데이터 증강 켜기. 기본 off면 기존 학습과 100%% 동일(비트 동일).')
    parser.add_argument('--aug-schedule', type=str, default='exp', choices=['exp', 'ramp', 'constant'],
                        help='증강 확률 스케줄. exp=지수 점증(공식 기본), ramp=선형, constant=고정')
    parser.add_argument('--aug-max-prob', type=float, default=0.55, help='p_max — 마지막 epoch에서의 증강 확률(논문 기본 0.55)')
    parser.add_argument('--aug-delay', type=int, default=0, help='앞쪽 몇 epoch을 증강 없이 보낼지(D). 과적합이 시작되는 epoch 직전에 두는 것이 목적')
    parser.add_argument('--aug-exp-decay', type=float, default=5.0, help='--aug-schedule exp의 감쇠 계수 c (공식 기본 5.0)')
    parser.add_argument('--aug-flip', action=argparse.BooleanOptionalAction, default=True,
                        help='좌-우(phase-encode) 플립. 끄려면 --no-aug-flip')
    parser.add_argument('--aug-mask', action=argparse.BooleanOptionalAction, default=True,
                        help='ACS 보존 마스크 추가 언더샘플. 끄려면 --no-aug-mask')
    parser.add_argument('--aug-weight-flip', type=float, default=1.0, help='flip 확률 가중치 (실제 확률 = p × 이 값). 0이면 flip만 끔')
    parser.add_argument('--aug-weight-mask', type=float, default=1.0, help='mask 증강 확률 가중치 (실제 확률 = p × 이 값 × (1-orig-mask-prob))')
    parser.add_argument('--aug-mask-thin-max', type=int, default=4, help='주변부 추가 언더샘플 최대 배수(ACS는 항상 보존)')
    parser.add_argument('--aug-orig-mask-prob', type=float, default=0.5,
                        help='원본(제공 고정) 마스크를 그대로 쓸 확률 — test 분포 정합·OOD 방지용')
```

- [ ] **Step 2: `load_data.py`의 `AugmentConfig(...)` 인자 교체** — `warmup_epochs`·`ramp_epochs` 두 줄을 지우고 대체

```python
    cfg = AugmentConfig(
        on=aug_on,
        schedule=getattr(args, "aug_schedule", "exp"),
        delay=getattr(args, "aug_delay", 0),
        total_epochs=int(getattr(args, "num_epochs", 1)),   # T = 런 길이 → 스케줄이 자동 스케일
        exp_decay=getattr(args, "aug_exp_decay", 5.0),
        max_prob=getattr(args, "aug_max_prob", 0.55),
        flip=getattr(args, "aug_flip", True),
        mask=getattr(args, "aug_mask", True),
        weight_flip=getattr(args, "aug_weight_flip", 1.0),
        weight_mask=getattr(args, "aug_weight_mask", 1.0),
        phase=getattr(args, "aug_phase", False),
        noise=getattr(args, "aug_noise", False),
        mask_thin_max=getattr(args, "aug_mask_thin_max", 4),
        orig_mask_prob=getattr(args, "aug_orig_mask_prob", 0.5),
        noise_sigma=getattr(args, "aug_noise_sigma", 0.0),
        seed=getattr(args, "seed", 430),
    )
```

- [ ] **Step 3: 인자가 파싱되고 config에 도달하는지 확인 (GPU 미사용)**

Run:
```bash
python -c "
import sys; sys.argv = ['x','--aug-on','--aug-delay','2','--no-aug-flip','-e','10']
from train import parse
from utils.data.transforms import AugmentConfig, MRIAugmentor
a = parse()
cfg = AugmentConfig(on=a.aug_on, schedule=a.aug_schedule, delay=a.aug_delay,
                    total_epochs=a.num_epochs, exp_decay=a.aug_exp_decay,
                    max_prob=a.aug_max_prob, flip=a.aug_flip, mask=a.aug_mask)
m = MRIAugmentor(cfg); m.set_epoch(9)
print('flip =', cfg.flip, '| p at ep9 =', m.probs())
"
```
Expected: `flip = False | p at ep9 = (0.546..., 0.0, 0.273...)` — `--no-aug-flip`이 먹고 `p`가 0이 아니다.

- [ ] **Step 4: 기존 테스트 회귀 확인**

Run: `python -m pytest tests/test_augment.py -v`
Expected: 13 passed.

- [ ] **Step 5: 커밋**

```bash
git add train.py utils/data/load_data.py
git commit -m "feat(aug): train.py --aug-* 인자 신설 + load_data에 스케줄 파라미터 전달"
```

---

### Task 5: epoch 전달(`set_epoch`) + `p` 로깅

`transforms.py`는 "지금 몇 epoch인지" 모른다. 이 통로가 없으면 스케줄이 영원히 epoch 0이다. 그리고 이번 사안의 근본 원인이 **무증상 실패**였으므로 매 epoch `p`를 찍어 눈에 보이게 한다.

`persistent_workers`를 쓰지 않으므로(`load_data.py:260-268`) DataLoader가 epoch마다 워커를 새로 띄운다 → 메인 프로세스 dataset에 심은 epoch이 워커 복사본에 전달된다.

**Files:**
- Modify: `utils/learning/train_part.py:389-391`

**Interfaces:**
- Consumes: `DataTransform.set_epoch(epoch)`(기존), Task 3의 `MRIAugmentor.probs()`
- Produces: 학습 로그의 `[aug] epoch=… p=… p_flip=… p_mask=…` 줄 — Task 6의 스모크 게이트가 이 줄을 확인한다.

- [ ] **Step 1: 루프 첫 줄에 배선 추가** — `print(f'Epoch #...')` 바로 다음

```python
        # 증강 스케줄에 epoch 전달(implementation.md §6b). val 로더는 is_train=False라 augmentor가 없다.
        # persistent_workers를 안 쓰므로 epoch마다 워커가 새로 떠 이 값이 워커 복사본에 실린다.
        tf = getattr(train_loader.dataset, "transform", None)
        if tf is not None:
            tf.set_epoch(epoch)
            augmentor = getattr(tf, "augmentor", None)
            if augmentor is not None:
                p, p_flip, p_mask = augmentor.probs()
                print(f"[aug] epoch={epoch} p={p:.4f} p_flip={p_flip:.4f} p_mask={p_mask:.4f}")
```

- [ ] **Step 2: 배선 테스트 추가** — `tests/test_augment.py` 맨 아래

```python
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
```

- [ ] **Step 3: 실행해서 통과 확인**

Run: `python -m pytest tests/test_augment.py -v`
Expected: 14 passed.

- [ ] **Step 4: 증강 off일 때 로그·동작이 그대로인지 확인 (GPU 미사용)**

Run: `python -m pytest tests/test_augment.py -k "off_by_default or val_transform" -v`
Expected: 2 passed. `--aug-on`이 없으면 `augmentor is None`이라 `[aug]` 줄도 안 찍힌다.

- [ ] **Step 5: 커밋**

```bash
git add utils/learning/train_part.py tests/test_augment.py
git commit -m "feat(aug): 매 epoch set_epoch 배선 + 증강 확률 p 로깅"
```

---

### Task 6: baseline 대조 + ON arm 슬롯 작성

**Files:**
- Create: `shell_scripting/shell22.sh`
- Read: 과적합이 관측된 런의 `val_loss_log*.npy`, 해당 shell 슬롯

**Interfaces:**
- Consumes: Task 4의 CLI 플래그, Task 5의 `[aug]` 로그
- Produces: `shell22.sh`(사람이 실행) · `--aug-delay` 확정값 · baseline 재사용 가능 여부 판정

- [ ] **Step 1: 과적합 런 식별 + config 대조**

사람에게 **어느 런인지(net-name/fold)** 확인한 뒤, 그 런의 shell 슬롯과 현재 HEAD의 config를 비교한다:

```bash
git log --oneline -20 -- shell_scripting/
grep -n "train.py" shell_scripting/shell*.sh
```

판정 기준: **fold · kfold-seed · seed · num-cascades · n-feat0 · sens-n-feat0 · n-history · lr · fg-loss · area-weight · grad-clip · accum-steps · coil-compress · readout-crop**이 전부 같으면 그 런을 OFF arm으로 **재사용한다**(GPU 21h 절약). 하나라도 다르면 baseline도 재학습해야 하므로 **사람에게 알리고 예산 승인을 받는다**(21h → 42h).

- [ ] **Step 2: `--aug-delay` 값 결정**

val 로그에서 val이 꺾인 epoch을 읽는다(GPU 미사용):

```bash
python -c "
import numpy as np, sys
log = np.load(sys.argv[1])           # (N,2): [epoch, val_loss]
for e, v in log: print(int(e), f'{v:.6f}')
best = int(log[np.argmin(log[:,1]), 0])
print('best epoch =', best, '→ --aug-delay =', max(0, best - 1))
" ../result/<net_name>/val_loss_log_<tag>.npy
```

`--aug-delay = max(0, best_epoch - 1)`로 정한다(증강 확률이 과적합 구간을 덮게 하는 것이 목적). 값을 Step 3의 스크립트에 박고, 근거를 스크립트 주석에 남긴다.

- [ ] **Step 3: `shell22.sh` 작성**

[`shell21.sh`](../../../shell_scripting/shell21.sh)를 복제해 헤더 주석을 이 실험 목적으로 바꾸고, **train.py 호출에 증강 인자만 추가**한다. `FOLD`·`K_FOLD`·`KFOLD_SEED`·나머지 config는 baseline과 **완전 동일**하게 유지한다(변경점 1개 원칙).

```bash
  --aug-on \
  --aug-schedule exp \
  --aug-max-prob 0.55 \
  --aug-delay <Step 2에서 결정한 값> \
  --aug-exp-decay 5.0 \
  --aug-mask-thin-max 4 \
  --aug-orig-mask-prob 0.5 \
```

헤더 주석에 반드시 적을 것: **변경점은 증강 인자뿐 · baseline은 `<net_name>` 런 · 판정은 fg-val 서명 3개 · `SMOKE=1` 먼저**.

- [ ] **Step 4: CPU 게이트 통과 확인 (Claude가 실행)**

Run: `python -m pytest tests/test_augment.py -v`
Expected: 14 passed. 여기서 실패하면 슬롯을 사람에게 넘기지 않는다.

- [ ] **Step 5: 커밋**

```bash
git add shell_scripting/shell22.sh
git commit -m "chore(queue): shell22 — 증강 ON arm 슬롯 (fold N, baseline 대비 증강 인자만 추가)"
```

- [ ] **Step 6: 사람에게 스모크 실행 인계** (⚠️ Claude가 실행하지 않는다)

전달할 내용:
1. `SMOKE=1 bash shell_scripting/shell22.sh` (120 iter)
2. 확인할 것 세 가지 — ① exit 0 · OOM 없음 ② 로그에 `[aug] … p=` 가 찍히고 **`p > 0`** ③ **iter time을 baseline 로그와 비교** (증강은 샘플당 FFT 2회(15×640×368)를 dataloader에 추가한다. `--num-workers 4`에서 CPU 병목이면 21h가 늘어난다 — 증가폭이 크면 `--num-workers` 상향 후 재측정)
3. ②가 실패하면 **본 학습을 시작하지 않는다**(증강 0회로 21시간을 태우는 것을 막는 게 이 게이트의 유일한 목적)
4. 통과하면 `bash shell_scripting/shell22.sh` 본 학습(≈21h)

---

### Task 7: 판정 리포트 + 문서 정정

**Files:**
- Create: `training_anal/2026-<완주일>-aug-analysis.md`
- Modify: `training_anal/2026-07-13-run10-longtrain-analysis.md`(정정 배너), `strategy/implementation.md`(§6 완료 처리), `strategy/augmentation.md`(구현 상태 갱신), `CLAUDE.md`(§10에서 항목 제거)

**Interfaces:**
- Consumes: Task 6 ON arm의 `val_loss_log`, baseline의 `val_loss_log`

- [ ] **Step 1: 서명 3개를 표로 정리**

`training_anal` 관례(직전 대비 diff 표)를 따르고, 다음 3개를 baseline과 나란히 적는다: **① best-epoch val ② val이 꺾이는 epoch ③ best epoch index.** ①이 없고 ②③만 있으면 "과적합은 늦췄으나 점수 이득 없음"으로 기록한다.

- [ ] **Step 2: `recon_eval` 2차 판정 (채점 가능한 환경에서, 사람이 실행)**

k-fold 서버엔 leaderboard 데이터가 없다(shell21 주석). 채점 가능한 환경에서 `acc4/acc8 × full/bbox` 4숫자를 받아 **bbox 비퇴행을 하드 게이트**로 판정한다.

- [ ] **Step 3: 설계 문서 §B4 표에 따라 승/패를 한 줄로 판정**

| 결과 | 다음 수 |
|---|---|
| bbox 유의 하락 | 즉시 탈락·롤백 |
| acc8만 상승, acc4 하락 | `--aug-mask-thin-max 2`로 1회 재시도 |
| 전체 패배 | `--aug-max-prob 0.3`으로 1회 재시도. 그 이상은 안 한다 |
| 승리 | flip·mask 분리 A/B(`--aug-weight-flip 0` / `--aug-weight-mask 0`) → 이후 epoch 연장 |

- [ ] **Step 4: 문서 정정**

- `training_anal/2026-07-13-run10-longtrain-analysis.md` 상단에 배너: **"이 결론('오래 = 좋음')은 작은 VarNet [16,32]×4 기준이다. 현재 백본(PromptMR+ nc10/nf48)은 10ep 내 과적합이 관측됐으므로 전이되지 않는다."**
- `strategy/implementation.md` §6a·6b를 완료 처리(구현 위치·커밋 해시 명시).
- `strategy/augmentation.md`의 구현 상태 배너를 "배선 완료 + A/B 결과" 기준으로 갱신.
- `CLAUDE.md` §10에서 "증강 활성화 배선(`--aug-*` + `set_epoch`)" 항목 제거.

- [ ] **Step 5: 커밋**

```bash
git add training_anal/ strategy/ CLAUDE.md docs/superpowers/
git commit -m "docs(aug): 증강 ON/OFF A/B 결과 분석 + 배선 완료 반영, run10 결론에 백본 정정 배너"
```

---

## 실행 순서 요약

Task 1~5는 GPU 없이 로컬/서버 어디서든 끝난다(`pytest`만 필요). Task 6 Step 6부터 GPU가 필요하고 **사람이 실행**한다. Task 7은 ON arm 완주 후.

GPU 예산: baseline 재사용 가능 시 **≈21h**(ON arm 1개), 재사용 불가 시 ≈42h.
