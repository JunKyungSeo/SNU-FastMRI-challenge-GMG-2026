# 변환별 최대확률 분리(--flip-max-prob / --affine-max-prob) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `--aug-max-prob` 하나가 flip·affine 공통이던 것을 `--flip-max-prob`/`--affine-max-prob`로 분리해 비율을 다르게 줄 수 있게 한다 (사용자 요청 2026-08-08). **기본값 None = `--aug-max-prob`로 fallback** — 기존 셸 인자와 100% 호환.

**Architecture:** 스케줄 곡선(exp/ramp/constant·delay)은 공유하고 도달점만 변환별로 스케일한다: `prob()`에 `p_max=None` 파라미터를 추가해 `prob(cfg.flip_max_prob)`/`prob(cfg.affine_max_prob)`로 같은 곡선을 각자의 최대값으로 계산. fallback 해석은 AugmentConfig 생성 시 1회 확정(None→max_prob, **0.0은 "그 변환만 확률 0"으로 유효값** — `or` 금지, None 검사 필수). 추첨 횟수는 불변(값 비교만 바뀜)이라 RNG 스트림 구조·재현성 영향 없음. `check_resume_compat`의 일치 필드에 증강 인자가 원래 없음을 확인했으므로(train_part.py:329-332) resume 영향도 없다.

**Tech Stack:** 기존 numpy·pytest 구조 그대로. 신규 의존성 없음.

## Global Constraints

- **로컬 torch 없음** — pytest는 서버 게이트[사람], 로컬은 `py -3.10 -m py_compile`까지 (선례 유지).
- 기존 셸(`_aug_shared.sh`/`_aug_acc4.sh` 등)은 **무수정** — fallback으로 지금과 동일하게 동작해야 한다(둘 다 0.5). 비율 변경은 사용자가 나중에 셸 인자로 결정.
- `--no-aug-affine`(구조적 off, rng 미소비)은 그대로 유지 — `--affine-max-prob 0`(추첨은 소비, 발동 0)과 용도가 다르다.
- 기존 테스트 무수정 통과: 스케줄 테스트들은 `aug.prob()`(무인자)를 쓰므로 기본 동작(=cfg.max_prob)이 유지되어야 한다.
- 커밋: kindofwater / bedso@snu.ac.kr (확인됨). **AI attribution 금지.**

## File Structure

- Modify `utils/data/transforms.py` — AugmentConfig(2필드), `prob()` 시그니처, `__call__` 게이트 2개
- Modify `train.py` — 인자 2개 (`--aug-scale-max` 다음)
- Modify `utils/data/load_data.py` — AugmentConfig 전달 2필드
- Modify `utils/learning/train_part.py` — `[aug]` 로그가 변환별 실제 값 출력
- Modify `scripts/tests/test_augment.py` — 신규 테스트 3건

---

### Task 1: 확률 분리 — 전체 배선 + 테스트 (단일 태스크)

**Files:**
- Modify: `utils/data/transforms.py`, `train.py`, `utils/data/load_data.py`, `utils/learning/train_part.py`
- Test: `scripts/tests/test_augment.py`

**Interfaces:**
- Consumes: `AugmentConfig(max_prob, affine, ...)`, `MRIAugmentor.prob()`, `_flip_lr`, 테스트 헬퍼 `affine_cfg`/`make_phantom_kspace`/`make_mask`/`make_target` (전부 기존)
- Produces: `AugmentConfig.flip_max_prob`/`.affine_max_prob` (resolved float), `MRIAugmentor.prob(p_max=None)`, CLI `--flip-max-prob`/`--affine-max-prob` (dest: `flip_max_prob`/`affine_max_prob`)

- [ ] **Step 1: 실패하는 테스트 작성** — `scripts/tests/test_augment.py` 끝에 추가:

```python
# ---------------------------------------------------------------------------
# 변환별 최대확률 분리(--flip-max-prob/--affine-max-prob) — 2026-08-08
# ---------------------------------------------------------------------------

def test_split_max_prob_falls_back_to_shared():
    """미지정(None)이면 기존 --aug-max-prob와 동일 — 기존 셸 인자 100% 호환 계약."""
    cfg = affine_cfg(max_prob=0.5)
    aug = MRIAugmentor(cfg)
    assert cfg.flip_max_prob == 0.5 and cfg.affine_max_prob == 0.5
    assert aug.prob(cfg.flip_max_prob) == 0.5
    assert aug.prob() == 0.5                     # 무인자 = 기존 동작(스케줄 테스트 보호)


def test_split_max_prob_gates_independently():
    """flip 1.0/affine 0.0 → flip만, flip 0.0/affine 1.0 → affine만 발동해야 한다.
    (0.0은 유효값 — None fallback과 혼동해 max_prob로 되살아나면 이 테스트가 잡는다)"""
    K, _ = make_phantom_kspace()
    M = make_mask(W=368)
    T = make_target()
    Mb = (M >= 1).astype(M.dtype)

    flip_only = MRIAugmentor(affine_cfg(flip_max_prob=1.0, affine_max_prob=0.0))
    out = flip_only(K, M, T, None, None, "p.h5", 0)
    exp = flip_only._flip_lr(K * Mb, T, None, Mb)[1]     # affine 미발동이라 K 원본 기준
    assert np.array_equal(out[2], np.ascontiguousarray(exp))

    aff_only = MRIAugmentor(affine_cfg(flip_max_prob=0.0, affine_max_prob=1.0))
    out2 = aff_only(K, M, T, None, None, "p.h5", 0)
    assert np.array_equal(np.asarray(out2[1]), M)        # flip 미발동 — 마스크 원본 그대로
    assert out2[2].shape == (384, 384) and not np.array_equal(out2[2], T)  # affine은 발동(재생성)


def test_split_max_prob_scales_schedule_shape():
    """스케줄 곡선은 공유, 도달점만 다름 — 두 확률의 비가 epoch 내내 일정해야 한다."""
    cfg = affine_cfg(schedule="exp", delay=2, total_epochs=10,
                     flip_max_prob=0.4, affine_max_prob=0.1)
    aug = MRIAugmentor(cfg)
    for e in range(3, 10):
        aug.set_epoch(e)
        pf, pa = aug.prob(cfg.flip_max_prob), aug.prob(cfg.affine_max_prob)
        assert pf > 0.0
        assert pa == pytest.approx(pf * 0.25, rel=1e-9)
```

- [ ] **Step 2: (로컬) 문법 확인** — RED는 서버 게이트로 이월

Run: `py -3.10 -m py_compile scripts/tests/test_augment.py`
Expected: exit 0. (서버 pytest 기준 기대 RED: `AugmentConfig`에 `flip_max_prob` 속성 부재 AttributeError / `prob()` takes 1 positional argument TypeError 계열 3건 FAIL)

- [ ] **Step 3: transforms.py 수정 — 세 군데**

(a) `AugmentConfig.__init__`의 `self.scale_max = ...` 줄 **다음**에 추가:

```python
        # 변환별 최대확률(2026-08-08) — None이면 max_prob로 fallback(기존 셸 인자 100% 호환).
        # 0.0은 "그 변환만 확률 0"인 유효값이므로 `or` 대신 None 검사로 해석한다.
        v = kw.get("flip_max_prob", None)
        self.flip_max_prob = float(self.max_prob if v is None else v)
        v = kw.get("affine_max_prob", None)
        self.affine_max_prob = float(self.max_prob if v is None else v)
```

(b) `prob()` 메서드 전체를 다음으로 교체 (기존 본문에서 p_max 결정부만 파라미터화 — 나머지 로직·주석 동일):

```python
    def prob(self, p_max=None):
        """이번 epoch에서 증강이 걸릴 확률. 학습 로그와 __call__이 같은 값을 쓰도록 한 곳에서 계산.
        MRAugment 공식 schedule_p와 동일: D=delay, T=total_epochs, c=exp_decay/(T-D).
        p(t<D)=0이고 p(T)=p_max로 정규화되므로 런 길이가 달라도 증강량이 일관된다.
        (epoch은 0..T-1이라 마지막 epoch의 p는 p_max에 살짝 못 미친다 — 상류와 동일.)
        p_max 인자(2026-08-08): 변환별 도달점(cfg.flip_max_prob/affine_max_prob)으로 같은
        곡선을 스케일할 때 쓴다. None이면 cfg.max_prob(기존 동작 — 스케줄 테스트가 이 계약을 쓴다)."""
        c = self.cfg
        if not c.on:
            return 0.0
        D, T, t = int(c.delay), int(c.total_epochs), self.epoch
        p_max = float(c.max_prob if p_max is None else p_max)
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

(c) `__call__`에서 확률 계산·게이트 부분 교체. 기존:

```python
        p = self.prob()
        if p <= 0.0:
            return kspace_c * M, M, target, bbox, grappa
```

교체:

```python
        p_flip = self.prob(self.cfg.flip_max_prob)
        p_affine = self.prob(self.cfg.affine_max_prob) if self.cfg.affine else 0.0
        if p_flip <= 0.0 and p_affine <= 0.0:
            return kspace_c * M, M, target, bbox, grappa
```

그리고 affine 게이트의 `(rng.random() < p)`를 `(rng.random() < p_affine)`으로, flip 게이트의 `if rng.random() < p:`를 `if rng.random() < p_flip:`으로 바꾼다 (각 1곳, 주석은 그대로).

- [ ] **Step 4: train.py — `--aug-scale-max` 인자 다음에 추가**

```python
    parser.add_argument('--flip-max-prob', type=float, default=None,
                        help='flip 전용 p_max. 미지정이면 --aug-max-prob를 따른다')
    parser.add_argument('--affine-max-prob', type=float, default=None,
                        help='affine 전용 p_max. 미지정이면 --aug-max-prob를 따른다. 0이면 affine만 확률 0')
```

- [ ] **Step 5: load_data.py — AugmentConfig 호출에서 `scale_max=` 줄 다음에 추가**

```python
        flip_max_prob=getattr(args, "flip_max_prob", None),
        affine_max_prob=getattr(args, "affine_max_prob", None),
```

- [ ] **Step 6: train_part.py — `[aug]` 로그 블록 교체.** 기존:

```python
            p = augmentor.prob()
            p_aff = p if augmentor.cfg.affine else 0.0
            print(f"[aug] epoch={epoch} p_flip={p:.4f} p_affine={p_aff:.4f}")
```

교체:

```python
            p_flip = augmentor.prob(augmentor.cfg.flip_max_prob)
            p_aff = augmentor.prob(augmentor.cfg.affine_max_prob) if augmentor.cfg.affine else 0.0
            print(f"[aug] epoch={epoch} p_flip={p_flip:.4f} p_affine={p_aff:.4f}")
```

- [ ] **Step 7: (로컬) 문법 확인**

Run: `py -3.10 -m py_compile utils/data/transforms.py train.py utils/data/load_data.py utils/learning/train_part.py scripts/tests/test_augment.py`
Expected: exit 0

- [ ] **Step 8: Commit**

```bash
git add utils/data/transforms.py train.py utils/data/load_data.py utils/learning/train_part.py scripts/tests/test_augment.py
git commit -m "feat(aug): flip/affine 최대확률 분리 — --flip-max-prob/--affine-max-prob (기본 --aug-max-prob fallback)"
```

---

### Task 2: [사람] 서버 검증 게이트

- [ ] 서버 분할 터미널: `python -m pytest scripts/tests/test_augment.py scripts/tests/test_remask.py -v` → 전부 PASS (신규 3건 포함)
- [ ] 비율을 실제로 바꿀 때: `_aug_<단계>.sh`의 `AUG_ARGS`에 예: `--flip-max-prob 0.5 --affine-max-prob 0.3` 추가. 학습 로그 `[aug] epoch=0 p_flip=0.5000 p_affine=0.3000`으로 확인.

## 계획에서 뺀 것 (ponytail)

- **스케줄 모양까지 변환별 분리**: 요청은 비율뿐 — 곡선 분리는 필요해지면 schedule 인자를 같은 방식으로 복제.
- **`--aug-max-prob` 제거/개명**: 기존 셸·재현 문서가 참조 — fallback 소스로 유지.
- **resume 가드에 증강 인자 추가**: 기존에도 증강 인자는 가드 밖(팀 선례) — 이번 범위 아님.
