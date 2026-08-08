# affine 후반 상승 스케줄(코사인 램프) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 전문가 단계에서 affine 확률을 계단(0→0.3)이 아니라 **후반에 코사인으로 0→0.5까지 올린 뒤 유지**하도록 한다. flip은 0.5 고정 유지(사용자 확정 2026-08-09).

**Architecture:** `prob()`는 **한 글자도 건드리지 않는다**(기존 스케줄 계약·테스트 보호). 대신 affine 전용 `affine_prob()`를 새로 두고, `--affine-peak-epoch`가 지정됐을 때만 코사인 램프(`p = p_max·(1−cos(πu))/2`, u=(t−D)/(P−D))로 계산하고 P 이후 p_max를 유지한다. 미지정이면 지금처럼 flip과 같은 곡선을 탄다(하위호환). flip과 affine의 곡선이 갈라져야 하는 이유: flip은 shared 30 epoch 내내 0.5였으므로 전문가에서 0부터 올리면 **없애려는 계단(0→0.3)보다 큰 계단(0.5→0→0.5)** 이 새로 생긴다.

**Tech Stack:** numpy(이미 import), 기존 pytest 구조. 신규 의존성 없음.

## Global Constraints

- **epoch은 단계 내 로컬 번호**(`--init-from`으로 시작하는 전문가 런은 0부터 센다). 전역 epoch = 30 + 로컬.
- **acc4는 절대 전역 지점을 복사하지 않고 비율로 옮긴다** — repo 선례(final_run_todo.md의 LR 절: "acc8의 '전역 38'을 그대로 가져오면 안 된다… 절대 지점이 아니라 비율로 옮겼다"). acc8 램프 = 단계 길이의 25%→50% → acc4(10 epoch)는 로컬 3→5.
- 확정 값: **acc8 `--affine-max-prob 0.5 --affine-delay 5 --affine-peak-epoch 10`** (전역 35→40), **acc4 `--affine-max-prob 0.5 --affine-delay 3 --affine-peak-epoch 5`** (전역 33→35). flip은 두 단계 모두 `--flip-max-prob 0.5` 유지.
- shared는 **건드리지 않는다** — 이미 실행 중이라 어차피 적용되지 않는다.
- tail 2단계는 같은 `_aug_*.sh`를 source하므로 로컬 epoch 0~1 < delay → **affine이 0으로 꺼진다**(의도된 결과, 사용자 확인함). 코드로 강제하지 않는다.
- **로컬 torch 없음** — pytest는 서버 게이트[사람]. 로컬은 `py -3.10 -m py_compile`까지.
- ⚠️ `scripts/tests/test_final_stage_args.py`에 **커밋 안 된 수정이 이미 워킹트리에 있다**(AUG_PREFIXES에 `--flip`·`--affine` 추가). Task 2가 그 파일의 EXPECTED까지 갱신해 **함께 커밋**한다.
- 커밋: kindofwater / bedso@snu.ac.kr. **AI attribution 금지.**

## File Structure

- Modify `utils/data/transforms.py` — AugmentConfig 2필드, `affine_prob()` 신설, `__call__` 1줄
- Modify `train.py` — CLI 인자 2개 (`--affine-max-prob` 다음)
- Modify `utils/data/load_data.py` — AugmentConfig 전달 2줄
- Modify `utils/learning/train_part.py` — `[aug]` 로그가 새 곡선을 반영
- Modify `scripts/tests/test_augment.py` — 신규 테스트 4건
- Modify `shell_scripting/_aug_acc4.sh`, `_aug_acc8.sh` — 확정 값
- Modify `scripts/tests/test_final_stage_args.py` — EXPECTED 갱신(+워킹트리의 접두사 수정 동반 커밋)

---

### Task 1: affine_prob() 코사인 램프 + 배선 + 테스트

**Files:**
- Modify: `utils/data/transforms.py`, `train.py`, `utils/data/load_data.py`, `utils/learning/train_part.py`
- Test: `scripts/tests/test_augment.py`

**Interfaces:**
- Consumes: `AugmentConfig(max_prob, delay, affine, affine_max_prob, ...)`, `MRIAugmentor.prob(p_max=None)`, 테스트 헬퍼 `affine_cfg`(기존)
- Produces: `AugmentConfig.affine_delay`(int, resolved)·`.affine_peak_epoch`(int|None), `MRIAugmentor.affine_prob()`, CLI `--affine-delay`/`--affine-peak-epoch`. Task 2의 셸이 이 인자 이름을 그대로 쓴다.

- [ ] **Step 1: 실패하는 테스트 작성** — `scripts/tests/test_augment.py` 끝에 추가:

```python
# ---------------------------------------------------------------------------
# affine 후반 상승(코사인 램프) — 2026-08-09. flip은 constant 유지, affine만 갈라진다.
# ---------------------------------------------------------------------------

def test_affine_peak_epoch_absent_keeps_shared_curve():
    """미지정이면 지금까지대로 flip과 같은 곡선 — 하위호환 계약."""
    cfg = affine_cfg(schedule="constant", max_prob=0.5, affine_max_prob=0.3)
    aug = MRIAugmentor(cfg)
    assert cfg.affine_peak_epoch is None
    for e in (0, 3, 9):
        aug.set_epoch(e)
        assert aug.affine_prob() == aug.prob(cfg.affine_max_prob) == 0.3


def test_affine_cosine_ramp_shape():
    """delay 전 0 · peak에서 max · 이후 유지 · 그 사이는 단조증가하는 코사인."""
    cfg = affine_cfg(schedule="constant", flip_max_prob=0.5, affine_max_prob=0.5,
                     affine_delay=5, affine_peak_epoch=10, total_epochs=20)
    aug = MRIAugmentor(cfg)

    for e in (0, 4, 5):                       # delay까지(포함) 0
        aug.set_epoch(e)
        assert aug.affine_prob() == 0.0
    aug.set_epoch(10)
    assert aug.affine_prob() == pytest.approx(0.5)
    aug.set_epoch(19)                          # peak 이후 유지
    assert aug.affine_prob() == pytest.approx(0.5)

    ps = []
    for e in range(5, 11):
        aug.set_epoch(e)
        ps.append(aug.affine_prob())
    assert all(b > a for a, b in zip(ps[:-1], ps[1:])), f"단조증가가 아니다: {ps}"

    # 코사인 램프의 대칭성 p(D+k) + p(P−k) = p_max. (중점 u=0.5는 t=7.5라 정수 epoch에
    # 안 떨어진다 — 그래서 한 점이 아니라 대칭쌍으로 못박는다. 아래 총노출이 정확히
    # 떨어지는 근거이기도 하다.)
    def at(e):
        aug.set_epoch(e)
        return aug.affine_prob()
    assert at(6) + at(9) == pytest.approx(0.5)
    assert at(7) + at(8) == pytest.approx(0.5)


def test_affine_ramp_leaves_flip_constant():
    """flip은 --aug-schedule(constant)을 그대로 따른다 — affine 램프에 딸려가면 안 된다.
    (딸려가면 shared 내내 0.5였던 flip이 전문가 시작에서 0으로 떨어져 더 큰 계단이 생긴다)"""
    cfg = affine_cfg(schedule="constant", flip_max_prob=0.5, affine_max_prob=0.5,
                     affine_delay=5, affine_peak_epoch=10, total_epochs=20)
    aug = MRIAugmentor(cfg)
    for e in (0, 5, 10, 19):
        aug.set_epoch(e)
        assert aug.prob(cfg.flip_max_prob) == 0.5


def test_affine_exposure_matches_design_numbers():
    """실전 값의 총노출(epoch 환산)을 못박는다 — A/B 없이 도는 런이라 의도를 테스트에 남긴다.
    acc8(20ep, 5→10): 6.0 = 기존 0.3 고정과 동일(재배치). acc4(10ep, 3→5): 2.75 ≈ 기존 3.0."""
    def exposure(total, delay, peak):
        cfg = affine_cfg(schedule="constant", affine_max_prob=0.5, total_epochs=total,
                         affine_delay=delay, affine_peak_epoch=peak)
        aug = MRIAugmentor(cfg)
        s = 0.0
        for e in range(total):
            aug.set_epoch(e)
            s += aug.affine_prob()
        return s

    assert exposure(20, 5, 10) == pytest.approx(6.0)      # = 0.3 × 20
    assert exposure(10, 3, 5) == pytest.approx(2.75)      # ≈ 0.3 × 10
```

- [ ] **Step 2: (로컬) 문법 확인** — RED는 서버 게이트로 이월

Run: `py -3.10 -m py_compile scripts/tests/test_augment.py`
Expected: exit 0. (서버 pytest 기준 기대 RED: `AttributeError: 'MRIAugmentor' object has no attribute 'affine_prob'` 계열 4건 FAIL)

- [ ] **Step 3: transforms.py — AugmentConfig에 2필드 추가.** `self.affine_max_prob = ...` 줄 **다음**, `self.seed = ...` 줄 **앞**에:

```python
        # affine 전용 스케줄(2026-08-09) — peak_epoch을 지정하면 delay~peak을 코사인으로 올라간
        # 뒤 유지한다(--aug-schedule과 무관). 미지정이면 flip과 같은 곡선(기존 동작).
        # delay 미지정이면 공통 delay를 따른다.
        v = kw.get("affine_delay", None)
        self.affine_delay = int(self.delay if v is None else v)
        v = kw.get("affine_peak_epoch", None)
        self.affine_peak_epoch = None if v is None else int(v)
```

- [ ] **Step 4: transforms.py — `prob()` 메서드 **바로 다음**에 `affine_prob()` 추가** (`prob()` 본문은 건드리지 않는다):

```python
    def affine_prob(self):
        """affine 전용 확률. peak_epoch 미지정이면 flip과 같은 곡선(기존 동작).
        지정하면 delay(D)부터 peak(P)까지 **코사인으로 올라가 이후 p_max를 유지**한다:
            p(t<D)=0 · p(t≥P)=affine_max_prob · 그 사이 p = p_max·(1−cos(πu))/2, u=(t−D)/(P−D)
        양 끝 기울기가 0이라 계단도, 꺾임도 없다. --aug-schedule(flip 곡선)과 독립인 이유:
        flip은 shared 30 epoch 내내 0.5였으므로 전문가에서 같이 0부터 올리면 없애려던 계단보다
        큰 계단(0.5→0→0.5)이 새로 생긴다. epoch은 **단계 내 로컬**(전역 = 30 + 로컬)."""
        c = self.cfg
        if (not c.on) or (not c.affine):
            return 0.0
        if c.affine_peak_epoch is None:
            return self.prob(c.affine_max_prob)
        D, P, t = int(c.affine_delay), int(c.affine_peak_epoch), self.epoch
        if t < D:
            return 0.0
        if (t >= P) or (P - D <= 0):
            return float(c.affine_max_prob)
        u = (t - D) / (P - D)
        return float(c.affine_max_prob * (1.0 - np.cos(np.pi * u)) / 2.0)
```

- [ ] **Step 5: transforms.py — `__call__`의 affine 확률 계산 1줄 교체.** 기존:

```python
        p_affine = self.prob(self.cfg.affine_max_prob) if self.cfg.affine else 0.0
```

교체(affine 꺼짐 처리는 `affine_prob()` 안에 있다):

```python
        p_affine = self.affine_prob()
```

- [ ] **Step 6: train.py — `--affine-max-prob` 인자 다음에 추가**

```python
    parser.add_argument('--affine-delay', type=int, default=None,
                        help='affine이 오르기 시작하는 epoch(단계 내 로컬). 미지정이면 --aug-delay')
    parser.add_argument('--affine-peak-epoch', type=int, default=None,
                        help='affine이 --affine-max-prob에 도달하는 epoch(이후 유지). 지정하면 '
                             'delay~peak 코사인 상승. 미지정이면 flip과 같은 곡선')
```

- [ ] **Step 7: load_data.py — AugmentConfig 호출에서 `affine_max_prob=` 줄 다음에 추가**

```python
        affine_delay=getattr(args, "affine_delay", None),
        affine_peak_epoch=getattr(args, "affine_peak_epoch", None),
```

- [ ] **Step 8: train_part.py — `[aug]` 로그의 p_aff 계산 1줄 교체.** 기존:

```python
            p_aff = augmentor.prob(augmentor.cfg.affine_max_prob) if augmentor.cfg.affine else 0.0
```

교체:

```python
            p_aff = augmentor.affine_prob()
```

- [ ] **Step 9: (로컬) 문법 확인**

Run: `py -3.10 -m py_compile utils/data/transforms.py train.py utils/data/load_data.py utils/learning/train_part.py scripts/tests/test_augment.py`
Expected: exit 0

- [ ] **Step 10: Commit**

```bash
git add utils/data/transforms.py train.py utils/data/load_data.py utils/learning/train_part.py scripts/tests/test_augment.py
git commit -m "feat(aug): affine 전용 코사인 램프 — --affine-delay/--affine-peak-epoch (flip은 constant 유지)"
```

---

### Task 2: 전문가 단계 셸 값 확정 + 동결 테스트 갱신

**Files:**
- Modify: `shell_scripting/_aug_acc4.sh`, `shell_scripting/_aug_acc8.sh`
- Modify: `scripts/tests/test_final_stage_args.py` (⚠️ AUG_PREFIXES 수정이 **이미 워킹트리에 있다** — 커밋 안 됨. EXPECTED만 갱신해 함께 커밋한다)

**Interfaces:**
- Consumes: Task 1의 CLI 인자명 `--affine-delay`/`--affine-peak-epoch`
- Produces: 없음(값 확정)

- [ ] **Step 1: `_aug_acc8.sh`의 AUG_ARGS 줄 교체** (전역 35→40 = 로컬 5→10):

```bash
AUG_ARGS=( --aug-on --aug-schedule constant --aug-max-prob 0.5 --flip-max-prob 0.5 --affine-max-prob 0.5 --affine-delay 5 --affine-peak-epoch 10 --aug-delay 0 --remask 8 --mask-aug offset )
```

- [ ] **Step 2: `_aug_acc4.sh`의 AUG_ARGS 줄 교체** (비율 이식 — 전역 33→35 = 로컬 3→5):

```bash
AUG_ARGS=( --aug-on --aug-schedule constant --aug-max-prob 0.5 --flip-max-prob 0.5 --affine-max-prob 0.5 --affine-delay 3 --affine-peak-epoch 5 --aug-delay 0 --remask 4 --mask-aug offset )
```

- [ ] **Step 3: `test_final_stage_args.py`의 EXPECTED에서 acc4·acc8 항목 교체** (shared 항목은 그대로 둔다):

```python
    "acc4": {"--aug-on": True, "--aug-schedule": "constant", "--aug-max-prob": "0.5",
             "--flip-max-prob": "0.5", "--affine-max-prob": "0.5",
             "--affine-delay": "3", "--affine-peak-epoch": "5",
             "--aug-delay": "0", "--mask-aug": "offset", "--remask": "4"},
    "acc8": {"--aug-on": True, "--aug-schedule": "constant", "--aug-max-prob": "0.5",
             "--flip-max-prob": "0.5", "--affine-max-prob": "0.5",
             "--affine-delay": "5", "--affine-peak-epoch": "10",
             "--aug-delay": "0", "--mask-aug": "offset", "--remask": "8"},
```

- [ ] **Step 4: 셸 문법 + 인자 대조 (로컬 가능 — bash만 필요, torch·GPU 불필요)**

Run:
```bash
bash -n shell_scripting/_aug_acc4.sh && bash -n shell_scripting/_aug_acc8.sh && \
for s in acc4 acc8; do echo "--$s--"; bash -c "source shell_scripting/_aug_$s.sh; printf '%s\n' \"\${AUG_ARGS[@]}\""; done
```
Expected: 문법 오류 없음. 출력에 acc8은 `--affine-delay 5`·`--affine-peak-epoch 10`, acc4는 `3`·`5`가 있고, 양쪽 `--affine-max-prob 0.5`.
(⚠️ `test_final_stage_args.py` 자체를 이 Windows 로컬에서 돌리면 스텁 실행·cp949 인코딩 문제로 실패한다 — 서버 몫이다. 위 대조로 값 일치만 확인한다.)

- [ ] **Step 5: Commit**

```bash
git add shell_scripting/_aug_acc4.sh shell_scripting/_aug_acc8.sh scripts/tests/test_final_stage_args.py
git commit -m "feat(aug): 전문가 단계 affine 후반 상승 값 확정(acc8 5→10 · acc4 3→5, max 0.5) + 동결 테스트 갱신"
```

---

### Task 3: [사람] 서버 게이트

- [ ] `python -m pytest scripts/tests/test_augment.py scripts/tests/test_final_stage_args.py scripts/tests/test_remask.py -v` → 전부 PASS (신규 4건 포함). torch 필요 = 서버에서.
- [ ] acc8 단계가 시작되면 로그 확인: `[aug] epoch=0 p_flip=0.5000 p_affine=0.0000` → epoch 5까지 0 → epoch 10에 `p_affine=0.5000` → 이후 유지.
- [ ] shared는 지금 실행 중이라 이 변경의 영향을 받지 않는다(프로세스가 시작 시점 코드를 물고 있다). 확인할 것 없음.

## 계획에서 뺀 것 (ponytail — 이유와 복귀 조건)

- **`prob()` 일반화(peak/cosine을 공용 스케줄로)**: 필요한 건 affine 한 갈래뿐인데 공용을 건드리면 기존 스케줄 테스트·shell22/24 이력이 전부 사정권에 든다. flip에도 램프가 필요해지면 그때 통합.
- **`--flip-delay`/`--flip-peak-epoch`**: flip은 constant 0.5 유지가 결론 — 대칭성만을 위한 인자는 안 만든다.
- **tail 단계 전용 증강 파일**: tail은 로컬 epoch 0~1이라 delay에 걸려 affine이 자동으로 0이 된다. 별도 파일은 같은 결과를 두 번 선언하는 것.
- **shared 재시작으로 affine 소급 적용**: 8.6일 재학습 비용이 이득 상한보다 크다. 다음 run에서 처음부터 켜는 것으로.
