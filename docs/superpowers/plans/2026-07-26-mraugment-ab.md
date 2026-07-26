# 증강 활성화 배선 + 증강 ON/OFF A/B 실행 플랜

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 죽어 있는 `MRIAugmentor`를 실제로 켤 수 있게 배선하고, 증강 ON/OFF를 동일 fold·seed에서 비교해 과적합 완화 여부를 판정한다.

**Architecture:** 증강 로직(`transforms.py`)은 그대로 두고, ① 스케줄 함수를 MRAugment 공식 형태(delay `D` · 총 epoch `T` 정규화)로 교체 ② 변환별 가중치를 분리 ③ `train.py` 인자와 `train_part.py`의 `set_epoch` 통로를 연결 ④ 매 epoch `p`를 로깅. 그 위에서 shell 슬롯 하나(≈21h)로 ON arm만 새로 학습하고, 과적합이 관측된 기존 런을 OFF arm으로 재사용한다.

**Tech Stack:** Python 3.10.12(서버) · numpy · PyTorch 2.3.1+cu121 · pytest · argparse

**설계 문서:** [`docs/superpowers/specs/2026-07-26-mraugment-ab-design.md`](../specs/2026-07-26-mraugment-ab-design.md)

## Global Constraints

- ⚠️ **개발 환경(로컬 노트북)에는 torch·pytest·skimage가 없다**(numpy 2.5.1·h5py만, Python 3.13). `utils.data.transforms`가 모듈 최상단에서 `torch`를 import하므로 **로컬에서는 테스트를 실행할 수 없다.**
  - 구현 태스크(1~6)에서 **`pytest`를 실행하려 시도하지 말 것** — 실패는 환경 문제이며 BLOCKED 사유가 아니다.
  - 로컬 검증 수단은 **`python -m py_compile <file>`**(import 없이 문법만 검사) 와 **`bash -n <script>`** 뿐이다.
  - 실제 실행 검증은 **Task 7의 서버 핸드오프**에서 사람이 한 번에 한다.
- **테스트 위치는 `scripts/tests/`** — 이 repo의 기존 관례(`conftest.py`가 repo root를 `sys.path`에 넣는다). `tests/`는 빈 디렉터리이므로 쓰지 않는다.
- **GPU를 쓰는 명령은 Claude가 절대 직접 실행하지 않는다** (`CLAUDE.md` §6). 슬롯 스크립트 작성까지만 하고 실행은 사람이 한다. `pytest`·`git`은 GPU를 안 쓰므로 큐에 넣지 않고 서버의 별도 터미널에서 실행한다.
- **기본값은 전부 off** — `--aug-on` 없이 돌린 학습은 변경 전과 동일해야 한다.
- 커밋 메시지·PR에 **AI attribution을 넣지 않는다** (`CLAUDE.md` §9). 커밋 identity는 이미 `kindofwater <bedso@snu.ac.kr>`로 설정돼 있다.
- 작업 브랜치는 **`jkMRplus`**. 브랜치를 바꾸지 않는다.
- 증강은 **train 로더에만** 붙는다. val/forward에는 절대 붙지 않는다.
- `argparse`가 `ArgumentDefaultsHelpFormatter`를 쓰므로 **help 문자열의 `%`는 `%%`로 이스케이프**해야 한다(`train.py:43` 선례).
- 비목표(건드리지 않는다): phase/noise/라인 지터 활성화 · translation · TTA · consistency 손실 · GRAPPA prior · bbox 가중 손실 · 합성 k-space.

---

### Task 1: 서버 검증 핸드오프 스크립트

로컬에서 아무것도 실행할 수 없으므로 **검증 도구를 먼저 만든다.** 이 스크립트 하나가 Task 2~6의 결과를 서버에서 일괄 검증하는 게이트다(Task 7). 아직 존재하지 않는 테스트 파일을 참조하는 것은 정상이다 — Task 2가 만든다.

**Files:**
- Create: `scripts/verify_aug.sh`

**Interfaces:**
- Consumes: (Task 2가 만들 예정인) `scripts/tests/test_augment.py`, (Task 5가 만들 예정인) `train.py`의 `--aug-*` 인자
- Produces: `bash scripts/verify_aug.sh` — 사람이 서버에서 실행하는 단일 검증 명령. exit 0 = 전부 통과.

- [ ] **Step 1: 스크립트 작성**

```bash
#!/usr/bin/env bash
# ============================================================
# 증강(MRIAugmentor) 배선 검증 — GPU 불필요, 큐에 넣지 말고 그냥 실행한다.
#
#   목적: 로컬(torch 없음)에서 작성한 증강 배선이 서버에서 실제로 동작하는지 한 번에 확인.
#   실행: bash scripts/verify_aug.sh          # repo 어디서 실행해도 됨
#
#   검사 3종:
#     [1] 단위 테스트 — flip 정확성·ACS 보존·bbox 동기·결정성·스케줄·가중치·배선 불변식
#     [2] CLI 도달성  — train.py의 --aug-* 인자가 AugmentConfig까지 전달되는지
#     [3] off 동등성  — --aug-on 없이 만든 DataTransform이 기존과 같은 6-튜플인지
#
#   전부 통과하면 마지막에 "ALL CHECKS PASSED"가 찍히고 exit 0.
#   실패하면 어느 검사인지 출력하고 exit 1 → 그 출력을 그대로 Claude에게 붙여주면 된다.
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1
echo "[verify_aug] repo root = $(pwd)"
echo "[verify_aug] python     = $(python --version 2>&1)"
echo

FAILED=0

echo "=== [1/3] 단위 테스트 (scripts/tests/test_augment.py) ==="
python -m pytest scripts/tests/test_augment.py -v
if [ $? -ne 0 ]; then echo "[FAIL] 단위 테스트"; FAILED=1; else echo "[ok] 단위 테스트"; fi
echo

echo "=== [2/3] CLI 인자가 AugmentConfig까지 도달하는지 ==="
python - <<'PY'
import sys
sys.argv = ["train.py", "--aug-on", "--aug-delay", "2", "--no-aug-flip", "-e", "10"]
from train import parse
from utils.data.transforms import AugmentConfig, MRIAugmentor

a = parse()
cfg = AugmentConfig(
    on=a.aug_on, schedule=a.aug_schedule, delay=a.aug_delay,
    total_epochs=a.num_epochs, exp_decay=a.aug_exp_decay,
    max_prob=a.aug_max_prob, flip=a.aug_flip, mask=a.aug_mask,
    weight_flip=a.aug_weight_flip, weight_mask=a.aug_weight_mask,
    orig_mask_prob=a.aug_orig_mask_prob,
)
m = MRIAugmentor(cfg)
m.set_epoch(9)
p, p_flip, p_mask = m.probs()
print(f"  flip={cfg.flip} (기대 False) | p={p:.4f} p_flip={p_flip:.4f} p_mask={p_mask:.4f}")
assert cfg.flip is False, "--no-aug-flip이 반영되지 않았다"
assert p > 0.5, f"10ep 런의 마지막 epoch인데 p가 너무 낮다: {p}"
assert p_flip == 0.0, f"flip이 꺼졌는데 p_flip이 0이 아니다: {p_flip}"
print("  OK")
PY
if [ $? -ne 0 ]; then echo "[FAIL] CLI 도달성"; FAILED=1; else echo "[ok] CLI 도달성"; fi
echo

echo "=== [3/3] 증강 off일 때 기존과 동일한 6-튜플인지 ==="
python - <<'PY'
import numpy as np
from utils.data.transforms import DataTransform

mask = np.zeros(368, dtype=np.float32); mask[::4] = 1.0; mask[170:199] = 1.0
rng = np.random.default_rng(0)
img = rng.standard_normal((4, 64, 368)) + 1j * rng.standard_normal((4, 64, 368))
kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img, axes=(-2, -1)),
                                     axes=(-2, -1), norm="ortho"), axes=(-2, -1)).astype(np.complex64)
target = rng.random((384, 384)).astype(np.float32)

out = DataTransform(isforward=False, max_key="max")(mask, kspace, target, {"max": 1.0}, "knee_acc4_1.h5", 0)
print(f"  반환 길이 = {len(out)} (기대 6)")
assert len(out) == 6, f"플래그 전부 off인데 {len(out)}-튜플"
print("  OK")
PY
if [ $? -ne 0 ]; then echo "[FAIL] off 동등성"; FAILED=1; else echo "[ok] off 동등성"; fi
echo

if [ "$FAILED" -eq 0 ]; then
  echo "=========================================="
  echo " ALL CHECKS PASSED — 증강 배선 검증 완료"
  echo "=========================================="
  exit 0
else
  echo "=========================================="
  echo " 실패한 검사가 있다 — 위 출력을 그대로 전달할 것"
  echo "=========================================="
  exit 1
fi
```

- [ ] **Step 2: 문법 검사 (로컬에서 가능)**

Run: `bash -n scripts/verify_aug.sh`
Expected: 출력 없음(exit 0). 오류가 나면 bash 문법 문제이므로 고친다.

- [ ] **Step 3: 커밋**

```bash
git add scripts/verify_aug.sh
git commit -m "chore(aug): 증강 배선 서버 검증 스크립트 추가 (단위 테스트·CLI 도달성·off 동등성)"
```

---

### Task 2: 기존 증강 동작을 회귀 테스트로 고정

증강 코드는 삭제된 노트북에서 1회 눈으로만 검증됐다. Task 3·4가 이 코드를 고치므로 **먼저 현재 동작을 테스트로 못 박는다.** 이 테스트들은 서버에서 **바로 통과할 것을 의도한다** — TDD의 red 단계가 아니라 리팩터링 안전망(characterization test)이다.

**Files:**
- Create: `scripts/tests/test_augment.py`

**Interfaces:**
- Consumes: `utils.data.transforms`의 `AugmentConfig`, `MRIAugmentor`, `DataTransform`, `_detect_acs`, `_fftc`, `_ifftc` (기존 `scripts/tests/conftest.py`가 repo root를 `sys.path`에 넣어준다 — 추가 설정 불필요)
- Produces: 헬퍼 `make_mask()`, `make_kspace()`, `make_target()`, `flip_only_cfg()`, `mask_only_cfg()` — Task 3·4·6의 테스트가 재사용한다.

- [ ] **Step 1: 테스트 파일 작성**

```python
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
```

- [ ] **Step 2: 문법 검사 (로컬에서 가능한 유일한 검증)**

Run: `python -m py_compile scripts/tests/test_augment.py && echo SYNTAX_OK`
Expected: `SYNTAX_OK`. **`pytest`를 실행하려 하지 말 것** — 로컬에 torch가 없어 import 자체가 실패한다(Global Constraints).

- [ ] **Step 3: 커밋**

```bash
git add scripts/tests/test_augment.py
git commit -m "test(aug): MRIAugmentor 불변식 회귀 테스트 추가 (flip 정확성·ACS 보존·bbox 동기·결정성·off 기본값)"
```

---

### Task 3: 스케줄을 MRAugment 공식 형태로 교체

현재 `_prob()`는 `warmup_epochs=10` 미만이면 0을 반환하고 램프가 `ramp_epochs=15`(25ep 런 가정)에 고정돼 있다. 실제 런은 `-e 10`(epoch 0~9)이라 **전 구간 `p=0`** → 증강이 0회 발생한다. 공식 repo(`MathFLDS/MRAugment`, `mraugment/data_augment.py`)의 `schedule_p()`와 동일하게 **총 epoch `T`로 정규화**한다.

**Files:**
- Modify: `utils/data/transforms.py:178-200` (`AugmentConfig` 필드), `utils/data/transforms.py:215-225` (`_prob`)
- Modify: `scripts/tests/test_augment.py` (테스트 추가)

**Interfaces:**
- Consumes: Task 2의 헬퍼
- Produces: `AugmentConfig(delay, total_epochs, exp_decay, schedule, max_prob, ...)` · `MRIAugmentor._prob() -> float`

- [ ] **Step 1: `AugmentConfig` 필드 교체** — `utils/data/transforms.py:183-187`의 스케줄 5줄을 대체

```python
        # 스케줄 — MRAugment 공식 repo(mraugment/data_augment.py: schedule_p)와 동일 형태.
        #   p(t<D)=0, p(T)=p_max. T로 정규화하므로 런 길이가 바뀌어도 증강량이 스케일된다.
        self.schedule = kw.get("schedule", "exp")          # 'exp' | 'ramp' | 'constant'
        self.delay = kw.get("delay", 0)                    # D: 증강 없이 보내는 앞쪽 epoch 수
        self.total_epochs = kw.get("total_epochs", 1)      # T: args.num_epochs (load_data가 채움)
        self.exp_decay = kw.get("exp_decay", 5.0)          # 논문/공식 repo 기본 c=5.0
        self.max_prob = kw.get("max_prob", 0.55)           # 논문 기본 p_max=0.55
```

`warmup_epochs`·`ramp_epochs`는 **삭제**한다(대체됨). 이 두 이름을 참조하는 곳은 `load_data.py`뿐이며 Task 5에서 함께 고친다.

- [ ] **Step 2: `_prob()` 교체** — `utils/data/transforms.py:215-225`

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

- [ ] **Step 3: 테스트 추가** — `scripts/tests/test_augment.py` 맨 아래. 기대값은 로컬 numpy로 검산했다(`T=10,D=2,c=5,p_max=0.55` → `p(3..9) = 0.2573, 0.3951, 0.4688, 0.5083, 0.5294, 0.5407, 0.5468`).

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
```

- [ ] **Step 4: 문법 검사**

Run: `python -m py_compile utils/data/transforms.py scripts/tests/test_augment.py && echo SYNTAX_OK`
Expected: `SYNTAX_OK`

- [ ] **Step 5: 커밋**

```bash
git add utils/data/transforms.py scripts/tests/test_augment.py
git commit -m "fix(aug): 증강 확률 스케줄을 총 epoch 정규화 형태로 교체 (10ep 런에서 p가 끝까지 0이던 문제)"
```

---

### Task 4: 변환별 가중치 분리 + 중복 draw 정리

지금은 flip·mask가 같은 `p`를 공유하고, mask만 `rng.random()`을 2회 뽑아 실효 확률이 `p·(1-orig_mask_prob)`다. 공식 repo의 `--aug_weight_*`(= `p_i = p·w_i`)에 대응하는 가중치를 넣고 draw를 하나로 합친다. **확률값은 기존과 동일**(순수 정리 + 가중치 추가)하며, 증강이 실행된 적이 없어 RNG 소비 변화로 인한 재현성 부채는 없다.

**Files:**
- Modify: `utils/data/transforms.py` (`AugmentConfig`에 가중치 2개, `MRIAugmentor`에 `probs()`, `__call__:267-303`)
- Modify: `scripts/tests/test_augment.py`

**Interfaces:**
- Consumes: Task 3의 `_prob()`
- Produces: `MRIAugmentor.probs() -> (p, p_flip, p_mask)` — Task 6의 로깅과 Task 1의 검증 스크립트가 쓴다.

- [ ] **Step 1: `AugmentConfig`에 가중치 추가** — `self.flip`/`self.mask` 정의 바로 아래

```python
        # 변환별 가중치(공식 repo --aug_weight_*): 실제 확률 = p · w_i. 한쪽만 0으로 두면 분리 A/B가 된다.
        self.weight_flip = kw.get("weight_flip", 1.0)
        self.weight_mask = kw.get("weight_mask", 1.0)
```

- [ ] **Step 2: `probs()` 추가** — `set_epoch` 바로 아래

```python
    def probs(self):
        """(p, p_flip, p_mask). 학습 로그와 __call__이 같은 값을 쓰도록 한 곳에서 계산."""
        c = self.cfg
        p = self._prob()
        p_flip = min(1.0, max(0.0, p * float(c.weight_flip)))
        p_mask = min(1.0, max(0.0, p * float(c.weight_mask) * (1.0 - float(c.orig_mask_prob))))
        return p, p_flip, p_mask
```

- [ ] **Step 3: `__call__`의 확률 계산·mask·flip 블록 교체**

`__call__` 안의 `p = self._prob()` / `if p <= 0.0: return ...` / mask 블록 / flip 블록을 아래로 교체한다(phase·noise 블록은 `p`를 그대로 쓰므로 손대지 않는다):

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

- [ ] **Step 4: 테스트 추가** — `scripts/tests/test_augment.py` 맨 아래

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
```

- [ ] **Step 5: 문법 검사**

Run: `python -m py_compile utils/data/transforms.py scripts/tests/test_augment.py && echo SYNTAX_OK`
Expected: `SYNTAX_OK`

- [ ] **Step 6: 커밋**

```bash
git add utils/data/transforms.py scripts/tests/test_augment.py
git commit -m "refactor(aug): 변환별 가중치(p_i = p·w_i) 도입 + mask 중복 draw 정리"
```

---

### Task 5: `train.py` 증강 인자 + `load_data.py` 전달

**Files:**
- Modify: `train.py` (`--init-from` 인자 다음, `args = parser.parse_args()` 앞에 블록 추가)
- Modify: `utils/data/load_data.py:194-208` (`AugmentConfig(...)` 인자)

**Interfaces:**
- Consumes: Task 3·4의 `AugmentConfig` 필드명(`delay`/`total_epochs`/`exp_decay`/`weight_flip`/`weight_mask`)
- Produces: CLI 플래그 `--aug-on --aug-schedule --aug-max-prob --aug-delay --aug-exp-decay --aug-flip/--no-aug-flip --aug-mask/--no-aug-mask --aug-weight-flip --aug-weight-mask --aug-mask-thin-max --aug-orig-mask-prob` — Task 1의 검증 스크립트와 Task 8의 shell 슬롯이 쓴다.

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

- [ ] **Step 2: `load_data.py`의 `AugmentConfig(...)` 인자 교체** — `warmup_epochs`·`ramp_epochs` 두 줄을 지우고 아래로 대체

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

- [ ] **Step 3: 문법 검사**

Run: `python -m py_compile train.py utils/data/load_data.py && echo SYNTAX_OK`
Expected: `SYNTAX_OK`

- [ ] **Step 4: 커밋**

```bash
git add train.py utils/data/load_data.py
git commit -m "feat(aug): train.py --aug-* 인자 신설 + load_data에 스케줄 파라미터 전달"
```

---

### Task 6: epoch 전달(`set_epoch`) + `p` 로깅

`transforms.py`는 "지금 몇 epoch인지" 모른다. 이 통로가 없으면 스케줄이 영원히 epoch 0이다. 그리고 이번 사안의 근본 원인이 **무증상 실패**였으므로 매 epoch `p`를 찍어 눈에 보이게 한다.

`persistent_workers`를 쓰지 않으므로(`load_data.py:260-268`) DataLoader가 epoch마다 워커를 새로 띄운다 → 메인 프로세스 dataset에 심은 epoch이 워커 복사본에 전달된다.

**Files:**
- Modify: `utils/learning/train_part.py:389-391`
- Modify: `scripts/tests/test_augment.py`

**Interfaces:**
- Consumes: `DataTransform.set_epoch(epoch)`(기존), Task 4의 `MRIAugmentor.probs()`
- Produces: 학습 로그의 `[aug] epoch=… p=… p_flip=… p_mask=…` 줄 — Task 8의 스모크 게이트가 이 줄을 확인한다.

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

- [ ] **Step 2: 배선 테스트 추가** — `scripts/tests/test_augment.py` 맨 아래

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

- [ ] **Step 3: 문법 검사**

Run: `python -m py_compile utils/learning/train_part.py scripts/tests/test_augment.py && echo SYNTAX_OK`
Expected: `SYNTAX_OK`

- [ ] **Step 4: 커밋**

```bash
git add utils/learning/train_part.py scripts/tests/test_augment.py
git commit -m "feat(aug): 매 epoch set_epoch 배선 + 증강 확률 p 로깅"
```

---

### Task 7: 서버 검증 게이트 (⚠️ 사람이 실행)

Task 2~6은 로컬에서 문법 검사만 받았다. **여기서 처음으로 실제 실행 검증**을 한다. Claude는 인계 문구를 준비하고 결과를 받아 처리한다.

**Files:** 없음(실행·판정만)

- [ ] **Step 1: 사람에게 인계**

전달 문구:

> 서버(별도 터미널, 큐 아님)에서 repo root에서 아래 한 줄을 실행해 주세요:
>
> ```bash
> bash scripts/verify_aug.sh
> ```
>
> 마지막에 `ALL CHECKS PASSED`가 나오면 알려주시고, 실패하면 **출력 전체를 그대로** 붙여주세요.

- [ ] **Step 2: 결과 처리**

- `ALL CHECKS PASSED` → Task 8로.
- 실패 → 실패 출력을 근거로 fix 라운드를 돌린다(구현 태스크의 fix 루프와 동일). **로컬에서 재현할 수 없으므로 매 수정마다 서버 재실행이 1회 필요하다** — 여러 수정을 모아 한 번에 검증한다.

---

### Task 8: baseline 대조 + ON arm 슬롯 작성

**Files:**
- Create: `shell_scripting/shell22.sh`
- Read: 과적합이 관측된 런의 `val_loss_log*.npy`, 해당 shell 슬롯

**Interfaces:**
- Consumes: Task 5의 CLI 플래그, Task 6의 `[aug]` 로그
- Produces: `shell22.sh`(사람이 실행) · `--aug-delay` 확정값 · baseline 재사용 가능 여부 판정

- [ ] **Step 1: 과적합 런 식별 + config 대조**

사람에게 **어느 런인지(net-name/fold)** 확인한 뒤, 그 런의 shell 슬롯과 현재 HEAD의 config를 비교한다:

```bash
git log --oneline -20 -- shell_scripting/
grep -n "train.py" shell_scripting/shell*.sh
```

판정 기준: **fold · kfold-seed · seed · num-cascades · n-feat0 · sens-n-feat0 · n-history · lr · fg-loss · area-weight · grad-clip · accum-steps · coil-compress · readout-crop**이 전부 같으면 그 런을 OFF arm으로 **재사용한다**(GPU 21h 절약). 하나라도 다르면 baseline도 재학습해야 하므로 **사람에게 알리고 예산 승인을 받는다**(21h → 42h).

- [ ] **Step 2: `--aug-delay` 값 결정**

val 로그에서 val이 꺾인 epoch을 읽는다(서버에서, GPU 미사용):

```bash
python -c "
import numpy as np, sys
log = np.load(sys.argv[1])           # (N,2): [epoch, val_loss]
for e, v in log: print(int(e), f'{v:.6f}')
best = int(log[np.argmin(log[:,1]), 0])
print('best epoch =', best, '→ --aug-delay =', max(0, best - 1))
" ../result/<net_name>/val_loss_log_<tag>.npy
```

`--aug-delay = max(0, best_epoch - 1)`로 정한다(증강 확률이 과적합 구간을 덮게 하는 것이 목적). 이 값은 스크립트에 박지 않는다 — `shell22.sh`는 `AUG_DELAY` 환경변수를 필수로 요구하고(`: "${AUG_DELAY:?…}"` 형태의 가드), 없으면 exit 2로 거부한다. 근거는 스크립트 주석에 남긴다.

- [ ] **Step 3: `shell22.sh` 작성**

[`shell21.sh`](../../../shell_scripting/shell21.sh)를 복제해 헤더 주석을 이 실험 목적으로 바꾸고, **train.py 호출에 증강 인자만 추가**한다. `FOLD`·`K_FOLD`·`KFOLD_SEED`·나머지 config는 baseline과 **완전 동일**하게 유지한다(변경점 1개 원칙).

⚠️ **SMOKE 경로는 스케줄을 다르게 줘야 한다.** 이 저장소의 모든 SMOKE 슬롯은 `EPOCHS=1`(shell21.sh)로 도는데, `exp`/`ramp` 스케줄은 `p(epoch=0) = 0`이 정확히 나온다(`delay` 이내이거나 `t<D`). 그러면 스모크 런은 `[aug] epoch=0 p=0.0000`을 찍고 `p <= 0.0` 조기 리턴으로 **증강을 0회** 수행한 채 끝나버려서, Step 6의 스모크 게이트 기준("로그에 `p > 0`")을 원천적으로 만족할 수 없다. 그래서 `shell21.sh`와 같은 `SMOKE` 분기 안에서 스케줄 인자 자체를 다르게 준다:

```bash
if [ "$SMOKE" = "1" ]; then
    # ⚠️ epoch=0(1-epoch 스모크)에서 exp/ramp는 p=0 그대로라 증강이 0회 실행된다.
    #    constant + delay=0 → p=p_max로 강제해야 스모크가 실제로 증강 경로를 태운다.
    AUG_SCHEDULE="constant"; AUG_DELAY=0
else
    AUG_SCHEDULE="exp"
    : "${AUG_DELAY:?Step 2에서 구한 값을 AUG_DELAY=<값>으로 넘겨야 한다 — 자동추정/기본값 금지, 없으면 exit 2}"
fi
```

```bash
  --aug-on \
  --aug-schedule "$AUG_SCHEDULE" \
  --aug-max-prob 0.55 \
  --aug-delay "$AUG_DELAY" \
  --aug-exp-decay 5.0 \
  --aug-mask-thin-max 4 \
  --aug-orig-mask-prob 0.5 \
```

**본 학습(SMOKE=0)은 그대로 `--aug-schedule exp`에 Step 2의 delay를 쓴다** — 위 override는 스모크 게이트 전용이다.

헤더 주석에 반드시 적을 것: **변경점은 증강 인자뿐 · baseline은 `<net_name>` 런 · 판정은 fg-val 서명 3개 · `SMOKE=1` 먼저(단, SMOKE는 `--aug-schedule constant --aug-delay 0`로 강제해야 증강이 실제로 걸린다)**.

- [ ] **Step 4: 문법 검사**

Run: `bash -n shell_scripting/shell22.sh && echo SYNTAX_OK`
Expected: `SYNTAX_OK`

- [ ] **Step 5: 커밋**

```bash
git add shell_scripting/shell22.sh
git commit -m "chore(queue): shell22 — 증강 ON arm 슬롯 (fold N, baseline 대비 증강 인자만 추가)"
```

- [ ] **Step 6: 사람에게 스모크 실행 인계** (⚠️ Claude가 실행하지 않는다)

전달할 내용:
1. `SMOKE=1 bash shell_scripting/shell22.sh` (120 iter) — ⚠️ **shell22.sh의 SMOKE 분기가 `--aug-schedule constant --aug-delay 0`을 쓰는지 먼저 확인한다.** 1-epoch 스모크에서 본 학습용 `exp`/`ramp` 스케줄을 그대로 쓰면 `p(epoch=0)=0`이라 증강이 무조건 0회 실행되고, 아래 ②를 절대 만족할 수 없는 게이트가 되어 사람이 GPU 서버까지 왕복하는 시간을 헛되이 쓰게 된다. 이 override가 이 스모크 게이트의 존재 이유다.
2. 확인할 것 세 가지 — ① exit 0 · OOM 없음 ② 로그에 `[aug] … p=` 가 찍히고 **`p > 0`**(constant+delay=0라면 `p = p_max = 0.55`여야 한다) ③ **iter time을 baseline 로그와 비교** (증강은 샘플당 FFT 2회(15×640×368)를 dataloader에 추가한다. `--num-workers 4`에서 CPU 병목이면 21h가 늘어난다 — 증가폭이 크면 `--num-workers` 상향 후 재측정)
3. ②가 실패하면 **본 학습을 시작하지 않는다**(증강 0회로 21시간을 태우는 것을 막는 게 이 게이트의 유일한 목적)
4. 통과하면 `AUG_DELAY=<Step 2에서 구한 값> bash shell_scripting/shell22.sh` 본 학습(≈21h, `SMOKE` 미설정 시 `--aug-schedule exp`로 돈다). **`AUG_DELAY`가 없으면 스크립트가 즉시 exit 2로 거부한다** — 서버에서 헛걸음하지 않도록 값을 미리 준비해서 넘길 것.

---

### Task 9: 판정 리포트 + 문서 정정

**Files:**
- Create: `training_anal/2026-<완주일>-aug-analysis.md`
- Modify: `training_anal/2026-07-13-run10-longtrain-analysis.md`(정정 배너), `strategy/implementation.md`(§6 완료 처리), `strategy/augmentation.md`(구현 상태 갱신), `CLAUDE.md`(§10에서 항목 제거)

**Interfaces:**
- Consumes: Task 8 ON arm의 `val_loss_log`, baseline의 `val_loss_log`

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

| 태스크 | 어디서 | 누가 |
|---|---|---|
| 1~6 (코드·테스트 작성) | 로컬 (문법 검사만) | Claude |
| **7 (서버 검증 게이트)** | **서버, 큐 밖** | **사람** — `bash scripts/verify_aug.sh` |
| 8 Step 1~5 (슬롯 작성) | 로컬 | Claude |
| **8 Step 6 (스모크·본학습)** | **서버, 큐** | **사람** — `SMOKE=1 bash shell22.sh` → 본 학습 |
| 9 (리포트·문서) | 로컬 | Claude |

GPU 예산: baseline 재사용 가능 시 **≈21h**(ON arm 1개), 재사용 불가 시 ≈42h.
