# Resume Training (이어학습) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** SDUM 학습을 마지막 완료 epoch의 checkpoint에서 이어서(`--resume`) 재개한다. 중단→`recon_eval` 채점→재개 loop 지원.

**Architecture:** `train()`에 `--resume` 경로 추가. `model.pt`(마지막 epoch) 로드 → model/optimizer/start_epoch/best_val_loss 복원. cosine scheduler는 **직렬화 없이 fast-forward**로 재현(base-lr 시점에 생성 후 `start_epoch × steps_per_epoch`번 step). scheduler 생성·steps 계산·config 가드·checkpoint 로드를 순수 helper로 분리해 GPU 없이 단위 테스트. `save_model`은 **불변**(이미 epoch/args/optimizer/best_val_loss 저장).

**Tech Stack:** PyTorch 2.3.1+cu121, numpy, pytest. Python 3.10.

## Global Constraints

- Pascal(GTX 1080 8GB) → **FP32 전용**(AMP/BF16 금지), **batch=1**.
- **GPU 실행은 `shell_scripting/` 큐로만** (비-GPU: pytest·git은 바로 실행).
- `save_model`(train_part.py:84-97)·`recon_eval.py`·`utils/common/metrics.py`는 **수정 금지**(제출 호환).
- seed=430 고정. resume는 `(checkpoint, seed)`로 결정론적(무중단 run과 byte-동일은 목표 아님).
- resume 범위 = **원래 계획 epoch까지만**(cosine horizon = 원래 `num_epochs`). 끝난 run 연장·mid-epoch 정확재개·자동resume은 범위 밖.
- 커밋 identity는 repo local config(`kindofwater`). 커밋 메시지 끝에 `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

### Task 1: `build_scheduler` helper 추출 (behavior-preserving refactor)

기존 `train()` 안 인라인 scheduler 생성 로직을 순수 helper로 뽑아 테스트·재사용 가능하게 한다. 동작 불변.

**Files:**
- Modify: `utils/learning/train_part.py` (현재 인라인 scheduler = 라인 129-140 부근)
- Test: `tests/test_resume.py` (신규)

**Interfaces:**
- Produces: `build_scheduler(optimizer, args, len_loader) -> torch.optim.lr_scheduler.LambdaLR | None`
  - `args.model=='sdum'` & `args.warmup_iters>0` 일 때만 LambdaLR(linear warmup→cosine), 아니면 `None`.

- [ ] **Step 1: 실패 테스트 작성** — `tests/test_resume.py`

```python
import math
from types import SimpleNamespace
import torch
from utils.learning.train_part import build_scheduler


def _dummy_opt(lr):
    p = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.AdamW([p], lr=lr)


def test_build_scheduler_none_for_varnet():
    args = SimpleNamespace(model="varnet", warmup_iters=200, num_epochs=2)
    assert build_scheduler(_dummy_opt(1e-3), args, 100) is None


def test_build_scheduler_none_when_warmup_zero():
    args = SimpleNamespace(model="sdum", warmup_iters=0, num_epochs=2)
    assert build_scheduler(_dummy_opt(1e-3), args, 100) is None


def test_build_scheduler_warmup_then_cosine():
    args = SimpleNamespace(model="sdum", warmup_iters=10, num_epochs=1)
    opt = _dummy_opt(2e-4)
    sch = build_scheduler(opt, args, 100)          # total_iters = 100
    assert sch is not None
    # 생성 직후(step 0): lr = base * (0+1)/warmup
    assert abs(opt.param_groups[0]["lr"] - 2e-4 * (1 / 10)) < 1e-12
    # warmup 끝(step 10)까지: cosine progress 0 → lr = base
    for _ in range(10):
        sch.step()
    assert abs(opt.param_groups[0]["lr"] - 2e-4) < 1e-9
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_resume.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_scheduler'`

- [ ] **Step 3: helper 추가 + train() 배선**

`utils/learning/train_part.py` 상단(함수 밖, `save_model` 위 등 아무 모듈 스코프)에 추가:

```python
def build_scheduler(optimizer, args, len_loader):
    """SDUM & warmup_iters>0이면 linear warmup→cosine LambdaLR, 아니면 None(상수 LR)."""
    if getattr(args, "model", "varnet") != "sdum" or getattr(args, "warmup_iters", 0) <= 0:
        return None
    total_iters = args.num_epochs * len_loader
    warmup = args.warmup_iters

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_iters - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

그리고 `train()` 안의 기존 인라인 블록(라인 129-140 부근)을 아래로 교체:

```python
    scheduler = build_scheduler(optimizer, args, len(train_loader))
    if scheduler is not None:
        print(f"[scheduler] warmup={args.warmup_iters} → cosine, "
              f"total_iters={args.num_epochs * len(train_loader)}, base_lr={args.lr}")
```

- [ ] **Step 4: 통과 확인**

Run: `pytest tests/test_resume.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: 커밋**

```bash
git add utils/learning/train_part.py tests/test_resume.py
git commit -m "refactor(train): extract build_scheduler helper (behavior-preserving, testable)"
```

---

### Task 2: `steps_per_epoch` + fast-forward가 올바른 lr에 도달함을 검증

resume 시 scheduler를 `start_epoch × steps_per_epoch`번 step해 재개 지점 lr로 맞춘다. `--max-iters`면 epoch당 step이 `len_loader`가 아니라 `min(max_iters, len_loader)`임을 반영.

**Files:**
- Modify: `utils/learning/train_part.py`
- Test: `tests/test_resume.py`

**Interfaces:**
- Consumes: `build_scheduler` (Task 1)
- Produces: `steps_per_epoch(args, len_loader) -> int`

- [ ] **Step 1: 실패 테스트 작성** — `tests/test_resume.py`에 append

```python
from utils.learning.train_part import steps_per_epoch


def test_steps_per_epoch():
    assert steps_per_epoch(SimpleNamespace(max_iters=0), 100) == 100
    assert steps_per_epoch(SimpleNamespace(max_iters=30), 100) == 30
    assert steps_per_epoch(SimpleNamespace(max_iters=200), 100) == 100


def test_fast_forward_lands_at_correct_lr():
    args = SimpleNamespace(model="sdum", warmup_iters=10, num_epochs=2, max_iters=0)
    len_loader = 50
    k = 1 * steps_per_epoch(args, len_loader)       # 1 epoch fast-forward = 50 step
    opt = _dummy_opt(2e-4)
    sch = build_scheduler(opt, args, len_loader)    # total_iters = 100
    for _ in range(k):
        sch.step()
    total = args.num_epochs * len_loader
    progress = (k - args.warmup_iters) / max(1, total - args.warmup_iters)
    expected = 2e-4 * 0.5 * (1.0 + math.cos(math.pi * progress))
    assert abs(opt.param_groups[0]["lr"] - expected) < 1e-9
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_resume.py -k "steps_per_epoch or fast_forward" -v`
Expected: FAIL — `cannot import name 'steps_per_epoch'`

- [ ] **Step 3: helper 추가**

`utils/learning/train_part.py`에 추가:

```python
def steps_per_epoch(args, len_loader):
    """--max-iters면 epoch당 실제 train step은 min(max_iters, len_loader)."""
    mi = getattr(args, "max_iters", 0)
    return min(mi, len_loader) if mi and mi > 0 else len_loader
```

- [ ] **Step 4: 통과 확인**

Run: `pytest tests/test_resume.py -k "steps_per_epoch or fast_forward" -v`
Expected: PASS (2 passed)

- [ ] **Step 5: 커밋**

```bash
git add utils/learning/train_part.py tests/test_resume.py
git commit -m "feat(train): steps_per_epoch + verify scheduler fast-forward lands at correct lr"
```

---

### Task 3: `check_resume_compat` 가드

resume 시 저장 args와 현재 args의 아키텍처·horizon 일치를 강제(다르면 즉시 ValueError).

**Files:**
- Modify: `utils/learning/train_part.py`
- Test: `tests/test_resume.py`

**Interfaces:**
- Produces: `check_resume_compat(saved_args, args) -> None` (불일치 시 `ValueError`)

- [ ] **Step 1: 실패 테스트 작성** — append

```python
import pytest
from utils.learning.train_part import check_resume_compat


def _args(**kw):
    base = dict(model="sdum", num_cascades=4, channels=[16, 32], num_epochs=4, warmup_iters=200)
    base.update(kw)
    return SimpleNamespace(**base)


def test_check_resume_compat_ok():
    check_resume_compat(_args(), _args())          # no raise


def test_check_resume_compat_arch_mismatch():
    with pytest.raises(ValueError):
        check_resume_compat(_args(num_cascades=4), _args(num_cascades=6))


def test_check_resume_compat_horizon_mismatch():
    with pytest.raises(ValueError):
        check_resume_compat(_args(num_epochs=4), _args(num_epochs=2))
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_resume.py -k check_resume_compat -v`
Expected: FAIL — `cannot import name 'check_resume_compat'`

- [ ] **Step 3: helper 추가**

```python
def check_resume_compat(saved_args, args):
    """resume 시 저장 config와 현재 config가 일치해야 함(아키텍처·cosine horizon)."""
    for f in ("model", "num_cascades", "channels", "num_epochs", "warmup_iters"):
        s, c = getattr(saved_args, f, None), getattr(args, f, None)
        if s != c:
            raise ValueError(
                f"--resume 불일치: '{f}' 저장={s} 현재={c}. 같은 config로 재개해야 함(원래 계획까지만)."
            )
```

- [ ] **Step 4: 통과 확인**

Run: `pytest tests/test_resume.py -k check_resume_compat -v`
Expected: PASS (3 passed)

- [ ] **Step 5: 커밋**

```bash
git add utils/learning/train_part.py tests/test_resume.py
git commit -m "feat(train): check_resume_compat guard (arch/horizon mismatch → error)"
```

---

### Task 4: `load_resume_checkpoint` + `load_val_log` helpers

checkpoint 로드(존재확인 + compat 가드)와 val_loss_log 로드(append용)를 순수 helper로. GPU 불필요 — fake checkpoint dict로 테스트.

**Files:**
- Modify: `utils/learning/train_part.py`
- Test: `tests/test_resume.py`

**Interfaces:**
- Consumes: `check_resume_compat` (Task 3)
- Produces:
  - `load_resume_checkpoint(exp_dir, args) -> dict` (`model.pt` 로드; 없으면 `FileNotFoundError`; compat 가드)
  - `load_val_log(file_path) -> np.ndarray` (shape `(k,2)`; `file_path + '.npy'` 없으면 `(0,2)`)

- [ ] **Step 1: 실패 테스트 작성** — append

```python
import numpy as np
from pathlib import Path
from utils.learning.train_part import load_resume_checkpoint, load_val_log


def test_load_resume_checkpoint_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_resume_checkpoint(tmp_path, _args())


def test_load_resume_checkpoint_roundtrip(tmp_path):
    saved = _args()
    torch.save(
        {"epoch": 2, "args": saved, "model": {}, "optimizer": {},
         "best_val_loss": 0.11, "exp_dir": str(tmp_path)},
        tmp_path / "model.pt",
    )
    ckpt = load_resume_checkpoint(tmp_path, _args())     # compat OK
    assert ckpt["epoch"] == 2 and abs(ckpt["best_val_loss"] - 0.11) < 1e-9


def test_load_resume_checkpoint_compat_fail(tmp_path):
    torch.save({"epoch": 2, "args": _args(num_cascades=4), "model": {},
                "optimizer": {}, "best_val_loss": 0.11, "exp_dir": str(tmp_path)},
               tmp_path / "model.pt")
    with pytest.raises(ValueError):
        load_resume_checkpoint(tmp_path, _args(num_cascades=6))


def test_load_val_log(tmp_path):
    base = str(tmp_path / "vl")
    assert load_val_log(base).shape == (0, 2)
    np.save(base, np.array([[0, 0.12]]))                 # base + '.npy'
    out = load_val_log(base)
    assert out.shape == (1, 2) and abs(out[0, 1] - 0.12) < 1e-9
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_resume.py -k "resume_checkpoint or val_log" -v`
Expected: FAIL — `cannot import name 'load_resume_checkpoint'`

- [ ] **Step 3: helper 추가**

`utils/learning/train_part.py` (상단 import에 `import os`, `import numpy as np`는 이미 있음):

```python
def load_resume_checkpoint(exp_dir, args):
    """model.pt(마지막 epoch) 로드 + config 일치 확인. 없으면 FileNotFoundError."""
    ckpt_path = exp_dir / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"--resume인데 checkpoint 없음: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    check_resume_compat(ckpt["args"], args)
    return ckpt


def load_val_log(file_path):
    """val_loss_log 이어쓰기용. np.save는 '.npy'를 붙이므로 그 경로를 확인."""
    p = file_path + ".npy"
    return np.load(p) if os.path.exists(p) else np.empty((0, 2))
```

- [ ] **Step 4: 통과 확인**

Run: `pytest tests/test_resume.py -k "resume_checkpoint or val_log" -v`
Expected: PASS (4 passed)

- [ ] **Step 5: 커밋**

```bash
git add utils/learning/train_part.py tests/test_resume.py
git commit -m "feat(train): load_resume_checkpoint + load_val_log helpers"
```

---

### Task 5: `--resume` 배선 (train.py + train()) + GPU 스모크

helper들을 `train()`에 연결하고 `--resume` 플래그를 노출. 순서(base-lr optimizer → scheduler 생성 → optimizer state 로드 → fast-forward)를 지킨다. 이 통합은 GPU가 필요하므로 pytest 대신 **큐 스모크**로 검증.

**Files:**
- Modify: `train.py` (인자 파싱, `--max-iters`/`--run-tag` 근처)
- Modify: `utils/learning/train_part.py` (`train()` 라인 122-146 부근)

**Interfaces:**
- Consumes: `build_scheduler`, `steps_per_epoch`, `load_resume_checkpoint`, `load_val_log` (Task 1-4)

- [ ] **Step 1: `--resume` 플래그 추가** — `train.py` `--run-tag` 라인 다음에

```python
    parser.add_argument('--resume', action='store_true', help='<net_name>/checkpoints/model.pt(마지막 epoch)에서 이어학습. 같은 config로만.')
```

- [ ] **Step 2: `train()` 배선** — `utils/learning/train_part.py`

(a) model 생성·`model.to(device)` 직후에 resume checkpoint 로드 + model 복원:

```python
    resume_ckpt = None
    if getattr(args, "resume", False):
        resume_ckpt = load_resume_checkpoint(args.exp_dir, args)
        model.load_state_dict(resume_ckpt["model"])
```

(b) 기존 `best_val_loss = 1.` / `start_epoch = 0` 를 resume 반영으로 교체:

```python
    best_val_loss = 1.
    start_epoch = 0
    if resume_ckpt is not None:
        start_epoch = resume_ckpt["epoch"]
        best_val_loss = resume_ckpt["best_val_loss"]
        print(f"[resume] {args.exp_dir/'model.pt'} → start_epoch={start_epoch} best_val={best_val_loss:.4f}")
```

(c) `scheduler = build_scheduler(...)`(Task 1에서 교체한 블록) **다음**에 optimizer state 로드 + fast-forward 추가:

```python
    if resume_ckpt is not None:
        optimizer.load_state_dict(resume_ckpt["optimizer"])   # scheduler 생성 후에 로드(initial_lr=base 유지)
        if scheduler is not None:
            for _ in range(start_epoch * steps_per_epoch(args, len(train_loader))):
                scheduler.step()
            print(f"[resume] scheduler fast-forward {start_epoch}ep, lr={optimizer.param_groups[0]['lr']:.3e}")
```

(d) `val_loss_log = np.empty((0, 2))` 를 resume 시 이어쓰기로 교체 — 기존 fname 계산(`run_tag`)을 재사용:

```python
    run_tag = getattr(args, "run_tag", "")
    fname = f"val_loss_log_{run_tag}" if run_tag else "val_loss_log"
    val_log_path = os.path.join(args.val_loss_dir, fname)
    val_loss_log = load_val_log(val_log_path) if resume_ckpt is not None else np.empty((0, 2))
```

> ⚠️ 주의: epoch 루프 안에 이미 있는 `fname`/`file_path` 계산과 중복되지 않게, 루프 안 저장부는 `val_log_path`를 재사용하도록 정리(같은 값). start_epoch==num_epochs면 루프가 비어 즉시 종료됨(이미 완료된 run) — 정상.

- [ ] **Step 3: 단위 테스트 회귀 확인 (비-GPU)**

Run: `pytest tests/test_resume.py -v`
Expected: PASS (전체 통과 — helper 회귀 없음)

Run: `python -c "import train; import utils.learning.train_part"` (문법/임포트)
Expected: 에러 없음

- [ ] **Step 4: GPU 스모크 (큐)** — `shell_scripting/` 슬롯에 작성해 실행

빈 슬롯(예: `shellB.sh`, `cp _template.sh`)에 2개 명령을 순차로:

```bash
# 1) 2-epoch 계획 중 epoch 0만 돌고 죽은 상황을 흉내: -e 2 지만 아래를 epoch0 후 Ctrl-C/kill
python train.py --model sdum --num-cascades 4 --channels 16 32 \
    -b 1 -l 2.4e-4 -e 2 -r 50 --max-iters 60 --readout-crop --warmup-iters 20 \
    -n sdum_resume_smoke --run-tag resume_smoke -t /root/Data/train/ -v /root/Data/val/
# → epoch0 끝(val 저장)나면 kill. 그 뒤 아래로 재개:
python train.py --model sdum --num-cascades 4 --channels 16 32 \
    -b 1 -l 2.4e-4 -e 2 -r 50 --max-iters 60 --readout-crop --warmup-iters 20 --resume \
    -n sdum_resume_smoke --run-tag resume_smoke -t /root/Data/train/ -v /root/Data/val/
```

Expected(로그 확인):
- 재개 실행이 `[resume] ... start_epoch=1` 과 `[resume] scheduler fast-forward 1ep, lr=...` 출력.
- epoch 루프가 `Epoch #  1` 부터 시작(epoch0 재실행 안 함).
- loss 유한, `val_loss_log_resume_smoke.npy`가 2 epoch(0,1) 행을 가짐(append 성공).
- 재개 lr이 무중단 run의 epoch1 시작 lr과 근사.

- [ ] **Step 5: 커밋**

```bash
git add train.py utils/learning/train_part.py
git commit -m "feat(train): --resume — load last checkpoint, fast-forward scheduler, append val_log"
```

---

### Task 6: 워크플로우 문서 — 중단→eval→재개

`strategy/collaboration_workflow.md`에 절차 추가.

**Files:**
- Modify: `strategy/collaboration_workflow.md`

- [ ] **Step 1: "고려할 점" 섹션 근처에 절차 추가**

```markdown
## 학습 중단 → 채점 → 재개 (resume)

긴 학습을 중간에 끊고 채점해보거나, 죽은 학습을 이어갈 때:

1. **중단**: 큐 스킵(다음 슬롯으로) 또는 `pkill -f train.py`. epoch 경계 저장이라 **직전 완료 epoch**의 `model.pt`가 남는다(진행 중이던 epoch은 버려짐).
2. **채점**: `python recon_eval.py -n <net_name> -p /root/Data/leaderboard` → SSIM_full/bbox/시간 3축.
3. **재개**: 같은 명령에 `--resume`만 추가해 다시 큐로. `-n`·config(`--num-cascades`/`--channels`/`--num-epochs`/`--warmup-iters`)는 **원래와 동일해야** 함(다르면 거부됨). cosine은 원래 horizon 이어서 감쇠.

> resume는 "원래 계획 epoch까지" 이어감. 더 길게 하려면 fresh run으로 새로.
```

- [ ] **Step 2: 커밋**

```bash
git add strategy/collaboration_workflow.md
git commit -m "docs(workflow): 학습 중단→recon_eval 채점→--resume 재개 절차"
```

---

## Self-Review (플랜 작성자 체크)

- **Spec coverage**: 스펙 §1 트리거(Task 5), §2 복원(Task 4·5), §3 scheduler fast-forward(Task 1·2·5), §4 val_log append(Task 4·5), §5 재현성(Global Constraints), §6 config 가드(Task 3·4), §7 워크플로우(Task 6), 테스트(Task 1-5). ✅ 전부 매핑됨.
- **Placeholder scan**: 모든 코드 스텝에 실제 코드 존재, "TODO/적절히 처리" 없음. GPU 스모크(Task5-4)는 pytest 불가라 큐 절차+기대 로그로 명시. ✅
- **Type consistency**: `build_scheduler`/`steps_per_epoch`/`check_resume_compat`/`load_resume_checkpoint`/`load_val_log` 시그니처가 Task 1-4 정의와 Task 5 사용에서 일치. `resume_ckpt` 키(`model/optimizer/epoch/best_val_loss/args`)는 `save_model`(train_part.py:84-97) 저장 키와 동일. ✅
