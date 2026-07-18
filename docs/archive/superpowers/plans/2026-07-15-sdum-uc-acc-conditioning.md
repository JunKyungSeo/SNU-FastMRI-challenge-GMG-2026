# SDUM Universal Conditioning (acc4/acc8 + cascade-index) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** SDUM cascade에 Universal Conditioning을 켜서, acc4/acc8(protocol label)과 cascade-index(timestep)를 Restormer 블록에 주입한다. 단일 `--uc` 플래그로 on/off 하며 기본은 off(baseline 보존).

**Architecture:** NVIDIA에서 vendoring된 UC 배선(`restormer.py`의 `time_embed`/`label_emb`/GDFN `emb_layers`)은 이미 완전하나 config에서 꺼져 있고, cascade-index 경로는 adapter가 `timestep`을 안 넘겨 죽어 있다. 이 계획은 (1) config factory가 `uc` 인자로 `time_cond`+`label_cond`를 함께 켜게 하고, (2) adapter가 `timestep=0`을 넘겨 cascade-index를 살리고 3-튜플 리턴을 처리하며, (3) `--uc` 플래그를 학습·추론·resume 배선에 연결한다. acc 값은 기존 마스크 휴리스틱(`_extract_acc`)을 그대로 쓴다 — 데이터 파이프라인은 손대지 않는다.

**Tech Stack:** PyTorch, MONAI(complex/fft), 기존 SDUM vendored 코드(`utils/model/sdum/`), pytest(`scripts/`).

## Global Constraints

- **기본 off:** `--uc` 미지정 시 `time_cond=label_cond=False`. 기존 학습·체크포인트·baseline 동작이 바이트 동일하게 보존돼야 한다.
- **단일 플래그:** label(acc)과 cascade-index를 분리하지 않는다. `--uc` 하나로 둘 다 켠다.
- **`time_cond`는 `label_cond`의 전제:** `emb_channels`가 `time_cond`에만 물려 있어([restormer.py:659](../../../utils/model/sdum/restormer.py#L659)) `label_cond`만 켜면 `nn.Linear(32, None)`으로 크래시. 따라서 `uc`는 항상 둘을 동시에 켠다.
- **acc 소스 = 마스크 휴리스틱:** `_extract_acc`([adapter.py:16](../../../utils/model/sdum/adapter.py#L16)) 유지. transform/train/test 언패킹은 수정 금지.
- **batch_size=1 전제:** acc·label을 배치 0번으로 확정하는 현행 동작 유지. batch>1 가드는 범위 밖(사용자 확정: 배치 키울 일 없음).
- **로컬 torch 없음:** 모든 `pytest`/smoke run은 **GPU 환경(Colab/서버)에서 실행**. 로컬에선 편집만.
- **UC 경계 resume 불가:** UC on/off는 파라미터 집합이 달라 서로 resume 불가. `--resume` 호환성 검사에 `uc`를 포함시켜 조용한 오작동 대신 명시적 에러를 낸다.
- **마스크 증강 동시 사용 금지:** `_aug_mask`가 frac을 낮춰 acc4→acc8 오판을 유발. UC 실험은 증강 off 상태에서만(현재 기본 off). 이 제약을 adapter 주석과 `--uc` help에 명시.

---

## File Structure

- `utils/model/sdum/config.py` — `tiny_8gb_args`에 `uc` 인자 추가. `uc=True`면 `time_cond`+`label_cond`를 켠다. (UC 설정의 단일 진입점)
- `utils/model/sdum/adapter.py` — `SDUMReconModel`이 `self.uc`를 저장하고, UC 시 `timestep=0`을 넘기고 3-튜플 리턴을 언패킹. (모델 호출 계약)
- `train.py` — `--uc` argparse 플래그. (사용자 진입점)
- `utils/learning/train_part.py` — `tiny_8gb_args(..., uc=args.uc)` 전달 + `check_resume_compat`에 `uc` 추가. (학습 배선)
- `utils/learning/test_part.py` — `forward`/`load_model`이 `uc`를 전달·복원. (추론 배선)
- `scripts/test_sdum_uc.py` — UC 전용 pytest (빌드/forward/conditioning-active/off-보존). (검증)

---

### Task 1: config factory가 `uc` 인자로 UC를 켠다

**Files:**
- Modify: `utils/model/sdum/config.py:53-54` (`tiny_8gb_args`)
- Test: `scripts/test_sdum_uc.py` (신규)

**Interfaces:**
- Consumes: 기존 `SdumArgs` 데이터클래스(`time_cond`/`label_cond` 필드 존재, [config.py:36-37](../../../utils/model/sdum/config.py#L36)).
- Produces: `tiny_8gb_args(uc: bool = False, **overrides) -> SdumArgs`. `uc=True`면 반환된 args의 `time_cond`와 `label_cond`가 둘 다 `True`. `uc=False`(기본)면 둘 다 `False`. `time_cond`/`label_cond`를 `overrides`로 명시하면 그 값이 우선.

- [ ] **Step 1: Write the failing test**

Create `scripts/test_sdum_uc.py`:

```python
import torch
from utils.model.sdum.config import tiny_8gb_args
from utils.model.sdum.adapter import build_sdum

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_uc_flag_sets_both_conds():
    off = tiny_8gb_args(num_cascades=2, channels=[16, 32])
    assert off.time_cond is False and off.label_cond is False
    on = tiny_8gb_args(num_cascades=2, channels=[16, 32], uc=True)
    assert on.time_cond is True and on.label_cond is True


def test_uc_explicit_override_wins():
    # uc=True지만 time_cond를 명시로 끄면 명시가 우선(회귀 방지용 계약)
    a = tiny_8gb_args(uc=True, time_cond=False)
    assert a.time_cond is False and a.label_cond is True
```

- [ ] **Step 2: Run test to verify it fails**

Run (GPU 환경): `pytest scripts/test_sdum_uc.py::test_uc_flag_sets_both_conds -v`
Expected: FAIL — `tiny_8gb_args() got an unexpected keyword argument 'uc'`

- [ ] **Step 3: Write minimal implementation**

Modify `utils/model/sdum/config.py`, replace `tiny_8gb_args`:

```python
def tiny_8gb_args(uc: bool = False, **overrides) -> SdumArgs:
    # uc=True → UC 켬(time_cond+label_cond). time_cond는 emb_channels 할당 전제라 반드시 동반.
    # 명시 override가 있으면 그 값이 우선(setdefault).
    if uc:
        overrides.setdefault("time_cond", True)
        overrides.setdefault("label_cond", True)
    return replace(SdumArgs(), **overrides) if overrides else SdumArgs()
```

- [ ] **Step 4: Run tests to verify they pass**

Run (GPU 환경): `pytest scripts/test_sdum_uc.py::test_uc_flag_sets_both_conds scripts/test_sdum_uc.py::test_uc_explicit_override_wins -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add utils/model/sdum/config.py scripts/test_sdum_uc.py
git commit -m "feat(sdum): tiny_8gb_args uc flag enables time_cond+label_cond"
```

---

### Task 2: adapter가 cascade-index를 배선하고 3-튜플 리턴을 처리한다

**Files:**
- Modify: `utils/model/sdum/adapter.py:22-59` (`SDUMReconModel.__init__`, `forward`)
- Test: `scripts/test_sdum_uc.py` (Task 1에서 생성)

**Interfaces:**
- Consumes: `tiny_8gb_args(uc=True)` (Task 1). `Cascaded_SkipConnected_MRI_Recon.forward(x, mask, mask_type, acc_factor, acq_type, ..., timestep=None)` — `timestep is not None`이면 `(x, cas_skips, sensitivity_maps)` 3-튜플, 아니면 텐서를 반환([latent_recon.py:138-141](../../../utils/model/sdum/latent_recon.py#L138)).
- Produces: `SDUMReconModel.forward(masked_kspace, mask) -> (B,384,384)` 텐서. UC on이면 내부에서 `timestep=0`을 넘겨 `actual_timestep=i`(cascade 번호)로 조건화. UC off면 기존 텐서-리턴 경로를 그대로 사용(바이트 동일).

- [ ] **Step 1: Write the failing test**

Append to `scripts/test_sdum_uc.py`:

```python
def _fake_batch(B=1, C=15, H=384, W=372):
    ksp = torch.randn(B, C, H, W, 2, dtype=torch.float32)
    mask = torch.zeros(B, 1, 1, W, 1, dtype=torch.uint8)
    mask[..., ::4, :] = 1
    mask[..., W // 2 - 15 : W // 2 + 15, :] = 1
    ksp = ksp * mask.permute(0, 1, 4, 2, 3).reshape(B, 1, 1, W, 1).to(ksp.dtype)
    return ksp.to(DEV), mask.to(DEV)


def test_uc_forward_shape_and_finite():
    model = build_sdum(tiny_8gb_args(num_cascades=2, channels=[16, 32], uc=True)).eval().to(DEV)
    ksp, mask = _fake_batch()
    with torch.no_grad():
        out = model(ksp, mask)
    assert out.shape == (1, 384, 384), out.shape
    assert torch.isfinite(out).all()


def test_uc_off_still_forwards():
    # baseline 경로 보존: uc off도 정상 (B,384,384)
    model = build_sdum(tiny_8gb_args(num_cascades=2, channels=[16, 32])).eval().to(DEV)
    ksp, mask = _fake_batch()
    with torch.no_grad():
        out = model(ksp, mask)
    assert out.shape == (1, 384, 384) and torch.isfinite(out).all()


def test_uc_conditioning_is_actually_used():
    """silent no-op 방지: UC 파라미터(time_embed/label_emb/emb_layers)에 grad가 흘러야 한다.
    grad가 전부 0/None이면 조건화가 forward에서 안 쓰인 것(과거 함정)."""
    model = build_sdum(tiny_8gb_args(num_cascades=2, channels=[16, 32], uc=True)).train().to(DEV)
    ksp, mask = _fake_batch()
    tgt = torch.randn(1, 384, 384, device=DEV)
    out = model(ksp, mask)
    torch.nn.functional.l1_loss(out, tgt).backward()
    cond_named = [(n, p) for n, p in model.named_parameters()
                  if ("time_embed" in n or "label_emb" in n or "emb_layers" in n)]
    assert cond_named, "UC 파라미터가 생성되지 않음"
    nonzero = [n for n, p in cond_named
               if p.grad is not None and torch.any(p.grad != 0)]
    assert nonzero, "UC 파라미터에 grad가 안 흐름 → 조건화가 forward에서 미사용(silent no-op)"
```

- [ ] **Step 2: Run test to verify it fails**

Run (GPU 환경): `pytest scripts/test_sdum_uc.py::test_uc_forward_shape_and_finite -v`
Expected: FAIL — `TypeError: tuple indices must be integers` (또는 `too many indices`) — adapter가 3-튜플에 `out[:, 0]`을 적용해서 터짐

- [ ] **Step 3: Write minimal implementation**

Modify `utils/model/sdum/adapter.py`. In `__init__` (after `self.acq_type = ...`, ~line 30) add:

```python
        # UC(Universal Conditioning) on 여부. on이면 forward에서 timestep을 넘겨 cascade-index 조건화.
        # ⚠️ acc는 _extract_acc(마스크 frac) 휴리스틱. 마스크 증강을 켜면 frac이 낮아져 acc4→acc8 오판 →
        #    UC 실험은 마스크 증강 off에서만.
        self.uc = bool(getattr(args, "time_cond", False) or getattr(args, "label_cond", False))
```

In `forward`, replace the model-call block ([adapter.py:49-54](../../../utils/model/sdum/adapter.py#L49)):

```python
        if self.uc:
            # timestep=0 → latent_recon가 actual_timestep = 0*T + i = i (cascade 번호)로 사용.
            # timestep!=None이면 (x, cas_skips, sensitivity_maps) 3-튜플 반환 → [0]으로 텐서만.
            out = self.cascaded(x, mask_b, self.mask_type, acc, self.acq_type, timestep=0)
            out = out[0]
        else:
            out = self.cascaded(x, mask_b, self.mask_type, acc, self.acq_type)

        # non-flow 경로는 영상 (B,T,C,H,W,2) 반환
        out = out[:, 0]                           # T 제거 -> (B,C,H,W,2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run (GPU 환경): `pytest scripts/test_sdum_uc.py -v`
Expected: PASS (test_uc_forward_shape_and_finite, test_uc_off_still_forwards, test_uc_conditioning_is_actually_used 모두 포함 전부 통과)

- [ ] **Step 5: Commit**

```bash
git add utils/model/sdum/adapter.py scripts/test_sdum_uc.py
git commit -m "feat(sdum): adapter wires timestep=0 for cascade-index UC + 3-tuple return"
```

---

### Task 3: `--uc` 플래그를 학습·추론·resume 배선에 연결한다

**Files:**
- Modify: `train.py:42` 부근 (argparse), 
- Modify: `utils/learning/train_part.py:181` (`tiny_8gb_args` 호출), `train_part.py:133` (`check_resume_compat`)
- Modify: `utils/learning/test_part.py:26-27` (`load_model`), `test_part.py:86` (`forward`)
- Test: 통합 smoke run (아래 Step에 명시)

**Interfaces:**
- Consumes: `tiny_8gb_args(..., uc=...)` (Task 1), UC-aware `SDUMReconModel` (Task 2).
- Produces: `python train.py --model sdum --uc ...`가 UC-on 모델을 학습하고, 체크포인트의 `args.uc=True`가 저장된다. `test_part.load_model`은 저장된 `saved.uc`로 동일 아키텍처를 재구성한다. `--resume`은 `uc` 불일치 시 명시적 에러.

- [ ] **Step 1: train.py에 `--uc` 플래그 추가**

Modify `train.py`, `--fg-loss` 인자 아래([train.py:42](../../../train.py#L42))에 추가:

```python
    parser.add_argument('--uc', action='store_true',
        help='SDUM Universal Conditioning: acc4/acc8(label)+cascade-index(timestep) 조건화. '
             '기본 off. ⚠️ 마스크 증강과 동시 사용 금지(frac 변동→acc 오판). UC on/off는 resume 호환 안 됨.')
```

- [ ] **Step 2: train_part.py가 uc를 모델에 전달하고 resume 호환 검사에 포함**

Modify `utils/learning/train_part.py:181`:

```python
        sdum_args = tiny_8gb_args(num_cascades=args.num_cascades, channels=list(args.channels),
                                  uc=getattr(args, "uc", False))
```

Modify `utils/learning/train_part.py:133`, `check_resume_compat`의 필드 튜플에 `"uc"` 추가:

```python
    for f in ("model", "num_cascades", "channels", "num_epochs", "warmup_iters", "lr", "uc"):
```

- [ ] **Step 3: test_part.py가 저장된 uc로 모델을 재구성**

Modify `utils/learning/test_part.py:26-27` (`load_model`, SDUM 분기):

```python
        model = build_sdum(tiny_8gb_args(
            num_cascades=saved.num_cascades, channels=list(saved.channels),
            uc=getattr(saved, "uc", False)))
```

Modify `utils/learning/test_part.py:86` (`forward`):

```python
        sdum_args = tiny_8gb_args(num_cascades=args.num_cascades, channels=list(args.channels),
                                  uc=getattr(args, "uc", False))
```

- [ ] **Step 4: 통합 smoke run — UC 학습이 돌고 체크포인트에 uc가 저장되는지**

Run (GPU 환경, 데이터 경로는 실제 값으로):

```bash
python train.py --model sdum --uc --num-cascades 2 --channels 16 32 \
  -e 1 --max-iters 3 -n uc_smoke \
  -t /Data/train/ -v /Data/val/
```

Expected: 크래시 없이 1 epoch(3 iter) 학습 완료, `../result/uc_smoke/checkpoints/model.pt` 생성.

검증 스니펫 (GPU 환경):

```bash
python -c "import torch; a=torch.load('../result/uc_smoke/checkpoints/best_model.pt', weights_only=False)['args']; print('uc=', getattr(a,'uc',None))"
```

Expected 출력: `uc= True`

- [ ] **Step 5: resume 비호환이 명시적 에러를 내는지 확인**

Run (GPU 환경): UC-off 체크포인트를 UC-on으로 resume 시도 → `check_resume_compat`가 막아야 함.

```bash
python train.py --model sdum --num-cascades 2 --channels 16 32 -e 1 --max-iters 3 -n uc_off_smoke -t /Data/train/ -v /Data/val/
python train.py --model sdum --uc --resume --num-cascades 2 --channels 16 32 -e 1 --max-iters 3 -n uc_off_smoke -t /Data/train/ -v /Data/val/
```

Expected: 두 번째 명령이 `ValueError: --resume 불일치: 'uc' 저장=False 현재=True` 로 중단.

- [ ] **Step 6: Commit**

```bash
git add train.py utils/learning/train_part.py utils/learning/test_part.py
git commit -m "feat: --uc flag wires UC through train/infer/resume for SDUM"
```

---

### Task 4: UC on/off A/B 실험 실행 (효과 측정)

**Files:**
- 코드 변경 없음. 실행·기록만.

**Interfaces:**
- Consumes: Task 1-3 완료된 `--uc` 파이프라인.
- Produces: UC off vs on의 val(SSIM) 비교 수치. best_model 체크포인트 2개.

- [ ] **Step 1: off-arm 학습 (baseline)**

Run (GPU 환경, 실제 데이터·epoch로):

```bash
python train.py --model sdum --num-cascades 4 --channels 32 64 \
  -e 5 -n uc_off -t /Data/train/ -v /Data/val/
```

Expected: 5 epoch 완료, `val_loss_log_.npy`에 epoch별 val 기록. (마스크 증강 off 확인 — 기본값)

- [ ] **Step 2: on-arm 학습 (UC)**

Run (GPU 환경, off-arm과 동일 config + `--uc`):

```bash
python train.py --model sdum --uc --num-cascades 4 --channels 32 64 \
  -e 5 -n uc_on -t /Data/train/ -v /Data/val/
```

Expected: 5 epoch 완료. UC 파라미터 추가로 param 수 소폭 증가(로그의 모델 생성부에서 확인 가능).

- [ ] **Step 3: val 비교**

Run (GPU 환경):

```bash
python -c "
import numpy as np
off = np.load('../result/uc_off/val_loss_log_.npy')
on  = np.load('../result/uc_on/val_loss_log_.npy')
print('off best (1-SSIM):', off[:,1].min())
print('on  best (1-SSIM):', on[:,1].min())
print('delta SSIM:', off[:,1].min() - on[:,1].min())  # 양수면 UC가 개선
"
```

Expected: `delta SSIM`이 양수면 UC가 val을 개선. (논문 Table 3f 기준 +0.38dB 수준을 기대하되, knee 단일 modality라 조건 신호가 acc4/acc8 1비트뿐이므로 이보다 작을 수 있음.) 음수/무변화면 UC가 이 세팅에선 무이득 — 그 결과도 유효한 결론.

- [ ] **Step 4: recon_eval A/B (선택, 채점 정합 SSIM_full 확정)**

Run (GPU 환경): 두 best_model에 대해 `recon_eval.py`를 각각 돌려 실채점 SSIM_full·bbox 비교. (val은 학습 지표라 실채점과 다를 수 있음.)

---

## Self-Review

**1. Spec coverage:**
- 단일 `--uc` 플래그 → Task 3 Step 1. ✅
- 기본 off·baseline 보존 → Task 1(기본값 False), Task 2(uc off 분기 유지), Task 2 `test_uc_off_still_forwards`. ✅
- acc = 마스크 휴리스틱, 데이터 파이프라인 미수정 → 어느 태스크도 transform/load_data 수정 안 함. ✅
- cascade-index(timestep) + acc(label) 둘 다 → Task 1(둘 다 켬) + Task 2(timestep=0 배선). ✅
- `time_cond`가 `label_cond` 전제 → Task 1이 항상 둘을 동시에 켬. ✅
- resume 비호환 명시적 에러 → Task 3 Step 2(`check_resume_compat`에 uc) + Step 5(검증). ✅
- 추론 경로도 UC 적용 → Task 3 Step 3(`load_model`/`forward`가 uc 복원). ✅
- silent no-op 방지 → Task 2 `test_uc_conditioning_is_actually_used`(grad 흐름 검증). ✅
- 마스크 증강 동시 금지 경고 → adapter 주석(Task 2) + `--uc` help(Task 3). ✅
- 효과 측정 → Task 4 A/B. ✅

**2. Placeholder scan:** TBD/TODO/"적절히 처리" 없음. 모든 코드 스텝에 실제 코드·명령·기대 출력 포함. ✅

**3. Type consistency:**
- `tiny_8gb_args(uc=False, **overrides)` — Task 1 정의, Task 2/3에서 `uc=...`로 호출. 일치. ✅
- `SDUMReconModel.self.uc` — Task 2 정의·사용. 일치. ✅
- `timestep=0` 키워드 — `Cascaded_SkipConnected_MRI_Recon.forward(timestep=None)` 시그니처와 일치([latent_recon.py:106](../../../utils/model/sdum/latent_recon.py#L106)). ✅
- `getattr(args,"uc",False)` — argparse args(train)와 `SdumArgs`(config)에 모두 안전(하위호환). ✅

---

**참고 — 왜 데이터 파이프라인을 안 건드리나:** acc는 [adapter.py:47](../../../utils/model/sdum/adapter.py#L47)에서 마스크로 산출되고, cascade가 mask_type·acc·acq를 보고 protocol label을 [restormer.py:1287](../../../utils/model/sdum/restormer.py#L1287)에서 자체 계산한다. label 공간은 `accelerations=[4,8]`로 이미 준비돼 있다([config.py:46](../../../utils/model/sdum/config.py#L46)). 따라서 `label_cond`만 켜지면 acc4/acc8 조건화가 추가 배선 없이 살아난다.

**참고 — label stride 무해 확인:** [restormer.py:1287](../../../utils/model/sdum/restormer.py#L1287)의 label 인코딩은 `mask_idx * n_mask * n_acq + acc_idx * n_acq + acq_idx`. 현 설정(`n_mask=1, n_acq=1, n_acc=2`)에서 `label = acc_idx ∈ {0,1}`로 acc4/acc8이 정확히 분리된다. (mask family를 추가하면 stride가 어긋날 수 있으나 범위 밖.)
