# PromptMR+ MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 작지만 충실한 PromptMR+("+" 4개 기능 전부 ON)를 `utils/model/promptmr/`로 vendoring하고 `--model promptmr`로 배선하여, GTX 1080 8GB에서 학습·채점되게 만든다.

**Architecture:** 하이브리드 vendoring — 아키텍처는 공식 `hellopipu/PromptMR-plus`(v2)를 정본으로, 실행 형태는 이미 SNU 데이터로더에 맞춰진 2025 clone(`../2025_4th_clone/FastMRI_challenge_sMaRtIfy_final/utils/model/promptmr_plus.py`)을 base로 삼는다. PromptMR은 k-space native(자체 SME + data-consistency + 내부 instance-norm)라 어댑터가 얇다 — `forward(masked_kspace, mask) → (B,384,384)` 계약을 backbone이 사실상 그대로 만족한다.

**Tech Stack:** PyTorch 2.3.1+cu121, einops, 기존 vendored `utils/model/fastmri/`, reentrant gradient checkpointing.

**Spec:** `docs/superpowers/specs/2026-07-19-promptmr-plus-mvp-design.md`

## Global Constraints

_모든 task는 아래를 암묵적으로 포함한다. 값은 spec에서 그대로 옮김._

- **GPU = GTX 1080 8GB (Pascal) → FP32 전용** (AMP/BF16/FP16 금지), **batch=1**.
- **사전학습 가중치·외부 데이터 금지** — 처음부터 학습. HuggingFace 자동 다운로드 vendor 금지(PromptMR+ 코드는 random init이라 자동 다운로드 없음 → 준수).
- **knee 전용, Cartesian acc4/acc8, 출력 정확히 384×384.**
- **`recon_eval.py`·`utils/common/metrics.py`는 고정·수정 금지**(organizer 재실행). 채점 = `0.5·SSIM_full + 0.5·SSIM_bbox + 추론시간`.
- **환경**: Python 3.10.12, torch 2.3.1+cu121. **로컬(이 세션)엔 torch 없음** → 로컬 검증은 `python -m py_compile` + grep만. **실제 실행·shape·학습 검증은 GPU 서버에서 `shell_scripting/` 큐 슬롯으로만.**
- **pytest 금지**(사용자 지시). TDD의 "실패 테스트" 대신 각 로컬 task는 `py_compile`+검토, 각 GPU task는 shell 슬롯 실행+기대출력으로 검증한다.
- **커밋**: AI/Claude attribution 절대 금지. 커밋 전 커밋 주체(teammate) 확인. identity는 local config로만.
- **라이선스**: vendoring하는 PromptMR+ 유래 파일에 **Rutgers non-commercial 저작권 고지 주석 유지**.
- **파괴적 행동(학습 kill, 큐 구조 변경, 체크포인트 삭제 등)은 사전 허락**.

---

## File Structure

**신규 생성:**
- `utils/model/reentrant_wrapper.py` — reentrant gradient-checkpointing 래퍼(clone에서 그대로).
- `utils/model/promptmr/__init__.py` — 패키지 마커.
- `utils/model/promptmr/promptmr.py` — vendored PromptMR+ backbone(clone base + v2 정합 수정). 단일 파일(clone이 이미 `batched_mask_center` inline + `import fastmri` repo-local + 384-crop image 반환으로 변형 완료 상태라 분할 불필요·surgical).
- `utils/model/promptmr/config.py` — tiny config dict factory.
- `utils/model/promptmr/adapter.py` — `build_promptmr(args)` 빌더.
- `shell_scripting/` — 검증용 슬롯 3개(probe/smoke/eval).

**수정:**
- `utils/model/fastmri/coil_combine.py` + `__init__.py` — `sens_expand`/`sens_reduce` 추가(유일한 fastmri 격차).
- `train.py` — `--model`/arch 플래그.
- `utils/learning/train_part.py` — 모델 빌드·optimizer 분기.
- `utils/learning/test_part.py` — `load_model`·`forward` 분기.
- `scripts/reconstruct.py` — 대응 플래그.

**무수정:** `recon_eval.py`, `utils/common/metrics.py`, `utils/model/varnet.py`, `utils/data/load_data.py`, `utils/data/transforms.py`.

> **Spec 대비 refinement**(계획 중 clone 소스 확인으로 확정): (1) backbone은 단일 파일 `promptmr.py`(spec의 `promptmr_v2.py`+`blocks.py` 분할 대신 — clone이 이미 self-contained). (2) fastmri는 재사용하되 `sens_expand`/`sens_reduce` 2개만 추가(spec §2의 "재사용"에 이 격차 보완). spec §2를 이에 맞춰 갱신함.

---

## Task 1: fastmri에 sens_expand/sens_reduce 추가

PromptMR backbone이 `fastmri.sens_expand`/`fastmri.sens_reduce`(num_adj_slices 인자)를 쓰는데 repo fastmri엔 없다. clone의 정의(정본 PromptMR ops)를 그대로 추가한다.

**Files:**
- Modify: `utils/model/fastmri/coil_combine.py` (끝에 2함수 추가)
- Modify: `utils/model/fastmri/__init__.py:17` (export 추가)

**Interfaces:**
- Consumes: 기존 `fastmri.fft2c`, `fastmri.ifft2c`, `fastmri.complex_mul`, `fastmri.complex_conj`.
- Produces: `sens_expand(x, sens_maps, num_adj_slices=1) -> Tensor`, `sens_reduce(x, sens_maps, num_adj_slices=1) -> Tensor`. Task 3의 backbone이 사용.

- [ ] **Step 1: `coil_combine.py` 끝에 두 함수 추가**

`utils/model/fastmri/coil_combine.py` 파일 맨 끝(`rss_complex` 아래)에 append:

```python

def sens_expand(x: torch.Tensor, sens_maps: torch.Tensor, num_adj_slices: int = 1) -> torch.Tensor:
    """
    Coil Expand with sensitivity maps. (PromptMR: image -> multi-coil k-space)
    """
    _, c, _, _, _ = sens_maps.shape
    return fastmri.fft2c(fastmri.complex_mul(x.repeat_interleave(c // num_adj_slices, dim=1), sens_maps))


def sens_reduce(x: torch.Tensor, sens_maps: torch.Tensor, num_adj_slices: int = 1) -> torch.Tensor:
    """
    Coil Combine with sensitivity maps. (PromptMR: multi-coil k-space -> image)
    """
    b, c, h, w, _ = x.shape
    x = fastmri.ifft2c(x)
    x = fastmri.complex_mul(x, fastmri.complex_conj(sens_maps))
    return x.view(b, num_adj_slices, c // num_adj_slices, h, w, 2).sum(dim=2, keepdim=False)
```

- [ ] **Step 2: `__init__.py`의 coil_combine import에 두 이름 추가**

`utils/model/fastmri/__init__.py`의 line 17을 수정:

```python
from .coil_combine import rss, rss_complex, sens_expand, sens_reduce
```

- [ ] **Step 3: 로컬 컴파일 검증**

Run: `python -m py_compile utils/model/fastmri/coil_combine.py utils/model/fastmri/__init__.py`
Expected: 에러 없이 종료(exit 0). *(py_compile은 import를 실행하지 않으므로 torch 없어도 통과.)*

- [ ] **Step 4: Commit**

```bash
git add utils/model/fastmri/coil_combine.py utils/model/fastmri/__init__.py
git commit -m "feat(fastmri): sens_expand/sens_reduce 추가 (PromptMR용 coil expand/reduce)"
```

---

## Task 2: reentrant_wrapper.py vendoring

gradient checkpointing 래퍼. clone 파일을 그대로 가져온다.

**Files:**
- Create: `utils/model/reentrant_wrapper.py`

**Interfaces:**
- Produces: `class ReentrantWrapper(nn.Module)` — `__init__(module, use_checkpoint=None)`, `forward(*args)`, 런타임에 `.use_checkpoint` 속성 토글 가능. `self.training`이 True일 때만 checkpoint. Task 3의 backbone이 UpBlock·sens_net에서 사용.

- [ ] **Step 1: 파일 생성 (clone에서 그대로 복사)**

Create `utils/model/reentrant_wrapper.py` — source: `../2025_4th_clone/FastMRI_challenge_sMaRtIfy_final/utils/model/reentrant_wrapper.py`. 내용:

```python
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential


class ReentrantWrapper(nn.Module):

    DEFAULT_CHECKPOINT = True

    class _DummyWrapper(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, *args):
            x, dummy = args[:-1], args[-1]
            return self.module(*x)

    def __init__(
        self,
        module: nn.Module,
        use_checkpoint: Optional[bool] = None
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint or self.DEFAULT_CHECKPOINT
        self._dummy_tensor = nn.Parameter(torch.ones(1, requires_grad=True))

        self.module = self._DummyWrapper(module)

    def forward(self, *args):
        if self.use_checkpoint and self.training:
            return checkpoint(self.module, *args, self._dummy_tensor, use_reentrant=True)
        else:
            return self.module(*args, self._dummy_tensor)
```

- [ ] **Step 2: 로컬 컴파일 검증**

Run: `python -m py_compile utils/model/reentrant_wrapper.py`
Expected: exit 0.

- [ ] **Step 3: Commit**

```bash
git add utils/model/reentrant_wrapper.py
git commit -m "feat(model): reentrant gradient-checkpointing 래퍼 vendoring"
```

---

## Task 3: PromptMR+ backbone vendoring (clone base + v2 정합)

clone의 `promptmr_plus.py`(768줄, 완결형)를 `utils/model/promptmr/promptmr.py`로 복사하고 **4개 지점만 수정**한다. 나머지는 그대로.

**Files:**
- Create: `utils/model/promptmr/__init__.py`
- Create: `utils/model/promptmr/promptmr.py`

**Interfaces:**
- Consumes: `import fastmri`(repo-local; Task 1의 sens_expand/reduce 포함), `from utils.common.utils import center_crop`, `from utils.model.reentrant_wrapper import ReentrantWrapper`(Task 2), `einops`.
- Produces: `class PromptMR(nn.Module)` with `__init__(num_cascades, num_adj_slices, n_feat0, feature_dim, prompt_dim, sens_n_feat0, sens_feature_dim, sens_prompt_dim, len_prompt, prompt_size, n_enc_cab, n_dec_cab, n_skip_cab, n_bottleneck_cab, no_use_ca=False, ..., mask_center=True, learnable_prompt=False, adaptive_input=False, n_buffer=4, n_history=0, use_sens_adj=True)` 및 `forward(masked_kspace (B,C,H,W,2), mask (B,1,1,W,1), num_low_frequencies=None, mask_type=("cartesian",), use_checkpoint=True, compute_sens_per_coil=True) -> Tensor (B,384,384)`. Task 4/5가 사용.

- [ ] **Step 1: 패키지 마커 생성**

Create `utils/model/promptmr/__init__.py`:

```python
# PromptMR+ backbone package.
```

- [ ] **Step 2: backbone 파일 복사**

`../2025_4th_clone/FastMRI_challenge_sMaRtIfy_final/utils/model/promptmr_plus.py` 전체(768줄)를 `utils/model/promptmr/promptmr.py`로 복사한다. import 3줄(`import fastmri`, `from utils.common.utils import center_crop`, `from utils.model.reentrant_wrapper import ReentrantWrapper`)은 repo에서 그대로 resolve됨(train.py가 `utils/model/`을 sys.path에 삽입).

- [ ] **Step 3: 파일 맨 위에 라이선스·출처 헤더 추가**

`utils/model/promptmr/promptmr.py` 맨 첫 줄 위에 삽입:

```python
# PromptMR+ backbone. Architecture: hellopipu/PromptMR-plus (v2).
# Rutgers Non-commercial Research License (c) 2024 Rutgers (Docket #2025-032).
# Retain this notice. Non-commercial research use only.
# Vendored/adapted from ../2025_4th_clone/.../promptmr_plus.py, reconciled to
# official v2: PromptUnet reduction 2->4; ReLU(inplace=False) kept for reentrant
# checkpointing safety; forward() mask handling rewritten for SNU (B,1,1,W,1) mask.
```

- [ ] **Step 4: 수정 A — PromptUnet reduction 기본값 2→4 (v2 정합)**

`PromptUnet.__init__`의 인자 기본값 줄을 수정. 찾기:

```python
                reduction=2,  # modified by sMaRtIfy
```

교체:

```python
                reduction=4,  # official v2 default
```

- [ ] **Step 5: 수정 B — forward()의 SNU mask 재정렬 블록 제거**

`PromptMR.forward` 안에서 아래 블록을 찾는다:

```python
        # Shape aligning to SNU FastMRI dataloader
        if mask.shape[2] == 1:
            if num_low_frequencies is None:
                num_low_frequencies = torch.tensor([round(0.08 * mask.shape[3]) for _ in range(mask.shape[0])]).cuda(non_blocking=True) # TODO Batch implementation
                
            mask = torch.stack([mask.squeeze(2).clone() for _ in range(masked_kspace.shape[2])], dim=2)

        self.sens_net_ckp.use_checkpoint = use_checkpoint
```

아래로 교체(reshape·`.cuda()` 삭제, mask는 그대로 통과, num_low_frequencies=None 유지):

```python
        # SNU dataloader mask: (B,1,1,W,1) byte. 이미 masked_kspace (B,C,H,W,2)에
        # 대해 ACS 추출(mask[:,0,0,:,0])·data-consistency(W축 torch.where) 양쪽에서
        # broadcast되므로 변형 없이 그대로 넘긴다. num_low_frequencies=None →
        # KspaceACSExtractor가 mask에서 실제 ACS 폭을 직접 도출한다.
        self.sens_net_ckp.use_checkpoint = use_checkpoint
```

> ReLU inplace(수정 C)와 out_chans(else in_chans, v2와 동일)는 clone 그대로 두면 됨 — 추가 편집 없음.

- [ ] **Step 6: 로컬 컴파일 검증**

Run: `python -m py_compile utils/model/promptmr/promptmr.py utils/model/promptmr/__init__.py`
Expected: exit 0. *(shape·런타임 검증은 Task 7 GPU에서.)*

- [ ] **Step 7: 잔존 참조 점검**

Run: `grep -nE "\.cuda\(|modified by sMaRtIfy|Shape aligning" utils/model/promptmr/promptmr.py`
Expected: 출력 없음(3개 수정이 모두 반영됨).

- [ ] **Step 8: Commit**

```bash
git add utils/model/promptmr/__init__.py utils/model/promptmr/promptmr.py
git commit -m "feat(promptmr): PromptMR+ backbone vendoring (clone base, v2 정합·SNU mask 수정)"
```

---

## Task 4: config.py — tiny config factory

spec §3의 tiny config를 PromptMR kwargs dict로 반환. CLI가 덮을 수 있는 3개(num_cascades, feature_dim, n_feat0)만 인자로.

**Files:**
- Create: `utils/model/promptmr/config.py`

**Interfaces:**
- Produces: `promptmr_tiny_config(num_cascades=6, feature_dim=(24,32,40), n_feat0=16, **overrides) -> dict`. 반환 dict의 키는 전부 `PromptMR.__init__`의 유효 인자. Task 5가 사용.

- [ ] **Step 1: 파일 작성**

Create `utils/model/promptmr/config.py`:

```python
# PromptMR+ tiny config (spec §3): 8GB·FP32·batch=1용 축소판. "+" 4개 전부 ON.
from typing import Sequence


def promptmr_tiny_config(num_cascades: int = 6,
                         feature_dim: Sequence[int] = (24, 32, 40),
                         n_feat0: int = 16,
                         **overrides) -> dict:
    """PromptMR(**promptmr_tiny_config(...)) 형태로 쓰는 kwargs dict를 반환.

    CLI 노출 노브 = num_cascades / feature_dim(--channels) / n_feat0.
    나머지(prompt·sens·CAB 개수, "+" 플래그)는 MVP 고정.
    """
    cfg = dict(
        num_cascades=num_cascades,
        num_adj_slices=1,                 # MVP: 단일 슬라이스(adjacent-slice는 후속)
        n_feat0=n_feat0,
        feature_dim=list(feature_dim),
        prompt_dim=[8, 16, 24],
        sens_n_feat0=8,
        sens_feature_dim=[12, 16, 20],
        sens_prompt_dim=[8, 12, 16],
        len_prompt=[5, 5, 5],
        prompt_size=[16, 8, 4],
        n_enc_cab=[2, 3, 3],
        n_dec_cab=[2, 2, 3],
        n_skip_cab=[1, 1, 1],
        n_bottleneck_cab=3,
        no_use_ca=False,
        learnable_prompt=True,            # 충실: prompt bank 학습 가능(upstream 코드 기본 False)
        adaptive_input=True,              # +
        n_buffer=4,                       # +
        n_history=3,                      # + (tiny; 논문 기본 11)
        use_sens_adj=True,                # +
    )
    cfg.update(overrides)
    return cfg
```

- [ ] **Step 2: 로컬 컴파일 검증**

Run: `python -m py_compile utils/model/promptmr/config.py`
Expected: exit 0.

- [ ] **Step 3: Commit**

```bash
git add utils/model/promptmr/config.py
git commit -m "feat(promptmr): tiny config factory (8GB·full '+' ON)"
```

---

## Task 5: adapter.py — build_promptmr

CLI args → PromptMR 인스턴스. backbone이 계약을 그대로 만족하므로 forward 래핑 불필요.

**Files:**
- Create: `utils/model/promptmr/adapter.py`

**Interfaces:**
- Consumes: `PromptMR`(Task 3), `promptmr_tiny_config`(Task 4). `args`에서 `num_cascades`/`channels`/`n_feat0`를 getattr.
- Produces: `build_promptmr(args) -> PromptMR` (nn.Module; `forward(kspace, mask) -> (B,384,384)`). Task 6이 사용.

- [ ] **Step 1: 파일 작성**

Create `utils/model/promptmr/adapter.py`:

```python
# PromptMR+ 빌더: CLI args -> PromptMR. backbone이 forward(kspace,mask)->(B,384,384)
# 계약을 이미 만족하므로 별도 forward 래퍼가 없다(k-space native, 자체 정규화).
from utils.model.promptmr.promptmr import PromptMR
from utils.model.promptmr.config import promptmr_tiny_config


def build_promptmr(args):
    cfg = promptmr_tiny_config(
        num_cascades=getattr(args, "num_cascades", 6),
        feature_dim=getattr(args, "channels", [24, 32, 40]),
        n_feat0=getattr(args, "n_feat0", 16),
    )
    return PromptMR(**cfg)
```

- [ ] **Step 2: 로컬 컴파일 검증**

Run: `python -m py_compile utils/model/promptmr/adapter.py`
Expected: exit 0.

- [ ] **Step 3: Commit**

```bash
git add utils/model/promptmr/adapter.py
git commit -m "feat(promptmr): build_promptmr 어댑터(빌더)"
```

---

## Task 6: `--model promptmr` 배선 (제거된 SDUM 배선 미러)

**Files:**
- Modify: `train.py` (argparse)
- Modify: `utils/learning/train_part.py` (build + optimizer 분기)
- Modify: `utils/learning/test_part.py` (`load_model` + `forward` 분기)
- Modify: `scripts/reconstruct.py` (argparse)

**Interfaces:**
- Consumes: `build_promptmr`(Task 5).
- Produces: `--model promptmr` 경로 — train/val/recon_eval/reconstruct 전부에서 PromptMR 구성. `check_resume_compat`(기존, `model/num_cascades/channels` 필드 이미 포함)와 호환.

- [ ] **Step 1: `train.py` — arch 플래그 추가**

`train.py`에서 `--sens_chans` 줄 바로 아래에 삽입:

```python
    parser.add_argument('--model', type=str, default='varnet', choices=['varnet', 'promptmr'], help='Backbone: varnet(기본) 또는 promptmr')
    parser.add_argument('--num-cascades', type=int, default=6, help='PromptMR cascade 수 T')
    parser.add_argument('--channels', type=int, nargs='+', default=[24, 32, 40], help='PromptMR PromptUnet feature_dim (3 level)')
    parser.add_argument('--n-feat0', type=int, default=16, help='PromptMR PromptUnet n_feat0 (stem 폭)')
```

- [ ] **Step 2: `train_part.py` — 모델 빌드 분기**

`utils/learning/train_part.py` `train()`에서 현재의 무조건 VarNet 생성:

```python
    model = VarNet(num_cascades=args.cascade,
                   chans=args.chans,
                   sens_chans=args.sens_chans)
    model.to(device=device)
```

를 아래로 교체:

```python
    if getattr(args, "model", "varnet") == "promptmr":
        from utils.model.promptmr.adapter import build_promptmr
        model = build_promptmr(args)
    else:
        model = VarNet(num_cascades=args.cascade,
                       chans=args.chans,
                       sens_chans=args.sens_chans)
    model.to(device=device)
```

- [ ] **Step 3: `train_part.py` — optimizer 분기 (AdamW, 논문 정합)**

같은 파일에서:

```python
    loss_type = SSIMLoss().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
```

를 아래로 교체:

```python
    loss_type = SSIMLoss().to(device=device)
    if getattr(args, "model", "varnet") == "promptmr":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    else:
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
```

- [ ] **Step 4: `test_part.py` — `load_model` 분기 (recon_eval 채점 경로)**

`utils/learning/test_part.py` `load_model()`의 현재 VarNet 생성:

```python
    saved = ckpt.get('args', args)  # 학습 때 저장된 args(아키텍처 인자 포함)
    model = VarNet(
        num_cascades=getattr(saved, 'cascade', args.cascade),
        chans=getattr(saved, 'chans', args.chans),
        sens_chans=getattr(saved, 'sens_chans', args.sens_chans))
```

를 아래로 교체:

```python
    saved = ckpt.get('args', args)  # 학습 때 저장된 args(아키텍처 인자 포함)
    if getattr(saved, 'model', 'varnet') == 'promptmr':
        from utils.model.promptmr.adapter import build_promptmr
        model = build_promptmr(saved)
    else:
        model = VarNet(
            num_cascades=getattr(saved, 'cascade', args.cascade),
            chans=getattr(saved, 'chans', args.chans),
            sens_chans=getattr(saved, 'sens_chans', args.sens_chans))
```

- [ ] **Step 5: `test_part.py` — `forward()` 분기 (reconstruct 경로)**

같은 파일 `forward()`의 현재 VarNet 생성:

```python
    model = VarNet(num_cascades=args.cascade,
                   chans=args.chans,
                   sens_chans=args.sens_chans)
    model.to(device=device)
```

를 아래로 교체:

```python
    if getattr(args, "model", "varnet") == "promptmr":
        from utils.model.promptmr.adapter import build_promptmr
        model = build_promptmr(args)
    else:
        model = VarNet(num_cascades=args.cascade,
                       chans=args.chans,
                       sens_chans=args.sens_chans)
    model.to(device=device)
```

- [ ] **Step 6: `scripts/reconstruct.py` — arch 플래그 추가**

`scripts/reconstruct.py` `parse()`에서 `--sens_chans` 줄 아래에 삽입:

```python
    parser.add_argument('--model', type=str, default='varnet', choices=['varnet', 'promptmr'], help='Backbone: varnet 또는 promptmr')
    parser.add_argument('--num-cascades', type=int, default=6, help='PromptMR cascade 수 T')
    parser.add_argument('--channels', type=int, nargs='+', default=[24, 32, 40], help='PromptMR feature_dim')
    parser.add_argument('--n-feat0', type=int, default=16, help='PromptMR n_feat0')
```

- [ ] **Step 7: 로컬 컴파일 검증**

Run: `python -m py_compile train.py utils/learning/train_part.py utils/learning/test_part.py scripts/reconstruct.py`
Expected: exit 0.

- [ ] **Step 8: 배선 일관성 점검**

Run: `grep -rn "build_promptmr\|model.*promptmr" train.py utils/learning/train_part.py utils/learning/test_part.py scripts/reconstruct.py`
Expected: train_part(build+optimizer), test_part(load_model+forward), train.py/reconstruct.py(choices)에서 promptmr 참조가 보임.

- [ ] **Step 9: Commit**

```bash
git add train.py utils/learning/train_part.py utils/learning/test_part.py scripts/reconstruct.py
git commit -m "feat: --model promptmr 배선 (train/test/reconstruct + AdamW 분기)"
```

---

## Task 7: GPU 검증 (서버 전용 — shell_scripting/ 큐)

> **실행 환경 주의:** 이 task의 실제 실행은 **GPU 서버**에서 이뤄진다(로컬엔 torch 없음). 각 step은 (a) 로컬에서 슬롯 파일 작성·커밋, (b) 서버에서 실행·기대출력 확인으로 나뉜다. 서버 실행은 사용자/서버 세션이 트리거한다. 슬롯 헤더 형식은 먼저 `shell_scripting/_template.sh`와 `shell_scripting/README.md`를 읽어 맞춘다. 데이터 경로(`/root/Data/...`)·net-name은 기존 슬롯과 동일 규약.

**Files:**
- Create: `shell_scripting/` 내 빈 슬롯 3개(probe/smoke/eval; 기존 비어있는 슬롯 재사용 가능).

**Interfaces:**
- Consumes: Task 6 완료된 `--model promptmr` 경로.
- Produces: peak GPU mem·loss·SSIM 3축 수치(→ `training_anal/`).

- [ ] **Step 1: (로컬) `_template.sh`·README 확인 후 build/shape probe 슬롯 작성**

빈 슬롯(예: `shell_scripting/shell2.sh`)에 목적 주석 + 시작 echo + `tee logs/...` 규약을 맞춰 아래 명령을 넣는다. `num_cascades=2` + `--max-iters 2`로 forward/backward가 8GB에 들어가는지만 본다:

```bash
python train.py --model promptmr --num-cascades 2 --n-feat0 16 --channels 24 32 40 \
  --readout-crop --num-epochs 1 --max-iters 2 --report-interval 1 \
  -n promptmr_probe -t /root/Data/train -v /root/Data/val
```

- [ ] **Step 2: (서버) probe 실행 → shape·OOM 확인**

Run(서버): `bash shell_scripting/shell2.sh`
Expected: OOM 없이 완주. 로그에 train step 2회 진행, val 출력의 reconstruction shape가 `(…, 384, 384)`. **첫 val SSIM(=1-loss 계열)이 ~0이 아님**(스케일 정합 확인). peak CUDA memory 기록. 실패 시(OOM) spec §6 레버 순서: `--readout-crop`(이미 ON) → coil-compress → `--num-cascades` 감소 → `--n-feat0 8`.

- [ ] **Step 3: (로컬) smoke train 슬롯 작성**

빈 슬롯에 tiny config 정식값으로:

```bash
python train.py --model promptmr --num-cascades 6 --n-feat0 16 --channels 24 32 40 \
  --readout-crop --num-epochs 1 --report-interval 100 --lr 2e-4 \
  --run-tag promptmr_smoke -n promptmr_smoke -t /root/Data/train -v /root/Data/val
```

- [ ] **Step 4: (서버) smoke 실행 → loss 감소·resume 확인**

Run(서버): `bash shell_scripting/<slot>.sh`
Expected: OOM 없이 최소 1 epoch(또는 `--max-iters`로 제한) 완주, **train loss가 초기 대비 감소**, `../result/promptmr_smoke/checkpoints/model.pt`·`best_model.pt` 저장. `--resume` 붙여 재실행 시 정상 로드(같은 config).

- [ ] **Step 5: (로컬) recon_eval 채점 슬롯 작성**

```bash
python recon_eval.py -n promptmr_smoke -p /root/Data/leaderboard
```

- [ ] **Step 6: (서버) 채점 실행 → 3축 수치 기록**

Run(서버): `bash shell_scripting/<slot>.sh`
Expected: `recon_eval.py`(고정)가 `load_model`로 PromptMR 재구성 성공, SSIM_full / SSIM_bbox / 추론시간 3축 산출. volume forward가 60s 경고를 넘는지 확인(속도도 점수 요소).

- [ ] **Step 7: (로컬) 결과 기록 + Commit**

`training_anal/2026-07-19-promptmr-mvp-first-run.md`에 3축 수치를 VarNet baseline(run11 등) 옆에 기록(핵심요약/ML 체크리스트 규약).

```bash
git add shell_scripting/ training_anal/2026-07-19-promptmr-mvp-first-run.md
git commit -m "chore(promptmr): MVP 검증 슬롯 + 첫 recon_eval 결과 기록"
```

---

## Self-Review (작성자 체크)

**Spec coverage:** §1 목표(build·smoke·score)=Task 7. §2 하이브리드 vendoring=Task 1~3(fastmri 격차·reentrant·backbone). §3 tiny config=Task 4. §4 얇은 어댑터=Task 5. §5 배선=Task 6. §6 8GB 전략(checkpointing·readout-crop·레버 순서)=Task 3(checkpointing 내장)·Task 7 Step 2. §7 GPU 검증=Task 7. §8 범위 밖(adjacent-slice 등)=미포함(의도적). 커버 완료.

**Placeholder scan:** 모든 코드 step에 실제 코드. `<slot>`/`net-name`은 슬롯 파일명(환경 의존)으로 정당한 자리표시. TODO/TBD 없음.

**Type consistency:** `build_promptmr`(Task 5)가 Task 3 `PromptMR` + Task 4 `promptmr_tiny_config` 사용 — 이름·시그니처 일치. `sens_expand/sens_reduce`(Task 1)가 Task 3 backbone의 `fastmri.sens_expand` 호출과 일치. `--num-cascades`/`--channels`/`--n-feat0`(Task 6)가 `build_promptmr`의 getattr 키(`num_cascades`/`channels`/`n_feat0`)와 일치.
