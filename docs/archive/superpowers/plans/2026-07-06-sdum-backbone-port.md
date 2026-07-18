# SDUM 백본 이식 구현 플랜

> **작업자(사람/에이전트)에게:** 이 플랜은 태스크 단위로 구현합니다. 각 스텝은 체크박스(`- [ ]`)로 진행 추적. 실행 방식은 문서 끝의 "실행 인계" 참고.

**목표:** 2026_GMG의 E2E-VarNet 백본을 **대폭 축소한 SDUM**(Restormer 기반 unrolled 모델)으로 교체하고, GTX 1080 8GB FP32에 맞게 우리 knee 데이터 파이프라인에 **k-space↔영상 어댑터**로 연결한다.

**구조:** NVIDIA의 Apache-2.0 SDUM 모델 코드(`restormer.py`, `latent_recon.py`, `varnet.py`)를 `utils/model/sdum/`에 vendor(복사·이식)하고 MONAI/einops/timm 의존성을 추가한다. SDUM cascaded 모델은 **영상 도메인**에서 `(B,T,C,H,W,2)` 텐서로 동작하고, 우리 파이프라인은 **k-space** `(masked_kspace, mask)`를 준다 → 얇은 `SDUMReconModel` 어댑터가 둘을 이어붙여 기존 `model(kspace, mask) -> (B,384,384)` 계약을 그대로 유지한다(→ `load_data.py`/`transforms.py` 수정 불필요). UC(universal conditioning)와 temporal 프레임은 config로 끔(`num_frames=1`, `time_cond=False`, `label_cond=False`). SWDC·per-cascade CSME·cascade skip·gradient checkpointing은 유지.

**기술 스택:** PyTorch(FP32), MONAI(`monai.apps.reconstruction`), einops, timm, pytest. NV의 `torch.utils.checkpoint(use_reentrant=False)` per-cascade 체크포인팅을 그대로 재사용.

## 전역 제약(Global Constraints)

- **하드웨어 타깃:** NVIDIA GTX 1080, **8GB VRAM, FP32 전용**(Pascal — BF16/AMP 이득 없음). Muon 옵티마이저 **안 씀** → **AdamW FP32** 사용.
- **과제 범위:** knee 전용; Cartesian **acc4/acc8**; 출력 **384×384** center-crop magnitude; 순위축 **SSIM_bbox**(bbox 손실은 *별도 플랜*).
- **사전학습 가중치·외부 데이터 금지**(대회 규정): NV HuggingFace 자동 다운로드 **반드시 비활성화**; `pretrained_csm=null`, `pretrained_recon=null`. 처음부터 학습.
- **모델 계약(불변):** `forward(masked_kspace, mask) -> image`; `masked_kspace`는 float32 `(B,C,H,W,2)`, `mask`는 byte `(B,1,1,W,1)`, 출력은 real `(B,384,384)`.
- **확정된 결정**(`strategy/model_decision_points.md` 참고): gradient checkpointing ON; CSME 옵션 (c) 주기적 재추정; SWDC 유지; UC 제거; batch_size=1; 입력 해상도는 readout-crop 384 외 추가 축소 안 함; coil compression 보류(off); GRAPPA prior·bbox 손실은 **이 플랜 범위 밖**(다른 팀원/별도 작업).
- **로컬 개발 현실:** VRAM 충분한 로컬 GPU 없음 + 로컬 데이터 없음 → **이 플랜의 모든 테스트는 CPU 합성텐서 smoke/shape/param 테스트**. 실제 8GB 적재 여부·정확도는 이후 VESSL/더 큰 GPU에서 OOM 탐침 스크립트(Task 4) + 본 학습(범위 밖)으로 검증.
- **라이선스 위생:** vendor한 파일은 Apache-2.0(라이선스 헤더 그대로 유지). 출처는 `utils/model/sdum/README.md`에 기록.

---

## 파일 구조

- `requirements.txt` (수정) — `monai`, `einops`, `timm` 추가.
- `utils/model/sdum/__init__.py` (생성) — `build_sdum`, `SDUMReconModel` export.
- `utils/model/sdum/restormer.py` (생성) — NV `scripts/models/restormer/restormer.py`에서 vendor.
- `utils/model/sdum/latent_recon.py` (생성) — NV `scripts/models/latent_recon.py`에서 vendor.
- `utils/model/sdum/nv_varnet.py` (생성) — NV `scripts/models/varnet.py`에서 vendor(우리 `utils/model/varnet.py`와 충돌 방지 위해 rename). `CoilSensitivityModel_DCAE` + `ComplexUnet_DCAE`만 필요.
- `utils/model/sdum/nv_utils.py` (생성) — 모델이 import하는 NV `scripts/utils.py` 헬퍼의 부분집합(`get_acs_image`, `reshape_channel_to_batch_dim`, `reshape_batch_channel_to_channel_dim`, MONAI에서 안 오는 `sensitivity_map_reduce/expand`). Task 1의 소스 정독으로 확정.
- `utils/model/sdum/config.py` (생성) — `SdumArgs` dataclass + `tiny_8gb_args()` 팩토리(우리 레버 값).
- `utils/model/sdum/adapter.py` (생성) — `SDUMReconModel(nn.Module)` 어댑터.
- `utils/model/sdum/README.md` (생성) — 출처 + 라이선스 노트.
- `scripts/probe_oom.py` (생성) — OOM 탐침/레버 스윕 하네스(GPU에서 실행).
- `tests/test_sdum_build.py` (생성) — 모델 빌드 + 파라미터 수.
- `tests/test_sdum_adapter.py` (생성) — 어댑터 forward shape/NaN 없음.
- `utils/learning/train_part.py` (수정) — 모델 생성 교체 + AdamW.
- `utils/learning/test_part.py` (수정) — 모델 생성 교체.
- `train.py` / `reconstruct.py` (수정) — SDUM 레버용 CLI 인자.

---

## Task 0: 의존성

**파일:**
- 수정: `requirements.txt`
- 테스트: `tests/test_sdum_build.py` (import 스모크, Task 2에서 확장)

**인터페이스:**
- 산출: 프로젝트 venv에서 `monai`, `einops`, `timm` import 가능.

- [ ] **스텝 1: requirements.txt에 의존성 추가**

`requirements.txt`에 아래를 추가(보수적으로 핀; MONAI ≥ 1.3에 `monai.apps.reconstruction` 있음):

```
monai>=1.3.0
einops>=0.7.0
timm>=0.9.0
pytest>=7.0.0
```

- [ ] **스텝 2: 설치**

실행: `pip install -r requirements.txt`
기대: 충돌 없이 설치(repo README 기준 Python 3.12.9).

- [ ] **스텝 3: import 스모크 테스트 작성**

`tests/test_sdum_build.py` 생성:

```python
def test_deps_importable():
    import monai  # noqa: F401
    import einops  # noqa: F401
    import timm  # noqa: F401
    from monai.apps.reconstruction.networks.nets.utils import divisible_pad_t  # noqa: F401
```

- [ ] **스텝 4: 테스트 통과 확인**

실행: `pytest tests/test_sdum_build.py::test_deps_importable -v`
기대: PASS.

- [ ] **스텝 5: 커밋**

```bash
git add requirements.txt tests/test_sdum_build.py
git commit -m "chore: add MONAI/einops/timm deps for SDUM port"
```

---

## Task 1: SDUM 모델 코드 vendor(복사·이식)

**파일:**
- 생성: `utils/model/sdum/__init__.py`, `restormer.py`, `latent_recon.py`, `nv_varnet.py`, `nv_utils.py`, `README.md`
- 테스트: `tests/test_sdum_build.py` (확장)

**인터페이스:**
- 소비: 이미 clone한 NV repo(`3_winter_FMRI/NV-SDUM-official`).
- 산출: `from utils.model.sdum.latent_recon import create_mri_recon_model` import 가능; `from utils.model.sdum.restormer import restormer_mri, Restormer` import 가능.

- [ ] **스텝 1: 모델 파일 3개를 그대로 복사(Apache 헤더 유지)**

```bash
mkdir -p utils/model/sdum
NV=../NV-SDUM-official/scripts   # clone 위치에 맞게 상대경로 조정
cp "$NV/models/restormer/restormer.py"  utils/model/sdum/restormer.py
cp "$NV/models/latent_recon.py"         utils/model/sdum/latent_recon.py
cp "$NV/models/varnet.py"               utils/model/sdum/nv_varnet.py
```

- [ ] **스텝 2: 모델이 import하는 정확한 헬퍼를 NV `scripts/utils.py`에서 확인(소스 정독)**

`../NV-SDUM-official/scripts/utils.py`를 읽고 다음 정의 위치를 찾는다: `get_acs_image`, `reshape_channel_to_batch_dim`, `reshape_batch_channel_to_channel_dim`, 그리고 `sensitivity_map_reduce`/`sensitivity_map_expand` 래퍼(주의: 일부는 `monai.apps.reconstruction.networks.nets.utils`에서 옴). 이 함수들만(그 import와 함께) `utils/model/sdum/nv_utils.py`로 복사한다. **HuggingFace `resolve_checkpoint_path`/`CHECKPOINT_HF_REGISTRY`는 절대 가져오지 않는다**(규정: 자동 다운로드 금지).

- [ ] **스텝 3: vendor한 파일들의 import 수정**

아래 정확한 수정을 적용(NV 파일들은 `scripts/`가 `sys.path`에 있다고 가정하는 bare import를 씀):

- `utils/model/sdum/latent_recon.py`: `from models.restormer.restormer import restormer_mri` → `from utils.model.sdum.restormer import restormer_mri`.
- `utils/model/sdum/restormer.py`: `from models.varnet import CoilSensitivityModel_DCAE` → `from utils.model.sdum.nv_varnet import CoilSensitivityModel_DCAE`; `from utils import *` → `from utils.model.sdum.nv_utils import get_acs_image, sensitivity_map_reduce, sensitivity_map_expand, reshape_channel_to_batch_dim, reshape_batch_channel_to_channel_dim`(이름은 스텝 2에서 찾은 것에 맞춤).
- `utils/model/sdum/nv_varnet.py`: `from utils import reshape_batch_channel_to_channel_dim, reshape_channel_to_batch_dim` → `from utils.model.sdum.nv_utils import ...`.

- [ ] **스텝 4: 패키지 init + 출처 README 생성**

`utils/model/sdum/__init__.py` 생성:

```python
from utils.model.sdum.adapter import SDUMReconModel, build_sdum  # noqa: F401
```

`utils/model/sdum/README.md` 생성:

```markdown
# Vendored SDUM model (NVIDIA NV-Raw2Insights-MRI)

출처: https://github.com/NVIDIA-Medtech/NV-Raw2insights-MRI (Apache 2.0).
restormer.py / latent_recon.py / nv_varnet.py / nv_utils.py 는 (라이선스 헤더 포함)
복사한 뒤 import 경로만 우리 패키지에 맞게 수정함.
사전학습 가중치는 사용하지 않음(대회 규정). HuggingFace 자동다운로드 코드는 의도적으로 vendor 안 함.
```

- [ ] **스텝 5: import 테스트 추가**

`tests/test_sdum_build.py`에 추가:

```python
def test_sdum_imports():
    from utils.model.sdum.latent_recon import create_mri_recon_model  # noqa: F401
    from utils.model.sdum.restormer import restormer_mri, Restormer  # noqa: F401
    from utils.model.sdum.nv_varnet import CoilSensitivityModel_DCAE  # noqa: F401
```

- [ ] **스텝 6: 테스트 통과 확인**

실행: `pytest tests/test_sdum_build.py::test_sdum_imports -v`
기대: PASS. (헬퍼 이름 누락으로 실패하면 스텝 3의 import 목록을 스텝 2 결과에 맞게 고치고 재실행.)

- [ ] **스텝 7: 커밋**

```bash
git add utils/model/sdum/ tests/test_sdum_build.py
git commit -m "feat: vendor NV SDUM model code (Apache-2.0) with rewired imports"
```

---

## Task 2: 8GB용 tiny config + 모델 팩토리

**파일:**
- 생성: `utils/model/sdum/config.py`
- 테스트: `tests/test_sdum_build.py` (확장)

**인터페이스:**
- 소비: Task 1의 `create_mri_recon_model(args)`.
- 산출: `SdumArgs` dataclass, `tiny_8gb_args(**overrides) -> SdumArgs`, `build_cascaded(args) -> nn.Module`(NV `Cascaded_SkipConnected_MRI_Recon` 반환).

- [ ] **스텝 1: 실패 테스트 작성(파라미터 수 상한)**

`tests/test_sdum_build.py`에 추가:

```python
def test_tiny_model_param_count():
    import torch  # noqa: F401
    from utils.model.sdum.config import tiny_8gb_args, build_cascaded

    args = tiny_8gb_args(num_cascades=4, channels=[32, 64])
    model = build_cascaded(args)
    n_params = sum(p.numel() for p in model.parameters())
    # tiny config는 230M "small" 변형보다 훨씬 작아야 함
    assert n_params < 30_000_000, f"too big: {n_params/1e6:.1f}M"
    assert n_params > 100_000, f"suspiciously tiny: {n_params}"
```

- [ ] **스텝 2: 테스트 실패 확인**

실행: `pytest tests/test_sdum_build.py::test_tiny_model_param_count -v`
기대: FAIL — `ModuleNotFoundError: utils.model.sdum.config`.

- [ ] **스텝 3: config.py 구현**

`utils/model/sdum/config.py` 생성. `SdumArgs`는 `restormer_mri.__init__`/`create_mri_recon_model`이 `args.`/`hasattr(args, ...)`로 읽는 모든 속성을 제공한다(`restormer.py:987-1128`, `latent_recon.py:280-323` 참고). UC off, temporal off, SWDC on, knee용 단일 패밀리 SWDC map:

```python
from dataclasses import dataclass, field, replace


@dataclass
class SdumArgs:
    # --- cascade / backbone (레버) ---
    model_type: str = "restormer"
    num_cascades: int = 4                 # A1
    channels: list = field(default_factory=lambda: [32, 64])   # A2 (2-level 기본)
    num_blocks: list = field(default_factory=lambda: [3, 6])
    num_heads: list = field(default_factory=lambda: [1, 2])
    num_refinement: int = 2
    mlp_ratio: int = 3                    # GDFN 확장
    norm_type: str = "ln"
    drop_path: float = 0.1
    hybrid_attn: bool = False
    post_norm: bool = False
    ms_refinement: bool = False
    # --- temporal / coils ---
    num_frames: int = 1                   # B6: 단일 슬라이스 (knee 2D)
    num_reduced_coils: int = 1            # CSME가 1개 coil-combined 영상으로 축소
    # --- CSME (A3) ---
    use_csm: bool = True
    use_acs_region: bool = True
    use_single_csm: bool = False          # 옵션 (c)는 후속; 아래 노트 참고
    # --- SWDC (B4) 유지 ---
    use_dc_weight_map: bool = True
    mask_specific_dc_weight_map: bool = True
    # --- UC (B5) 제거 ---
    time_cond: bool = False
    label_cond: bool = False
    num_experts: int = None
    top_k: int = 1
    # --- cascade skips ---
    enable_cas_skips: bool = True
    flow: bool = False
    # --- SWDC 패밀리/프로토콜 테이블 (knee: 단일 uniform 패밀리, acc4/acc8) ---
    dataset: str = "knee"                 # -> soft_dc가 (768,768) dc map 분기 사용
    train_mask_types: list = field(default_factory=lambda: ["uniform"])
    accelerations: list = field(default_factory=lambda: [4, 8])
    acq_types: list = field(default_factory=lambda: ["knee"])
    # --- 사전학습 없음 (규정) ---
    pretrained_csm: object = None
    pretrained_recon: object = None


def tiny_8gb_args(**overrides) -> SdumArgs:
    return replace(SdumArgs(), **overrides) if overrides else SdumArgs()


def build_cascaded(args: SdumArgs):
    from utils.model.sdum.latent_recon import create_mri_recon_model
    return create_mri_recon_model(args)
```

> CSME 옵션 (c) "주기적 재추정" 노트: NV는 `use_single_csm`(공유 vs per-cascade)와 `use_acs_region`(첫 cascade만 ACS)만 노출. 진짜 *k cascade마다 재추정*은 네이티브 플래그가 없음 → cascade 루프를 subclass하는 후속 태스크로 구현. 이 플랜에선 기본 `use_single_csm=False`(per-cascade CSME, 논문과 일치)로 두고, (c) 정교화는 `strategy/model_decision_points.md`에 TODO로 기록.

- [ ] **스텝 4: 테스트 통과 확인**

실행: `pytest tests/test_sdum_build.py::test_tiny_model_param_count -v`
기대: PASS. `args.` 속성 누락 에러가 나면, NV가 쓰는 기본값(`restormer.py`의 `hasattr(...) else <default>`)으로 `SdumArgs`에 그 속성을 추가하고 재실행.

- [ ] **스텝 5: 커밋**

```bash
git add utils/model/sdum/config.py tests/test_sdum_build.py
git commit -m "feat: SdumArgs + tiny 8GB factory (UC off, temporal off, SWDC on)"
```

---

## Task 3: k-space↔영상 어댑터 (`SDUMReconModel`)

**파일:**
- 생성: `utils/model/sdum/adapter.py`
- 테스트: `tests/test_sdum_adapter.py`

**인터페이스:**
- 소비: `build_cascaded(args)`(Task 2); NV cascaded forward 시그니처 `cascaded(x, mask, mask_type, acc_factor, acq_type)`(non-flow 경로, `latent_recon.py:95-141` — 뒤 인자들은 None 기본).
- 산출: `SDUMReconModel(nn.Module)` — `forward(masked_kspace, mask) -> (B,384,384)`; 모듈 레벨 `build_sdum(args) -> SDUMReconModel`.

- [ ] **스텝 1: NV의 정확한 텐서 관례 확인(소스 정독)**

`restormer_mri.forward`(`restormer.py:1238-1330`)와 NV `get_acs_image`, `sensitivity_map_reduce/expand`(Task 1 스텝 2)를 읽는다. 확인할 것: 모델 입력 `x`는 **언더샘플 영상** `(B,T,C,H,W,2)`; `ref_image`도 동일; `mask`는 `"b t c h w two -> (b t) c h w two"`로 rearrange되므로 coil축으로 broadcast 가능한 `(B,T,C,H,W,2)` boolean/byte여야 함; `soft_dc`는 FP32로 돌고 `dc_weight_map`을 `mask_type`으로 인덱싱. 우리 mask에서 acc 도출(frac ≤ 0.25 → acc8, else acc4)은 `strategy/candidates.md`와 동일.

- [ ] **스텝 2: 어댑터 실패 테스트 작성(CPU, 합성)**

`tests/test_sdum_adapter.py` 생성:

```python
import torch
from utils.model.sdum.config import tiny_8gb_args
from utils.model.sdum.adapter import build_sdum


def _fake_batch(B=1, C=15, H=384, W=372):
    # 우리 파이프라인 계약: masked_kspace (B,C,H,W,2) float32, mask (B,1,1,W,1) byte
    ksp = torch.randn(B, C, H, W, 2, dtype=torch.float32)
    mask = torch.zeros(B, 1, 1, W, 1, dtype=torch.uint8)
    mask[..., ::4, :] = 1                 # 대략 acc4 uniform
    mask[..., W // 2 - 15 : W // 2 + 15, :] = 1   # ACS 중앙
    ksp = ksp * mask.permute(0, 1, 4, 2, 3).reshape(B, 1, 1, W, 1).to(ksp.dtype)  # 대략 언더샘플
    return ksp, mask


def test_adapter_forward_shape_and_finite():
    args = tiny_8gb_args(num_cascades=2, channels=[16, 32])
    model = build_sdum(args).eval()
    ksp, mask = _fake_batch()
    with torch.no_grad():
        out = model(ksp, mask)
    assert out.shape == (1, 384, 384), out.shape
    assert torch.isfinite(out).all()
```

- [ ] **스텝 3: 테스트 실패 확인**

실행: `pytest tests/test_sdum_adapter.py -v`
기대: FAIL — `ModuleNotFoundError: utils.model.sdum.adapter`.

- [ ] **스텝 4: adapter.py 구현**

`utils/model/sdum/adapter.py` 생성. 우리 k-space 계약을 NV의 영상 도메인 호출로 변환하고 되돌린다. MONAI의 centered ifft, RSS 사용; **최종 crop은 기존 `utils/common/utils.py::center_crop(data, height, width)`를 재사용**(shape `(C,H,W)` 입력, crop+zero-pad, 존재 확인됨 — 인라인 불필요). ifft/rss 헬퍼 정확한 이름은 Task 1 스텝 2 / MONAI(`monai.data.fft_utils.ifftn_centered`, `monai.apps.reconstruction.mri_utils.root_sum_of_squares_t`, `monai.apps.reconstruction.complex_utils.complex_abs_t`)에서 확정:

```python
import torch
import torch.nn as nn
from monai.apps.reconstruction.complex_utils import complex_abs_t
from monai.apps.reconstruction.mri_utils import root_sum_of_squares_t
from monai.data.fft_utils import ifftn_centered

from utils.common.utils import center_crop
from utils.model.sdum.config import build_cascaded


def _extract_acc(mask_1d: torch.Tensor) -> int:
    # mask_1d: (W,) of {0,1}. frac<=0.25 -> acc8 else acc4 (strategy/candidates.md)
    frac = mask_1d.float().mean().item()
    return 8 if frac <= 0.25 else 4


class SDUMReconModel(nn.Module):
    """어댑터: (masked_kspace (B,C,H,W,2), mask (B,1,1,W,1)) -> image (B,384,384)."""

    def __init__(self, args, out_size=(384, 384)):
        super().__init__()
        self.cascaded = build_cascaded(args)
        self.out_size = out_size
        self.mask_type = str(args.train_mask_types[0])  # "uniform"
        self.acq_type = str(args.acq_types[0])          # "knee"

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, C, H, W, two = masked_kspace.shape
        # 언더샘플 영상 = ifft(masked_kspace); T=1 차원 추가 -> (B,T,C,H,W,2)
        img = ifftn_centered(masked_kspace, spatial_dims=2, is_complex=True)
        x = img.unsqueeze(1)                      # (B,1,C,H,W,2)
        # (B,1,1,W,1) 마스크에서 (B,1,C,H,W,2) boolean 마스크로 broadcast
        mask_b = mask.bool().view(B, 1, 1, 1, W, 1).expand(B, 1, C, H, W, two)
        acc = _extract_acc(mask.reshape(B, W)[0])

        out = self.cascaded(
            x, mask_b, self.mask_type, acc, self.acq_type
        )  # non-flow 경로는 영상 (B,T,C,H,W,2) 반환

        # magnitude로 coil-combine: NV 출력은 이미 영상 도메인 multicoil
        out = out[:, 0]                           # T 제거 -> (B,C,H,W,2)
        mag = root_sum_of_squares_t(complex_abs_t(out), spatial_dim=1)  # (B,H,W)
        # center_crop은 (C,H,W)를 받음 → B를 C자리로 취급, (B,384,384) 반환(폭 372→384 zero-pad 포함)
        return center_crop(mag, *self.out_size)


def build_sdum(args, out_size=(384, 384)) -> SDUMReconModel:
    return SDUMReconModel(args, out_size=out_size)
```

> ⚠️ 출력단 정확한 coil-combine은 NV cascaded forward가 k-space를 반환하는지 영상을 반환하는지에 달림. `restormer_mri.forward` 기준 반환 `x`는 영상 도메인 multicoil `(B,T,C,H,W,2)`(`sensitivity_map_expand` + `soft_dc` 후, soft_dc가 `ifftn_centered`로 끝남). 스텝 1 정독에서 다르게 나오면(예: coil-combined 단일 채널) RSS를 그에 맞게 단순화. 조정 후 재실행.

- [ ] **스텝 5: 테스트 통과 확인**

실행: `pytest tests/test_sdum_adapter.py -v`
기대: PASS — `(1,384,384)` finite. NV forward 내부에서 shape/rearrange 에러가 나면, `mask_b`/`x` 차원을 `rearrange("b t c h w two-> (b t) c h w two")`(`restormer.py:1254-1256`)와 `soft_dc` 마스크 broadcast에 맞춰 조정 후 재실행.

- [ ] **스텝 6: 커밋**

```bash
git add utils/model/sdum/adapter.py tests/test_sdum_adapter.py
git commit -m "feat: SDUM k-space<->image adapter preserving model(kspace,mask)->384 contract"
```

---

## Task 4: OOM 탐침/레버 스윕 하네스

**파일:**
- 생성: `scripts/probe_oom.py`

**인터페이스:**
- 소비: `tiny_8gb_args`, `build_sdum`(Task 2-3).
- 산출: CLI `python scripts/probe_oom.py --cascades N --channels C1 C2` — 합성 데이터로 1 forward+backward 돌려 peak CUDA 메모리 또는 "OOM" 출력.

- [ ] **스텝 1: 탐침 스크립트 작성**

`scripts/probe_oom.py` 생성:

```python
import argparse
import torch
from utils.model.sdum.config import tiny_8gb_args
from utils.model.sdum.adapter import build_sdum


def probe(cascades, channels, H=384, W=372, C=15):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args = tiny_8gb_args(num_cascades=cascades, channels=channels)
    model = build_sdum(args).to(dev).train()
    ksp = torch.randn(1, C, H, W, 2, device=dev)
    mask = torch.zeros(1, 1, 1, W, 1, dtype=torch.uint8, device=dev)
    mask[..., ::4, :] = 1
    mask[..., W // 2 - 15 : W // 2 + 15, :] = 1
    tgt = torch.randn(1, 384, 384, device=dev)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    try:
        out = model(ksp, mask)
        loss = torch.nn.functional.l1_loss(out, tgt)
        loss.backward()
        if dev == "cuda":
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"OK cascades={cascades} channels={channels} peak={peak:.2f}GB")
        else:
            print(f"OK (cpu) cascades={cascades} channels={channels}")
    except torch.cuda.OutOfMemoryError:
        print(f"OOM cascades={cascades} channels={channels}")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cascades", type=int, default=4)
    ap.add_argument("--channels", type=int, nargs="+", default=[32, 64])
    args = ap.parse_args()
    probe(args.cascades, args.channels)
```

- [ ] **스텝 2: CPU에서 실행되는지 확인(로직 스모크 — GPU 불필요)**

실행: `python scripts/probe_oom.py --cascades 2 --channels 16 32`
기대: `OK (cpu) cascades=2 channels=[16, 32]` 출력(느릴 수 있음, 정상).

- [ ] **스텝 3: 커밋**

```bash
git add scripts/probe_oom.py
git commit -m "feat: OOM-probe harness for 8GB lever sweep (run on GPU)"
```

> 실제 사용(이후 8GB/더 큰 GPU에서): `--cascades {4,6,8}` × `--channels {[16,32],[24,48],[32,64]}` 스윕 → peak ~7.5GB 이하로 `OK`인 가장 큰 조합 선택. 탈락 config는 로그. gradient checkpointing이 NV cascade 루프에 내장돼 있어 cascade 수를 늘려도 peak가 천천히 증가.

---

## Task 5: SDUM을 학습/추론에 배선

**파일:**
- 수정: `utils/learning/train_part.py`(모델 생성 + 옵티마이저)
- 수정: `utils/learning/test_part.py`(모델 생성)
- 수정: `train.py`, `reconstruct.py`(CLI 인자)
- 테스트: `tests/test_sdum_adapter.py`(1-step 학습 스모크 추가)

**인터페이스:**
- 소비: `build_sdum`, `tiny_8gb_args`; 기존 dataloader 6-튜플 `(mask, kspace, target, maximum, fname, slice)`, 손실 `SSIMLoss`.
- 산출: `train.py --model sdum --num-cascades N --channels C1 C2` 경로가 AdamW FP32로 SDUM 어댑터 학습.

- [ ] **스텝 1: 1-step 학습 스모크 테스트 작성(CPU)**

`tests/test_sdum_adapter.py`에 추가:

```python
def test_one_training_step():
    import torch
    from utils.model.sdum.config import tiny_8gb_args
    from utils.model.sdum.adapter import build_sdum

    args = tiny_8gb_args(num_cascades=2, channels=[16, 32])
    model = build_sdum(args).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    ksp, mask = _fake_batch()
    tgt = torch.randn(1, 384, 384)
    out = model(ksp, mask)
    loss = torch.nn.functional.l1_loss(out, tgt)
    opt.zero_grad(); loss.backward(); opt.step()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0 and all(torch.isfinite(g).all() for g in grads)
```

- [ ] **스텝 2: 통과 확인**

실행: `pytest tests/test_sdum_adapter.py::test_one_training_step -v`
기대: PASS(cascade 전체에 finite gradient 흐름).

- [ ] **스텝 3: `train.py`에 모델 선택 인자 추가**

`train.py` `parse()`에 추가(기본값은 VarNet 경로 보존):

```python
parser.add_argument("--model", type=str, default="varnet", choices=["varnet", "sdum"])
parser.add_argument("--num-cascades", type=int, default=4)
parser.add_argument("--channels", type=int, nargs="+", default=[32, 64])
```

- [ ] **스텝 4: `train_part.py`의 모델 생성 교체**

`train_part.py` `train()`의 `VarNet(...)` 생성부를 찾아 분기로 교체. AdamW FP32(Muon 미사용):

```python
if getattr(args, "model", "varnet") == "sdum":
    from utils.model.sdum.config import tiny_8gb_args
    from utils.model.sdum.adapter import build_sdum
    sdum_args = tiny_8gb_args(num_cascades=args.num_cascades, channels=list(args.channels))
    model = build_sdum(sdum_args).to(device)
else:
    model = VarNet(num_cascades=args.cascade, chans=args.chans, sens_chans=args.sens_chans).to(device)
```

그리고 옵티마이저가 `torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)`인지 확인(Adam이면 교체).

- [ ] **스텝 5: `test_part.py` / `reconstruct.py`에 동일 교체 반영**

`test_part.forward`의 모델 생성부에도 동일한 `if args.model == "sdum": build_sdum(...) else VarNet(...)` 분기를 적용하고, `reconstruct.py`에도 같은 CLI 인자 3개 추가.

- [ ] **스텝 6: 학습-step 테스트 재실행 + 빠른 실사용 스모크**

실행: `pytest tests/test_sdum_adapter.py -v`
기대: 전부 PASS. (실제 1-epoch 데이터 학습은 GPU 머신에서 — CPU 테스트 범위 밖.)

- [ ] **스텝 7: 커밋**

```bash
git add train.py reconstruct.py utils/learning/train_part.py utils/learning/test_part.py tests/test_sdum_adapter.py
git commit -m "feat: select SDUM backbone via --model sdum; AdamW FP32 wiring"
```

---

## 범위 밖(별도 플랜)

- **GRAPPA prior + residual**을 SDUM 백본에 추가(다른 팀원; 전략 B10). dataloader 튜플 확장(`return_grappa`) + 어댑터 입력 확장 필요 — `strategy/implementation.md §5b` 참고.
- **bbox 가중 손실**(`SSIM_full + λ·SSIM_bbox`) — 전략 B9; `return_bbox` 배선 필요.
- **SWDC 패턴 튜닝**(acc4/acc8-specific dc weight map) — 단일 "uniform" 패밀리 이상.
- **CSME 옵션 (c)** 진짜 k-cascade마다 재추정(cascade 루프 subclass).
- **Progressive cascade expansion** 커리큘럼(6→10→18) — 작은 T에선 아마 불필요.
- **전체 8GB 레버 스윕 + 정확도 학습**을 VESSL/GPU에서.

---

## 자기 검토(Self-Review)

- **스펙 커버리지:** `model_decision_points.md`의 확정 결정 → gradient checkpointing(NV cascade 루프로 유지, Task 3/4), CSME(Task 2, 옵션-c는 후속 표기), SWDC 유지(Task 2 `use_dc_weight_map`), UC 제거(Task 2 `time_cond/label_cond=False`), batch=1(픽스처/전역제약), coil-compression off(미활성), GRAPPA/bbox 범위 밖(명시). ✅
- **Placeholder 점검:** 두 "소스 정독" 스텝(Task 1 스텝 2, Task 3 스텝 1)은 NV 헬퍼 시그니처를 확정하는 실제 확인 작업 — 로직 placeholder 아님; 모든 코드 스텝은 실행 가능한 코드를 담음. 유일한 의도적 유보(`num_reduced_coils`/출력-RSS shape)는 실패하면 즉시 드러나는 테스트로 게이트됨.
- **타입 일관성:** `tiny_8gb_args`/`build_cascaded`/`build_sdum`/`SDUMReconModel.forward(masked_kspace, mask)` 이름이 Task 2-5에서 일관. 모델 계약이 `train_part.py`/`test_part.py`의 기존 `model(kspace, mask)` 호출과 일치.
- **리스크 노트:** 최대 불확실성은 NV forward의 정확한 mask/`(B,T,C,H,W,2)` broadcasting·출력 도메인 — Task 3에 격리, shape+finite 테스트 통과가 배선(Task 5)의 게이트.
