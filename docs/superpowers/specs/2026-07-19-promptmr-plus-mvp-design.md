# PromptMR+ 백본 구현 설계 (MVP increment)

- **날짜**: 2026-07-19
- **브랜치**: `promptMR_plus` (SDUM tip `f8a68a0`에서 분기, SDUM 전용 코드 제거 완료)
- **근거 자료**: 논문 `papers/09b_PromptMR_plus.pdf`(+ `09_PromptMR.pdf`), 공식 repo `github.com/hellopipu/PromptMR-plus`(v2), 기존 포팅 `../2025_4th_clone/FastMRI_challenge_sMaRtIfy_final/utils/model/promptmr_plus.py`
- **문서 규약**: 산문은 한국어, 기술용어·식별자·config·경로·contract는 영어 원문 유지(repo 관례).

---

## 0. 배경

PromptMR+ 는 **Deep Unrolled Model(DUM)**: T개 cascade(논문 기본 T=12, 가중치 공유 안 함), 각 cascade가 k-space에서 data-consistency 1스텝을 수행한다.

    kᵗ = kᵗ⁻¹ − ηᵗ·M⊙(kᵗ⁻¹ − y) + F·E·D·R·F⁻¹(kᵗ⁻¹)

- **R**(Reduce): 추정된 sensitivity map으로 coil-combine → 복소 영상
- **D**(Denoiser): 3-level channel-attention U-Net (`PromptUnet`)
- **E**(Expand): sens map 곱 → 코일별 영상, **F/F⁻¹**: centered FFT/iFFT
- sensitivity map은 별도 **SME network**가 ACS(center k-space)에서 **1회** 추정 후 전 cascade 재사용

**PromptMR → PromptMR+ 의 "+"는 백본 dim 변화 없이 플래그 4개**로 정의된다:
1. `adaptive_input=True` + `n_buffer=4`: denoiser 입력을 `xₜ` 하나에서 `[xₜ, AᴴAxₜ, Aᴴy, sₜ]`로 확장. `sₜ`는 cascade 간 이월되는 learnable memory (self-adaptive gradient).
2. `n_history=11`: momentum layer — 각 decoder level에서 이전 모든 cascade의 동일 level feature를 concat 후 `1x1 Conv + CAB`로 융합(cross-cascade skip).
3. `use_sens_adj=True`: 인접 슬라이스에 걸쳐 같은 coil을 채널로 쌓아 SME를 (2a+1)배 메모리 효율화(논문의 ~55% train 메모리 절감의 주동인).
4. prompt block: level별 learnable prompt bank(softmax 가중 → interp → 3x3)로 입력 유형(contrast/accel/view) 문맥 주입.

---

## 1. 목표와 완료 기준 (MVP)

작지만 **충실한 PromptMR+("+" 4개 전부 ON)** 를 `utils/model/promptmr/`로 vendoring하고 `--model promptmr`로 배선하여, 다음을 만족한다:

1. GTX 1080(8GB, **FP32 전용**, batch=1)에서 forward/backward가 **OOM 없이** 동작하고 출력이 `(1, 384, 384)`.
2. GPU에서 짧은 **smoke 학습** → **loss 감소** 확인, checkpoint 저장 + `--resume` 로드 정상.
3. **고정된 `recon_eval.py`** 하네스로 채점되어 **SSIM_full / SSIM_bbox / time** 3축 첫 수치를 VarNet baseline 옆에 남긴다.

> **비목표**: 이번 increment는 "VarNet을 이긴다"가 아니다. 그건 후속 scaling increment. 여기서는 **진짜 PromptMR+가 정확하고 8GB에 들어간다**를 증명한다.

**성공 판정(evidence-based)**: (1)~(3)의 로그·수치가 실제로 산출되어야 완료. 특히 첫 smoke의 SSIM이 ~0이 아니라 정상 범위(스케일 정합 확인용)여야 한다.

---

## 2. 베이스 소스 — 하이브리드 vendoring

정확성은 **공식 v2**에서, 실전성(SNU 어댑팅·8GB 튜닝)은 **2025 clone**에서 가져온다.

- `utils/model/promptmr/`에 vendoring:
  - `promptmr.py`(백본 단일 파일: `PromptMR`·`PromptMRBlock`·`NormPromptUnet`·`PromptUnet`·`SensitivityModel` + 블록류 `CAB`/`PromptBlock`/`Down`/`Up`/`SkipBlock`/`KspaceACSExtractor`). clone `promptmr_plus.py`가 이미 self-contained(블록·모델 한 파일)라 분할하지 않는다 — surgical.
  - `config.py`(tiny config dict factory), `adapter.py`(`build_promptmr`), `__init__.py`.
  - clone은 이미 **`batched_mask_center`를 inline**했고 `from data import transforms` 대신 repo-local `import fastmri`를 쓰며 384-crop image를 반환한다 → 공식 v2 vendoring에서 해야 할 변형이 이미 돼 있는 상태.
- **FFT/coil 라이브러리는 repo `utils/model/fastmri/` 재사용.** 단 유일한 격차 = `sens_expand`/`sens_reduce`(num_adj_slices 인자)가 repo fastmri엔 없음 → clone 정의(정본 PromptMR ops)를 `coil_combine.py`에 **2함수 추가**. `ifft2c`/`rss`/`complex_*`는 이미 존재.
- gradient checkpointing `reentrant_wrapper.py`를 `utils/model/`에 vendoring.
- **정합 수정(clone→공식 v2)**: PromptUnet `reduction` 2→4(v2), `ReLU(inplace=False)` 유지(reentrant checkpointing 안전), `out_chans … else in_chans`는 v2와 동일이라 유지, `forward()`의 SNU mask 재정렬 블록은 우리 `(B,1,1,W,1)` mask에 맞게 재작성(`.cuda()`·reshape 제거, `num_low_frequencies=None`로 ACS 자동 도출).
- gradient checkpointing용 `reentrant_wrapper.py`는 clone에서 가져온다.
- **clone의 의심 수정 2곳은 공식 v2와 대조 검증**: `reduction=2` 로컬 변경, `out_chans` fallback 분기(`adaptive_input=False`일 때 `in_chans`로 떨어지는 부분). 충돌 시 **공식 v2 우선**.
- ⚠️ **Rutgers non-commercial 저작권 고지**를 vendoring 파일에 유지(대회 제출물 라이선스 이슈). 대회=비상업 연구이므로 사용 허용, 단 고지 유지 의무.

---

## 3. 시작 config (tiny, full "+" ON)

첫 MVP 학습 config — "+" 플래그 4개 전부 ON, dim은 8GB 위해 강하게 축소:

```
num_cascades   = 6            adaptive_input = True,  n_buffer = 4
n_feat0        = 16           n_history      = 3      use_sens_adj = True
feature_dim    = [24, 32, 40]
prompt:  len_prompt=[5,5,5],  prompt_size=[16,8,4],   prompt_dim=[8,16,24]
sens_n_feat0   = 8,   sens_feature_dim=[12,16,20],    sens_prompt_dim=[8,12,16]
num_adj_slices = 1            use_checkpoint = True
```

- **OOM 확인 사다리**: 최초 build/shape probe는 `num_cascades=2` + `--max-iters` 소량으로 forward/backward가 fit함을 확인한 뒤 위 config로 올린다.
- **scale-up 노브**(후속 increment): `num_cascades` → `n_feat0` → `feature_dim`.
- optimizer **AdamW(lr=2e-4, weight_decay=0.01)** (논문 정합). LR schedule은 기존 `--warmup-iters` 레버로 warmup→cosine 선택 가능(기본 상수).
- loss **SSIMLoss**(`utils/common/loss_function.py`) — 대회 metric·논문과 일치.

---

## 4. 어댑터 — 얇게 (SDUM처럼 두껍지 않음)

PromptMR+는 **k-space native**(자체 SME + data-consistency + `NormPromptUnet` 내부 instance-norm)라, SDUM과 달리 **`÷std/×std` 스케일 래퍼도, image-domain 브리징도 불필요**하다. `PromptMR.forward(masked_kspace (B,C,H,W,2), mask, …)`가 **이미 384×384 center-crop RSS magnitude**를 반환 → contract를 사실상 그대로 만족.

어댑터 층이 하는 일:
- `config.py`: hyperparam dataclass(SDUM의 `SdumArgs` 대응) + factory `promptmr_tiny_args(**overrides)`. `pretrained=None`(no-pretrain 규칙).
- `adapter.py`: `build_promptmr(args)` — `PromptMR` 인스턴스 생성. mask 포맷 shim: SNU mask는 `(B,1,1,W,1)`; clone의 `mask.shape[2]==1` broadcast + `num_low_frequencies=round(0.08·W)` ACS 휴리스틱이 이와 맞는지 **검증**.
- **정합성 체크포인트(critical)**: 출력 스케일이 target 스케일에 안착해야 한다. `NormPromptUnet`의 normalize→net→denormalize가 이를 처리하므로 별도 래퍼 불필요 — **첫 smoke에서 SSIM이 ~0이 아님**으로 검증. 데이터로더 `--normalize`는 **쓰지 않는다**(모델 자체 정규화와 이중 정규화 방지).

> contract(재확인): `forward(kspace (B,C,H,W,2), mask (B,1,1,W,1)) → (B,384,384)`. 소비처 `train_part.py`·`test_part.py`(`load_model`/`recon_slice`/`forward`)·고정 `recon_eval.py` 모두 `model(kspace, mask)`만 호출.

---

## 5. 배선 지점 (제거된 SDUM 배선을 미러)

- **`train.py`** `parse()`: `--model {varnet,promptmr}`, `--num-cascades`, `--channels`(nargs='+'), 필요한 PromptMR dim 플래그 추가.
- **`utils/learning/train_part.py`**:
  - `train()` 모델 빌드 분기 → `from utils.model.promptmr.adapter import build_promptmr`.
  - optimizer 분기 → promptmr면 **AdamW(wd=0.01)**.
  - `build_scheduler`·`check_resume_compat`는 이미 model-agnostic(후자는 `model/num_cascades/channels` 필드 이미 포함) → 수정 불필요.
- **`utils/learning/test_part.py`**: `load_model()`·`forward()`에 promptmr 분기(저장된 ckpt args의 `saved.model`로 자기기술적 복원). **이것이 고정 `recon_eval.py`가 부르는 계약** → recon_eval 무수정.
- **`scripts/reconstruct.py`** `parse()`: 대응 플래그 추가.
- **무수정**: `recon_eval.py`, `utils/common/metrics.py`(고정·제출 시 organizer 재실행), `utils/model/varnet.py`, `utils/data/load_data.py`, `utils/data/transforms.py`(데이터 파이프라인은 model-agnostic, yield하는 `(kspace,mask)`가 이미 contract와 일치).

---

## 6. 8GB 메모리 전략

- **gradient checkpointing ON**(모든 cascade + 각 UpBlock upsample + sens net). Pascal FP32(no AMP)라 사실상 필수.
- **native ~15 coils 우선**. OOM 시 이미 배선된 `coil-compress`(SVD, `transforms.py`) → 8 virtual coils 레버 사용.
- **`--readout-crop` ON**(무손실 640→384 height, ~40% activation 절감). 언더샘플 축은 width(PE)라 readout(height) crop이 DC 정합을 깨지 않음. 데이터로더 `--normalize`는 OFF(§4).
- **OOM 레버 순서**(위 config가 안 들어가면): `--readout-crop` → `coil-compress 8` → `num_cascades 6→4` → `n_feat0 16→8`.
- 모든 probe는 **peak CUDA memory 로깅**.

---

## 7. 검증 (GPU 전용 — pytest 금지)

전부 `shell_scripting/` 큐 슬롯에서 GPU로 실행(비-GPU 명령은 직접). 단위 테스트/pytest는 이번 작업에서 돌리지 않는다.

1. **Build/shape probe**: `num_cascades=2`, `--max-iters 2` → forward/backward 동작, 출력 `(1,384,384)`, OOM 없음, peak mem 로깅.
2. **Smoke train**: §3 tiny config, capped iters 또는 1~2 epoch → **loss 감소**, checkpoint 저장, `--resume` 로드 확인.
3. **채점**: `python recon_eval.py -n <net_name> -p /root/Data/leaderboard` → SSIM_full / SSIM_bbox / time 산출, `training_anal/`에 VarNet 대비 기록.

---

## 8. 범위 밖 (명시적 후속 increment — 각자 spec→plan)

- **Adjacent-slice 2.5D**(`num_adj>1` + 데이터로더가 인접 슬라이스 공급하도록 `SliceData` 변경) — PromptMR+ 핵심 기능이나 데이터 파이프라인 변경 필요, 별도 처리.
- coil-compression 튜닝(몇 virtual coils가 SSIM/메모리 최적인지).
- 채널·cascade를 논문 규모로 **scaling** — VarNet 넘기기 목표.
- `--fg-loss`/`--bbox-loss` 학습(현재 VarNet best 레버) 적용.
- warmup→cosine schedule 튜닝.

---

## 9. 열린 결정 / 리스크

**열린 결정(기본값, 변경 시 재검토):**
- coils: native 15 우선, compress-8은 OOM fallback.
- loss: MVP는 plain SSIM(논문·metric 정합), fg/bbox 후속.
- `--readout-crop`: PromptMR+ 기본 ON.

**리스크:**
- **8GB fit 불확실성**: full "+" ON(특히 `n_history` momentum은 cascade 수에 따라 저장 feature 증가) + native 15 coils가 6 cascade에서도 안 들어갈 수 있음 → §6 레버 순서로 대응. 최악의 경우 `n_history`를 3→1로 임시 축소(단, "+" 유지 원칙과 충돌하므로 마지막 수단).
- **스케일 정합**: 내부 instance-norm의 denorm이 target 스케일과 안 맞으면 SSIM≈0 → §4 체크포인트로 조기 검출.
- **clone vs 공식 v2 미세 차이**(`reduction`, `out_chans`)로 인한 조용한 정확도 손실 → 대조 검증으로 차단.
- **속도**: unrolled + checkpointing recompute + FP32 → epoch 느림. recon_eval은 volume forward >60s 경고 → time 축 점수·제출 제약 주시(모델 크기/속도도 점수 요소).
