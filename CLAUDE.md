# CLAUDE.md — FastMRI 2026 Challenge 작업 컨텍스트 & 핸드오프

이 repo는 **2026 SNU FastMRI Challenge**(서울대 전기정보공학부, 6번째) 참가용 코드·리서치·전략 문서 모음이다.

이 파일은 두 부분으로 구성된다:
- **Part A — 대회 컨텍스트**: 과제·데이터·채점·규칙 (변하지 않는 배경 지식)
- **Part B — 작업 핸드오프**: 미래 세션(GPU 서버 포함)의 Claude가 먼저 읽어야 하는 노트.

> **⚡ GPU 서버라면 바로 여기부터: [`docs/gpu_quickstart.md`](docs/gpu_quickstart.md)** — clone→sanity→8GB config 찾기→학습까지 복붙 순서.

**진행 상황(2026-07-07 갱신):** SDUM 어댑터 forward는 **Colab GPU에서 검증 완료**(shape/finite/1-step 학습 통과). 단 **실제 1080 8GB 적재량은 미검증** → 학습 전 `probe_oom.py`로 config를 잡아야 함(quickstart 2단계). 아래 Part B §1도 참고.

---

# Part A — 대회 컨텍스트

## 대회 개요
- **주최**: 서울대 전기정보공학부 이종호 교수 연구실. 후원: 에어스메디컬(AIRS Medical), 클린벤처스.
- **과제명**: **Annotation-guided MRI Reconstruction** (역대 대회 최초, 레퍼런스 없는 open question).
- **대상**: **무릎(knee) MRI만** (작년까진 brain+knee, 올해는 brain 제외 knee 전용).
- **핵심 명제**: 전체 화질뿐 아니라 **병변(lesion) 위치에서 hallucination 없이 정확히 복원**하는가.

## 입력 / 출력
- **입력**:
  - 언더샘플된 k-space 데이터
  - **acc4 / acc8** 두 가속률 (1D 위상인코딩 축 언더샘플)
  - GRAPPA 복원 영상 (acc4/acc8 각각) — 이미 alias가 상당히 줄어든 초기해
- **출력**: alias-free 복원 영상.
- **데이터**: NYU fastMRI knee 계열. train/val split 제공되나 **자유 재조합 허용**.

## 데이터셋 상세 (공식 `tutorial.ipynb`, `2026_baby_varnet` 브랜치 기준)
- **파일 포맷**: HDF5(`.h5`). 모든 데이터는 `h5py`로 접근.
- **디렉토리 구조**:
  ```
  {DATA_DIR}/train/kspace/knee_acc4_1.h5   # k-space 파일
  {DATA_DIR}/train/image/knee_acc4_1.h5    # image 파일 (같은 이름끼리 페어)
  ```
  - 파일명 패턴: `knee_acc{4|8}_{idx}.h5`. **acc4 / acc8이 별도 파일**로 존재.
  - `train/` 외에 val split도 동일 구조로 제공(자유 재조합 허용).
- **k-space 파일 (`train/kspace/`) — 2개 key**:
  - `kspace`: **fully-sampled(완전샘플) complex k-space**. shape `(slices, coils, height, width)`. 예시: `(23, 15, 640, 368)`.
    - ⚠️ **저장된 kspace 자체는 언더샘플이 아니라 완전샘플**(2026-07-04 실측 확인: `recon(kspace)`가 `image_label`과 corr 1.0, `recon(kspace×mask)`가 `image_input`과 corr 1.0). mask==0 라인에도 신호 에너지가 그대로 남아있음.
    - **언더샘플은 로드 시점에 `kspace × mask`로 생성**한다(`transforms.py:227` 비증강 경로 / `:154` 증강 경로). 즉 `image_input = recon(kspace×mask)`, `image_label = recon(kspace)`. 파이프라인이 mask를 곱하므로 leakage 없음. **모델에 raw kspace를 그대로 넣으면 정답 유출**이니 항상 mask 적용 필수.
  - `mask`: 언더샘플링 마스크. shape `(width,)`, 값은 0/1 이진. 예시: `(368,)` → **1D 위상인코딩 축 언더샘플** 확인. 로드 시 이 마스크를 완전샘플 kspace에 곱해 언더샘플을 재현.
  - **multi-coil 데이터**(예시 15코일). 각 coil에 iFFT 후 **RSS(root-sum-of-squares)**로 하나의 복원 영상을 얻는 구조.
  - **coil sensitivity map은 데이터에 미제공**(2026-07-04 실측: kspace/image 파일 어디에도 sens/csm/smap key 없음, mask는 순수 1D 이진벡터). VarNet의 `SensitivityModel`(sens_net)이 ACS 구간에서 **내부 추정**한다.
  - height/width(640×368)는 원본 k-space 해상도로, image 도메인의 384×384와 다름.
- **image 파일 (`train/image/`) — 3개 key, 모두 shape `(slices, 384, 384)`**:
  - `image_grappa`: **GRAPPA로 복원한 영상** (alias 상당히 감소한 초기해).
  - `image_input`: 언더샘플 k-space에서 얻은 **aliased 영상** (모델 입력).
  - `image_label`: fully-sampled k-space에서 얻은 **alias-free 정답(GT) 영상**.
  - acc8 파일도 포맷·key·shape 동일. acc8은 언더샘플이 심해 aliasing이 더 강함.

## Test 데이터셋 (train/val과의 차이)
- **제공 항목**: aliased image(`image_input`), GRAPPA 복원 영상(`image_grappa`), **언더샘플된 k-space 데이터**.
  - ⚠️ **train/val과 결정적 차이**: test의 kspace는 **완전샘플 kspace + mask 조합이 아니라, 처음부터 언더샘플된 k-space**다. train/val처럼 "로드 시 mask를 곱해 언더샘플을 재현"하는 구조가 아님 → 완전샘플 kspace를 전제로 한 로직(예: mask 미적용 시 GT 유출 우려, label 생성)은 test에는 적용 불가/불필요.
- **병변 위치 정보(annotation) 없음** — test에는 bbox가 제공되지 않는다. 채점측이 내부적으로 SSIM_bbox를 계산.
- **입출력 공통 특성 (train/val/test 모두)**:
  - 입력은 모두 **multicoil 데이터 + 여러 slice**.
  - **이미지(파일)마다 slice 수, coil 수, 가로/세로 픽셀 수가 다름** → 모델/파이프라인은 가변 shape을 처리해야 함.
  - **출력 이미지 크기는 모두 동일** (384×384).
  - **병변이 없는 이미지가 다수 존재** (train/val/test 공통).
  - 모든 병변(bbox)은 **16×16 픽셀 이상** 크기.
- **재현성**: 모든 seed를 고정하여 **fully reconstructable**해야 함 (주최측이 재현 검증).

## 병변 Annotation
- **의료진이 직접 라벨링**한 bounding box (레드박스).
- **저장 위치**: **image 파일의 h5 `attrs['annotations']`에 JSON 문자열**로 저장 (`json.loads(image_f.attrs.get("annotations", "{}"))`).
- **JSON 스키마** — `{"슬라이스번호": [{"x", "y", "width", "height", "label"}, ...], ...}`:
  - key = 슬라이스 인덱스 문자열(예: `"13"`), value = 해당 슬라이스의 bbox 리스트.
  - 각 bbox: `x`, `y`(좌상단 좌표), `width`, `height`, `label`(pathology 문자열).
  - **좌표는 이미 image(384×384) 공간 기준** → `image_label` 위에 그대로 사각형 draw 가능(별도 리스케일 불필요).
- **22개 pathology 카테고리** (meniscus tear 등). **16×16 픽셀 이상 큰 bbox만** 제공.
- annotation 없는 파일/슬라이스도 존재(`attrs`에 `annotations` 키 자체가 없을 수 있어 `.get(..., "{}")` 기본값 처리). bbox 간 위치 관계도 참고 가치 있음.

## 채점 지표 (3축) — Final Score = 가중합
1. **SSIM_full** — 전체 화질. **배경을 마스크로 날린 영상 안에서만** SSIM 평가.
2. **SSIM_bbox** — annotation된 병변 영역 내부 SSIM. **순위를 가르는 핵심**
   (작년 데이터로 재채점 시 bbox 포함하니 등수가 크게 뒤바뀜: 3등→1등 사례).
3. **Inference time** — 올해 신규. 같은 SSIM이면 2~3배 빠른 모델에 보너스.
- **가중치(weight)는 미공개** → 3축 모두 균형 있게 최적화해야 안전.

## 하드웨어 / 제출 제약
- **최종 제출 서버 = GTX 1080 (VRAM 8GB, Pascal, Tensor Core 없음 → FP16/AMP 이득 없음, FP32 학습)**.
- 원논문의 A100/120M급 세팅은 그대로 못 씀 → **아이디어만 이식**.
- **최종 제출 모델은 반드시 주최측 제공 GPU 서버(VESSL)에서 학습**한 것만 평가.
  개인 GPU는 실험용, 최종은 서버에서 재학습. **고정 seed 재현(reproduce) 검증함** → 재현성 필수.

## 대회 규칙
- **External data 금지** (제공 train/val만 사용).
- **Pretrained weight 금지** (네트워크 구조는 사용 가능, 사전학습 가중치 위 학습 불가).
- **Leaderboard 데이터를 train에 사용 금지** (leaderboard 종속 PT 모델 금지).
- 레포는 **private** 권장 (공개 시 도용 위험).

## 확인 필요 / 미확정
- **채점 3축 가중치** (미공개) — 추론시간 비중이 크면 cascade 다이어트 우선순위↑.
- **TGVN 적용 가능성**: OT에 2nd contrast(PD/PDFS 쌍) 제공 언급 없음 → side-info 없이는 TGVN 핵심 이득 불가, 사실상 제외 가능성 높음. 데이터 열어 확인.
- SSIM_full의 **배경 마스크 정의**(threshold/ROI) — eval code(7/8 공개)에서 확인.
- 병변 없는 슬라이스 비율, bbox JSON 스키마 상세.

---

# Part B — 작업 핸드오프 (SDUM 백본 이식)

## 0. 현재 환경 구분 (중요)

- **지금까지의 작업 환경 = 개발용 로컬(Windows 노트북)**: torch/MONAI **미설치**, GPU 없음(MX450 2GB뿐), 데이터 없음.
  - 그래서 **로컬에서 한 검증 = 파이썬 AST 문법 파싱뿐.** `pytest`는 **한 번도 돌리지 않았다.**
- **GPU 서버로 이동하면 사용자가 명시적으로 알려준다.** 그 신호가 오기 전까지는 여기가 개발 환경이라고 가정.

## 1. GPU 서버로 옮기면 — ⚠️ 학습보다 먼저 "모델 돌려서 검증"

사용자가 "이제 GPU 서버다"라고 하면, **본 학습(train) 전에 아래 검증을 먼저** 수행한다. 순서대로:

```bash
# (0) 가상환경 권장 (전역 오염 방지). Python 3.12.x 권장(repo 타깃 3.12.9).
pip install -r requirements.txt          # torch/monai/einops/timm/pytest 등

# (1) import & 빌드 스모크
pytest tests/test_sdum_build.py -v       # 의존성 import + 모델 빌드 + 파라미터 수

# (2) ⭐ 어댑터 게이트 (가장 중요) — 이게 통과해야 그 다음으로 감
pytest tests/test_sdum_adapter.py -v     # forward shape (B,384,384) + finite + 1-step 학습

# (3) 8GB 적재 스윕 (실제 GPU에서만 의미)
python scripts/probe_oom.py --cascades 6 --channels 24 48
#   -> peak GB 또는 OOM 출력. peak ~7.5GB 이하로 OK인 가장 큰 조합 선택.

# (4) 위가 다 통과한 뒤에야 실제 학습
python train.py --model sdum --num-cascades 4 --channels 32 64 \
    -t <train_dir> -v <val_dir> -n sdum_test ...
```

**검증(1)~(3)이 통과하기 전에는 장시간 학습을 돌리지 말 것.**

## 2. 솔직한 리스크 (미검증 지점)

개발 환경에서 실행을 못 했기 때문에 아래는 **런타임 미검증**이다. GPU에서 처음 돌릴 때 오류가 날 수 있다:

- **가장 위험: 어댑터(`utils/model/sdum/adapter.py`).** 우리 파이프라인은 k-space `(B,C,H,W,2)` + mask `(B,1,1,W,1)`를 주는데, NV SDUM 모델은 **영상 도메인 `(B,T,C,H,W,2)`**를 받는다. 어댑터가 ifft·T=1 차원·mask broadcast·RSS·384 crop으로 잇는다.
  - **불확실 포인트 1**: NV `restormer_mri.forward`의 mask가 `"b t c h w two -> (b t) c h w two"`로 rearrange되는데, 우리 `mask_b`(B,1,C,H,W,2) broadcast가 `soft_dc`/`get_acs_region`의 기대 shape과 정확히 맞는지.
  - **불확실 포인트 2**: cascaded 출력이 영상 도메인 multicoil `(B,T,C,H,W,2)`이 맞는지(그래서 RSS로 합쳐 `(B,384,384)`가 되는지). 만약 coil-combined 단일 채널을 반환하면 RSS를 단순화해야 함.
  - **증상**: `test_sdum_adapter`가 shape/rearrange 에러로 실패. → **수정 가이드는 플랜의 Task 3 ⚠️ 노트** 참조: `docs/superpowers/plans/2026-07-06-sdum-backbone-port.md`.
- **의존성 해소 미검증**: NV 코드는 MONAI에 깊게 의존. `nv_utils.py`는 필요한 헬퍼 6개만 AST로 추출했는데, 실제 import 시 빠진 이름이 있으면 `ImportError`가 날 수 있음(그땐 `NV-SDUM-official/scripts/utils.py`에서 해당 함수를 추가 vendor).
- **MONAI 버전 호환**: `monai.apps.reconstruction.*` 경로가 설치된 MONAI 버전에서 동일한지 확인(≥1.3 가정).

## 3. 이 이식에서 확정된 설계 (배경)

- 기존 **E2E-VarNet 백본을 SDUM으로 교체**. `--model sdum`으로 선택(기본은 여전히 `varnet`이라 기존 경로 안 깨짐).
- 제약: **GTX 1080 8GB FP32**(Muon/BF16 안 씀 → **AdamW FP32**), knee 전용, acc4/acc8, 출력 384², 순위축 SSIM_bbox.
- **사전학습 가중치/외부데이터 금지**(대회 규정) → NV HuggingFace 자동다운로드 코드는 vendor 안 함. 처음부터 학습.
- 확정 결정(상세: `strategy/backbone.md`): gradient checkpointing ON(NV cascade 루프 내장) · CSME per-cascade · SWDC 유지 · **UC 제거**(`time_cond/label_cond=False`) · batch=1 · coil compression 보류 · num_frames=1(2D).
- **레버 스윕 대상**: `--num-cascades`(T), `--channels`(Restormer 폭). config 기본 `[32,64]`=2-level; `[C]` 하나면 1-level A/B(병변 과평활 대비, 상세: backbone.md).

## 4. 범위 밖 (별도 작업/다른 팀원)

- **GRAPPA prior + residual** (다른 팀원 진행 중), **bbox 가중 손실**, SWDC 패턴 튜닝, CSME 옵션(c) 주기적 재추정, progressive expansion. 상세: `strategy/` 문서들 + 플랜의 "범위 밖" 절.

## 5. 참고 문서

- 이식 플랜(태스크·수정 가이드): `docs/superpowers/plans/2026-07-06-sdum-backbone-port.md`
- 모델 결정 로그: `strategy/backbone.md`
- 원본 SDUM 코드(참고용, repo 밖): `../NV-SDUM-official/` (NVIDIA-Medtech/NV-Raw2insights-MRI, Apache-2.0)
- SDUM 논문: arXiv:2512.17137
