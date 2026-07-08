# CLAUDE.md — 이 repo에서 작업하는 Claude에게

미래 세션(GPU 서버 포함)의 Claude가 먼저 읽는 핸드오프 노트. **구조 + 현재 상태**만 담는다.

---

## 0. 지금 상황

- **환경 = GPU 서버.** SDUM 학습이 실제로 돌고 있다(로컬 노트북 아님).
- **팀 2명** (§4).

> 진행 중인 학습 상태·문제·수정은 계속 바뀌므로 이 파일에 적지 않는다. 최신 학습 로그 분석은 `training_anal/`, 수정 계획은 `docs/superpowers/plans/` 참조.

---

## 1. 환경 · 실행

- **GPU 서버(현재)**: Python **3.10.12**, torch **2.3.1+cu121**(CUDA 12.1, `cuda.is_available()=True`), monai 1.3.2, einops/timm/h5py/scikit-image/pytest 설치됨. GPU = **GTX 1080 8GB**(driver 550.90.12).
  - **Pascal → FP32 전용**(AMP/BF16 금지), **batch=1**.
  - README는 Python 3.12.9를 타깃으로 적지만 **서버 실제는 3.10.12에서 정상 동작**.
- **로컬 개발 노트북**: torch/MONAI 없음, GPU 없음. → 로컬에서 한 검증은 **문법 파싱 수준뿐**. 실제 실행/pytest/학습은 **전부 GPU 서버(또는 Colab)** 에서.
- **데이터**: `/root/Data/{train,val}/{image,kspace}/knee_acc{4,8}_{n}.h5` (train 각 1–85, val 각 1–15).
- **학습 전 검증 게이트**(오래 돌리기 전에):
  ```bash
  pytest tests/test_sdum_build.py -v      # import + 빌드 + 파라미터 수
  pytest tests/test_sdum_adapter.py -v     # forward (B,384,384) + finite + scale-equivariance + 1-step
  python scripts/probe_oom.py --cascades 6 --channels 24 48   # 8GB 적재 스윕 (peak ~7.5GB 이하 최대 조합)
  ```
- **GPU 작업은 큐 스크립트로만 실행** (`shell_scripting/`): GPU를 쓰는 명령(학습·recon 등)은 `shell_scripting/shellN.sh` **슬롯**(= 작업 1개 = 파일 1개)에 써서 돌린다. 각 슬롯은 맨 위 목적 주석 + 시작 echo + `logs/<슬롯>_<timestamp>.log` 자동 기록. 실행은 `bash shell1.sh ; bash shell2.sh ; ...`로 이어서(큐), 다음 작업은 빈 슬롯에 미리 채워둔다. 규칙·상세: `shell_scripting/README.md`.
  - ⚠️ **비-GPU 명령은 큐에 넣지 말고 그냥 바로 실행한다.** `h5_viewer.py`·`pytest`·`git` 등 가벼운 건 옆 분할 터미널에서 즉시 실행. 큐는 **GPU 독점 작업 전용**이다.

---

## 2. 대회 규칙 (repo 근거: `README.md`, `strategy/`)

- **knee 전용**, Cartesian **acc4/acc8**, 출력 **384×384**.
- **평가 = SSIM 두 가지 + 추론시간**:
  - `SSIM_full`: foreground mask 내부 평균 SSIM.
  - **`SSIM_bbox`(순위 결정축)**: fastMRI+ 병변 bounding-box 안에서만 계산.
  - 자가채점: `utils/common/metrics.py`의 `ssim_full`/`ssim_bbox`.
- **사전학습 가중치·외부 데이터 금지** → NV SDUM의 HuggingFace 자동다운로드는 **vendor 안 함**, 처음부터 학습.
- **annotation**: 각 image h5 `attrs['annotations']` JSON. 형식 `{"<slice>":[{x,y,width,height,label},...]}` (384² 공간 기준, w·h>16 박스만).
- **재현성**: seed 고정 → 재학습 시 같은 결과.
- **제출물**: 코드 repo(실행법 README) + loss 기록 + 모델 weight + 설명 ppt.
- phase1 = train/val만 제공. leaderboard 데이터는 추후 배포(그때 `reconstruct.py` 사용).

---

## 3. 구조 (필수만)

- **백본 선택**: `train.py --model {varnet(기본) | sdum}`. 기존 E2E-VarNet 경로는 그대로 유지되고, SDUM은 추가 경로.
- **SDUM = NV 영상도메인 cascaded 모델(vendored) + 어댑터**:
  - `utils/model/sdum/` — `nv_varnet.py`(cascade), `restormer.py`(regularizer), `latent_recon.py`, `nv_utils.py`, `config.py`, **`adapter.py`**.
  - **어댑터가 핵심 접합부**: 우리 파이프라인은 k-space `(B,C,H,W,2)`+mask를 주고, NV 모델은 영상도메인을 받음. `adapter.forward`가 **ifft → per-sample 스케일 정규화(÷s) → cascaded → RSS → 384 crop → 출력 ×s 복원**으로 이음. 계약: `forward(kspace,mask)->(B,384,384)`, scale-equivariant.
- **진입점/루프**: `train.py`(인자·seed) → `utils/learning/train_part.py`(`train`/`train_epoch`) / `test_part.py`.
- **데이터**: `utils/data/load_data.py`(로딩·GRAPPA·dataloader), `transforms.py`(readout crop·정규화·GRAPPA·coil compression·`MRIAugmentor`).
- **레버**: `--num-cascades`(T), `--channels`(Restormer 폭; `[32,64]`=2-level, `[C]`=1-level). 디버깅 레버: `--num-workers`, `--max-iters`.
- **확정 설계**(SPEC = `strategy/confirmed_model.md`, 인덱스 `strategy/README.md`): readout crop 640→384(무손실) · 전역 스칼라 정규화 ÷s · GRAPPA 영상 prior+residual · 감도맵 1회 추정 후 재사용 · **gradient checkpointing ON** · CSME per-cascade · SWDC 유지 · **UC 제거**(time/label_cond=False) · num_frames=1(2D) · coil compression 보류.

---

## 4. Git · 협업

- **remote**: `https://github.com/JunKyungSeo/FastMRI_SNU_2026_GMG.git` (HTTPS).
  - 이 서버엔 gh CLI·credential helper 없음 → **push엔 PAT 필요**(Username=GitHub 아이디, Password=PAT). `!` 셸은 TTY가 없어 대화형 프롬프트 불가 → 토큰을 URL에 넣거나 `~/.git-credentials`에 직접 기입.
- **브랜치**: `SDUM`(작업 브랜치, origin/SDUM과 동기) / `master`(안정·머지 대상). master가 `behind`로 보여도 그 커밋 내용은 대개 SDUM 히스토리에 포함돼 있음.
- **팀 2명**:
  - `kindofwater <bedso@snu.ac.kr>`
  - `seojunkyung0428 <sjkdanny@naver.com>` (= GitHub `JunKyungSeo`, repo 소유자)
  - 커밋 identity는 이 repo의 local config로 맞춘다(전역 오염 금지). 지금은 `kindofwater`로 설정돼 있음 — 다른 팀원이면 `git config user.{name,email}` 바꿔 쓸 것.
- Jupyter `.ipynb_checkpoints/`와 `viewer_result/` 대용량 산출물은 `.gitignore` 처리됨.

---

## 5. 범위 밖 (후순위 / 다른 팀원)

GRAPPA prior+residual 배선 · bbox 가중 손실 · 증강 활성화 배선(`--aug-*` + `set_epoch`) · SVD coil compression · progressive expansion · config 스윕. 상세: `strategy/candidates.md`, `strategy/implementation.md`.

---

## 6. 참고 문서

- 전략 인덱스·SPEC: `strategy/README.md` · `strategy/confirmed_model.md` · `strategy/model_decision_points.md`
- 학습 로그 분석·수정 계획(누적): `training_anal/` · `docs/superpowers/plans/`
- 원본 SDUM 코드(참고, repo 밖): `../NV-SDUM-official/` (Apache-2.0) · 논문 arXiv:2512.17137
