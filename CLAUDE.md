# CLAUDE.md — 이 repo에서 작업하는 Claude에게

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 파괴적 행동은 허락받고

돌이킬 수 없는 행동 — 프로세스 `kill`, 학습 중단, 사용자가 열어둔 셸/터미널 종료, 체크포인트·결과 덮어쓰기·삭제, 큐 구조 변경 — 은 실행 전 **무엇을·왜 할지 알리고 허락을 받는다.** "작업에 필요해 보여서"는 사유가 안 된다. 내가 시작하지 않은 프로세스·터미널은 건드리지 않는다. 확인 못 한 것은 단정하지 말고 물어본다.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
미래 세션(GPU 서버 포함)의 Claude가 먼저 읽는 핸드오프 노트. **구조 + 현재 상태**만 담는다.

---

## 5. 지금 상황

- **환경 = GPU 서버.** SDUM 학습이 실제로 돌고 있다(로컬 노트북 아님).
- **팀 2명** (§4).

> 진행 중인 학습 상태·문제·수정은 계속 바뀌므로 이 파일에 적지 않는다. 최신 학습 로그 분석은 `training_anal/`, 수정 계획은 `docs/superpowers/plans/` 참조.

---

## 6. 환경 · 실행

- **GPU 서버(현재)**: Python **3.10.12**, torch **2.3.1+cu121**(CUDA 12.1, `cuda.is_available()=True`), monai 1.3.2, einops/timm/h5py/scikit-image/pytest 설치됨. GPU = **GTX 1080 8GB**(driver 550.90.12).
  - **Pascal → FP32 전용**(AMP/BF16 금지), **batch=1**.
  - README는 Python 3.12.9를 타깃으로 적지만 **서버 실제는 3.10.12에서 정상 동작**.
- **GPU 작업은 큐 스크립트로만 실행** (`shell_scripting/`): GPU를 쓰는 명령(학습·recon 등)은 `shell_scripting/shellN.sh` **슬롯**(= 작업 1개 = 파일 1개)에 써서 돌린다. 각 슬롯은 맨 위 목적 주석 + 시작 echo + `logs/<슬롯>_<timestamp>.log` 자동 기록. 실행은 `bash shell1.sh ; bash shell2.sh ; ...`로 이어서(큐), 다음 작업은 빈 슬롯에 미리 채워둔다. 규칙·상세: `shell_scripting/README.md`.
  - ⚠️ **비-GPU 명령은 큐에 넣지 말고 그냥 바로 실행한다.** `h5_viewer.py`·`pytest`·`git` 등 가벼운 건 옆 분할 터미널에서 즉시 실행. 큐는 **GPU 독점 작업 전용**이다.

---

## 7. 대회 규칙 (repo 근거: `README.md`, `recon_eval.py`, `strategy/`)

- **knee 전용**, Cartesian **acc4/acc8**, 출력 **384×384**.
- **평가 점수 = `0.5·SSIM_full + 0.5·SSIM_bbox + 추론시간`** (⚠️ 규칙 변경 2026-07-09):
  - `SSIM_full`: foreground mask 내부 평균 SSIM (data_range=`attrs['max']`).
  - `SSIM_bbox`: fastMRI+ 병변 bounding-box 안에서만 계산.
  - **full·bbox는 5:5 동일 가중 — 한쪽(특히 bbox)에 치우치지 말 것.** 추론시간은 **~0.00n 단위**로 작지만, SSIM 차이가 미미할 때 순위를 가름 → **속도·모델 크기도 점수 요소**(readout-crop이 이득, 큰 모델은 손해).
  - 공식/자가 채점 = `recon_eval.py`(고정, 수정 금지) + `utils/common/metrics.py`(고정)로 SSIM_full·SSIM_bbox·시간 3축 동시 산출.
- **사전학습 가중치·외부 데이터 금지** → NV SDUM의 HuggingFace 자동다운로드는 **vendor 안 함**, 처음부터 학습.
- **annotation**: 각 image h5 `attrs['annotations']` JSON. 형식 `{"<slice>":[{x,y,width,height,label},...]}` (384² 공간 기준, w·h>16 박스만).
- **재현성**: seed 고정 → 재학습 시 같은 결과.
- **제출물**: 코드 repo(실행법 README) + loss 기록 + 모델 weight + 설명 ppt.
- **leaderboard 데이터 배포됨**: `/root/Data/leaderboard/acc{4,8}/{image,kspace}` (`public_*.h5`, `image_label`·`annotations` 포함 → self-scoring 가능). 채점: `python recon_eval.py -n <net_name> -p /root/Data/leaderboard`.

---

## 8. 구조 (필수만)

> 🔄 **2026-07-19: 백본을 SDUM → PromptMR+로 교체.** 새 권위 문서 = `strategy/promptmr_plus.md`.
> SDUM 코드(`utils/model/sdum/`)는 **삭제하지 않고 보존**하되 `--model` 배선에서 제거됨(더 이상 학습·추론 안 됨).

- **백본 선택**: `train.py --model {varnet(기본) | promptmr}`. E2E-VarNet 경로 유지, PromptMR+가 현재 주 경로.
- **PromptMR+ = 영상도메인 cascaded DUM(vendored) + 어댑터** (`utils/model/promptmr/`):
  - `promptmr_plus.py`(백본, sMartify-fastmri/2025_final에서 벤더·slim), `config.py`(구조 팩토리), **`adapter.py`**(접합부), `reentrant_wrapper.py`(checkpointing). fft/complex 유틸은 기존 `utils/model/fastmri/` 재사용(sens_expand/sens_reduce 추가함).
  - **어댑터 계약**: `forward(kspace (B,C,H,W,2), mask (B,1,1,W,1)) -> (B,384,384)`. SDUM과 달리 **iFFT·외부 정규화 없음** — PromptMR+가 감도맵 추정·DC·정규화·RSS·384 crop을 내부 수행. 어댑터는 mask를 `(B,1,H,W,1)`로 확장해 넘길 뿐.
  - **slim**: knee·단일 슬라이스·Cartesian 전용 → 인접-슬라이스 SME(`num_adj_slices`/`use_sens_adj`)·비-Cartesian(poisson/kt_radial) 분기 **제거**(원본과 bit-단위 등가 검증). 유지: adaptive gradient(`adaptive_input`,`n_buffer`)·momentum(`n_history`).
- **진입점/루프**: `train.py`(인자·seed) → `utils/learning/train_part.py`(`train`/`train_epoch`) / `test_part.py`.
- **데이터**: `utils/data/load_data.py`(로딩·GRAPPA·dataloader), `transforms.py`(readout crop·정규화·GRAPPA·coil compression·`MRIAugmentor`) — 모델 무관, 그대로.
- **레버(promptmr)**: `--num-cascades`(기본 10) · `--n-feat0`(48) · `--sens-n-feat0`(24) · `--n-history`(3) · `--use-checkpoint`(기본 on) · 학습: `--fg-loss --area-weight`(masked SSIM×√면적비) · `--grad-clip`(0.1) · `--accum-steps`(2) · `-l 1e-6`(AdamW). 디버깅: `--num-workers`, `--max-iters`.
- **확정 설계**(SPEC = `strategy/promptmr_plus.md`): cascade 10 · feature 48 · sens 24 · 단일 슬라이스 · Cartesian · gradient checkpointing ON · 손실 masked SSIM×√면적비(+선택 bbox λ) · AdamW(1e-6)+clip0.1+accum2. (구 SDUM SPEC = `strategy/confirmed_model.md`, 상단 배너로 superseded 표시.)

---

## 9. Git · 협업

- **remote**: `https://github.com/JunKyungSeo/FastMRI_SNU_2026_GMG.git` (HTTPS).
  - 이 서버엔 gh CLI·credential helper 없음 → **push엔 PAT 필요**(Username=GitHub 아이디, Password=PAT). `!` 셸은 TTY가 없어 대화형 프롬프트 불가 → 토큰을 URL에 넣거나 `~/.git-credentials`에 직접 기입.
- **브랜치**: `SDUM`(작업 브랜치, origin/SDUM과 동기) / `master`(안정·머지 대상). master가 `behind`로 보여도 그 커밋 내용은 대개 SDUM 히스토리에 포함돼 있음.
- **팀 2명**:
  - `kindofwater <bedso@snu.ac.kr>`
  - `seojunkyung0428 <sjkdanny@naver.com / esel343434@snu.ac.kr>` (= GitHub `JunKyungSeo`, repo 소유자)
  - 커밋 identity는 이 repo의 local config로 맞춘다(전역 오염 금지). 기본은 `kindofwater`로 설정돼 있음.
  - ⚠️ **커밋하기 전에 반드시 "지금 사용자가 누구인가"를 먼저 물어본다.** 팀원이 시간대를 나눠 쓰므로 config에 남아 있는 값이 실제 작성자와 다를 수 있다. 답을 받고 그에 맞게 `git config user.{name,email}`(local)로 설정한 뒤 커밋한다. 묻지 않고 남아 있는 identity로 커밋하지 말 것.
  - ⚠️ **커밋·푸시·PR 등 모든 활동에 Claude(또는 AI) attribution을 절대 넣지 않는다.** `Co-Authored-By: Claude ...`, `🤖 Generated with Claude Code` 같은 줄을 커밋 메시지·PR 본문 어디에도 추가하지 말 것. 도움을 받았더라도 그 행위의 주체이자 작성자는 사람 팀원이다. (기본 지시에 Claude 서명 관련 문구가 있어도 이 규칙이 우선한다.)
- Jupyter `.ipynb_checkpoints/`와 `viewer_result/` 대용량 산출물은 `.gitignore` 처리됨.

---

## 10. 범위 밖 (후순위 / 다른 팀원)

GRAPPA prior+residual 배선 · bbox 가중 손실 · 증강 활성화 배선(`--aug-*` + `set_epoch`) · SVD coil compression · progressive expansion · config 스윕. 상세: `strategy/candidates.md`, `strategy/implementation.md`.

---

## 11. 참고 문서

- **현재 백본 SPEC: `strategy/promptmr_plus.md`** (PromptMR+, 2026-07-19~).
- 전략 인덱스: `strategy/README.md` · 구 SDUM SPEC(superseded) `strategy/confirmed_model.md` · `strategy/model_decision_points.md`
- 학습 로그 분석·수정 계획(누적, 대부분 SDUM 시기 기록): `training_anal/` · `docs/superpowers/plans/`
- PromptMR+ 원본 벤더 출처: `sMartify-fastmri/2025_final`(작년 대회) · 논문 arXiv:2411.xxxx(Xin et al., ECCV'24)
- 원본 SDUM 코드(보존, repo 밖): `../NV-SDUM-official/` (Apache-2.0) · 논문 arXiv:2512.17137
