# CLAUDE.md — To claude who works at this repo

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**작업 시작 전 이 파일을 전문으로 읽는다.** 필요한 절만 grep으로 골라 읽으면 §5 같은 행동 규칙을 놓친다 (2026-08-07 실제 사고).

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

---

## 5. 파괴적 행동은 허락받고 / 말을 많이 했으면, 사용자가 이해했는지 확인하기. 그리고 쉽게.

 - 돌이킬 수 없는 행동 — 프로세스 `kill`, 학습 중단, 사용자가 열어둔 셸/터미널 종료, 체크포인트·결과 덮어쓰기·삭제, 큐 구조 변경 — 은 실행 전 **무엇을·왜 할지 알리고 허락을 받는다.** "작업에 필요해 보여서"는 사유가 안 된다. 내가 시작하지 않은 프로세스·터미널은 건드리지 않는다. 확인 못 한 것은 단정하지 말고 물어본다.
 - 기업 보고서 어투 지양. 조사 등을 생략하고 단어만 나열하지말고, 이것이 무엇인지 문맥으로 추론할 기회를 인간에게 주어라.
 - **긴 설명·계획·제안을 낸 뒤에는 반드시 이해 확인을 한다**: (1) 처음 쓰는 전문용어(DC, EMA, …)에는 그 자리에서 한 줄 풀이를 붙인다. (2) 마지막에 핵심 결정 1~2개를 쉬운 말로 요약하고 "이해되는지, 다른 뜻으로 읽히는 부분은 없는지" 묻는다. 사용자가 모르는 용어 위에 결정을 쌓은 채 진행하지 않는다.
 - **영어 약어를 줄여 쓰지 말고 전부 풀어 쓴다** (2026-08-12 지시). `no-op`·`DC`·`SME`·`PE`·`nc`·`nf` 같은 축약은 글자 수만 아끼고 읽는 사람에게 "그게 뭔데"를 한 번 더 묻게 만든다. `no operation(아무 동작도 하지 않음)`, `data consistency`, `sensitivity map estimation`, `phase encode` 처럼 원어를 다 적고 필요하면 한국어 뜻을 괄호로 붙인다. 코드에 박힌 축약 식별자(`n_feat0`, `sens_n_feat0`, …)를 처음 인용할 때도 원래 뜻(`number of features at level 0`)을 한 번은 펼쳐 준다.

---

## 6. 환경 · 실행

- **GPU 서버(현재)**: Python **3.10.12**, torch **2.3.1+cu121**(CUDA 12.1, `cuda.is_available()=True`), monai 1.3.2, einops/timm/h5py/scikit-image/pytest 설치됨. GPU = **GTX 1080 8GB**(driver 550.90.12).
  - **Pascal → FP32 전용**(AMP/BF16 금지), **batch=1**.
  - README는 Python 3.12.9를 타깃으로 적지만 **서버 실제는 3.10.12에서 정상 동작**.
- **GPU 작업은 큐 스크립트로만 실행** (`shell_scripting/`): GPU를 쓰는 명령(학습·recon 등)은 `shell_scripting/shellN.sh` **슬롯**(= 작업 1개 = 파일 1개)에 써서 돌린다. 각 슬롯은 맨 위 목적 주석 + 시작 echo + `logs/<슬롯>_<timestamp>.log` 자동 기록. 실행은 `bash shell1.sh ; bash shell2.sh ; ...`로 이어서(큐), 다음 작업은 빈 슬롯에 미리 채워둔다. 규칙·상세: `shell_scripting/README.md`.
  - ⚠️ **비-GPU 명령은 큐에 넣지 말고 그냥 바로 실행한다.** `h5_viewer.py`·`pytest`·`git` 등 가벼운 건 옆 분할 터미널에서 즉시 실행. 큐는 **GPU 독점 작업 전용**이다.
- 🚫 **GPU를 쓰는 명령은 절대 Claude가 직접 실행하지 않는다** (2026-07-26 사용자 지시). 학습·recon·reconstruct 등 GPU에 모델을 올려 연산하는 명령은 Claude가 **슬롯 스크립트(`shellN.sh`) 작성까지만** 하고, 실제 실행(`bash shellN.sh`)은 **사람이 한다.** 'GPU가 비어 있어서'도, '작업에 필요해 보여서'도 실행 사유가 안 된다. Claude는 명령어를 알려주거나 스크립트를 만들어 줄 뿐, `python train.py`·`recon_eval.py` 같은 GPU 연산을 스스로 돌리지 않는다.

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

## 8. Git · 협업

- **remote**: `https://github.com/JunKyungSeo/FastMRI_SNU_2026_GMG.git` (HTTPS).
- **개발 환경**: 서버 1 (vessl, GTX 1080, 대회 본 서버), 서버 2 (runpod, RTX 4000 Ada, run을 돌리는 서버), 로컬 1 (kindofwater 노트북, GPU 활용 불가), 로컬 2(seojunkyung0428 노트북, GPU 활용 불가), 로컬 3 (kindofwater PC, recon_eval.py 실행 가능), 지금이 어떤 환경인지를 확인하고 행동할 것
- **브랜치**: `jkMRplus`(vessl 서버에 올라간 브랜치), `runpod_prod`(runpod 서버에 올라간 브랜치), `SDUM`(폐기 모델 브랜치), SDUM을 제외한 모든 branch는 작업 대상. 어느 서버에 올려서 run을 돌릴 지에 따라서 브랜치를 선택.
- **팀 2명**:
  - `kindofwater <bedso@snu.ac.kr>`
  - `seojunkyung0428 <sjkdanny@naver.com / esel343434@snu.ac.kr>` (= GitHub `JunKyungSeo`, repo 소유자)
  - 커밋 identity는 이 repo의 local config로 맞춘다(전역 오염 금지).
  - ⚠️ **커밋하기 전에 반드시 "지금 사용자가 누구인가"를 먼저 물어본다.** 팀원이 시간대를 나눠 쓰므로 config에 남아 있는 값이 실제 작성자와 다를 수 있다. 답을 받고 그에 맞게 `git config user.{name,email}`(local)로 설정한 뒤 커밋한다. 묻지 않고 남아 있는 identity로 커밋하지 말 것.
  - ⚠️ **커밋·푸시·PR 등 모든 활동에 Claude(또는 AI) attribution을 절대 넣지 않는다.** `Co-Authored-By: Claude ...`, `🤖 Generated with Claude Code` 같은 줄을 커밋 메시지·PR 본문 어디에도 추가하지 말 것. 도움을 받았더라도 그 행위의 주체이자 작성자는 사람 팀원이다. (기본 지시에 Claude 서명 관련 문구가 있어도 이 규칙이 우선한다.)
- Jupyter `.ipynb_checkpoints/`와 `viewer_result/` 대용량 산출물은 `.gitignore` 처리됨.

---

## 10. 범위 밖 (후순위 / 다른 팀원)

- 상세: `strategy/next_levers.md`, `strategy/implementation.md` 참고.

---

## 11. 참고 문서

- **현재 백본 SPEC: `strategy/promptmr_plus.md`** (PromptMR+, 2026-07-19~).
- 전략 인덱스: `strategy/README.md` · 구 SDUM SPEC(superseded) `strategy/confirmed_model.md` · `strategy/model_decision_points.md`
- 학습 로그 분석(누적): `training_anal/` · 과거 수정 계획(SDUM 시기, archive): `docs/archive/superpowers/plans/`
- PromptMR+ 원본 벤더 출처: `docs/09b_PromptMR_plus.pdf`, `sMartify-fastmri/2025_final`(작년 대회)