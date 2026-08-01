# strategy/ — FastMRI 2026 전략 문서 인덱스

> 2026 SNU FastMRI Challenge (Annotation-guided Knee Reconstruction) 참가 전략 모음.
> 구조: **결정(SPEC) · 실행 · 후보 · 제외 · 근거**의 5역할로 분리. 결정은 `confirmed_model.md` 한 곳에서만 "정의", 나머지는 참조.
>
> 🔄 **작업 사이클(train→분석→코멘트→수정)은 [`collaboration_workflow.md`](collaboration_workflow.md).**

---

## Authoring Rules for `strategy/` — MUST FOLLOW (2026-08-01)

These rules bind **every file created or edited under `strategy/`**, and they bind the
AI agent doing the editing, not only the human. They exist because these documents are
read primarily by an agent that has **no access to the conversation in which a decision
was made** and cannot ask a follow-up question. Anything not written here is lost.

### R1 — Record what IS, never what MIGHT BE

Do **not** write forward-looking improvement proposals. Concretely, do not write:
"we could try X", "future work", "worth exploring later", "consider adding", "next step
would be", "추후 검토", "여력 되면".

Record only these three kinds of statement:

| Allowed | Required content |
|---|---|
| **Decision** | what was decided, the date, and the reason it was decided that way |
| **Measurement** | the numbers, the exact conditions they were measured under, and the script/log that produced them |
| **Constraint** | what is true and cannot be changed (scoring formula, hardware, competition rules) |

A proposal that has been neither decided nor measured does not belong in this folder.
If an idea was evaluated and dropped, that is a **decision** — record it with its reason
(this is what `rejected.md` is for). If an idea was measured and the result was
inconclusive, that is a **measurement** — record the numbers and say it was inconclusive.

### R2 — Optimize for AI readability, not human brevity

- **English is preferred** for new prose. Existing Korean text may stay; do not mass-translate.
- **Never abbreviate away the referent.** Write the full identifier: file path, symbol
  name, and line number — `utils/data/transforms.py::MRIAugmentor._flip_lr` (line 286),
  not "the flip function".
- **Every number carries its units and its conditions.** Not "ValLoss 0.1148" but
  "ValLoss 0.1148 (fold 2, acc8 only, 2000 iterations ≈ 1 epoch, seed 430, shell30)".
- **Every claim carries its evidence.** Name the log file, the script, or the commit hash.
  A claim with no traceable source must be labelled as an assumption.
- **Prefer explicit tables over compressed prose.** Repeating a noun is cheaper than
  making the reader resolve a pronoun.
- Do not rely on the reader having seen a sibling file. Restate the premise in one line,
  then link.

### R3 — Write for revision, and keep the revision history

- **Every file ends with a `## Change Log` section**, chronological, oldest entry first.
  **Write each entry long, not terse.** A future agent must be able to reconstruct the
  reasoning from the entry alone, without the conversation that produced it. Length is
  not a cost in this folder; a missing premise is. Use this shape:

  ```markdown
  - YYYY-MM-DD — <short title of the change>
    - **What changed**: the concrete edit, naming the section/claim that was touched.
    - **Why**: the reason the previous version was wrong, incomplete, or superseded.
      Include the reasoning, not just the conclusion.
    - **Evidence**: log file, script, commit hash, or measurement that justifies it.
      If there is none, say so explicitly and label the entry an assumption.
    - **Supersedes**: the earlier statement this replaces, quoted or referenced, so the
      reader can see what the position used to be and that it changed deliberately.
  ```

  A one-line entry is acceptable only for a pure typo or link fix.
- **When a statement is superseded, do not silently delete it.** Mark the old statement
  as superseded with its date, keep the original reasoning, and add the new statement
  next to it. A reader must be able to see that the position changed and why.
  Use `~~strikethrough~~` for a retracted number, and a dated `⚠️`/`🔁` note for a
  reversed conclusion.
- **Never leave an undated claim.** If you cannot date it, mark it `(date unknown)`.
- Structure sections so a single claim can be replaced without rewriting the file:
  one claim per bullet or per table row, not buried inside a paragraph.

### R1 and the two remaining forward-looking files — under human review, do not auto-edit

`next_levers.md` (ranked future levers) and the "전략 사각지대(가설)" half of
`gaps_and_blindspots.md` are **forward-looking by design**. Under a strict reading of R1
their content does not belong in `strategy/`.

**Status (2026-08-01): the human owner is reviewing these two files and will decide how
R1 applies to them.** Until that decision is recorded here, an AI agent must not
restructure, prune, or relocate their content on the basis of R1. Fixing a factually
wrong number or a stale claim inside them is still allowed, and still requires a
Change Log entry.

`candidates.md` was the third such file. It was **deleted on 2026-08-01** — see the
Change Log entry at the bottom of this file for the audit that justified it.

---

## 파일 역할 (읽는 순서)

| 파일 | 역할 | 언제 보나 |
|---|---|---|
| **[promptmr_plus.md](promptmr_plus.md)** | ✅ **현재 백본 SPEC (PromptMR+, 2026-07-19~)** | 백본·손실·학습 설정 |
| [confirmed_model.md](confirmed_model.md) | 📦 구 SDUM SPEC(superseded) — 모델 무관 요소는 유효 | 입력·정규화·GRAPPA 등 |
| **[implementation.md](implementation.md)** | 🔧 코드 배선 체크리스트(의존성 순서) | 배선하는 중 |
| **[next_levers.md](next_levers.md)** | 🔬 미결정 후보·열린 A/B(랭킹·기대이득·GPU시간) | 다음에 뭘 실험할지 |
| [augmentation_and_preprocessing.md](augmentation_and_preprocessing.md) | 🔖 증강·전처리 **결정 대기 항목**(사람 판단 필요, 일부 미반영) | 증강/전처리 켤지 정할 때 |
| **[gaps_and_blindspots.md](gaps_and_blindspots.md)** | 🕳️ 구현 정합성 갭(코드 확인) + 전략 사각지대(가설) | "우리가 뭘 놓쳤지?" |
| **[rejected.md](rejected.md)** | ❌ 제외된 것 + 이유 | 재논의 방지 |
| [model_strategy.md](model_strategy.md) · [augmentation.md](augmentation.md) · [preprocessing.md](preprocessing.md) · [related_work.md](related_work.md) | 📚 근거(rationale)·출처 아카이브 | "왜 이렇게 정했지?"·"선례 논문?" |

> ⚠️ **confirmed_model = 재구성 forward 파이프라인 구조만 확정.** 손실 함수·효율 레버·추가 증강은 아직 **next_levers.md**(미확정).

## 상태 대시보드

### ✅ 확정 (→ [confirmed_model.md](confirmed_model.md))
- 입력 2분기: **k-space readout(높이) 640→384 crop, 폭 native** + **GRAPPA 영상 도메인 유지(fft 안 함)**
- **전역 스칼라 정규화 ÷s** (입력유도, acc 무관 단일, k-space·GRAPPA·target 동일), 출력 ×s
- 증강: mask(k-space만)·flip(전부 동기), 스케줄 exp/ramp/constant (로직·CLI·epoch 배선 완료 2026-07-26, 기본 off — --aug-on으로 켠다. A/B는 shell22.sh ON arm 대기)
- 감도맵 **1회 추정 후 재사용** (mask=그대로 / flip=flip)
- main: VarNet cascade+DC, **GRAPPA 영상 prior 채널 + residual**

### 🔬 열린 후보 (→ [next_levers.md](next_levers.md))
> 랭킹·기대이득·GPU시간·검증여부는 `next_levers.md`(C1~C17)가 정본이다. 여기서 중복 정의하지 않는다.

**결론이 난 것 (더 이상 후보가 아님, 2026-08-01 대조)**
- `--fg-loss` (foreground 마스크 손실·검증 정합) → **채택**. run11에서 full 0.9161 / bbox 0.9095로 새 best.
- `--bbox-loss-lambda` (bbox 가중 손실) → **배선 완료·미채택**. run12에서 λ=0.5 퇴행 → 기본 0.0(off).
- coil compression (`--coil-compress`) → **배선 완료·미채택**.
- acc4/acc8 커리큘럼 → **채택**. MoE Option C(`--init-from`: 공유 사전학습 → acc별 미세조정)가 이것이다.
- 입력crop vs 출력crop → **입력crop 채택**(전 슬롯이 `--readout-crop`).
- SSIM_full 배경 마스크 정의 → **해결**(eval code 공개, `utils/common/metrics.py::foreground_mask`).
- acc-conditioned GRAPPA 가중(FiLM) → **기각**. MoE 전문가 분리가 같은 1비트 조건화의 상한(`next_levers.md` C5).
- k-space 랜덤 위상 · 라인 지터 · coil 노이즈 · 원본마스크 혼합비 → **코드에서 제거**(2026-07-28). `--aug-*`는 flip 계열만 남았다.

**미구현으로 남은 것** — GRAPPA 상보융합(`return_grappa`가 `train.py`에 없음) · 병변 슬라이스 오버샘플링 ·
중간 감독 · 고주파 집중 손실 · 배치 SME · progressive · FLAT · knowledge distillation ·
TensorRT(Pascal 이득 제한) · coil noise pre-whitening · ACS 저해상도 앵커 · TGVN(PD/PDFS 미확정)

### ❌ 제외 (→ [rejected.md](rejected.md))
- 기하: 위-아래 flip · 정수 translation · 회전/스케일/shear/elastic
- 강도: image 노이즈/밝기/gamma · cutout · MixUp/CutMix · 합성병변 삽입
- 구조: GRAPPA fft · 완전 384²(폭 pad) · x0 대체 · 2.5D 인접슬라이스 입력

## 하드 제약 (전 문서 공통)
knee 전용 · **`0.5·SSIM_full + 0.5·SSIM_bbox + 추론시간`** 채점(full·bbox 5:5, 2026-07-09 규칙변경) · GTX 1080 8GB FP32 · external/pretrained 금지 · 세부는 [`.claude/CLAUDE.md`](../.claude/CLAUDE.md).

## Change Log

- 2026-08-01 — Added the "Authoring Rules for `strategy/`" section
  - **What changed**: New top-level section between the intro block and "파일 역할",
    defining three binding rules for every file under `strategy/`: R1 (record what is,
    never what might be), R2 (optimize for AI readability, English preferred, never
    abbreviate away the referent), R3 (write for revision, keep a chronological
    Change Log). Also added this Change Log section to the README itself.
  - **Why**: These documents are consumed primarily by an AI agent that has no access to
    the conversation in which a decision was made and cannot ask a follow-up question.
    Two concrete failure modes had already occurred in this repo and motivated each rule:
    (a) a claim with no recorded measurement conditions — "bbox 내부 상관 0.08~0.59" from
    a 2-file sample with no control group — blocked the 2.5D adjacent-slice feature for a
    month until it was re-measured and refuted (see `model_strategy.md` §3); (b) stale
    comments asserting that augmentation was unwired survived the wiring commit and kept
    steering decisions, because nothing forced a dated supersede note.
  - **Evidence**: `model_strategy.md` §3 (re-measurement refuting the old correlation
    claim), `scripts/adjacent_slice_corr.py` (the script that produced it).
  - **Supersedes**: nothing — the folder previously had no authoring convention at all.

- 2026-08-01 — Amended R3 to require long-form Change Log entries, and changed the status
  of the three forward-looking files from "unresolved" to "under human review"
  - **What changed**: R3's Change Log template replaced a single-line
    `date — what — why — ref` format with a structured multi-line entry
    (What changed / Why / Evidence / Supersedes), and now states explicitly that length
    is not a cost here while a missing premise is. One-line entries are restricted to
    typo and link fixes. Separately, the R1-conflict subsection was retitled and now
    records that the human owner is reviewing `candidates.md`, `next_levers.md`, and the
    사각지대 half of `gaps_and_blindspots.md`; an AI agent must not restructure, prune, or
    relocate their content on R1 grounds until that decision is recorded here.
  - **Why**: A one-line entry cannot carry the premise of a decision, which defeats the
    purpose R3 exists for — the same failure mode R1 and R2 are guarding against. The
    three-file status was changed because ownership of that decision is now explicit, and
    leaving it marked "unresolved" invited an agent to act on its own reading of R1.
  - **Evidence**: instruction from the repository owner, this conversation, 2026-08-01.
  - **Supersedes**: the previous R3 template
    (`- YYYY-MM-DD — <what changed> — <why> — <commit or log reference>`) and the previous
    subsection heading "Why R1 conflicts with three existing files — resolve before
    editing them".

- 2026-08-01 — Deleted `strategy/candidates.md` and repointed every live reference to it
  - **What changed**: `strategy/candidates.md` was removed from the repository. The file-role
    table row now points to `next_levers.md`. The "🔬 열린 후보" dashboard section, which
    duplicated `candidates.md`'s contents, was rewritten to list what has since been
    **decided** versus what remains **unimplemented**, and to state that `next_levers.md`
    is the single authority for ranking. Fourteen inbound references in nine files were
    repointed: `CLAUDE.md` (line 116), `strategy/augmentation.md`, `strategy/model_strategy.md`,
    `strategy/preprocessing.md` (×2), `strategy/augmentation_and_preprocessing.md`,
    `strategy/confirmed_model.md`, `strategy/gaps_and_blindspots.md` (×3),
    `strategy/implementation.md` (×3), `strategy/model_decision_points.md` (×2),
    `strategy/README.md` (×4). References inside `docs/superpowers/plans/` and
    `docs/superpowers/specs/` were deliberately left untouched because those files are
    dated historical records, not current guidance.
  - **Why**: The file was created 2026-07-06 (`80f0a13`) and last edited 2026-07-12
    (`347fb43`), then went untouched for three weeks while the decisions it tracked were
    actually made. An item-by-item audit against the current code found **10 of its 24
    entries were already settled and the file still listed them as open**, so it was
    actively asserting false things: `--fg-loss` was recorded as "off·검증전" when run11
    had already adopted it as the new best; bbox weighted loss was listed as the "1순위"
    candidate after run12 had tried λ=0.5, regressed, and turned it off; 원본마스크 혼합비,
    k-space 랜덤 위상, 라인 지터, and coil 노이즈 were listed as tunable when the
    2026-07-28 decision had removed them from `utils/data/transforms.py` entirely; and
    acc-conditioned GRAPPA weighting (FiLM) was still recommended after `next_levers.md`
    C5 concluded MoE expert separation is the upper bound of that same one-bit
    conditioning. The 9 genuinely open items are all covered by `next_levers.md` with
    strictly more information (expected gain, GPU hours, whether the candidate was
    adversarially verified), making `candidates.md` an inferior duplicate. Under R1 a
    file whose decided entries are not recorded as decisions is worse than no file.
  - **Evidence**: `git log --follow strategy/candidates.md` for the dates; per-item grep
    of `train.py`, `utils/data/transforms.py`, `utils/data/load_data.py`, and
    `utils/model/promptmr/promptmr_plus.py` for implementation status;
    `training_anal/2026-07-14-run11-fgloss-analysis.md` (fg-loss adopted),
    `training_anal/2026-07-18-run12-bbox-uc-analysis.md` (bbox loss regressed),
    `strategy/next_levers.md` C5 (acc-conditioning killed).
  - **Supersedes**: `strategy/candidates.md` in its entirety, and the previous
    "🔬 열린 후보" dashboard bullets that mirrored it. Content deliberately not preserved:
    the paper citations that existed only in `candidates.md` (FiLM — Perez et al. AAAI 2018;
    Deeply-Supervised Nets — Lee et al. AISTATS 2015; Knowledge Distillation — Hinton et al.
    2015; SVD coil compression — Buehrer et al. MRM 2007 / Zhang et al. MRM 2013). The
    repository owner accepted this loss on 2026-08-01. The A/B judging rule that
    `candidates.md` stated in its header (SSIM_bbox non-regression as a hard gate,
    incremental one-change-at-a-time) was **not** lost — it is defined in
    `augmentation.md` lines 393 and 403, which `candidates.md` was merely citing.
