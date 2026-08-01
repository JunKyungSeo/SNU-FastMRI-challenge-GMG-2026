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

Two failures in this repository produced these rules; they are recorded here so the rules
are not later dismissed as generic style advice.

1. **A number without its conditions blocked a feature for a month.** `model_strategy.md`
   §3 asserted that lesion-bbox inter-slice correlation was 0.08–0.59, and `rejected.md`
   used it to reject 2.5D adjacent-slice input on 2026-07-02. The measurement came from a
   **2-file sample with no control group**, and the script that produced it was a scratchpad
   file that no longer exists. Re-measured on 2026-08-01 over 170 files with same-size
   control boxes, lesion correlation was *higher* than control — the low absolute value had
   been an artifact of computing Pearson on small patches. → **R2**: every number carries its
   conditions and a traceable source.
2. **Stale comments outlived the code they described and kept steering decisions.** After
   augmentation was wired on 2026-07-26, `load_data.py` and `augmentation_and_preprocessing.md`
   still stated that the `--aug-*` arguments did not exist and that augmentation was "dead",
   because nothing forced a dated supersede note. → **R1** and **R3**: a decision must be
   recorded as a decision, and a superseded claim must be visibly superseded, with its date.

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

- **The revision history goes inside the entry that changed, not in a section about the
  document.** Do not add a "Change Log", "수정 내역", or "이 문서의 한계" section. Narrating
  your own edits ("moved section 4 into L07", "retitled this subsection", "deleted the
  estimate column") is editorial bookkeeping — git already holds it, in a form that cannot
  drift out of sync with the file. What the document must hold is the **subject-matter**
  history: when the idea appeared, when it landed in code, when it ran, what it measured.
- **Write each entry long, not terse.** A future agent must be able to reconstruct the
  reasoning from the entry alone, without the conversation that produced it. Length is not a
  cost in this folder; a missing premise is. An entry that records a lifecycle should carry
  each stage with its own date and its own evidence, for example:

  ```markdown
  ### <item name>
  - **Idea**: date first recorded, and where. If the earliest trace is a commit rather than a
    document, say so.
  - **Code**: commit hash + date, or `not implemented`.
  - **Executed**: log file + date, or `not executed`.
  - **Result**: the numbers with their measurement conditions, or `not measured`.
  - **Status**: the current decision, dated, with the reason it was decided that way.
  ```

  A field that cannot be filled says `not implemented` / `not executed` / `not measured`
  rather than being omitted — an empty field cannot be told apart from an unchecked one.
- **When a statement is superseded, do not silently delete it.** Mark the old statement
  as superseded with its date, keep the original reasoning, and put the new statement next
  to it, in the same entry. A reader must be able to see that the position changed and why.
  Use `~~strikethrough~~` for a retracted number, and a dated `⚠️`/`🔁` note for a
  reversed conclusion.
- **Never leave an undated claim.** If you cannot date it, mark it `(date unknown)`.
- Structure sections so a single claim can be replaced without rewriting the file:
  one claim per bullet or per table row, not buried inside a paragraph.
- **Keep the file at one depth.** Every fact belongs inside the entry that acts on it. Do not
  create a "general findings" or "cross-cutting" section — a measurement parked away from the
  decision it changes is a measurement nobody applies. The concrete failure this rule comes
  from: `next_levers.md` carried a two-server wall-clock table showing a 2.5× difference in a
  general-findings section, while every entry above it still quoted GPU hours without naming
  a server.

### R1 and the two remaining forward-looking files — under human review, do not auto-edit

`next_levers.md` (ranked future levers) and the "전략 사각지대(가설)" half of
`gaps_and_blindspots.md` are **forward-looking by design**. Under a strict reading of R1
their content does not belong in `strategy/`.

**Status (2026-08-01): the human owner is reviewing these two files and will decide how
R1 applies to them.** Until that decision is recorded here, an AI agent must not
restructure, prune, or relocate their content on the basis of R1. Fixing a factually
wrong number or a stale claim inside them is still allowed, and must be dated where it is
made.

`candidates.md` was the third such file. It was **deleted on 2026-08-01** after an
item-by-item audit against the code found that 10 of its 24 entries were already decided
while the file still listed them as open — `--fg-loss` was recorded as unverified after
run11 had adopted it, bbox weighted loss was still the "1순위" candidate after run12 had
turned it off, and four augmentations it offered for tuning had been removed from
`utils/data/transforms.py` on 2026-07-28. Its surviving open items are all covered by
`next_levers.md` with more information. Commit `c4ad79b` holds the deletion and the
reference cleanup.

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
