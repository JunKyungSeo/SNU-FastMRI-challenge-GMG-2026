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

- **Every file ends with a `## Change Log` section**, chronological, oldest entry first:
  `- YYYY-MM-DD — <what changed> — <why> — <commit or log reference>`
- **When a statement is superseded, do not silently delete it.** Mark the old statement
  as superseded with its date, keep the original reasoning, and add the new statement
  next to it. A reader must be able to see that the position changed and why.
  Use `~~strikethrough~~` for a retracted number, and a dated `⚠️`/`🔁` note for a
  reversed conclusion.
- **Never leave an undated claim.** If you cannot date it, mark it `(date unknown)`.
- Structure sections so a single claim can be replaced without rewriting the file:
  one claim per bullet or per table row, not buried inside a paragraph.

### Why R1 conflicts with three existing files — resolve before editing them

`candidates.md` (open A/B candidates), `next_levers.md` (ranked future levers), and the
"전략 사각지대(가설)" half of `gaps_and_blindspots.md` are **forward-looking by design**.
Under a strict reading of R1 their content does not belong in `strategy/`. This is
unresolved: decide whether R1 exempts those three files, or whether their content moves
elsewhere, before editing them.

---

## 파일 역할 (읽는 순서)

| 파일 | 역할 | 언제 보나 |
|---|---|---|
| **[promptmr_plus.md](promptmr_plus.md)** | ✅ **현재 백본 SPEC (PromptMR+, 2026-07-19~)** | 백본·손실·학습 설정 |
| [confirmed_model.md](confirmed_model.md) | 📦 구 SDUM SPEC(superseded) — 모델 무관 요소는 유효 | 입력·정규화·GRAPPA 등 |
| **[implementation.md](implementation.md)** | 🔧 코드 배선 체크리스트(의존성 순서) | 배선하는 중 |
| **[candidates.md](candidates.md)** | 🔬 미결정 후보·열린 A/B | 다음에 뭘 실험할지 |
| [augmentation_and_preprocessing.md](augmentation_and_preprocessing.md) | 🔖 증강·전처리 **결정 대기 항목**(사람 판단 필요, 일부 미반영) | 증강/전처리 켤지 정할 때 |
| **[gaps_and_blindspots.md](gaps_and_blindspots.md)** | 🕳️ 구현 정합성 갭(코드 확인) + 전략 사각지대(가설) | "우리가 뭘 놓쳤지?" |
| **[rejected.md](rejected.md)** | ❌ 제외된 것 + 이유 | 재논의 방지 |
| [model_strategy.md](model_strategy.md) · [augmentation.md](augmentation.md) · [preprocessing.md](preprocessing.md) · [related_work.md](related_work.md) | 📚 근거(rationale)·출처 아카이브 | "왜 이렇게 정했지?"·"선례 논문?" |

> ⚠️ **confirmed_model = 재구성 forward 파이프라인 구조만 확정.** 손실 함수·효율 레버·추가 증강은 아직 **candidates.md**(미확정).

## 상태 대시보드

### ✅ 확정 (→ [confirmed_model.md](confirmed_model.md))
- 입력 2분기: **k-space readout(높이) 640→384 crop, 폭 native** + **GRAPPA 영상 도메인 유지(fft 안 함)**
- **전역 스칼라 정규화 ÷s** (입력유도, acc 무관 단일, k-space·GRAPPA·target 동일), 출력 ×s
- 증강: mask(k-space만)·flip(전부 동기), 스케줄 exp/ramp/constant (로직·CLI·epoch 배선 완료 2026-07-26, 기본 off — --aug-on으로 켠다. A/B는 shell22.sh ON arm 대기)
- 감도맵 **1회 추정 후 재사용** (mask=그대로 / flip=flip)
- main: VarNet cascade+DC, **GRAPPA 영상 prior 채널 + residual**

### 🔬 열린 후보 (→ [candidates.md](candidates.md))
- **구현됨(검증 전, off)**: **foreground 마스크 손실·검증 정합**(`--fg-loss`, 채점 SSIM_full 영역만; 2026-07-11) — A/B로 이득 확인 대기. [implementation §3c](implementation.md)
- **1순위 손실**: bbox 가중 손실 · 병변 슬라이스 오버샘플링
- **효율 A군(택1 A/B, SME비용)**: 배치 SME(잠정채택) / **coil compression(SVD, 데이터측 구현됨)** / progressive / FLAT
- **효율 B군(직교·조합가능)**: knowledge distillation(teacher→student) / TensorRT(배포, Pascal 이득 제한)
- **즉시 A/B**: 입력crop vs 출력crop · GRAPPA 상보융합(input⊕grappa)
- **대기**: TGVN(PD/PDFS 확인)  *(SSIM_full 배경 마스크는 eval code 공개로 정의 확정 → 위 "구현됨"으로 이동)*

### ❌ 제외 (→ [rejected.md](rejected.md))
- 기하: 위-아래 flip · 정수 translation · 회전/스케일/shear/elastic
- 강도: image 노이즈/밝기/gamma · cutout · MixUp/CutMix · 합성병변 삽입
- 구조: GRAPPA fft · 완전 384²(폭 pad) · x0 대체 · 2.5D 인접슬라이스 입력

## 하드 제약 (전 문서 공통)
knee 전용 · **`0.5·SSIM_full + 0.5·SSIM_bbox + 추론시간`** 채점(full·bbox 5:5, 2026-07-09 규칙변경) · GTX 1080 8GB FP32 · external/pretrained 금지 · 세부는 [`.claude/CLAUDE.md`](../.claude/CLAUDE.md).

## Change Log

- 2026-08-01 — Added the "Authoring Rules for `strategy/`" section (R1 record-what-is /
  R2 AI-readability / R3 revision history). Reason: these documents are consumed by an
  agent with no access to the originating conversation, so unstated context is lost and
  undated claims keep re-litigating settled decisions. Flagged the unresolved conflict
  between R1 and the three forward-looking files (`candidates.md`, `next_levers.md`,
  `gaps_and_blindspots.md` 사각지대 half).
