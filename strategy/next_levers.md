# next_levers.md — Lever Ledger (레버 원장)

> **What this file is**: a dated ledger of every performance lever considered on top of the
> PromptMR+ backbone. For each lever it records **when the idea was first written down, when
> it landed in code, when it was executed, and what the measurement said** — in that order.
> **What this file is not**: a ranked wish list. Ranking and expected-gain estimates were
> removed on 2026-08-01; they were unfalsifiable guesses that outlived the measurements
> that contradicted them. A lever with no execution date has no claimed gain here.
>
> **Uniform depth**: every fact in this file lives inside the lever entry it bears on, or in
> the Change Log if it describes the file itself. There is no separate "general findings"
> section — a measurement that floats free of a lever is a measurement nobody acts on.
>
> Authoring rules for this folder: [`README.md`](README.md) R1/R2/R3. This file is subject
> to R1's forward-looking restriction; the human owner exempted it as a designated
> proposal-tracking file on 2026-08-01.

---

## 0. How to read this file

Every lever entry carries the same five fields. A field that cannot be filled says so
explicitly rather than being omitted.

| Field | Meaning |
|---|---|
| **Idea** | Date the idea was first recorded, and where. If the earliest trace is a code commit rather than a document, that is stated. |
| **Code** | Commit hash + date the wiring landed, or `not implemented`. |
| **Executed** | Log file + server-clock date of the run, or `not executed`. |
| **Result** | Numbers with their measurement conditions, or `not measured`. |
| **Status** | `open` (no execution yet) / `closed-measured` (executed, decision made) / `closed-unmeasured` (dropped by reasoning alone) |

### ⚠️ Which dates are trustworthy

- **Execution dates are reliable.** They come from the log filename and the `[START]` line,
  both written by the server clock at run time.
- **Code-landing dates are approximate.** This branch has been rebased onto `origin/jkMRplus`
  many times, and rebase rewrites the committer date. Example of the damage: commit `12bbe7e`
  (2.5D adjacent-slice wiring) reports author date `2026-08-01 02:06`, but
  `shell_scripting/logs/shell30_f2_20260731_181035.log` shows a run that *used* `12bbe7e`
  starting `2026-07-31 18:10:35`. The commit therefore existed earlier than its recorded
  date. **When a log's `[BR]` line names a hash, that log's date is the harder upper bound
  on when that code existed.**

---

## 1. Baseline the levers are measured against

| Metric | Value | Conditions |
|---|---|---|
| SSIM_full | 0.9321 | MoE (acc4/acc8 expert split), no augmentation, leaderboard set |
| SSIM_bbox | 0.9300 | same run |
| Recon time | 734.3 ms/slice | same run |
| acc4 breakdown | full 0.9497 / bbox 0.9505 | `shell20_moe_reconstruct_20260726_060814.log` |
| **acc8 breakdown** | **full 0.9145 / bbox 0.9094** | same log — this is where the headroom is (gap 0.035 / 0.041) |

⚠️ **Scale warning that governs every entry below.** The measured gain from MoE expert
separation itself was **acc8 +0.0014 full / +0.0020 bbox**. No seed-to-seed variance has ever
been measured on this pipeline, so any lever whose effect is of that same order cannot be
declared a win or a loss from a single run. This is not a hypothetical concern: the 2.5D
adjacent-slice probe (L06) flipped sign between a 120-iteration and a 2000-iteration run.

---

## 2. Open levers — code exists or idea recorded, not yet executed

### L01 — Terminal LR decay (`--lr-decay-start-epoch`, `--lr-min-factor`)

- **Idea**: recorded 2026-07-31 in this document's predecessor version as candidate `C3`
  ("종단 LR 감쇠"). Earlier motivation: the MoE fine-tune stage ran `lr 5e-5 × 3 epoch` with
  effectively no change, i.e. the fine-tune had never actually done anything.
- **Code**: `c734b27` (author date 2026-08-01 05:49) added `--lr-decay-start-epoch` and
  `--lr-min-factor` to `train.py`, plus slots `shell_scripting/shell31.sh` and
  `shell_scripting/shell31_recon.sh`.
- **Executed**: `not executed` — no `shell31_*` log exists under `shell_scripting/logs/`.
- **Result**: `not measured`.
- **Status**: open.
- **External comparison that motivates it**: the 2025 winning clone — the repository this
  project's PromptMR+ backbone was vendored from — trained far longer with a decaying LR.
  Read from `2025_4th_clone/FastMRI_challenge_sMaRtIfy_final/` on 2026-08-01:

  | | sMaRtIfy final | This repo |
  |---|---|---|
  | epochs | **30** (+ 65 for the MoE stage) | 10 |
  | LR schedule | **StepLR(step_size 11, gamma 0.3)**, from epoch 5 | constant |
  | `num_cascades` | 8 | 10 |
  | `n_feat0` | **8** | **48** |
  | `feature_dim` | **[24, 32, 40]** | **[72, 96, 120]** |
  | `sens_n_feat0` | 8 | 24 |
  | MRAugment | `--aug_strength 0`, i.e. **off** | left-right flip only |
  | k-space mask augmentation | on, weight 1.0, from epoch 15, exp schedule | removed 2026-07-28 |
  | `num_adj_slices` | **1**, marked `TODO ... not implemented` in `train_part.py` | 1 (3 available since L06) |

  Two readings bear on this lever. First, with `--num-epochs 10` a `step_size` of 11 never
  fires, so **LR decay has genuinely never been exercised in this pipeline** — the claim in
  the 2026-07-31 version of this document that it had been tried was false. Second, their
  model is roughly 15× narrower than ours while training 3× longer, which is the opposite
  trade from the one this repo made; that asymmetry is unmeasured and is the reason L01 is
  worth running before any capacity lever (L03).

### L02 — `best_model` selection informed by SSIM_bbox

- **Idea**: recorded 2026-07-31 as candidate `C4`. Rationale: SSIM_bbox is half the score,
  but checkpoint selection only looks at validation loss.
- **Code**: `not implemented`. Verified 2026-08-01: `utils/learning/train_part.py::save_model`
  takes an `is_new_best` flag derived from `best_val_loss` alone (lines 323–335, 402–410).
  No SSIM_bbox term participates in the comparison.
- **Executed**: `not executed`.
- **Result**: `not measured`.
- **Status**: open.

### L03 — Extra cascades appended to the acc8 expert only, earlier stages frozen

- **Idea**: recorded 2026-07-31 as candidate `C7`. Rationale: acc4 appears saturated
  (full 0.9497) while acc8 does not (0.9145), yet both experts get identical capacity
  (`--num-cascades 10`).
- **Code**: `not implemented`.
- **Executed**: `not executed`.
- **Result**: `not measured`.
- **Status**: open.

### L04 — Frozen backbone + second-stage image-domain refiner

- **Idea**: recorded 2026-07-31 as candidate `C8`. Precedent cited at the time: E2E-VarNet
  0.8936 → 0.8953 with a 2M-parameter ShiftNet on fastMRI knee ×8; and SNU top-placing teams
  repeating the pattern (2022 3rd place froze VarNet and trained an image-domain network;
  2024 used VarNet + NAFNet).
- **Code**: `not implemented`.
- **Executed**: `not executed`.
- **Result**: `not measured`.
- **Status**: open.
- **Hard constraint discovered during the 2026-07-31 review, recorded here so it is not
  re-derived**: a 5-slice refiner variant must not be built. `recon_eval.py` lines 78–87 call
  `recon_slice` once per slice and time each call individually. If slice `s` required
  stage-I output for `s−2 … s+2`, that is five backbone invocations inside one timed call
  ≈ 3670 ms/slice, ≈139 s per volume. A single-slice 2-channel variant (recon, aliased) is
  the only shape that fits the timing harness.

### L05 — Loss mixing: SSIM + ℓ1, and later MS-SSIM

- **Idea**: recorded 2026-07-31 as candidate `C11`. Supporting number available at the time:
  on fastMRI knee ×8, SSIM loss 0.662 vs MS-SSIM+ℓ1 0.673 measured on the same anatomy and
  acceleration as ours, even though the evaluation metric was plain SSIM.
- **Code**: `05d8ed2` (author date 2026-07-31 14:14) added `--l1-weight` to `train.py`
  (default 0.0 = bit-identical to before), with the ℓ1 term divided by `data_range` so it
  cannot dominate SSIM numerically.
- **Executed**: **attempted and aborted, never actually ran.** `shell28.sh` `STAGE=all` was
  launched twice — `shell28_all_20260731_161204.log` and `shell28_all_20260731_161600.log` —
  and both exited immediately with
  `[STOP] ../result/promptmr_moe_f2_shared_aug/checkpoints/best_model.pt 없음`, because the
  slot's default `PROBE_NET` pointed at a checkpoint that does not exist on this server. The
  ℓ1 arm is stage B of that slot and therefore never started.
- **Result**: `not measured`.
- **Status**: open. To run it, override the probe target:
  `STAGE=l1 PROBE_NET=promptmr_moe_f2_acc8_aug_rm bash shell_scripting/shell28.sh`.

### L06 — 2.5D adjacent-slice input (`--num-adj-slices 3`)

- **Idea**: originally **rejected**, then revived. The rejection is recorded in
  `rejected.md` line 33 and dates to 2026-07-02, on the grounds that lesion-bbox
  inter-slice correlation was low (0.08–0.59) and adjacent-slice stacking would therefore
  blur lesions and damage SSIM_bbox.
- **Re-measured 2026-08-01**: the rejection basis was refuted. `scripts/adjacent_slice_corr.py`
  over 170 files / 7394 lesion boxes / 36761 control boxes found lesion-bbox correlation is
  **higher** than same-size control boxes at gap 1 (acc4 0.627 vs 0.510; acc8 0.624 vs 0.500)
  and gap-1 SSIM is identical (0.680 vs 0.682). The original low absolute values were a
  small-patch variance-normalization artifact; the original 2-file sample had no control
  group. Detail: `model_strategy.md` §3.
- **Code**: `12bbe7e` — wiring across five sites (`promptmr_plus.py` num_adj_slices +
  center-slice extraction + per-slice-group sensitivity RSS normalization,
  `load_data.py` slice-major coil-axis stacking with volume-edge clamping,
  `test_part.py` inference-side stacking, `train.py --num-adj-slices {1,3}`,
  `config.py`). Regression test: `scripts/tests/test_adj_slices.py`.
- **Executed**: twice, `shell_scripting/shell30.sh`, both on 2026-07-31.
  - `shell30_f2_20260731_181035.log` — 120 iterations per arm, fold 2, acc8 only.
  - `shell30_f2_20260731_191828.log` — 2000 iterations per arm (≈1 epoch), fold 2, acc8 only,
    finished 2026-08-01 03:14:09.
- **Result** (2000-iteration run, fold 2, acc8 only, seed 430):

  | | adj=1 | adj=3 | delta |
  |---|---|---|---|
  | ValLoss | 0.1157 | 0.1148 | −0.0009 |
  | TrainLoss | 0.09853 | 0.09885 | +0.0003 (worse) |
  | TrainTime, 2000 iter | 9359 s | 18126 s | **1.94×** |
  | ValTime, 536 slices | 351.6 s | 667.5 s | **1.90×** |
  | VRAM peak | 2.92 GB | 3.42 GB | +17% (of 8 GB) |

  **The effect does not resolve above noise: the sign flipped between the two runs.** At 120
  iterations adj=3 was worse (0.1566 vs 0.1543); at 2000 iterations it was better. The two
  arms have different first/last convolution shapes (2→6 channels) so their initializations
  differ, and ±0.001 sits inside that initialization noise.
- **Status**: closed-measured — **wiring kept, adoption deferred.** At equal GPU budget,
  adj=1 with twice the iterations is the better buy: in the same run adj=1 improved ValLoss
  from 0.1543 at 120 iterations to 0.1157 at 2000 iterations (−0.0386 over a 16.7× increase),
  so 2000→4000 iterations will almost certainly buy more than 0.0009. Default is
  `--num-adj-slices 1`, which is bit-identical to the pre-wiring code, so the option costs
  nothing to keep. **Re-open condition**: an equal-wall-clock A/B, giving adj=1 double the
  iterations. Comparing at equal iterations only shows that the slower model won at equal
  iterations, which is not a decision-relevant comparison.
- **Provenance note**: the upstream repository this backbone was vendored from left this
  feature unfinished — `2025_4th_clone/FastMRI_challenge_sMaRtIfy_final/utils/learning/train_part.py`
  line 173 hardcodes `num_adj_slices=1` with the comment
  `TODO Parameters considering adjacent slices are not implemented.` The slim vendoring in
  this repo inherited that gap; `12bbe7e` is the first implementation of it on either side.
- **VRAM measurement that this run contributed** (see also L10): peak 3.42 GB of 8 GB **with
  a 3× coil axis**, against 2.92 GB at `adj=1`. Memory was not close to binding in either arm.

### L07 — Measure the seed-to-seed noise floor

- **Idea**: recorded 2026-07-31 as candidate `C16`.
- **Code**: not applicable — this is a run, not a feature.
- **Executed**: `not executed`.
- **Result**: `not measured`.
- **Status**: open.
- **Why it gates other levers**: the position that put MoE first was a margin of
  +0.0012 full / +0.0016 bbox over the shared single model. No variance estimate exists for
  that margin. For reference, the 2025 winning team's own ablation reported a MoE gain of
  +0.0002. L01, L02, L03, L05 and L08 all have candidate effects of that magnitude, so
  without a noise floor a single run cannot decide any of them. L06 already demonstrated the
  failure mode concretely by flipping sign between two runs of the same arm pair.
- **Known noise source 1 — the two servers are not interchangeable.** Same config, same fold,
  measured:

  | Server | Data path | s/epoch | Hours/epoch |
  |---|---|---|---|
  | vessl GTX 1080 | `/root/Data` | 17,830 | **4.95** |
  | runpod RTX 4000 Ada | `/workspace/Data` | 7,250 | **2.01** |

  A 2.5× wall-clock difference means **any GPU-hour figure in this file is meaningless without
  naming the server**. Worse, it has already contaminated a comparison: the two arms of the
  flip-augmentation A/B ran on different servers, which is a fourth confound on top of the
  three already known for that result.
- **Known noise source 2 — TF32, unmeasured and currently unguarded.**
  `utils/common/utils.py::seed_fix` sets `cudnn.deterministic = True` and
  `cudnn.benchmark = False` but never touches `allow_tf32`, so PyTorch's defaults apply:
  `cudnn.allow_tf32` is `True`. Pascal (GTX 1080, sm_61) has no TF32 hardware and therefore
  runs convolutions in FP32; Ampere and later (sm_80+) run them at TF32 precision, a 10-bit
  mantissa. A cross-server A/B can therefore be comparing FP32 convolutions against TF32
  ones, at a magnitude comparable to the effects being measured — and PromptMR+ is
  convolution-dominated. `nvidia-smi -L` on both servers settles whether this applies.
  ⚠️ This is an **assumption from reading the defaults**, not a measurement; nobody has
  compared the two precisions on this pipeline.
- **Reproducibility consequence, independent of A/B validity**: the competition requires that
  a fixed seed reproduces the result. With `allow_tf32` left at its default, a final model
  trained on Ampere+ will not reproduce on a Pascal machine. If the submission is trained on
  an Ampere+ server, `torch.backends.cudnn.allow_tf32 = False` has to be set explicitly,
  at the cost of 2–3× slower convolutions.

### L08 — Weight averaging (SWA-lite / EMA)

- **Idea**: recorded 2026-07-31 as candidate `C10`.
- **Code**: `not implemented`.
- **Executed**: `not executed`.
- **Result**: `not measured`.
- **Status**: open.
- **Relevant property confirmed at the time**: PromptMR+ contains no BatchNorm-family layers,
  so averaged weights need no statistics recalibration. Training cost 0, inference cost 0 ms.

### L09 — Direct supervision on unacquired k-space lines

- **Idea**: recorded 2026-07-31 as candidate `C12`.
- **Code**: `not implemented`.
- **Executed**: `not executed`.
- **Result**: `not measured`.
- **Status**: open.
- **Enabling fact confirmed at the time**: the k-space stored in the training-pool h5 files is
  **full** k-space, not the masked product. The ground truth for the unacquired lines is
  therefore already on disk and currently unused.

### L10 — Adjacent slices used for sensitivity-map estimation only (batch SME)

- **Idea**: recorded 2026-07-31 as candidate `C14`.
- **Code**: `not implemented`. Verified 2026-08-01: `utils/model/promptmr/promptmr_plus.py`
  line 547 records the deliberate choice not to use `use_sens_adj`, because the 8 GB
  configuration estimates sensitivity per coil (`compute_sens_per_coil=True`).
- **Executed**: `not executed`.
- **Result**: `not measured`.
- **Status**: open.
- **Why L06's outcome does not settle this**: L06 measured lesion-texture correlation between
  adjacent slices. Sensitivity maps are low-frequency ACS-derived coil profiles with a
  different correlation structure, so neither L06's refutation of the old rejection nor L06's
  cost measurement transfers here.
- ⚠️ **This lever's stated purpose no longer holds: memory is not the binding constraint.**
  Batch SME was adopted (in `model_strategy.md` §3.2) as a memory optimization promising
  ≈55% less GPU memory for sensitivity estimation. Measured peak on the current configuration
  is **3.32 GB of 8 GB**, and L06's run peaked at **3.42 GB even with a 3× coil axis**. There
  is no memory pressure to relieve, so if this lever is run it has to be justified by
  *accuracy* (upstream PromptMR+ claims batch SME also improves the estimate), not by memory.
  The same collapsed premise invalidates the reasoning behind several other reduction
  decisions still recorded as necessary: `n_history` 3 instead of the upstream 11,
  `num_cascades` 10, and "coil compression is mandatory". `promptmr_plus.md` still presents
  those as forced by an 8 GB ceiling.

### L11 — Batch the per-coil sensitivity estimation on the inference path

- **Idea**: recorded 2026-07-31 as candidate `C17①`.
- **Code**: `not implemented`. Verified 2026-08-01:
  `utils/model/promptmr/adapter.py` line 37 passes `compute_sens_per_coil=True`
  unconditionally, so the sensitivity U-Net is invoked once per coil in sequence.
- **Executed**: `not executed`.
- **Result**: `not measured`.
- **Status**: open.
- **Claimed to be numerically identical** at the time of the 2026-07-31 review, on the basis
  that no operation in that path mixes across the batch dimension. Gradient checkpointing is
  already disabled at inference, so the sequential loop has no memory justification there.
  ⚠️ The numerical-identity claim has not been re-verified since; treat it as an assumption.

### L12 — Re-open bbox weighting by reweighting the SSIM map instead of adding a loss term

- **Idea**: recorded 2026-07-31 as candidate `C13`.
- **Code**: `not implemented` in the reweighting form. The additive form exists:
  `--bbox-loss-lambda` landed in `80975c5` (author date 2026-07-14 21:59).
- **Executed**: the additive form was executed as run12 — see L15 in the closed section.
- **Result**: `not measured` for the reweighting form.
- **Status**: open.

---

## 3. Closed levers — executed or dropped, kept as history

### L13 — Mask regeneration for the acc8 expert (`--remask 8 --remask-offset uniform`)

- **Idea**: the enabling observation is that training-pool k-space is full k-space, so a
  legitimate acc8 mask can be applied to acc4 volumes to manufacture new (input, GT) pairs.
  Recorded as candidate `C1`, and argued to be the highest-priority lever: equispaced
  undersampling has exactly `acc` distinct aliasing phase patterns
  (`e^{2πi·o·m/acc}`, set by the offset), the training-pool offset is deterministic
  (`(W//2) % acc`), and the acc8 expert had only ever seen 3 of the 8 patterns while scoring
  draws offsets at random.
- **Code**: `9cf519c` (author date 2026-07-29 11:32) — `--remask` / `--remask-offset` in
  `train.py`, generator `utils/data/transforms.py::equispaced_mask` and
  `canonical_offset`, loader support in `utils/data/load_data.py`. The mask generator was
  verified bit-exact against the competition mask family (step 4|8, centre fraction 0.08)
  across 280 volumes of the training pool plus leaderboard.
- **Executed**: 2026-07-31, `shell_scripting/shell28.sh` `STAGE=offset` —
  `shell28_offset_20260731_171753.log`. The probe forces offsets 0–7 on a fixed checkpoint
  and scores each, with zero training, so it measures the ceiling of what remask could
  recover before spending the A/B.
- **Result** (acc8, 20 volumes, hold-out fold 2):

  | Offset group | SSIM_full | SSIM_bbox |
  |---|---|---|
  | canonical (offsets the model trained on) | 0.9024 | 0.9201 |
  | all other offsets | 0.9069 | 0.9272 |
  | gap | **−0.0045** | **−0.0071** |

  The gap is **negative**: the model scored *higher* on the offsets it had never trained on.
  That is the opposite of the hypothesis, and it means the model is insensitive to mask
  offset — there is no offset-coverage deficit to recover.
- **Status**: closed-measured. The offset half of this lever is dead. The remaining
  justification for `--remask` is the volume-count half (80 → 160 training volumes for the
  acc8 expert), which the probe does not address and which was never separated from the
  offset argument.
- **Consequence recorded for other readers**: any earlier claim that a `REMASK=1` run's gain
  demonstrates "mask diversity" must be re-read as "twice the optimizer steps", since the
  offset mechanism is now measured to contribute nothing.

### L14 — 180° rotation augmentation

- **Idea**: 2026-07-31, added as a slot by a teammate. Reasoning recorded in
  `shell_scripting/shell29.sh`: unlike a 90° rotation, 180° does not swap the readout and
  phase-encode axes, the mask transform is the same `roll(flip(M),1)` the existing left-right
  flip already uses, and it is a closed form with no interpolation (measured
  relative max error 2.2e-07). Mechanically it could have been enabled immediately; the open
  question was only whether it produces *good* data, because the up-down component moves the
  joint line from the upper third of the image to the lower third.
- **Code**: `not implemented` as an augmentation. The probe script
  `scripts/probe_orient.py` and the slot `shell_scripting/shell29.sh` were written to decide
  whether to implement it. No `--aug-rot180` flag exists.
- **Executed**: 2026-07-31, `shell29_20260731_190011.log` — checkpoint
  `promptmr_moe_f2_aug_rm`, acc8, 10 volumes, four orientations scored.
- **Result**:

  | Orientation | SSIM_full | SSIM_bbox | Δfull | Δbbox |
  |---|---|---|---|---|
  | original | 0.8931 | 0.9123 | — | — |
  | left-right flip (control) | 0.8925 | 0.9115 | +0.0006 | +0.0008 |
  | up-down flip | 0.8904 | 0.9088 | +0.0027 | +0.0034 |
  | 180° | 0.8902 | 0.9092 | +0.0029 | +0.0031 |

  The left-right control gap is +0.0008, near zero as expected for a checkpoint trained with
  left-right flip augmentation — which validates the probe. The 180° gap of +0.0031 is the
  **ceiling** on what a 180° augmentation could teach, and it is below the 0.010 threshold the
  slot set in advance for justifying a 60-hour A/B. Since the 180° gap ≈ the up-down gap, the
  only new information in a 180° rotation is the up-down component.
- **Status**: closed-measured. Not implemented; `utils/data/transforms.py` was not touched.
- ⚠️ The gap is an upper bound on teachable information, **not** a predicted leaderboard gain.
  The scoring set contains no inverted knees.

### L15 — bbox weighted loss, additive form (`--bbox-loss-lambda`)

- **Idea**: SSIM_bbox is the ranking axis, so weight the lesion region directly:
  `L = (1 − SSIM_full) + λ·(1 − SSIM_bbox)`. Recorded in `model_strategy.md` §2.1.
- **Code**: `80975c5` (author date 2026-07-14 21:59), `--bbox-loss-lambda`, default 0.0 which
  is bit-identical to the previous behaviour. Valid-box determination matches the scoring
  function (`utils/common/metrics.py::ssim_bbox`, `win_size=7`).
- **Executed**: run12 — `training_anal/2026-07-18-run12-bbox-uc-analysis.md`.
- **Result**: regressed on all three scoring axes. Default left at 0.0 (off).
- **Status**: closed-measured, with a **caveat that keeps L12 open**: run12 changed three
  things at once. The design document specified λ=0.1 but the run actually used λ=0.5; it was
  combined with a second lever (UC) in the same run; and the backbone at the time was SDUM,
  not PromptMR+. The isolated effect of bbox weighting has therefore never been measured.

### L16 — Foreground-masked loss and validation (`--fg-loss`)

- **Idea**: compute the SSIM loss and the validation metric only inside the scoring
  foreground mask, so that validation predicts the leaderboard instead of being inflated by
  background. Recorded in `preprocessing.md` B5.
- **Code**: `347fb43` (author date 2026-07-12 01:46), `--fg-loss`, reusing the scoring
  definition `utils/common/metrics.py::foreground_mask` rather than redefining it.
- **Executed**: run11 — `training_anal/2026-07-14-run11-fgloss-analysis.md`. Identical config
  to run10 with only `--fg-loss` added.
- **Result**: full 0.9161 / bbox 0.9095 / 1514 ms/slice, versus run10's
  0.9118 / 0.9021 / 1523 ms. Improved on all three axes; the feared bbox regression did not
  occur. Also established that foreground-masked validation predicts the leaderboard score.
- **Status**: closed-measured — **adopted.** Every current slot passes `--fg-loss`.

### L17 — Acceleration-rate band U{7,8,9} instead of exactly 8

- **Idea**: recorded 2026-07-31 as candidate `C2`.
- **Code**: `not implemented`.
- **Executed**: `not executed`.
- **Result**: `not measured`.
- **Status**: closed-unmeasured — dropped by reasoning. Scoring uses exactly acc4 or acc8.
  Training on aliasing periodicities that never appear at scoring time trades away
  specialization, and the only measurement available on that axis says specialization won
  (MoE expert separation beat the shared model).

### L18 — Explicit mask / acceleration-rate conditioning as a replacement for MoE

- **Idea**: recorded 2026-07-31 as candidate `C5`.
- **Code**: `not implemented`.
- **Executed**: `not executed`.
- **Result**: `not measured`.
- **Status**: closed-unmeasured — dropped by reasoning. MoE already consumes the same one-bit
  condition via 100% parameter separation, which is the upper bound of conditioning on that
  bit; a few hundred FiLM parameters have no structural route past it. The best realistic
  outcome is a tie, whose only benefit is shipping one checkpoint instead of two — not a
  scoring axis. This decision also retired the `acc-conditioned GRAPPA weighting` proposal
  recorded in `confirmed_model.md` line 92 and `model_decision_points.md` line 100.

### L19 — Move preprocessing into `prep_volume` to shorten the timed section

- **Idea**: recorded 2026-07-31 as candidate `C17②`.
- **Status**: closed-unmeasured — **rejected as measurement avoidance.** The repository owner
  stated that preprocessing belongs inside the reported recon time. Not implemented, and not
  to be revisited on efficiency grounds.

---

## Change Log

- 2026-07-31 — File created as "다음 레버 — PromptMR+ 위에서 무엇을 할 것인가"
  - **What changed**: New file, added in commit `05d8ed2` alongside the `shell28.sh` probe
    slot. Contained 17 candidates (C1–C17) organized by priority, each with an expected-gain
    range and a GPU-hour estimate, split into "검증 완료 (생존 6)", "탈락 (검증됨)", and
    "미검증".
  - **Why**: to consolidate the output of a seven-branch investigation (two papers, the 2019
    and 2020 fastMRI competitions, past SNU placing teams, recent SOTA, and an audit of this
    repository's own code and history) into one ranked list.
  - **Evidence**: commit `05d8ed2`, author date 2026-07-31 14:14.
  - **Supersedes**: nothing.

- 2026-08-01 — Recorded the 2.5D adjacent-slice measurement
  - **What changed**: The re-judgement table entry for 2.5D adjacent slices was updated
    twice: first to record that the old rejection basis had been refuted, then to record the
    shell30 cost/benefit measurement and the resulting decision to defer adoption.
  - **Why**: the rejection had been based on a 2-file sample with no control group, and the
    replacement measurement produced the opposite result; then the implementation measurement
    showed the cost was 1.9× for an effect that did not resolve above noise.
  - **Evidence**: commits `c41ebd3` and `e520104`; `scripts/adjacent_slice_corr.py`;
    `shell_scripting/logs/shell30_f2_20260731_191828.log`.
  - **Supersedes**: the entry's original text, `bbox 내부 상관 0.08~0.59`.

- 2026-08-01 — Rewrote the entire file as a dated lever ledger
  - **What changed**: The document was restructured from a priority-ranked candidate list
    into a ledger of 19 levers (L01–L19), each carrying five explicit fields —
    **Idea / Code / Executed / Result / Status** — with dates and evidence for each. Open
    levers are in §2, executed-or-dropped levers moved to §3 as history. The per-candidate
    **expected-gain ranges and priority ordering were deleted**. A "how to read this file"
    section was added in §0, including an explicit warning that code-landing dates are
    unreliable because rebasing rewrote committer dates, and that log dates are the harder
    evidence. Cross-cutting measurements were consolidated into §4, and §4.3 was added with
    the 2025 winning clone's hyperparameters.
  - **Why**: the previous structure violated all three authoring rules in
    [`README.md`](README.md). Under **R1** it recorded proposals as if they were findings:
    every candidate carried an expected-gain range produced before any measurement existed,
    and two of those ranges were later contradicted by the only probes that ran — L13's
    offset gap came back **negative** (−0.0045 full / −0.0071 bbox) where the document had
    predicted acc8 +0.002~0.006, and L14's ceiling came back at +0.0031 against a 0.010
    threshold. Under **R2** the entries did not say when anything happened or which log
    produced a number, so a reader could not tell a measured value from an estimate. Under
    **R3** there was no change log at all and no per-lever history, so the file could not
    record that a lever had been executed without rewriting the claim in place and losing the
    prior position. Three specific errors that the rewrite corrects: the file listed the
    ℓ1-mixing probe as available when both `shell28.sh STAGE=all` launches had aborted on a
    missing checkpoint and the ℓ1 arm never started; it did not record that L13's probe had
    already killed the offset argument; and it did not record that L14 had already been
    measured and folded.
  - **Evidence**: `git log --diff-filter=AM -S<symbol>` for each code-landing date;
    `shell_scripting/logs/shell28_offset_20260731_171753.log` (L13 result),
    `shell_scripting/logs/shell28_all_20260731_161204.log` and `..._161600.log` (L05 aborted),
    `shell_scripting/logs/shell29_20260731_190011.log` (L14 result),
    `shell_scripting/logs/shell30_f2_20260731_181035.log` and `..._191828.log` (L06 result),
    `training_anal/2026-07-14-run11-fgloss-analysis.md` (L16),
    `training_anal/2026-07-18-run12-bbox-uc-analysis.md` (L15); code status of L02, L10, L11
    verified by reading `utils/learning/train_part.py`,
    `utils/model/promptmr/promptmr_plus.py`, and `utils/model/promptmr/adapter.py` on
    2026-08-01.
  - **Supersedes**: the entire prior structure — sections "0. 결론", "1. 우선순위" (including
    the expected-gain and GPU-hour columns), "2. P1 상세", "3. acc8 — 헤드룸이 몰린 곳",
    "4. 재판정 대상", "5. 조사 중 드러난 것들", "6. 문서 교정 필요", "7. 이 문서의 한계".
    The candidate identifiers C1–C17 are retired; each lever entry names the C-number it came
    from so older commits and analysis documents remain traceable. **Content deliberately
    dropped**: section "6. 문서 교정 필요", a list of eight incorrect statements in other
    documents. It is dropped from this file because it tracks other files' defects rather
    than levers; the corrections it called for must be made in those files directly, and
    four of them (`preprocessing.md:16`, `model_strategy.md:66`, `rejected.md:30`,
    `augmentation.md:109` — all downstream of the premise that GRAPPA cannot be recomputed)
    remain unfixed as of this rewrite.

- 2026-08-01 — Flattened the file: dissolved the two free-floating sections into the lever
  entries they bear on
  - **What changed**: sections "4. Cross-cutting facts established while investigating these
    levers" and "5. Known limits of this file" were removed as sections, and their content was
    moved into the entry that acts on it. Specifically: the two-server wall-clock table
    (17,830 vs 7,250 s/epoch) and the TF32 precision hazard moved into **L07**, which exists
    precisely to quantify measurement noise, and the TF32 note gained the reproducibility
    consequence (a model trained on Ampere+ will not reproduce on Pascal unless
    `cudnn.allow_tf32` is explicitly disabled). The "memory is no longer binding" measurement
    (peak 3.32 GB of 8 GB, and 3.42 GB in L06's 3×-coil run) moved into **L10**, whose entire
    stated justification was memory savings — so the fact now sits where it invalidates the
    lever's premise, along with the note that `promptmr_plus.md` still presents `n_history` 3,
    `num_cascades` 10, and mandatory coil compression as forced by an 8 GB ceiling. The 2025
    winning clone's hyperparameter table moved into **L01**, where the 30-epoch + StepLR
    comparison is the actual argument for the lever, with the `num_adj_slices=1` TODO row
    cross-referenced from **L06** as provenance. A "Uniform depth" clause was added to the
    header stating the rule.
  - **Why**: a fact parked in a "general findings" section is a fact nobody acts on. The
    two-server difference is the clearest case — it had been sitting in §5.1 as an observation
    while every lever entry above it still quoted GPU hours without naming a server. Moving
    each fact next to the decision it changes also makes the file uniformly deep: one kind of
    unit (the lever entry) instead of entries plus prose commentary at a different altitude.
  - **Evidence**: instruction from the repository owner, 2026-08-01. No numbers were altered
    in the move; the source logs and commits cited in each relocated passage are unchanged.
  - **Supersedes**: sections 4 and 5 of the previous version of this file. Content from the
    former §5 that had no lever to attach to is absorbed here instead: of the 17 candidates in
    the 2026-07-31 version, **8 had been through adversarial verification and 9 had not**, and
    verification repeatedly corrected the source material by large factors — L13's GPU-hour
    estimate was off by 2.4×, L04's inference-cost estimate by two orders of magnitude, and
    the claim that LR decay had already been tried was simply false (now recorded in L01).
    The unverified entries should be expected to move by similar amounts, so file:line must be
    confirmed before acting on any of them. The former §5 also noted that the eight
    verification agents ran without a working safety classifier. GRAPPA self-computation
    remains tracked separately in [`grappa_selfcompute.md`](grappa_selfcompute.md).
