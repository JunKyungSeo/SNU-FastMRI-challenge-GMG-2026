# CLAUDE.md — `training_anal/` 학습 분석 작성 규칙

이 폴더는 **SDUM 학습 run별 분석**을 모은다. 미래 세션의 Claude가 새 run 로그를 분석할 때 **아래 규칙을 반드시 따른다.**

> 상위 핸드오프는 repo 루트 `../CLAUDE.md`. 여기선 "분석을 어떻게 쓰나"만 다룬다.

---

## 0. 핵심 원칙 — 분석은 항상 "직전 run과의 비교"다

한 run을 고립해서 보지 말 것. 우리 작업은 **run마다 뭔가를 바꿔 가설을 검증하는 연쇄**다. 그래서 모든 분석은 **직전 run·직전 분석을 이어받아** 쓴다. 아래 두 섹션은 **필수**(빠지면 분석이 아님):

1. **이번 run이 직전과 무엇이 달라졌나** — 명령/config/코드 diff를 표로. *"무엇을 바꿔서 무엇을 검증하려 했는지"* 를 명시.
2. **직전 분석에서 지적했거나 고치려던 문제가 이번에 어떻게 됐나** — 항목별 추적 표. 각 항목을 `✅해결 / ⬜여전 / 🔁재발 / 🐛새로발견` 로 판정.

---

## 2. 파일 규칙

- **파일명**: `YYYY-MM-DD-run{N}-{shortname}-analysis.md` (예: `2026-07-07-run3-learncheck-analysis.md`). `{shortname}` = `-n` 실험 이름의 핵심.
- **1 run = 1 파일.** 기존 분석을 덮어쓰지 말고 새로 추가.
- 새 분석을 쓸 때마다 아래 **§4 run 이력**에 한 줄 append.

## 3. 표준 섹션 (권장 순서)

1. 제목 + **한 줄 결론**(가장 중요한 성과/실패 하나).
2. **핵심 요약**(한눈에 보는 요약) — ✅됨 / ⚠️주의 / ❌안됨 / 🐛버그 를 번호로. *(예전 이름 "TL;DR" — 같은 것, 이름만 쉽게)*
3. **실행 조건 + 직전 대비 diff** ← 필수(원칙 1).
4. **직전 분석 지적사항 추적** ← 필수(원칙 2).
5. **손실 / 속도·효율 / (있으면) 지표** 분석.
6. **일반 ML 구현 점검** ← 이 run 분석과 **별개로**, "표준 ML이면 당연히 있어야 할 것들"이 우리 파이프라인에 다 들어와 있는지 매번 훑는다. 체크리스트 = 아래 **§3.5**.
7. **다음 액션 + 다음 학습 고려사항** — (a) 지금 바로 할 것(P0/P1/P2 to-do) + (b) *이 run 결과가 시사하는* 다음 학습에서 더 파볼 점(가설·레버; 확정 아닌 "탐색거리").
8. **run 누적 요약 표**(run1→…→이번).

### 3.5 일반 ML 구현 점검 체크리스트  (§3-6번 — 매 분석에 포함)

이 run 성과와 **무관하게**, 표준 ML 학습이면 갖춰야 할 요소가 파이프라인에 들어와 있는지 매번 표로 훑는다.
상태: **✅있음 / ⚠️부분 / ⬜없음(=구현 후보) / 🚫제약상 불가(이유 명시)**.

| 범주 | 항목 | 상태 | 근거/비고 (파일:줄 or 이유) |
|---|---|---|---|
| 재현성 | seed 고정 + cudnn deterministic | | |
| 최적화 | LR 스케줄(warmup/decay) | | |
| 최적화 | optimizer · weight decay | | |
| 최적화 | gradient clipping | | |
| 최적화 | 유효 batch(grad accumulation) | | |
| 체크포인트 | best/last 저장 | | |
| 체크포인트 | resume(이어학습) | | |
| 검증 | 매 epoch validation | | |
| 검증 | best-model 선택 기준 | | |
| 지표 | 대회 지표(SSIM_full/bbox) 로깅 | | |
| 지표 | baseline(zero-filled) 대비 | | |
| 데이터 | augmentation 배선 | | |
| 안정성 | NaN/발산 감지·중단 | | |
| 효율 | grad checkpointing · num_workers | | |
| 효율 | mixed precision(AMP/BF16) | 🚫 | Pascal(1080) FP32 강제 = 금지 |

> ⬜(없음)이 곧 결함은 아니다 — **"왜 아직 없나 / 지금 필요한가"**를 §7 다음 고려사항에서 판단하고, 있어야 하는데 빠졌으면 다음 액션 P#로 올린다.

## 4. 정직성 규칙 (반드시 지킬 것)

- **확정 vs 의심 구분.** 코드로 확인한 건 파일:줄 근거를 달고 "확정", 아니면 "의심".
- **지표 함정을 항상 명시**:
  - SSIM은 MRI에서 **zero-filled만 해도 0.7~0.85** → 낮은 loss가 실력이 아닐 수 있음. 베이스라인 대비로만 판단.
  - **train loss ≠ val loss**, **whole-image SSIM ≠ SSIM_bbox(순위축)**. 무엇을 보고 있는지 매번 밝힐 것.
- **하니스/코드 자체의 버그도 분석 대상.** 로그 숫자가 이상하면 코드를 열어 확인(예: run3에서 `--max-iters` × 평균 분모 버그로 TrainLoss가 15× 축소된 것). 로그를 곧이곧대로 믿지 말 것.
- 과장 금지: "안 망가짐"과 "잘한다"를 구분해서 쓴다.

---

## 5. run 이력 (새 분석마다 한 줄 append)

| run | 이름 | 핵심 변경 | 결과 한 줄 | 분석 |
|---|---|---|---|---|
| 1 | `sdum_run1` | SDUM 첫 GPU 학습(풀 epoch) | loss ~1.0 고정 = 안 배움. 스케일 불일치 의심. | [run1](2026-07-07-run1-analysis.md) |
| 2 | `sdum_scalefix` | 어댑터 per-sample 스케일 정규화(`c56351d`) | loss 1.0→~0.1(SSIM≈0.9). P0 해결. 속도 그대로(num_workers 무효). | [run2](2026-07-07-run2-scalefix-analysis.md) |
| 3 | `sdum_learncheck` | `--max-iters 300` × 5 epoch + **매 epoch val** | val SSIM 0.842→0.855 ↑(학습 확인). 🐛 TrainLoss 평균 버그 발견. bbox/baseline 여전 미측정. | [run3](2026-07-07-run3-learncheck-analysis.md) |
| 4–5 | `sdum_scalefix` (풀 epoch) | run3의 `--max-iters` 제거 → 풀 4651 iter 완주. run5 = `nw 6` 속도 실험 | 첫 풀 epoch: TrainLoss **0.1146**·Val **0.882**(whole SSIM). num_workers 무효 확정. 🐛 `val_loss_log` 2버그(SUM 저장·덮어쓰기) → **수정함**. | [run4–5](2026-07-08-run4-5-fullepoch-analysis.md) |
| 6 | `sdum_smoke` (smoke) | **+`--readout-crop`, +cosine/warmup 스케줄러** 코드 게이트(`--max-iters 30`) | 두 코드 크래시 없이 동작, **~4.0 s/iter → 1ep≈5.2h(≈40%↓)** 실측. 🐛 `validate()`가 `--max-iters` 무시 발견. base LR 2e-4(수정 전). | [run6](2026-07-08-run6-smoke-readoutcrop-sched-analysis.md) |
| 7 | `sdum_run5`* | +LR **2.4e-4**, readout-crop, cosine warmup 200, 풀 2ep | 속도 **~5.2h/ep(확실, crop)**. Train 0.116→0.100 / Val SSIM 0.881→**0.885**(=2nd epoch 몫, crop·cosine 품질기여 matched-epoch로 ≈0). **under-training**·bbox 미측정. *dir는 태그충돌로 sdum_run5. | [run7](2026-07-09-run7-crop-cosine-analysis.md) |
| 8 | `sdum_run8` | (예정, shell6) run7 + **cascades 4→6 + channels [24,48]** — 헤드룸을 용량에 투자. readout-crop 유지 | (훈련 전) OOM borderline~medium([24,48]×6 ~7.1–8.1GB), ~18–20h. | [run8(pre)](2026-07-09-run8-capacity-analysis.md) |

> ⚠️ **이름 충돌 주의:** run2와 run4–5는 **둘 다 `-n sdum_scalefix`**(같은 명령). `../result/sdum_scalefix/`는 마지막 run이 덮어쓴 상태. (run6은 셸스크립트 확인용 중복이라 분석 제외 — 별도 최소 기록 예정.)
