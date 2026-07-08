# run6 (smoke) 분석 — `--readout-crop` + cosine/warmup 스케줄러 코드 검증 (2026-07-08)

**한 줄 결론:** 새 코드 2개(입력단 readout-crop, cosine+warmup 스케줄러)가 **크래시 없이 동작**하고, readout-crop으로 **~4.0 s/iter → 1 epoch ≈ 5.2h** (run4 ~9h 대비 **≈40% 단축**, extra.md 예측과 일치). 이건 "안 망가짐 + 속도 확보" 확인이지 "성능 좋다"는 아님(30 iter뿐).

> 이 run은 **정식 run이 아니라 9h run7 전 게이트용 스모크(=run6)**다. `--max-iters 30 -e 1`로 train 30 iter만 돌렸다.
>
> 📌 **넘버링 정정:** 이 스모크 = **run6**, 뒤이어 도는 게 **run7**(단 결과 dir는 태그 충돌 잔재로 `sdum_run5`), 다음 큐 슬롯 = **run8**. (규칙: 모든 GPU 학습에 순차 run 번호.)

---

## §0 TL;DR

- ✅ **readout-crop 배선 동작** — cropped 경로로 loss 유한, 크래시 없음. (extra.md의 "최대 속도 레버"가 실제로 켜짐)
- ✅ **스케줄러 배선 동작** — `[scheduler] warmup=10 → cosine, total_iters=4651` 출력, per-iter step 정상.
- ✅ **속도 ~40% 단축 실측** — steady ~4.0 s/iter → 1 epoch ≈ 5.2h (run4 ~9h 대비). *단 30 iter 외삽·val 제외 추정치.*
- ⚠️ **base LR = 2e-4로 돌아감** — 이 스모크는 LR 수정(2.4e-4) **전에** 시작돼 2e-4. run7부터 논문값 2.4e-4 적용됨(코드/슬롯 이미 수정).
- 🐛 **`validate()`가 `--max-iters`를 무시** — train은 30 iter(~2분)에 끝났으나 **full val set을 다 forward**(~5분+)해서 스모크가 총 11분 걸림. SIGTERM으로 val 중단.
- ❌ **순위 지표(SSIM_full/bbox) 여전 미로깅** — 이번엔 cv2 이슈로 item 1·2 스킵. whole-image loss만.

---

## §1 실행 조건 + 직전(run4–5) 대비 diff  *(필수)*

| 항목 | run4–5 (직전, 풀 epoch) | 이 스모크 | 무엇을 검증하려 했나 |
|---|---|---|---|
| 해상도 | full 640×W (crop 안 됨) | **640→384 입력 crop** (`--readout-crop`) | crop 경로 안 깨지나 + 속도이득 실측 |
| LR 스케줄 | 상수 2e-4 (스케줄러 없음) | **warmup 10 → cosine** (`--warmup-iters 10`) | 스케줄러 배선이 실제로 붙나 |
| base LR | 2e-4 | 2e-4 *(수정 전)* | — (LR값 검증 아님) |
| iters | full 4651 | `--max-iters 30` | 코드 경로만 빠르게 |
| epoch | 1 (풀) | `-e 1` (30 iter에서 break) | — |
| 이름 | `sdum_scalefix` | `sdum_smoke` (`--run-tag smoke`) | 결과 분리 |

**목적:** item 3(`--readout-crop`)·item 4(스케줄러) 코드가 9h run 전에 안 깨지는지 + readout-crop 속도이득이 예측(~40%)대로인지 게이트.

---

## §2 직전 분석/extra.md 지적사항 추적  *(필수)*

| 지적 (출처) | 판정 | 근거 |
|---|---|---|
| ⭐ readout crop이 SDUM 경로에 안 걸림 = 최대 속도 레버 (extra.md A) | ✅**해결** | `--readout-crop` 플래그 추가·플러밍 확인, 스모크에서 ~4s/iter(≈40%↓) 실측 |
| ⭐ 고정 LR (상수 2e-4)가 정체 원인 (extra.md C) | ✅**해결(배선)** | cosine+warmup LambdaLR 배선, `[scheduler]…` 출력으로 활성 확인. *효과(정체 풀림)는 run5 풀 epoch에서 판정* |
| LR을 논문(2.4e-4)과 다르게 2e-4로 씀 | 🐛→✅ | 이번 세션에 발견, shell4/5/6 + 예시 `-l 2.4e-4`로 수정. run7부터 적용(스모크 자체는 2e-4) |
| 🐛 `val_loss_log` SUM 저장·run별 덮어쓰기 (run4–5 §5-A/B) | ✅**해결** | 이전 커밋에서 per-subject 정규화 + `--run-tag` 분리 |
| SSIM_full/bbox 로깅 없음 = 순위 지표 못 봄 (extra.md D, item 1·2) | ⬜**여전** | cv2(libGL) 이슈로 이번 스킵. val 로그는 whole-image SSIMLoss만 |
| 🐛 `validate()`가 `--max-iters` 무시 | 🐛**새로발견** | `train_epoch`만 early-break(`train_part.py:24`), `validate()`엔 상한 없음 → 스모크가 full val 지불 |

---

## §3 손실

30 iter train loss(각 5 iter 로그): `0.1989 → 0.2458 → 0.1324 → 0.1149 → 0.1631 → 0.1853`.

- **유한·NaN 없음·flat 1.0 아님** → cropped 입력 + 스케줄러 경로가 정상 신호를 낸다. (run1의 "loss 1.0 고정" 같은 파탄 아님)
- **수렴/성능 판단 불가** — 30 iter뿐 + warmup 구간(1~10 iter는 LR이 base의 1/10~10/10로 램프업)이라 값이 낮은 게 실력이 아님. **"안 망가짐"만 확정.**
- run4의 TrainLoss 0.1146과 **직접 비교 금지**(iter 수·LR·셔플 순서 다름).

## §4 속도 · 효율

| | 값 | 비고 |
|---|---|---|
| iter 0 | 4.72s | CUDA init·캐싱 포함(1회성) |
| steady | **~20.0s / 5 iter = ~4.0 s/iter** | iter 5~25 안정 |
| 1 epoch(4651) 외삽 | **≈ 5.17 h** | run4 ~9h/epoch 대비 **≈43% 단축** |

- extra.md의 "readout-crop ~40% 단축" 예측이 **실측으로 확인**됨(무손실 crop이 공간연산을 줄인 결과).
- ⚠️ **정직성:** 30 iter 외삽 + validation 시간 제외. 실제 epoch 벽시계는 val·디스크 I/O 포함이라 다를 수 있음 → **run7 풀 epoch 로그로 확정**.
- VRAM: GPU util 99%, mem ~7.5GB (crop이 640 대비 메모리를 **줄임** → OOM 위험 없음).

## §5 다음 액션

- **P0 (진행 중):** **run7** = `-l 2.4e-4 --readout-crop --warmup-iters 200 -e 2` 시작됨(결과 dir `sdum_run5` = 태그충돌 잔재). 완주 후 분석 — ⓐ TrainLoss가 run4 0.1146보다 내려가나, ⓑ cosine tail로 val 정체가 풀리나, ⓒ 실제 epoch 벽시계.
- **P1:** `validate()`에 스모크용 iter 상한(예: val 20슬라이스) → 앞으로 스모크가 진짜 몇 분에 끝남. *실제 run은 full val 유지(순위 지표 필요).*
- **다음 슬롯:** shell6 = **run8**(대안 5b, readout-crop 없음, `-r 100`).
- **P2:** SSIM_full 로깅 (metrics.py cv2→scipy 후). "SSIM_full 먼저" 우선순위대로 bbox는 뒤로.

---

## §6 run 누적 요약

| run | 이름 | 핵심 변경 | 결과 한 줄 |
|---|---|---|---|
| 1 | `sdum_run1` | SDUM 첫 풀 epoch | loss ~1.0 고정 = 안 배움 |
| 2 | `sdum_scalefix` | 어댑터 스케일 정규화 | loss→~0.1, P0 해결 |
| 3 | `sdum_learncheck` | `--max-iters 300`×5ep + val | val SSIM 0.842→0.855↑. TrainLoss 평균 버그 발견 |
| 4–5 | `sdum_scalefix` 풀 | max-iters 제거, 풀 4651 완주 | TrainLoss 0.1146 / Val 0.882(whole). val_loss_log 2버그 수정 |
| **6** (smoke) | `sdum_smoke` | **+readout-crop, +cosine/warmup** (30 iter 게이트) | **두 코드 동작·크래시 없음, ~40% 속도↓ 실측.** validate() max-iters 무시 발견 |
| 7 (진행중) | `sdum_run5`* | +LR 2.4e-4 논문값, 풀 2ep | (진행 중) *dir는 태그충돌 잔재 |

> ⚠️ 이 스모크(run6)는 base LR **2e-4**(수정 전). 정식 run7부터 논문값 **2.4e-4**. 스모크는 성능이 아니라 **코드·속도 게이트**로만 해석할 것.

---

## 💬 코멘트 (사람)

> max iter 시 trainloss에 문제가 있다는 분석을 training_anal 사이에서 본 거 같은데 그게 이번에도 영향을 미쳤는지 궁금하고, crash 나는지 보려고 넣은 거니까 이정도면 굿

**→ 답변 (max-iters TrainLoss 버그, 이번 스모크 영향 여부):**

맞습니다 — 그 버그는 **run3 분석**에서 발견한 것: `train_epoch`가 `--max-iters`로 조기 break하는데(`train_part.py:24`) epoch 끝 평균을 `total_loss / len_loader`(=full **4651**)로 나눔(`:47`) → **epoch-요약 "TrainLoss=" 값이 ~155배(4651/30) 축소**된다. 코드엔 아직 그대로(§5-C "보류").

**이번 스모크(run6)엔 영향 없음** — 근거:
- 그 버그는 **epoch 끝 "TrainLoss=" 요약줄에만** 영향. 이번 스모크는 **validation 중 kill**돼서 그 줄이 **아예 안 찍힘**(로그 `grep -c TrainLoss` = **0**).
- 우리가 본 값은 **per-iter `Loss=`(0.19~0.25)** 뿐 — raw `loss.item()`이라 **분모 버그와 무관**.
- 즉 스모크 결론(크래시 없음·~40% 속도)은 이 버그와 **독립**.

**run7(진행 중, 풀 run)엔 애초에 안 걸림** — `--max-iters`가 없으니 early break 자체가 없다 → epoch-요약 TrainLoss 정상 계산. (버그는 `--max-iters` 쓸 때만 발현)

→ crash 체크 목적 달성 → **"굿" 동의.** ✅ (근본 수정은 §5-C대로 보류, `--max-iters` 스모크에서만 요약 TrainLoss를 안 믿으면 됨)
