# SDUM run3 (`sdum_learncheck`) 학습 로그 분석 (2026-07-07)

run2에서 스케일 정규화로 "학습이 성립"함을 봤지만 **val을 못 돌려(인터럽트) "진짜 배우는지"는 미확정**이었다. run3는 `--max-iters 300`으로 epoch를 짧게 끊어 **5 epoch를 돌리며 매 epoch validation**을 붙인 첫 run이다.

> **한 줄 결론:** ✅ **드디어 "배운다"가 확인됐다** — val SSIM이 epoch마다 오른다(0.842→0.855, 매 epoch NewRecord). 단 (1) 리포트되는 **TrainLoss는 `--max-iters` 때문에 15.5× 축소된 버그값**이고, (2) 순위축 **SSIM_bbox·zero-filled 베이스라인은 여전히 미측정**, (3) 속도는 그대로(7s/iter)에 **val이 캡 안 돼 21분/epoch**.

---

## 0. TL;DR

1. ✅ **학습 성립 확인 (run2의 최우선 숙제).** ValLoss **0.1584 → 0.1448** (whole-image SSIM 0.842 → 0.855), epoch 0·1 **연속 NewRecord**. held-out val에서 오르므로 진짜 학습이다.
2. 🐛 **버그 발견: TrainLoss가 틀렸다.** `--max-iters 300`으로 끊는데 평균을 **full loader(4651)로 나눠** 15.5× 축소. 리포트 0.010 → **실제 ≈0.16**. 코드 확정: [`train_part.py:47`](../utils/learning/train_part.py#L47) + [`:24`](../utils/learning/train_part.py#L24).
3. ⚠️ **아직 미측정: SSIM_bbox(순위축) + zero-filled 베이스라인.** val은 whole-image SSIM이라 병변 실력을 못 본다. val 0.84가 zero-filled 바닥보다 얼마나 위인지도 불명.
4. ❌ **속도 그대로 + 새 병목.** train 7.0s/iter(compute-bound, **3연속 확인**). val은 `--max-iters`에 안 걸려 **매 epoch 21분** 풀 val → "빠른 디버깅"의 새 병목.

---

## 1. 실행 조건 + run2 대비 무엇이 달라졌나 (필수)

```bash
python train.py --model sdum --num-cascades 4 --channels 16 32 \
  -b 1 -l 2e-4 -e 5 --max-iters 300 -r 50 --num-workers 4 \
  -n sdum_learncheck -t /root/Data/train/ -v /root/Data/val/
```

| 항목 | run2 (scalefix) | run3 (learncheck) | 왜 바꿨나 |
|---|---|---|---|
| epochs | 1 | **5** | epoch 간 val 추세를 보려고 |
| `--max-iters` | 없음(풀 epoch) | **300** | 9h/epoch 회피 — Task4 레버를 실제로 적용 |
| validation | 인터럽트로 **미실행** | **매 epoch 실행** | run2가 남긴 최우선 숙제 |
| report-interval | 10 | 50 | 로그 정리 |
| config / lr / workers | 4×[16,32] / 2e-4 / 4 | **동일** | 통제변인(모델은 그대로 두고 절차만 검증) |

**핵심:** run3 = *"`--max-iters`로 짧게 여러 epoch + 매 epoch val"* → run2 분석의 최우선 액션(**val로 학습 확정**)과 속도 레버(**`--max-iters`**)를 **동시에** 적용한 run이다.

---

## 2. 직전(run2) 분석 지적사항 — 이번에 어떻게 됐나 (필수)

| run2에서 지적/시도한 것 | run3 결과 | 상태 |
|---|---|---|
| A. val로 학습 여부 확정 (최우선) | val 붙임 → SSIM 0.842→0.855 상승 | ✅ **해결** |
| A. **SSIM_bbox**(순위축) 측정 | `validate()`가 whole-image SSIM만 계산(주석 안 씀) | ⬜ 여전히 미측정 |
| A. zero-filled 베이스라인 비교 | 안 함 | ⬜ 여전 |
| B. `--max-iters`로 빠른 사이클 | 300 적용 → train 35분/epoch | ✅ 적용(단 val이 안 캡됨, §4.2) |
| B. checkpointing OFF / readout crop 확인 | 안 건드림 | ⬜ 여전 |
| 속도 compute-bound(num_workers 무효) | train 7.0s/iter 재확인 | ✅ **3연속 재확인** |
| — (신규) | TrainLoss 평균 버그가 `--max-iters`와 충돌해 드러남 | 🐛 **새로 발견** |

---

## 3. 손실 분석

### 3.1 🐛 TrainLoss 버그 — 먼저 짚어야 함

로그의 epoch 요약이 이상하다: per-iter loss는 0.08~0.23인데 **TrainLoss = 0.01027(ep0), 0.008869(ep1)**. 개별 iter보다 10배 이상 작은 평균은 불가능 → 코드 확인:

- [`train_part.py:24`](../utils/learning/train_part.py#L24): `--max-iters 300`이면 iter 300에서 `break` (300 iter만 누적).
- [`train_part.py:47`](../utils/learning/train_part.py#L47): `total_loss = total_loss / len_loader` 인데 `len_loader = len(data_loader) = 4651` (full epoch 길이).

**→ 300 iter 합을 4651로 나눔 = 15.5× 축소.** 교정하면:
- ep0: 0.01027 × (4651/300) = **≈ 0.159**
- ep1: 0.008869 × (4651/300) = **≈ 0.138**

교정값이 per-iter(~0.13~0.16) 및 ValLoss(0.16, 0.14)와 일치 → **버그 확정.** 영향: **`--max-iters`를 쓴 모든 run의 리포트 TrainLoss는 거짓**이며 run 간 비교 불가. **고쳐야 함**(§5).

### 3.2 ValLoss — 신뢰할 신호, 오른다 ✅

ValLoss는 full val set 기준이라 믿을 수 있다:

| epoch | ValLoss | ≈ SSIM | 비고 |
|---|---|---|---|
| 0 | 0.1584 | 0.842 | NewRecord |
| 1 | 0.1448 | 0.855 | NewRecord |

- 2 epoch 연속 하강 + NewRecord → **held-out에서 실제로 좋아진다 = 진짜 학습.** run2는 val이 없어 못 했던 확정을 run3가 해냈다.
- 개선폭은 **작다**(SSIM +0.013/epoch). epoch당 train 300 iter뿐이고 config도 최소급이라 예상 범위.

### 3.3 ⚠️ 그래도 "잘한다"는 아직 — bbox·baseline 미측정

- ValLoss는 [`train_part.py:76`](../utils/learning/train_part.py#L76)의 `ssim_loss(target, recon)` = **whole-image SSIM**(주석 box 안 씀). **순위축 SSIM_bbox가 아니다** → 병변 복원 실력은 여전히 안 보임.
- **SSIM 바닥 함정(run2에서 강조):** MRI는 zero-filled만 해도 SSIM 0.7~0.85. **val 0.842가 바닥 바로 위일 수도 있다.** zero-filled 베이스라인을 재보기 전엔 "0.85 = 좋다"라고 말 못 함.

---

## 4. 속도 / 효율 분석

### 4.1 train 속도 — 7s/iter, compute-bound (3연속 확인)

TrainTime 2093s / 300 iter = **6.98 s/iter.** run1(7.0)·run2(6.9)와 동일. `--num-workers 4`도 여전히 무효 → **연산 병목 확정**(run2 결론 재확인). 300 iter epoch의 train 부분만 **≈35분**.

### 4.2 ⚠️ val이 `--max-iters`에 안 걸린다 → 21분/epoch

[`validate()`](../utils/learning/train_part.py#L58)는 **full val_loader**를 돈다(`--max-iters` break 없음). ValTime = **1286s ≈ 21분/epoch.**

→ `--max-iters`로 train을 35분으로 줄여도 **val 21분이 그대로 붙어** 1 epoch ≈ 56분. "빠른 디버깅"의 새 병목은 이제 **val**이다. 학습 여부만 볼 땐 **val을 subset·격epoch로 캡**해야 한다(§5).

### 4.3 "ForwardTime"은 오해의 소지 — 실은 recon 저장 시간

로그의 `ForwardTime = 0.4032s → 4.9942s`는 모델 forward가 아니다. [`train_part.py:155-158`](../utils/learning/train_part.py#L155): **new-best일 때 `save_reconstructions()` 소요시간**을 측정해 그렇게 이름 붙였을 뿐. 0.4 vs 5.0s 차이는 **디스크 I/O 노이즈**(모델 속도와 무관). 이름이 오해를 부르므로 지표로 쓰지 말 것.

---

## 5. 다음 액션

- 🐛 **[P0-fix] TrainLoss 평균을 실제 iter 수로 나누게 수정** ([`train_part.py:47`](../utils/learning/train_part.py#L47)). 안 고치면 `--max-iters` run들의 train loss가 전부 거짓이라 학습 곡선 판단 불가.
- **[P0] SSIM_bbox를 `validate()`에 추가**(순위축) + **zero-filled 베이스라인 1회 측정.** 이 둘이 있어야 "배운다"→"잘한다" 판정 가능. (§3.3)
- **[P1] val을 캡** (`--max-iters` 확장 or 격epoch val or val subset) → 56분/epoch 병목 해소. (§4.2)
- **[P1] 속도 레버 여전히 열림**: gradient checkpointing OFF / readout crop 640→384가 SDUM 경로에 실제 걸리는지 확인.
- **[P2] 학습 확인됐으니** 지표(위 P0) 정비 후 → 더 긴 학습 + config 스윕(`--num-cascades`/`--channels`). `[16,32]×4`는 병변 과평활 우려라 bbox 지표 확보 후 키울 것.

---

## 6. run1 → run2 → run3 요약

| 항목 | run1 | run2 (scalefix) | run3 (learncheck) |
|---|---|---|---|
| 손실(train) | ~1.0 고정 | 0.03~0.28 (per-iter) | per-iter 동일, **epoch요약 버그값** |
| val | — | — (인터럽트) | **0.1584→0.1448 (SSIM 0.842→0.855) ↑** |
| 학습 여부 | ❌ 안 배움 | 🟡 성립(미확정) | ✅ **확인(val 상승)** |
| 속도 | 7.0s/iter | 6.9s/iter | 7.0s/iter (compute-bound 확정) |
| 새 발견 | 스케일 의심 | num_workers 무효 | 🐛 TrainLoss 버그 · val 미캡 · bbox 미측정 |

> **성과:** val로 "배운다"를 확정. **다음 게이트:** TrainLoss 버그 수정 + **SSIM_bbox·zero-filled 베이스라인**으로 "잘한다"를 판정, 그리고 val 캡으로 사이클 완성.
