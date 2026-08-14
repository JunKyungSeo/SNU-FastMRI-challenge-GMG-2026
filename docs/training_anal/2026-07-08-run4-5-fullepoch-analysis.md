# SDUM run4–5 (`sdum_scalefix`, 풀 epoch) 학습 로그 분석 (2026-07-08)

run3(`--max-iters 300`으로 짧게 5 epoch)에서 "배운다"를 확인한 뒤, run4–5는 **`--max-iters`를 빼고 진짜 풀 epoch(4651 iter, `-e 1`)을 완주**시킨 run들이다. 로그: `shell_scripting/logs/run4_shell1` / `run5_shell2`.

> **run6 제외:** shell3(run6)은 run4의 완전 복제로 **셸스크립트 동작 확인용**이었다 → 이 분석에서 제외(별도 최소 기록 예정, §코멘트 참조).

> **한 줄 결론:** run4가 **처음으로 풀 epoch을 완주**해 신뢰할 TrainLoss(0.1146)·ValLoss(0.118 = whole-image SSIM 0.882)를 얻었다. run5는 `--num-workers 4→6` 실험으로 **속도차 0.057% → compute-bound 재확인**. 덤으로 harness 버그 2개(`val_loss_log`)를 발견해 **이번에 수정**했다.

> ⚠️ **검증 노트:** 수치는 워크플로우로 adversarial 재검증. loss는 run4=run5 **byte-동일**(seed 430 + cudnn.deterministic) — worker 수·shuffle과 무관.

---

## 0. TL;DR

1. ✅ **첫 풀 epoch 완주 (run4).** TrainLoss **0.1146**(정상값), ValLoss **0.118**(whole-image SSIM 0.882), TrainTime **32361s ≈ 9.0h** + ValTime 21.4분. run3 축약 없이 얻은 첫 신뢰 지표.
2. ✅ **TrainLoss 버그 미발동 확인.** `--max-iters` 없어 run3의 평균-분모 버그 안 걸림 → 0.1146은 참 평균(per-iter·ValLoss와 정합). 버그 코드는 그대로(§5-C, 보류).
3. 🔁 **num_workers 무효 확정(5번째) + compute-bound.** nw=4(run4) vs nw=6(run5): TrainTime 32361 vs 32342s = **0.057% 차**. 6.96 s/iter. val은 또 **미캡(21.4분)** — run3 지적 재발.
4. 🐛 **harness 버그 2개 발견 → 이번에 수정.** `val_loss_log.npy`가 (a) 정규화 전 **합(3.54=0.118×30)** 저장, (b) 매 run **덮어씀**. → §5-A(정규화 후 저장)·§5-B(run별 파일명) **코드 수정 완료**.
5. ⬜ **"안 망가짐"이지 "잘한다" 아직.** ValLoss는 whole-image SSIM(주석 없음) = **순위축 SSIM_bbox 아님**, zero-filled 베이스라인도 **미측정**. 0.882의 상당분은 MRI SSIM 바닥일 수 있음.

---

## 1. 실행 조건 + run3 대비 무엇이 달라졌나 (필수)

```bash
python train.py --model sdum --num-cascades 4 --channels 16 32 \
  -b 1 -l 2e-4 -e 1 -r 10 --num-workers {4|6} \
  -n sdum_scalefix -t /root/Data/train/ -v /root/Data/val/
```

| 항목 | run3 (learncheck) | run4–5 (scalefix 풀 epoch) | 왜 바꿨나 |
|---|---|---|---|
| `--max-iters` | 300 | **없음(0) → 풀 4651 iter** | run3 축약을 풀어 **참 풀-epoch TrainLoss/Val 측정** |
| epochs | 5 | 1 | 풀 epoch 1회(9시간)면 충분 |
| report-interval | 50 | 10 | — |
| net_name | `sdum_learncheck` | **`sdum_scalefix`** (= run2 명령) | — |
| num_workers | 4 | **run4:4, run5:6** | run5 = nw 4→6 속도 실험 |
| config | 4×[16,32] | **동일** | 통제변인 |

**run4 vs run5 (유일한 차이 = num_workers):**
| | run4 = shell1 (nw 4) | run5 = shell2 (nw 6) |
|---|---|---|
| per-iter loss | 기준 | **byte-동일** |
| TrainTime | 32361.16s | 32342.73s |
| ValLoss | 0.118 | 0.118 |
| 결과 | 풀 epoch 완주(exit 0) | 완주(같은 결과) |

→ nw 4↔6 시간차 **18.4s(0.057%)** = 유의미한 이득 없음. loss가 byte-동일한 건 seed_fix(cudnn.deterministic)가 **worker 수·shuffle과 무관하게** 재현성을 보장하기 때문(그래서 nw 실험이 timing에만 격리됨).

---

## 2. 직전(run3) 분석 지적사항 — 이번에 어떻게 됐나 (필수)

| run3에서 지적/시도 | run4–5 결과 | 상태 |
|---|---|---|
| P0 TrainLoss 평균 버그(`--max-iters`×분모) | 풀 epoch라 미발동 → 0.1146 정상. **코드는 그대로**(`train_part.py:47`+`:24`) | ⬜ 휴면(§5-C 보류) |
| P0 **SSIM_bbox**(순위축) 측정 | validate가 여전히 whole-image SSIM | ⬜ 여전 미측정(P1로 이동, §7) |
| P0 **full SSIM** 정확도(baseline·정합) | zero-filled 베이스라인 미측정, whole vs SSIM_full 정합 미확인 | ⬜ → **새 1순위(§7)** |
| P1 val 캡(21분 병목) | ValTime 1284.58s 그대로 | 🔁 재발 |
| P1 checkpointing OFF / readout crop | 안 건드림 | ⬜ 미착수 |
| P1 `--num-workers` 효과 | nw 4 vs 6 = 0.057% → **무효 확정** | ✅ 결론(음성) |
| P2 풀 학습 + config 스윕 | 풀 epoch은 됨, config `[16,32]×4` 그대로 | 🟡 부분 |
| — (신규) | `val_loss_log` SUM버그·덮어쓰기 | 🐛 발견 → **수정함(§5)** |

---

## 3. 손실·검증 분석

### 3.1 TrainLoss 0.1146 — 신뢰 가능한 첫 풀-epoch 값 ✅
`--max-iters` 없어 4651 iter 전부 돌고 `total_loss/len_loader`가 참 개수(4651)로 나뉜다 → **run3의 15.5× 축소 버그 미발동**. 0.1146은 per-iter(~0.11)·ValLoss(0.118)와 정합.

### 3.2 풀-epoch loss 추세 — 초반 급강 후 얕은 표류
run4 로그 500-iter 창 평균(466개 every-10th 샘플, 참 TrainLoss 0.1146):

| 구간 | 평균 | | 구간 | 평균 |
|---|---|---|---|---|
| 0–500 | 0.154 | | 2500–3000 | 0.105 |
| 500–1000 | 0.110 | | 3000–3500 | **0.101** |
| 1000–1500 | 0.122 | | 3500–4000 | 0.110 |
| 1500–2000 | 0.110 | | 4000–4500 | 0.111 |
| 2000–2500 | 0.115 | | 4500–4680 | 0.116 |

→ 첫 ~500 iter에 0.154→0.11 급강, 이후 **완전 평탄이 아니라 얕은 하향 표류**(2500–3500 저점 0.10대). "초반에 쉬운 부분을 잡고 이후 매우 완만히 개선"으로 읽는 게 정확.

### 3.3 ValLoss 0.118 = whole-image SSIM 0.882 — 해석 주의
- run3 ep0 0.1584(0.842) → ep1 0.1448(0.855) → **run4 0.118(0.882)**.
- ⚠️ **"누적 학습" 아님:** `train()`은 항상 scratch 시작(`best_val_loss=1.`, `start_epoch=0`, resume 없음). run4는 run3 가중치를 잇지 않는다. 정확히는 *"scratch 모델을 풀 epoch(4651 슬라이스, run3의 ~7.7배)로 학습하니 whole-image val loss가 더 낮다"* — 독립 run 간 비교이지 한 곡선이 아님.
- ⚠️ **SSIM 바닥 함정:** MRI는 zero-filled만으로 SSIM 0.7~0.85. **베이스라인 미측정**이라 0.882가 실제로 "좋은"지 미확정.
- ⚠️ **"NewRecord"는 무의미:** `best_val_loss` 초기값 1.0이라 **정상 run이면 무조건 발동** — run4/5가 run3를 이겼다는 뜻이 아니다.

---

## 4. 속도 / 효율 분석

- **s/iter:** 32361.16 / 4651 = **6.96 s/iter** (run1 7.0 · run2 6.9 · run3 6.98 · run4 6.96 → **compute-bound 확정**).
- **풀 epoch wall-clock:** train 9.0h + val 21.4분 ≈ **9.35h/epoch**.
- **num_workers 4 vs 6:** 32361.16s vs 32342.73s = **0.057% 차 → 이득 없음**. GPU가 데이터 대기로 노는 게 아니라 연산으로 꽉 참. (run5 nw 실험 = 음성 결과, 그래도 정보.)
- **val 미캡 재발:** `validate()`는 `--max-iters` 무관하게 풀 val(30 subject) → 21.4분/epoch. run3 병목 그대로.

---

## 5. 🐛 버그 / 아티팩트 + 조치

**A. `val_loss_log.npy`가 정규화 전 합(sum) 저장 → ✅ 수정함.** 파일값 `[[0, 3.5405]]` = 0.118 × 30(subject). 원인: `train_part.py`가 val_loss를 `/num_subjects` **하기 전에** append·save. → **수정**: validate 직후 정규화하고 저장(이제 0.118 저장). (num_subjects=30 확정: val h5 30개.)

**B. `val_loss_log.npy` 매 run 덮어쓰기 → ✅ 수정함.** 고정 경로라 run5가 run4를 덮어씀(run3 5-epoch 곡선도 소실). → **수정**: `--run-tag`로 파일명에 run 식별자(`val_loss_log_<tag>.npy`) 붙여 분리 저장. (사용법: 슬롯 명령에 `--run-tag run4` 추가.)

**C. run3 TrainLoss 평균 버그 잔존(휴면) → 보류(사용자 지정).** `:47 total_loss/len_loader` + `:24 max_iters break`. 이번엔 `--max-iters` 미사용이라 무해. **다음에 `--max-iters` 쓰기 전 수정 필요**.

**D. 셸 로그가 python 명령을 안 찍음 → ✅ 수정함.** run4/5 로그에 명령 라인 0개(그래서 nw=6이 로그만으론 안 보였음). → **수정**: `_template.sh`에 `set -x` 추가(실행 명령을 로그에 그대로 기록) + `shell_scripting/README.md` 반영. (기존 슬롯은 `cp _template.sh shellN.sh` 또는 수동으로 갱신해야 적용.)

---

## 6. 프로세스 교훈

- **결정론 → 같은 `-n`·config 재실행은 동일 결과 복제.** seed_fix(cudnn.deterministic)로 loss가 nw·shuffle 무관하게 byte-동일. 그래서 **run6처럼 같은 명령을 또 돌리면** 9시간짜리 같은 결과만 나온다(그건 셸스크립트 확인용이었음, §코멘트).
- **단일 GPU 직렬 실행:** run4 [DONE] 05:08:59 = run5 [START] 05:08:59. `shell1/2/3` 이름은 병렬처럼 보이지만 **직렬 큐**. 의미 있는 실험(예: nw)만 슬롯에 넣을 것.

---

## 7. 다음 액션 (우선순위 — **full SSIM 먼저**)

- **[P0] full SSIM을 먼저 정확히 잡기** (사용자 지정 1순위):
  1. ✅ `val_loss_log` 버그(§5-A/B) 수정 완료 → 이제 val 곡선 신뢰 가능.
  2. **zero-filled 베이스라인 whole-image SSIM 측정** → 0.882가 바닥 대비 실제로 얼마 위인지.
  3. validate의 whole-image `ssim_loss`가 대회 **SSIM_full(foreground mask)** 정의와 맞는지 확인·정합.
- **[P1] 그다음 SSIM_bbox**(순위축) — full SSIM 정리 후.
- **[P1] val 캡**(21분 병목) + 속도 레버(checkpointing OFF / readout crop 640→384 실적용 확인).
- **[P1] §5-C TrainLoss 평균 버그** — 다음 `--max-iters` 사용 전 수정.
- **[P2] 풀 epoch 확인됨** → config 스윕(`--num-cascades`/`--channels`). `[16,32]×4`는 병변 과평활 우려.

---

## 8. run1 → … → run4 요약

| 항목 | run1 | run2 | run3 | run4(=run5) |
|---|---|---|---|---|
| 명령 | 풀 epoch | scalefix(중단) | learncheck(`--max-iters 300`×5) | **scalefix 풀 epoch 완주** |
| TrainLoss | ~1.0 | 0.03~0.28(per-iter) | 0.010(버그값) | **0.1146(정상)** |
| ValLoss(whole SSIM) | — | — | 0.1584→0.1448(0.842→0.855) | **0.118(0.882)** |
| 학습 | ❌ 안 배움 | 🟡 성립 | ✅ 확인 | ✅ 풀 epoch 확인 |
| 속도 | 7.0 | 6.9 | 6.98 | 6.96 s/iter |
| 새 발견 | 스케일 의심 | nw 무효(추정) | 🐛 TrainLoss 버그 | 🐛 val_loss_log 2버그(→수정)·nw 무효 확정 |

> **성과:** 첫 풀 epoch로 신뢰 TrainLoss/Val 확보 + num_workers 무효 확정 + val_loss_log 버그 2개 수정.
> **다음 게이트:** ① **full SSIM 정확히**(zero-filled 베이스라인 + SSIM_full 정합) ② 그다음 SSIM_bbox ③ val 캡·속도 레버.

---

## 💬 코멘트

> **사용자 (2026-07-08):**
> - 이 중복 실행은 train 자체가 아니라 **셸스크립트 기능 확인용**이었다. run6도 내가 잘못 판단해 분석을 지시한 것 → **이 문서의 run6 관련 내용은 삭제** (run6 종료 후 "shellscripting 확인용 중복코드"로만 별도 기록).
> - 목표 우선순위: SSIM_bbox보다 **full SSIM을 먼저 정확히 잡는 것**이 1순위.
> - 버그 조치 방침: §5-A 자체 수정 / §5-B **run 번호 붙여 따로 저장**하도록 코드 수정 / §5-C 보류 / §5-D 즉시 수정(template·README 반영).
> - 위 문제 수정 후 앞으로 할 일 재점검.
>
> **→ 조치 완료 (2026-07-08):** run6 내용 삭제·문서를 run4–5로 재작성 · §5-A(정규화 후 저장)·§5-B(`--run-tag`)·§5-D(`set -x`) 수정 · §5-C 보류 · §7 우선순위를 full SSIM 먼저로 변경.
