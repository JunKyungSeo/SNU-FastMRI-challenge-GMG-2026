# run12 분석 — bbox 가중 손실(C1) + UC 묶음, run11 대비 3축 퇴행 (2026-07-18)

> ✅ **완료.** run12 10ep 완주 + recon_eval 채점 끝.

**한 줄 결론:** 현 best인 run11(fg-loss)에 **bbox 손실 λ=0.5(C1) + UC 를 동시에** 얹었더니 recon_eval **3축이 전부 퇴행** — full 0.9161→**0.9114**, bbox 0.9095→**0.9045**, 시간 1514→**1535ms**. bbox 손실이 노린 **SSIM_bbox마저 떨어져** 게이트("bbox↑ AND full 비퇴행")를 양쪽 다 위반. **run11 유지 · run12 롤백.** 단 두 레버 동시라 bbox 탓/UC 탓 **분리 불가** → UC 단독(run13, 진행 중)이 판별.

## §0 핵심 요약

- ❌ **C1(bbox 손실) 실패**: 목표는 SSIM_bbox↑였는데 오히려 **0.9095→0.9045 (−0.0050)**. λ=0.5 bbox 가중이 병변 박스를 살리기는커녕 aggregate bbox를 깎음. 게이트 위반.
- ❌ **full도 퇴행**: 0.9161→**0.9114 (−0.0047)**. bbox 가중이 전체 최적화를 왜곡한 것으로 의심(손실이 병변 슬라이스에 과편중).
- ❌ **시간 손해**: UC param(time/label 임베딩) 추가 → 1514→**1535ms (+21)**. 채점 시간축에서 순손실.
- ⚠️ **confound — 2 레버 동시(원인 분리 불가)**: run12 = run11 + `--bbox-loss-lambda 0.5` + `--uc`. 3축 퇴행이 **bbox 때문인지 UC 때문인지, 상호작용인지 격리 안 됨.** → **run13**(run11 + `--uc`만, 진행 중)이 UC 순효과를 준다: run13≈run11이면 UC 중립 → bbox가 run12 원흉; run13<run11이면 UC도 유해.
- ✅ **val이 실채점을 예측 적중**: run12 fg-val(foreground-masked) best **0.8996** (run11 0.9041 대비 **−0.0045**) → recon full **−0.0047**과 거의 일치. fg-val이 신뢰 예측기임을 재확인(run11에서 세운 결론 지속).
- ⚠️ **acc8(어려운 8배)이 더 타격**: full acc8 **−0.0067** / bbox acc8 **−0.0059** 로 acc4(−0.0029 / −0.0041)보다 하락 폭이 큼. 언더샘플이 심할수록 손해 집중.

## §1 직전(run11) 대비 diff  *(필수)*

| 항목 | run11 (현 best) | **run12** | 비고 |
|---|---|---|---|
| `--fg-loss` | ON | **ON (유지)** | BASELINE=run11 채택 → fg-loss 계승 |
| `--bbox-loss-lambda` | 0.0 (off) | **0.5 (C1 ON)** | ← 추가 레버 ① 병변 bbox 손실 가중 |
| `--uc` | off | **ON** | ← 추가 레버 ② acc4/acc8 + cascade-index 조건화 |
| 그 외 | [16,32]×4, 10ep, readout-crop, LR 2.4e-4, warmup 200 | **동일** | seed 430 |

**명령(로그 확인):** `--fg-loss --bbox-loss-lambda 0.5 --uc` (shell3 log L7, `[config] BASELINE=run11`).
**목적:** bbox 가중으로 무너진 병변 4개 label(ACL sprain H/L·Subchondral edema·Bone-Fracture, Stage-0 진단) 구제 = SSIM_bbox↑ + UC로 acc 조건화.
⚠️ **단일-레버 A/B 컨벤션 위반(2 레버 동시).** 사용자 결정(GPU 아껴 묶음) — 결과 해석에 confound 상존(§0).

## §2 직전 분석(run11) 지적 추적  *(필수)*

| 지적 (출처) | 판정 | 근거 |
|---|---|---|
| **bbox 하락 감시** (run11 §0·§7, @esel34) | 🔁**재발·악화** | fg-loss는 bbox 안 깎았는데(run11 bbox↑), **명시적 bbox 손실을 넣은 run12에서 bbox가 오히려 하락**(0.9095→0.9045). 역설. |
| "bbox 떨어지면 bbox 가중손실 추가 검토" (run11 §7) | ❌**시도→실패** | 바로 그 bbox 가중손실(C1 λ=0.5)이 run12 = aggregate bbox를 못 올림. |
| fg-val이 실채점 예측 (run11 §0) | ✅**지속** | run12도 fg-val(−0.0045)이 recon full(−0.0047) 예측. |
| single-lever A/B 컨벤션 (training_anal §0) | 🐛**위반** | 2 레버 동시 → 원인 분리 불가. run13로 사후 격리. |
| 작은+오래 우월 (run10) | (계승) | config 동일([16,32]×4·10ep). |

## §3 손실 · 지표

**fg-masked val 궤적 (1−SSIM, 낮을수록 좋음) — run11 vs run12:**

| epoch | run11 val | run12 val | run12−run11 |
|---|---|---|---|
| 0 | 0.1092 | 0.1157 | +0.0065 |
| 3 | 0.09894 | 0.1041 | +0.0052 |
| 6 | 0.09652 | 0.1013 | +0.0048 |
| 9 | 0.09587 | 0.10039 | +0.0045 |
| **best** | **0.09587**(ep8) | **0.10039**(ep9) | **+0.0045 (run12 열세)** |

→ run12가 **매 epoch run11보다 val 열세**, 격차가 처음부터 끝까지 유지(수렴해도 안 좁혀짐). "학습만 더 하면 따라잡는다"가 아님 = 손실 설계 자체의 손해.
※ TrainLoss는 run12가 높음(0.166→0.122 vs run11 0.126→0.095)이나 **bbox λ항이 손실에 더해진 값이라 run11과 절대 비교 무의미.**

**recon_eval 3축 (leaderboard 실채점, 0.5·full+0.5·bbox+시간):**

| run | full | bbox | ms/slice | 0.5f+0.5b | acc4 full/bbox | acc8 full/bbox |
|---|---|---|---|---|---|---|
| run10 (whole-loss) | 0.9118 | 0.9021 | 1523 | 0.90695 | 0.9361 / 0.9305 | 0.8875 / 0.8738 |
| **run11 (fg-loss)** | **0.9161** | **0.9095** | **1514** | **0.91280** | 0.9390 / 0.9361 | 0.8933 / 0.8828 |
| run12 (fg+bbox+UC) | 0.9114 | 0.9045 | 1535 | 0.90795 | 0.9361 / 0.9320 | 0.8866 / 0.8769 |

- **run12 vs run11**: full **−0.0047**, bbox **−0.0050**, 시간 **+21ms** → 3축 전부 열세. 조합점수 0.9128→0.9080.
- **run12 vs run10**: full −0.0004(거의 동), bbox +0.0024, 시간 +12ms → SSIM 조합은 +0.001로 미세 우위지만 **시간은 손해**. 즉 run12는 run11이 확보한 이득을 도로 반납하고 run10 수준으로 후퇴(+속도만 나빠짐).
- **핵심 역설**: bbox 손실을 켰는데 aggregate SSIM_bbox가 **떨어졌다.** 목표 4개 label이 국소적으로 올랐어도 나머지 박스가 더 깎였을 가능성 → **채점은 aggregate라 어쨌든 실패**. (라벨별 분해는 §7 P1.)

## §3.5 일반 ML 구현 점검 (델타)

| 항목 | 상태 | 근거/비고 |
|---|---|---|
| 대회 지표 로깅(bbox) | ⚠️부분 | bbox **손실**은 배선됨(C1). 단 **val에는 여전히 bbox 미로깅**(fg-full만) → bbox 퇴행을 val로 못 봄(recon_eval에서야 드러남). |
| UC(조건화) | ✅배선·동작 | smoke `SMOKE uc= True` + grad-flow pytest 통과. 단 **효과는 (묶음 기준) 음성**, 시간만 +21ms. |
| best-model 선택 | ✅ | fg-masked val 기준(채점축 근사). run12 best=ep9. |
| 나머지(seed·resume·sched·grad-ckpt) | ✅계승 | run10/11과 동일. AMP는 🚫(Pascal FP32). |

## §7 다음 액션 + 다음 학습 고려사항

**지금 바로 (P0/P1):**
- **P0 — run12 롤백. run11(fg-loss)이 best 라인.** run12는 제출 후보 아님.
- **P0 — run13(run11+UC, 진행 중) 채점**으로 UC 순효과 격리:
  - run13 ≈ run11 → **UC 중립** → run12 퇴행의 원흉 = **bbox 손실(C1)**. bbox λ=0.5 폐기/재설계.
  - run13 < run11 → **UC도 유해** → UC 폐기(원래 확정설계 "UC 제거"로 회귀 근거). SPEC과 일치.
- **P1 — bbox 라벨별 재분해**: `scripts/bbox_ssim_breakdown.py`의 RECON을 `sdum_run12`로 → 목표 4개 label(ACL sprain 등)이 국소적으로라도 올랐는지 확인. (올랐어도 aggregate 실패지만, C1 방향 자체가 틀렸는지 판단 근거.)

**다음 학습 고려(가설·탐색거리, 확정 아님):**
- bbox 손실이 원흉이면: λ↓(0.5→0.2?) 또는 "가중" 대신 **cap/정규화 방식 재설계**. 단 fg-loss가 이미 bbox를 올려둔 상태(run11 bbox 0.9095)라 **추가 bbox 손실의 한계효용이 음수**일 수 있음 → 폐기가 유력.
- UC가 유해면: SPEC의 "UC 제거" 재확정. knee는 조건 신호가 acc4/acc8 1비트뿐이라 논문(11 contrast) 이득이 안 옮겨온 것 → **param↑(시간손해)만 남음.**
- **교훈**: run11까지의 승리 공식(작은 모델·긴 epoch·채점정합 fg-loss)에 **레버를 더 얹는 게 이 지점에선 순손해.** 다음은 "빼기"(UC 제거) 또는 fg-loss 라인 위 미세조정으로.

## §8 run 누적 요약 (run1 → run12)

| run | 핵심 | 실채점(full/bbox/ms) | 판정 |
|---|---|---|---|
| 10 | [16,32]×4·10ep, whole-loss | 0.9118 / 0.9021 / 1523 | epoch>용량 확인 |
| 11 | +fg-loss(B5) | **0.9161 / 0.9095 / 1514** | **best (fg-loss 승)** |
| 12 | +bbox λ0.5(C1) +UC | 0.9114 / 0.9045 / 1535 | **3축 퇴행 → 롤백** |
| 13 | run11 +UC (진행 중) | (대기) | UC 순효과 격리용 |
