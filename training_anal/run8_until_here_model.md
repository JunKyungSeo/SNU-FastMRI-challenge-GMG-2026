# run8 시점까지 모델 거시 판단 — SDUM vs 2025 우승작 비교 (2026-07-09)

run4~8까지의 SDUM 상태를, 작년 4팀(sMaRtIfy, 오픈 리더보드 1위) 솔루션과 **거시적으로** 비교해 "지금 어디에 집중해야 하나"를 정리. (모델·하이퍼파라미터 사실은 두 repo 코드/문서에서 직접 추출.)

---

## 1. SDUM(2026 현재) vs 2025_4th_clone 비교

| 항목 | 2025 (PromptMR+, 오픈 리더보드 1위) | 2026 SDUM (현재) |
|---|---|---|
| 백본 | PromptMR+ 언롤드(VarNet 계열), cascade 8(orig 12), 채널 ~6배 축소 | SDUM 언롤드 Restormer, cascade 4~6, ch [16,32]/[24,48] |
| 용량 철학 | **OOM 경계까지 최대로** ("파라미터 하나만 더 늘려도 CUDA OOM") | 보수적 — 문서상 **용량 한계 아님**(underfit) |
| **학습 epoch** | **65** (baseline 30 + MoE 35) | **1~2** (run9=4 예정) |
| epoch당 시간 | ~2.7h (~9740s train + ~650s val) | ~5~9h (readout-crop 후 ~5h) |
| **총 학습량** | **~175시간, ~49만 스텝 규모** | **~10~20시간, ~1~4만 스텝** |
| iters/epoch | ~4937 (batch 1, 슬라이스=iter) | 4651 |
| lr | 2e-4, StepLR(16·27ep ×0.3) | 2.4e-4, warmup→cosine (**2 epoch에 LR 0 도달**) |
| optimizer | AdamW, wd 0.01 | AdamW, wd 1e-3 |
| loss | SSIMLoss × √(유효면적) 가중 + gamyun 이미지마스크 | SSIMLoss 통짜 (**전경마스크·bbox 없음**) |
| augmentation | MRAugment(기하), **≈epoch 30부터 ON**(규제) | 설계됐지만 **OFF**(전 run 미사용) |
| 과적합 관리 | 별도 서버로 epoch별 일반화 측정→과적합 시점에 aug ON | train≈val(**underfit**) — 아직 관리 불필요 |
| 부가 | 4-way MoE(brain/knee×acc4/8) + ResNet50 라우터 | **knee 전용**(MoE 거의 무의미), GRAPPA prior 설계만 |
| 결과 | 최종 SSIM 0.9748 | SSIM_full 0.887 / SSIM_bbox 0.899 (run4, 2 epoch·이른 값) |

### 핵심 한 줄
**가장 큰 차이는 모델이 아니라 학습량.** 2025는 65 epoch(~7일), SDUM은 1~2 epoch — **~10~30배 덜 학습.** 나머지 차이(백본·용량·스케줄)는 이거에 비하면 부차적. SDUM 자체 진단(train 0.885 ≈ val 0.882 → underfit, 아직 하강 중인데 cosine이 LR을 0으로 조기 종료)도 동일하게 "덜 학습됨"을 가리킴.

---

## 2. 2025_4th_clone 하이퍼파라미터 (상세)

- **batch** 1 (배치>1 미구현), num_workers 0
- **epoch 65** = baseline 30 + MoE 35, **iters/epoch ~4937**, epoch당 ~2.7h → 총 ~175h
- **lr 2e-4**; Stage1 StepLR(×0.3 @16·27ep); Stage2 SequentialLR(LinearLR 워밍업 3ep → Constant → StepLR ×0.3/5ep)
- **optimizer** AdamW, **weight_decay 0.01**
- **loss** SSIMLoss(win7, k1=.01, k2=.03) × √(유효면적) 스칼라 가중 + gamyun 이미지마스크(무효영역 0)
  - ablation(README): 이미지마스크 +0.0004, √면적가중 +0.0008, MRAugment +0.0009, MoE +0.0002
- **augmentation** MRAugment(translation/rotation/shear/scale + fliph만; flipv·rot90·readout-flip 제외) — baseline OFF, MoE에서 exp 스케줄로 **≈epoch 30→50에 p=0.5까지 램프**
- **정규화** wd 0.01 + aug + best-ckpt-by-val; dropout·gradient clip 없음; grad checkpointing(메모리용)
- **부가 기법** 4-way MoE, ResNet50 라우터(GRAPPA 이미지로 학습, train/val 반전), 2단계 커리큘럼(단일→MoE warm-start), hacker 마스크 과적합, 최종제출엔 val을 학습에 흡수(use_val_for_final)
- **핵심 판단 근거(README §3):** 모델을 OOM 경계까지 키운 뒤, **별도 서버로 epoch별 일반화를 측정해 과적합이 시작되는 ~epoch 30에 augmentation을 켬.** 너무 늦게(45ep) 켜면 최종 SSIM 0.9743으로 하락.

> ⚠️ **2025 관련성 주의**: 2025는 **brain+knee & 순수 SSIM 채점**. 2026은 **knee 전용 + SSIM_bbox(절반) + 추론시간**. → MoE/라우터는 2026엔 거의 무의미. 2025 플레이북은 **뼈대(최대용량+장기학습+후반 augmentation)로만** 쓰되 bbox·시간 축으로 재조정 필요.

---

## 3. 어디에 집중해야 하나 (우선순위)

1. **[압도적 1순위] 학습을 훨씬 오래.** 레퍼런스 65 epoch vs 지금 1~2. 지배적 격차이자 가장 싼 고효율 레버. cosine 지평을 2 epoch가 아니라 **전체 계획 길이(예: 20~40+ epoch)** 로 잡기. readout-crop으로 ~5h/epoch → 20 epoch ≈ 4일(다일 커밋). **용량은 여기부터 손대지 말 것.**
2. **[2순위] SSIM_bbox를 검증에 배선.** 점수 절반인데 **지금 측정조차 안 함**(통짜 SSIM만). metrics.py로 매 epoch **SSIM_full(전경마스크)+SSIM_bbox** 로깅, best_model을 **실제 채점 지표**로 선택. 현재 점수 50%를 눈 감고 달리는 중.
3. **[3순위] bbox 가중 손실**(bbox_paper 01/02 근거). 순위축(SSIM_bbox)에 **용량보다 직접적** 레버. 단 2번(측정)이 선행.
4. **[4순위] ML 위생**: gradient clipping, NaN 감지, **resume**(다일 학습 + 인스턴스 재시작 대비). run7 §3.5가 지적한 공백.
5. **용량 확대(모델 키우기): 지금은 낮은 우선순위.** 용량 벽 증거 없음 + 키우면 **추론시간 점수 손해** + underfit 악화. 충분히 학습시킨 뒤 SSIM_bbox가 정체 & train≈val 유지될 때만 재검토.
6. **augmentation: 나중.** 2025처럼 **과적합이 나타날 때(train/val 격차 벌어질 때)** ON. 지금 underfit이라 켜면 학습만 느려짐.

### 결론
지금 SDUM은 **"레퍼런스보다 ~30배 덜 학습된 underfit"** 상태. 모델을 만지기 전에 **(1) 오래 학습 + (2) SSIM_bbox 측정 배선** 이 유일하게 중요한 두 가지.

---

*근거: SDUM측 — config.py/adapter.py/train_part.py, training_anal/2026-07-09-run7·run8 분석, strategy/confirmed_model.md. 2025측 — 2025_4th_clone/README, hacker.sh, train.py/train_moe.py, promptmr_plus.py, mraugment.*
