# run7 분석 — readout-crop + cosine/warmup + LR 2.4e-4 (2026-07-09)

**한 줄 결론:** readout-crop이 **속도를 ~40% 줄인 것(9h→5.2h/epoch)만이 확실한 성과**다. 정확도 개선(val 0.8812→0.8846, **+0.003**)은 **matched-epoch로 보면 crop·cosine 기여 ≈0**(2번째 epoch 몫) + N=1 노이즈 바닥 근처라 **유의하다 말하기 어렵다**. 게다가 **순위축 SSIM_bbox는 여전히 미측정** → 이 +0.003이 병변에 닿았는지조차 모른다.

> ⚠️ **정직성 정정:** 이전 세션에서 "run7 loss 0.037까지 하강"이라 한 건 **epoch1 iter3500 단일 per-iter 값**(batch=1 단일슬라이스 노이즈)이었다. **실제 epoch 평균 train = 0.100**. 과장이었고 정정한다.

---

## §0 핵심 요약 (한눈에 보는 요약)

- ✅ **속도 = 유일한 확실 이득**: readout-crop으로 **~5.21h/epoch**(18,756s), run4-5 ~9h 대비 ~40%↓. 8GB에 ~6.4GB 안착. (cosine은 FLOP 중립이라 속도엔 0 기여 → **속도는 100% crop 몫**.)
- ⚠️ **정확도 개선은 modest를 넘어 "거의 없음"**: val SSIM 0.8812(ep0)→0.8846(ep1). 근데 **ep0(crop+cosine, 1ep) 0.8812 ≈ run4-5(no-crop/no-cosine, 1ep) 0.882** → matched-epoch면 crop+cosine 품질 기여 **≈0**, +0.003은 순전히 **2번째 epoch** 덕.
- ⚠️ **새 채점(2026-07-09 규칙변경) = `0.5·SSIM_full + 0.5·SSIM_bbox + 추론시간`**. 우리 val whole-image는 **SSIM_full의 거친 대리(=점수 절반)** 일 뿐 — foreground-mask·bbox·추론시간은 미측정. full·bbox는 **5:5 동일 가중**이라 bbox만 볼 게 아니고, 추론시간(~0.00n)도 SSIM 접전 시 순위를 가름. → **recon_eval(shell7)로 3축 한 번에** 봐야 함.
- ⚠️ **N=1·노이즈 바닥 미상**: per-slice loss 0.037~0.29 진동. 15 subject에서 +0.003은 유의성 검정 없음 = **run-to-run 노이즈 범위**.
- ✅ **과적합 아님(팀 판단 맞음) → under-training**: train·val 둘 다 계속 하강 중. cosine이 2 epoch만에 LR→0으로 **수렴 전 조기 종료**. (capacity wall 아님)
- 🐛 **train/val 지표가 서로 다른 구현**: train=conv2d `SSIMLoss`(loss_function.py), val=skimage `structural_similarity`(utils.py). + train 0.100은 epoch running-mean이라 end-state 과소. → 0.015 갭은 **soft한 숫자**.

---

## §1 실행 조건 + 직전(run4-5) 대비 diff  *(필수)*

| 항목 | run4-5 (직전) | run7 | 검증하려던 것 |
|---|---|---|---|
| 해상도 | full 640 | **640→384 crop** (`--readout-crop`) | 속도이득 + 품질 유지 |
| LR 스케줄 | 상수 2e-4 | **warmup200→cosine** | 정체 tail 뽑기 |
| base LR | 2e-4 | **2.4e-4**(논문값) | 논문 정합 |
| epoch | 1 | **2** | cosine 전체 horizon |
| 이름 | `sdum_scalefix` | `sdum_run5`* | *태그충돌 잔재 |

**목적:** readout-crop 속도이득 확보 + cosine으로 run4-5의 정체를 뚫는지.

## §2 직전 분석/extra.md 지적 추적  *(필수)*

| 지적 (출처) | 판정 | 근거 |
|---|---|---|
| readout crop 미배선 = 최대 속도 레버 (extra.md A) | ✅**해결** | ~5.2h/epoch 실측(~40%↓) |
| 고정 LR 정체 원인 (extra.md C) | ⚠️**배선했으나 효과 미미** | cosine 켜짐. 단 matched-epoch 품질 기여 ≈0 |
| LR 논문값 2.4e-4 | ✅**해결** | 적용 확인 |
| 🐛 `val_loss_log` SUM/덮어쓰기 (§5-A/B) | ✅**해결·검증** | 이번 로그가 per-subject 정규화·run별 저장 정상 동작 |
| SSIM_full/bbox 로깅 없음 (extra.md D) | ⬜여전(학습중 로깅) → **사후 채점은 가능** | validate엔 미배선. 단 `recon_eval.py`로 checkpoint 사후 3축 채점 가능(cv2 headless로 해결) |
| baseline(zero-filled) 대비 | ⬜**여전** | extra.md 1회 측정만(full 0.833/bbox 0.820) |

## §3 손실 · 속도 · 지표

**손실 (epoch 평균):**
| epoch | Train | Val(=1-SSIM) | Val SSIM |
|---|---|---|---|
| 0 | 0.1159 | 0.1188 | 0.8812 |
| 1 | **0.1001** | **0.1154** | **0.8846** |

- **귀속 (적대적 검증 결과)**: 속도 = crop 100%(cosine FLOP 중립). **품질 = confounded + matched-epoch로 ≈0** — ep0(0.8812) ≈ run4-5(0.882)라, crop·cosine 어느 쪽도 1-epoch 시점에 품질을 못 샀고, +0.003은 2번째 epoch 몫. crop-only vs cosine-only ablation 없이는 분리 불가(하지만 anchor가 "둘 다 미미"를 시사).
- **추론시간 관점(새 규칙)**: readout-crop은 공간을 40% 줄여 **추론시간 점수에도 이득**(작지만 접전 시 유효). 반대로 run8의 큰 모델은 추론시간 손해 → 새 규칙에선 "capacity vs 속도" 트레이드가 점수에 직접 반영됨.
- **정직성**: 전부 whole-image(= SSIM_full의 대리, data_range도 gt.max()로 채점식과 다름). **실제 채점 3축(full/bbox/시간) 미측정** → recon_eval(shell7)로 처음 봄. N=1, 노이즈 바닥 미상.

## §3.5 일반 ML 구현 점검 (코드 접지)

| 범주 | 항목 | 상태 | 근거 |
|---|---|---|---|
| 재현성 | seed+cudnn deterministic | ✅있음 | utils.py:76-83 seed_fix(430), worker seed load_data.py:12-18 |
| 최적화 | LR 스케줄(warmup/cosine) | ✅있음 | train_part.py:130-140 (sdum&warmup>0 조건부) |
| 최적화 | optimizer·weight decay | ✅있음 | AdamW wd=1e-3 (train_part.py:117-120) |
| 최적화 | **gradient clipping** | ⬜**없음** | train_part.py:35-37 clip 없음 |
| 최적화 | 유효batch(grad accum) | ⬜없음 | 매 iter step, batch=1 |
| 체크포인트 | best/last 저장 | ✅있음 | save_model model.pt+best_model.pt (train_part.py:84-97) |
| 체크포인트 | **resume(이어학습)** | ⬜**없음** | start_epoch=0 하드코딩, load 경로 없음 |
| 검증 | 매 epoch validation | ✅있음 | train_part.py:143-147 |
| 검증 | best-model 선택 기준 | ⚠️부분 | val_loss<best 이나 **whole-image SSIM 기준**(bbox 아님) |
| 지표 | **대회 SSIM_full/bbox 로깅** | ⚠️부분 | metrics.py 구현 있으나 **미배선**, validate는 whole만 |
| 지표 | baseline(zero-filled) 대비 | ⬜없음 | 학습 중 비교 없음 |
| 데이터 | augmentation 배선 | ⚠️부분 | 인프라 있으나 `--aug-*`+set_epoch 없어 off |
| 안정성 | **NaN/발산 감지** | ⬜**없음** | isfinite 체크 0건 |
| 효율 | grad checkpointing·workers | ✅있음 | latent_recon.py:61.. + num_workers 4 |
| 효율 | mixed precision | 🚫불가 | Pascal FP32 강제(의도) |

> **주목 갭(run9 후보)**: ④gradient clipping·⑬NaN 감지(긴 run 안전), ⑦resume, ⑩+⑨**SSIM_bbox 로깅·모델선택**, ⑫augmentation.

## §4 다음 액션 + 다음 학습 고려사항 (run9)

**P0 — 지금 바로 (학습 아님, 최고 EV):**
- ⭐ **recon_eval로 실제 3축 채점(shell7)**: run7(`-n sdum_run5`)·run8(`-n sdum_run8`)을 `recon_eval.py -p /root/Data/leaderboard`로 **SSIM_full + SSIM_bbox + 추론시간** 한 번에 측정 → zero-filled(full 0.833/bbox 0.820)과 비교. 여태 whole-image 대리만 봤으니 **실제 점수를 처음 보는 것**. (cv2는 opencv-python-headless로 해결됨.)

**다음 학습 고려사항 (이 run이 시사):**
- 🔎 **run7은 under-training**이지 capacity 한계가 아님(둘 다 하강 중 LR이 0됨). → **가장 싼 고EV 레버 = 현재 capacity로 더 오래 학습**(epoch↑ 또는 cosine `total_iters`를 늘려 하강 중에 LR이 0 안 되게).
- ⚠️ **run8(cascades6+[24,48]) 재평가**: capacity 베팅은 run7 데이터로 **약하게 정당화**됨 — capacity wall 증거 없음. 더 큰 모델은 epoch/wall-clock을 줄여 **under-training을 오히려 악화**. run8은 이미 도니 capacity 신호는 보되, **run9 우선순위는 "capacity 고정 + 더 오래 → plateau 확인 후 필요시 확장"**.
- 🎯 **손실이 순위축을 안 겨냥**: `SSIMLoss`는 whole-image 균일(bbox 가중 없음). bbox 측정서 정체 확인되면 **bbox-weighted loss**가 capacity보다 직접적 레버.
- ✅ **readout-crop 유지**(유일한 확실 이득). schedule **길이**를 run9 튜너블로.
- 📏 **노이즈 바닥 확립**: sub-0.005 델타를 믿기 전 per-subject val 분산 / multi-seed 반복.

---

## §5 run 누적 요약

| run | 이름 | 핵심 변경 | 결과 한 줄 |
|---|---|---|---|
| 4–5 | `sdum_scalefix` 풀 | 풀 epoch | Train 0.1146 / Val 0.882 (1ep) |
| 6 | `sdum_smoke` | +readout-crop, +스케줄러 (30iter 게이트) | 코드 동작·~40%↓ 실측 |
| **7** | `sdum_run5`* | +crop, +cosine warmup200, +LR2.4e-4, 2ep | **속도 ~5.2h/ep(확실). 정확도 val 0.881→0.885(=2nd epoch 몫, crop·cosine 기여 matched-epoch로 ≈0). under-training. bbox 미측정.** |
| 8 | `sdum_run8` | (진행중) +cascades6+[24,48] | VRAM ~6GiB(OOM 기우). capacity 베팅은 run7 데이터로 약함 |

> ⚠️ **결론**: run7의 진짜 교훈은 "cosine이 정체를 뚫었다"가 **아니라** "아직 **under-training**이고 **실제 채점(full+bbox+시간)을 한 번도 안 봤다**"는 것. 다음: **recon_eval로 3축 채점(shell7) + 현재 capacity로 더 오래 학습(shell8=run9)**. bbox 한쪽이 아니라 **full·bbox 5:5 + 추론시간** 균형으로 판단할 것.
