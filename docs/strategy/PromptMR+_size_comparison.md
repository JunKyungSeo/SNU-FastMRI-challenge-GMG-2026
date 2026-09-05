# PromptMR+ 공식 repo vs 2025_4th_clone — 메모리 축소 비교

공식 [hellopipu/PromptMR-plus](https://github.com/hellopipu/PromptMR-plus)(ECCV 2024, CMRxRecon2024 1위) vs SNU FastMRI 챌린지 클론(`sMaRtIfy_final` = `ver2`, 두 클론 동일). VRAM 때문에 줄인 하이퍼파라미터 정리.

## TL;DR

| 질문 | 답 |
|---|---|
| 클론이 메모리 줄인 주 방법 | **모델 크기 축소 (~15×)** — 런타임 트릭이 아님 |
| 런타임 트릭(checkpointing, per-coil sens, batch=1, 384 crop) | **공식에도 이미 있던 것** → 클론은 그대로 유지 |
| 근거 | 공식 README: 트릭 다 켜도 fm-knee 학습 **≥17GB** 필요 |
| 원본 수치 출처 | 공식 `configs/model/pmr-plus.yaml` (검증 완료) |
| 유일한 주석 오기 | `len_prompt`: 공식 **[5,5,5]** ↔ 클론 실제 [3,3,3](주석엔 "원래 [3,3,3]") |

---

## 1. 모델 아키텍처 — 줄인 것 (메모리 목적)

| 파라미터 | 공식 | 클론 | 축소 | 설명 |
|---|---|---|---|---|
| **num_cascades** | 12 | **8** | 1.5× | 언롤 깊이. 활성값 ~선형 → 최대 요인 |
| **n_feat0** | 48 | **8** | **6.0×** | 메인 UNet 기본폭. 활성값 ~폭² |
| **feature_dim** | [72,96,120] | **[24,32,40]** | 3.0×/레벨 | 메인 레벨폭 |
| **prompt_dim** | [24,48,72] | **[8,16,24]** | 3.0×/레벨 | 프롬프트 채널폭 |
| **prompt_size** | [64,32,16] | **[16,8,4]** | 4×/변(면적 16×) | 프롬프트 텐서 공간크기 |
| **len_prompt** | [5,5,5] | **[3,3,3]** | 1.67× | 프롬프트 성분 수 |
| **sens_n_feat0** | 24 | **8** | 3.0× | Sens 네트워크 기본폭 |
| **sens_feature_dim** | [36,48,60] | **[12,16,20]** | 3.0×/레벨 | Sens 레벨폭 |
| **sens_prompt_dim** | [12,24,36] | **[4,8,12]** | 3.0×/레벨 | Sens 프롬프트폭 |
| **n_history** | 11 | **3** | 3.67× | cascade 간 history 상태 |
| **num_adj_slices** | 5 | **1**(하드코딩) | 5.0× | 입출력 채널 10→2. ⚠️ "adjacency 미구현" 주석 |

## 2. 모델 아키텍처 — 유지한 것 (변경 없음)

| 파라미터 | 값 (공식=클론) |
|---|---|
| n_enc_cab / n_dec_cab / n_skip_cab | [2,3,3] / [2,2,3] / [1,1,1] |
| n_bottleneck_cab | 3 |
| n_buffer | 4 |
| adaptive_input / use_sens_adj / no_use_ca | True / True / False |

→ **폭·언롤·history·인접 슬라이스만 축소. 블록 깊이·버퍼·구조 플래그는 원본 그대로.**

---

## 3. 학습 루프 / 데이터 / 런타임 트릭

| 항목 | 공식 | 클론 | 메모리 목적? |
|---|---|---|---|
| batch_size | 1 | 1 | — 동일 |
| precision / AMP | 미지정(fp32 추정) | fp32 | — 둘 다 fp32 |
| accumulate_grad_batches | 없음 | 없음 | — |
| GPU 수 | devices 2 (DDP) | 단일(추정) | ✗ 하드웨어 (DDP는 per-GPU 안 줄임) |
| max_epochs | 45 | 5 | ✗ 학습 길이 |
| optimizer / lr | AdamW(wd 1e-2) / 1e-4 | AdamW / 2e-4 | ✗ 튜닝 |
| lr_scheduler | StepLR(35, 0.1) | StepLR(11, 0.3) | ✗ 튜닝 |
| 크롭 384 | uniform_resolution(입력단) | center_crop(출력단) | ✗ **챌린지 출력 규격** |
| num_workers | 4 | 0(CUDA fork 회피) | ✗ 호스트 RAM |
| mask/accel | RandomMaskFunc [4,8] | HDF5 mask+MaskAugmentor | ✗ 데이터셋 |
| **gradient checkpointing** | fm-knee.yaml `true`* | 기본 `true` | ✅ 둘 다 ON (축소 아닌 유지) |
| **compute_sens_per_coil** | fm-knee.yaml `true` | 기본 `true` | ✅ 둘 다 ON |

\* 공식은 `model.yaml` 기본 `false` → **`fm-knee.yaml`에서만 true 오버라이드**. `fm-brain.yaml`은 미설정(기본 false). 즉 공식조차 데이터셋별로 트릭 on/off가 갈림. 클론은 코드 기본으로 항상 ON.

> 크롭 384: 양쪽 동일(챌린지 목표 크기). 단 클론은 cascade **이후** 출력단 적용 → cascade 연산 중 peak 메모리는 못 줄임.

---

## 4. 메모리 목적 분류 (요약)

| 분류 | 항목 |
|---|---|
| ✅ 진짜 메모리 축소 | n_feat0, feature_dim, sens 3종, prompt_dim, prompt_size, len_prompt(부분), num_cascades, n_history, num_adj_slices(부분) |
| ✗ 메모리 아님 | 384 crop(출력 규격), max_epochs, lr/optimizer/scheduler, num_workers, mask/accel, GPU 수 |
| ➖ 동등(원본 유지) | batch_size 1, fp32, accumulate 없음, CAB 깊이, n_buffer, checkpointing/per-coil |

**한 줄 결론:** 공식은 "메모리 효율적 *설계*"(암묵 history `s_t`, momentum 재사용, 인접 SME, checkpointing)를 제공하고, 클론은 그걸 **거의 그대로 물려받은 뒤 모델을 ~15× 축소**해 소비자 GPU에 맞췄다. 인접 SME(공식 간판 트릭)만 `num_adj_slices=1`로 포기.

---

## 5. 파라미터 수 감소 추정 (~15×, torch 없이 해석적)

| 구성 | 축소 근거 | 배율 |
|---|---|---|
| 메인 conv | 폭 3×(→conv당 ~9×) × cascade 1.5× | ~13× |
| Prompt 텐서/block | 공식 ~829k → 클론 ~10.4k (len_prompt 5→3 반영) | ~80× (비중 작음) |
| Sens 네트워크 | 폭 3×(~9×) + 입력채널 5→1 | 추가 |
| **종합** | 지배 요인 = 폭(~9×) × cascade(1.5×) | **~13~16× (≈15×)** |

원본이 수천만 파라미터급이면 클론은 **낮은 한 자릿수 백만대**. 정확값은 torch로 두 설정 `sum(p.numel())` 필요.

---

## 6. 출처 / 확인 필요

| 구분 | 내용 |
|---|---|
| 공식 아키텍처 | `configs/model/pmr-plus.yaml` |
| 공식 학습 설정 | `configs/base.yaml`, `configs/train/pmr-plus/{fm-knee,fm-brain}.yaml` |
| 공식 VRAM(≥17GB) | `README.md` |
| 클론 아키텍처 | [train.py:77-96](../../2025_4th_clone/FastMRI_challenge_sMaRtIfy_final/train.py#L77-L96) |
| 클론 모델 생성(adj=1) | [train_part.py:171-190](../../2025_4th_clone/FastMRI_challenge_sMaRtIfy_final/utils/learning/train_part.py#L171-L190) |
| 클론 384 crop | [promptmr_plus.py:674](../../2025_4th_clone/FastMRI_challenge_sMaRtIfy_final/utils/model/promptmr_plus.py#L674) |
| 클론 checkpointing | [reentrant_wrapper.py](../../2025_4th_clone/FastMRI_challenge_sMaRtIfy_final/utils/model/reentrant_wrapper.py) |

| 미확정(추후 확인) | 상태 |
|---|---|
| 공식 precision/AMP, accumulate | 어떤 config에도 없음 → fp32·미사용 **추정** |
| 공식 coil 수 / 총 파라미터 수 | 명시 없음 |
| 클론 학습 GPU 개수 | 추출값 없음(공식 devices=2) |
| 파라미터 ~15× | 해석 추정 → torch 실측 권장 |

> 논문(09b PDF): 아키텍처 수치는 Supplementary B로 이관돼 본문에 없음. 논문 확인 사항은 `2a+1=5`(인접), Table 5(history 11에서 최고), Strategy C(sens 메모리 1/5) 정도.
