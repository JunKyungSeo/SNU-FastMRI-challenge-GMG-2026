# 확정 파이프라인 (Pipeline) — FastMRI 2026

> 상태: **확정 (2026-07-04)** · 이 문서가 **재구성 파이프라인 구조의 단일 기준(source of truth)**.
> ⚠️ **백본(main network)은 여기서 정의하지 않는다 — 교체 가능하므로 [`backbone.md`](backbone.md)에서 관리**(현재: SDUM). 이 문서는 **백본이 무엇이든 성립하는 고정 계약**(입력 분기·crop·정규화·증강·감도맵·GRAPPA prior·출력·채점)만 담는다.
> 세부 근거는 [`model_strategy.md`](research/model_strategy.md) · [`augmentation.md`](research/augmentation.md) · [`preprocessing.md`](research/preprocessing.md) · 배선은 [`implementation.md`](implementation.md).
> 실측 검증(2026-07-04): readout crop 무손실·정규화·GRAPPA 정렬을 acc4/acc8 · slice 13에서 확인(검증 스크립트는 전략 검증용 임시라 삭제, 결과만 아래 표에 유지).

---

## 파이프라인 (좌 → 우)

```
train+val 병합 ─ K-fold ─┐
                         │
image_GRAPPA(384²,mag) ──┼───────────────[ ÷s ]──→ (flip aug) ──→ main network ──[ ×s ]──→ output 384²
                         │  (fft 안 함)      ↑
원본 kspace(640×W,15ch)──┴─ ifft_h→crop 640→384→fft_h ─→ (mask aug) ┘
                            (readout만, 폭 native)
```

## 구성 요소 (확정 사항)

### 1. 입력 2분기
- **원본 k-space** (640×W·15coil·complex): **readout(높이)만 640→384 crop**(`ifft_h → crop → fft_h`). **폭(phase-encode)은 native(acc4 368 / acc8 372) 유지.** 배치 크기 통일이 필요하면 **k-space에서 mask=0으로 pad**(영상 도메인 pad 금지).
- **image_GRAPPA** (384²·magnitude): **영상 도메인 유지. fft 하지 않음.**

### 2. 정규화 `÷s` (합류 지점, 증강 전)
- **입력 유도 전역 스칼라 1개.** zero-filled RSS 영상의 robust percentile(예: p99)에서 산출. **`maximum` attr 사용 금지**(label 통계 → 추론 시 없음).
- **k-space · GRAPPA · target에 동일한 s로 나눔.** 출력단에서 `×s`로 역정규화.
- **acc4/acc8/기타 구분 없이 전부 동일한 방식**(가속률 무관 단일 규칙).
- **코일별 아님(전역 1개)** → 코일 상대비 보존 → sens_net 무해.

### 3. 증강 (MRAugment, 지수 스케줄)
- 기하변환 g를 **한 곳에서 정의** → `{k-space, GRAPPA, target, bbox}`에 **동일 적용**.
- **mask 증강 = k-space에만**(ACS 100% 보존, 주변부만 thinning). GRAPPA엔 미적용.
- **flip = 전부 동기**(k-space·GRAPPA·target·bbox). 좌-우(width)만.
- warmup(첫 10~15ep off) → 지수형 확률 점증. 샘플별 결정적 RNG(재현성).

### 4. 감도맵 (sens map 재사용)
- **sens_net 1회 추정**(ACS 중앙 라인 기반) → 증강 view에 **재사용**:
  - **mask 증강 → 맵 그대로**(ACS 보존이라 byte-identical).
  - **flip → 맵도 flip**(원본 맵을 뒤집어 사용, U-Net 재실행 없음).
- 이득 실현 조건: 감도추정을 **per-view forward에서 분리**(canonical source에서 1회 추정 후 g 적용).
- ⚠️ flip view에 안 뒤집은 맵을 쓰면 coil-combine/DC가 조용히 오염 → "flip이면 맵도 flip" 필수.

### 5. Main network (백본은 교체 가능 → [`backbone.md`](backbone.md))
- **백본(unrolled 재구성 네트워크) = [`backbone.md`](backbone.md)에서 관리** (현재: SDUM / 이전: E2E-VarNet 베이스라인). 아래는 **어떤 백본을 쓰든 지켜야 하는 파이프라인 계약**이다.
- **DC(data consistency)는 원 측정 k-space 기준**(readout-crop 후에도 정확히 성립, 실측 0.00e+00). unrolled 백본이면 종류 무관하게 이 기준을 따른다.
- **GRAPPA = 영상 도메인 prior 채널 + residual**: `out = grappa + f([image_input, grappa, …])`. 백본에 **얹는 구체 방식**은 backbone.md(백본별로 다름).
- GRAPPA(384²) ↔ k-space(384×폭) 격자 정합: GRAPPA 폭을 native로 crop, 또는 **출력단 384² refinement head**에서 융합(격자 동일).

### 6. 출력
- `×s` 역정규화 → `center_crop(384,384)` → alias-free 384² 영상.

---

## 실측 근거 (acc4 / acc8 · slice 13)

| 항목 | acc4 | acc8 | 판정 |
|---|---|---|---|
| readout-crop 후 샘플라인 vs 측정값 오차 | 0.00e+00 | 0.00e+00 | **DC 정확 성립** |
| readout-crop recon corr (vs image_input) | 1.0000 | 1.0000 | **무손실** |
| 메모리(픽셀, native 640×W 대비) | 60.0% | 60.0% | **40% 절감** |
| 완전 384²(폭 pad) 시 k-space 에너지 번짐 | 1.353 | 1.331 | **폭 pad 금지** |
| crop으로 버려지는 readout 에너지 | 10.9% | 12.6% | 채점 384²라 무관 |

- readout crop은 현재 파이프라인(출력단 `center_crop`)과 **수학적으로 동일**(상대오차 ~1e-16). 즉 crop을 입력단으로 앞당겨도 선형 recon 손실 0.
- 버려지는 11~13%는 실제 조직이나 **정답 `image_label` 자체가 384² crop본**이라 채점 대상 아님.

---

## 남은 실험 / 미확정
- **A/B: 입력 crop vs 출력 crop** — 비선형 U-Net 문맥 차이만 SSIM으로 확인(readout는 완전샘플이라 alias 유입 없음 → 이론상 차이 작음).
- **SSIM_full 배경 마스크 정의** — eval code(7/8 공개) 후 동기화.
- **채점 3축 가중치** 미공개 → SSIM_bbox 비퇴행 게이트를 1차 기준.

## 구현 현황 (상세: [`implementation.md`](implementation.md))

### ✅ 데이터 측 구현·검증 완료 (2026-07-04, utils/data/) — 전부 플래그 게이트, 기본 off=기존 비트 동일
- **GRAPPA 로딩(train/val)**: `load_data.py` `return_grappa` 플래그 → `image_grappa` 로드, flip 동기 정렬.
- **정규화 ÷s**: `transforms.py` `normalize` 플래그 → `global_scale()` 로 k-space·GRAPPA·target 동일 ÷s, `scale` 반환.
- **readout crop**: `readout_crop` 플래그 → `readout_crop()`(높이만 640→384, 폭 native). 실측 recon corr 1.0.

### ⬜ 미배선 (network·학습 측 = 동료 담당)
- **GRAPPA prior + residual**: `varnet.py` `VarNet.forward` 보조 채널 경로 + residual(out=grappa+f).
- **감도맵 재사용**: sens 추정을 per-view forward에서 분리(mask=그대로/flip=flip).
- **소비부 확장**: train_part.py/test_part.py 배치 언패킹 확장 + 출력 `×s` 역정규화.
- **증강 스케줄 배선**: train.py `--aug-*` 인자 + train_part.py `set_epoch`.
- **acc-conditioned GRAPPA 가중**(후보): mask frac≤0.25→acc8/else acc4로 conditioning → 융합 가중 학습. [`decisions.md`](decisions.md).

### ⏳ 보류 (외부조건)
- **forward(추론) GRAPPA 로딩**: leaderboard 데이터 구조 미공개 → 공개 후 배선. 제출 전 필수 확인.
