# 확정 모델 (Confirmed Architecture) — FastMRI 2026

> ⚠️ **SUPERSEDED (2026-07-19): 백본이 SDUM → PromptMR+로 교체됨. 현재 SPEC = [`promptmr_plus.md`](promptmr_plus.md).**
> 아래 내용은 SDUM 시기의 확정 파이프라인 기록으로 보존한다(입력 2분기·readout crop·GRAPPA·감도맵 등
> **모델 무관 요소는 여전히 유효**). 백본·손실·옵티마이저 관련 서술은 `promptmr_plus.md`가 우선한다.

> 상태: **확정 (2026-07-04)** · 이 문서가 파이프라인 구조의 **단일 기준(source of truth)**.
> 세부 근거는 [`model_strategy.md`](model_strategy.md) · [`augmentation.md`](augmentation.md) · [`preprocessing.md`](preprocessing.md) · [`todo.md`](todo.md).
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
> ⚠️ **STALE (2026-07-11): 어댑터 self-normalize로 사실상 대체됨 — 켜지 말 것.** 이 SPEC은 SDUM 어댑터가 self-normalize를 넣기 전에 작성됨. 현재 `adapter.py:40-56`이 **per-sample ÷std → ×std**(scale-equivariant)를 이미 수행한다. 앞단에서 전역 `÷s`를 해도 어댑터가 std로 재정규화하며 씻어내 **loss·SSIM에 ≈no-op**(캐스케이드 입력이 수학적으로 동일). 즉 이 절이 원했던 "DC·손실 스케일 안정"은 **이미 어댑터가 제공**한다. `--normalize`는 train.py에 인자 미등록이고, `--readout-crop` help도 "함께 켜지 말 것(어댑터 self-normalize)"이라 명시. 켜면 (1) 6-튜플 언패킹 크래시, (2) 실익 ≈0, (3) 추론경로 ×s 미배선 시 스케일 버그 리스크만 남는다. 배경: `training_anal/`(run 분석) 및 3-1 논의.

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

### 5. Main network
- **E2E-VarNet cascade + DC.** DC는 **원 측정 k-space 기준**(readout-crop 후에도 정확히 성립, 실측 0.00e+00).
- **GRAPPA = 영상 도메인 prior 채널 + residual**: `out = grappa + f([image_input, grappa, …])`.
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
- **정규화 ÷s**: `transforms.py` `normalize` 플래그 → `global_scale()` 로 k-space·GRAPPA·target 동일 ÷s, `scale` 반환. ⚠️ **어댑터 self-normalize(`adapter.py:40-56`, per-sample ÷std)로 대체 → 켜지 말 것**(§2 STALE 주석 참조).
- **readout crop**: `readout_crop` 플래그 → `readout_crop()`(높이만 640→384, 폭 native). 실측 recon corr 1.0.

### ⬜ 미배선 (network·학습 측 = 동료 담당)
- **GRAPPA prior + residual**: `varnet.py` `VarNet.forward` 보조 채널 경로 + residual(out=grappa+f).
- **감도맵 재사용**: sens 추정을 per-view forward에서 분리(mask=그대로/flip=flip).
- **소비부 확장**: train_part.py/test_part.py 배치 언패킹 확장 + 출력 `×s` 역정규화.
- ~~**acc-conditioned GRAPPA 가중**(후보): mask frac≤0.25→acc8/else acc4로 conditioning → 융합 가중 학습.~~ → **2026-08-01 기각**: MoE(acc4/acc8 전문가 분리)가 같은 1비트 조건을 100% 파라미터 분리로 쓰는 조건화의 상한이라, conditioning 몇백 파라미터가 이를 넘을 구조적 경로가 없다. 근거: [`next_levers.md`](next_levers.md) C5.

### ⏳ 보류 (외부조건)
- **forward(추론) GRAPPA 로딩**: leaderboard 데이터 구조 미공개 → 공개 후 배선. 제출 전 필수 확인.
