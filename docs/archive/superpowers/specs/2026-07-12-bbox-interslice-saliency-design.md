# 설계: bbox · 인접슬라이스 · 자가-saliency 3축 강화 로드맵

작성일 2026-07-12 · 대상 대회 FastMRI SNU 2026 (knee, acc4/acc8) · 목표 우승

> 이 문서는 브레인스토밍 산출물(설계/로드맵)이다. 확정 SPEC은 아니며, 각 Stage는 승인 후
> 개별 spec → 구현 plan으로 분화한다. 근거는 `strategy/`(model_strategy.md·candidates.md·rejected.md)와
> 코드 탐색(2026-07-12)에 있다.

---

## 1. 문제 정의 (왜 이 작업인가)

현재 파이프라인(`--model sdum` = cascaded Restormer, 2D single-slice, `SSIMLoss`)은 전체 이미지·k-space
복원과 고주파 보존에는 최선을 다하지만, **점수를 실제로 가르는 두 신호를 전혀 쓰지 못한다.**

1. **bbox(병변 위치·크기·종류)** — 점수의 50%인 `SSIM_bbox`를 직접 겨냥하는 신호. `return_bbox` 배관만
   구현돼 있고 **손실에 미연결**. 병변 유형(label)은 데이터에 있으나 채점·학습 어디에도 미사용.
2. **인접 슬라이스 상관** — 한 파일 = 한 무릎 볼륨이라 연속 슬라이스가 같은 해부구조인데, 현재
   `num_frames=1`(2D)로 **볼륨 정보 미활용**. (인접 배치 SME는 전략상 채택했으나 SDUM엔 미배선.)

## 2. 반드시 인지할 두 제약 (탐색으로 확정)

### 제약 A — bbox는 추론 시 모델 입력이 아니다
`reconstruct.py` / `utils/learning/test_part.py`는 **k-space+mask만** 입력받는다. 주최측은 숨긴
annotation으로 채점하고, leaderboard의 annotation은 **자가채점용**일 뿐이다.
→ **bbox 활용 = 학습 시점 신호**(손실 가중·샘플링·보조 태스크). 추론에서 활용하려면 모델이
병변 위치를 **스스로 예측**해야 한다(Stage C).

### 제약 B — 순진한 적용은 SSIM_bbox를 깎는다
팀 실측(`model_strategy.md §3`): 인접 슬라이스 **전역** Pearson은 높지만(acc4 gap1 **0.87**),
**bbox 내부(병변) 상관은 0.08~0.59로 낮다**. 그래서 naive 2.5D 입력 스태킹은 이미 **기각**됨
(병변 blur → SSIM_bbox 악화, `rejected.md`).
→ **인접 정보는 SSIM_full(전역 구조)에만 쓰고 병변부에선 꺼지도록, bbox 신호는 병변부에만 집중**시켜
두 메커니즘을 **분리(decouple)**하는 것이 이 설계의 핵심 원리.

### 채점 (근거: `recon_eval.py`, `utils/common/metrics.py`)
`총점 = 0.5·SSIM_full + 0.5·SSIM_bbox + 추론시간`.
- `SSIM_bbox`: 유효 박스(w,h > win_size=7)들의 **단순 평균**(acc4·acc8 각각 계산 후 평균).
- full·bbox 동일 5:5 → **두 축을 동시에** 밀어야 하고, 추론시간·모델 크기는 tie-break.

## 3. 현재 상태 (코드 근거)

| 요소 | 현황 | 파일 |
|---|---|---|
| 백본 | SDUM cascaded Restormer(image-domain, T=1 2D), diffusion off | `utils/model/sdum/{adapter,latent_recon,restormer,config}.py` |
| 손실 | `SSIMLoss`(1−SSIM), `--fg-loss`로 foreground 마스크 정합 | `utils/common/loss_function.py` |
| bbox 배관 | `return_bbox`·`_parse_bbox`·`_bbox_collate`·flip-sync **구현됨**, 손실 **미연결** | `load_data.py`, `transforms.py` |
| 인접슬라이스 | 미사용(T=1). VolumeSampler·temporal 코드 존재하나 no-op | `load_data.py`(per-slice) |
| bbox label | 데이터에 존재, **완전 미사용** | image h5 `attrs['annotations']` |

## 4. 공통 원칙 (repo 컨벤션)

- **기본 off 플래그**로 게이트 → 끄면 기존 학습과 **비트 동일**(`--fg-loss` 선례가 8자리 일치로 검증).
- **직전 best 위에 하나만 추가하는 incremental A/B**. 판정 = `recon_eval.py` leaderboard 자가채점,
  **하드 게이트 = SSIM_bbox 비퇴행** + (bbox 강화 단계는) SSIM_full 비퇴행.
- seed 고정 재현성 유지. GPU 작업은 `shell_scripting/shellN.sh` 큐 슬롯으로만(CLAUDE.md §6).
- **메모리 예산**: GTX 1080 8GB · FP32 · batch=1 · gradient checkpointing on → 신규 모듈은
  **경량·저해상도·인접은 값싼 단계에만**.

---

## 5. 추천 로드맵

```
Stage 0 (진단) → Stage A (bbox 손실+오버샘플) → Stage S (인접슬라이스 로더 인프라)
                                              → Stage C (자가-saliency refinement)
                                              → Stage D (인접 융합/일관성)
```

A가 먼저인 이유: 50% 축 직접 겨냥·저위험·배관 존재, 그리고 **C의 saliency 지도학습 신호를 여는 전제**.
S(로더 인프라)는 D의 전제이자 배치-SME(§3.2)도 여는 공용 인프라.

### Stage 0 — 진단 (구현 아님, 먼저)
- val/leaderboard 자가채점을 **병변 유형별·acc별로 분해**해 어느 클래스·어느 가속(acc8 유력)이
  SSIM_bbox를 끌어내리는지 측정. `recon_eval.py`의 `ssim_bbox` 루프를 label별 집계하는 임시 스크립트.
- 산출물: λ(Stage A)·saliency 집중 대상(Stage C)·유형 가중치의 근거. 채점 코드 미수정.

### Stage A — bbox 가중 손실 + 병변 슬라이스 오버샘플링 (저위험 기반)
**대상**: `train.py`(인자), `load_data.py`(sampler), `train_part.py`(언패킹+손실),
`loss_function.py`(bbox 항). transforms는 이미 bbox emit(`_parse_bbox`, tuple 맨 끝).

- **bbox 가중 손실**: `SSIMLoss`(win=7, k1/k2 채점 동일)의 conv 기계를 재사용해 각 박스 crop
  `output[y:y+h, x:x+w]`·`target[...]`에서 `1−SSIM`을 계산, 박스 평균. 박스 없는 슬라이스 → 항=0.
  총손실 `L = L_full + λ·L_bbox`(`L_full`은 현행 `--fg-loss` 경로 그대로). 출력·타깃·maximum은 현재
  손실 호출에 흐르는 것과 동일 공간 사용. λ는 Stage 0 근거로 스윕.
- **병변 오버샘플링**: `create_data_loaders`에서 `--lesion-oversample` on일 때 `WeightedRandomSampler`
  (shuffle 대체). 병변 슬라이스 ×k, 그외 1. 병변 슬라이스는 dataset init 시 각 파일 `attrs['annotations']`
  1회 읽어 판정. `generator`(seed)로 재현성. 픽셀 미변형 → **hallucination 위험 0**.
- **재사용**: `return_bbox`/`_bbox_collate`/`_parse_bbox`/flip-sync(`MRIAugmentor._flip_lr`) 완비 →
  `--return-bbox` 켜고 손실 소비부만 배선. tuple 언패킹 확장은 `--fg-loss` 패턴 그대로
  (순서: `...,[scale],[grappa],[fg_mask],[bbox]`).
- **게이트**: SSIM_bbox↑ & SSIM_full 비퇴행. 끄면 비트 동일.

### Stage S — 인접 슬라이스 로더 인프라 (공용 전제)
- `SliceData`가 옵션(`--neighbor-radius a`)으로 center + ±a 이웃 슬라이스의 kspace(및 일관성용 target)를
  같은 파일 인접 인덱스에서 로드(경계 clamp). 한 파일=한 볼륨이라 인접=같은 무릎(탐색 확정;
  ⚠️ acc4_N과 acc8_N은 동일 피험자 보장 없음 → 볼륨 내부만 인접 취급).
- 이 인프라가 **Stage D**와 배치-SME(§3.2)를 함께 연다. 기본 off = 기존 6-tuple 유지.

### Stage C — 자가예측 saliency 집중 refinement (백본 변경, 추론서 bbox 간접 사용)
**대상**: `utils/model/sdum/{restormer,latent_recon,adapter,config}.py`, `train_part.py`(손실 항).

- **saliency 헤드**: 마지막 cascade 직전 feature/recon에서 1채널 saliency σ∈[0,1](384²) 산출 경량 conv.
- **지도 신호 (열린 A/B, §6-1)**: 기본 권장 = **recon 오차맵 |out−target|**(전 슬라이스 무료·조밀)
  + **bbox 마스크 강조**(Stage A 배관). "고오차 영역 예측"이 병변+난이도 영역을 포괄하고 미주석
  슬라이스에도 신호가 있어 일반화에 유리.
- **집중 refinement**: `out = base + σ·refine(base)`(refine=경량 residual 블록), 이후 **SWDC 유지**.
  → 병변/고오차 예측 영역에 용량 집중. **추론 시 σ 자가예측(진짜 bbox 불필요, 제약 A 우회)**, 비용 소량.
- **날조 방지**: residual+DC 보존, saliency 정규화, 게이트 = bbox 비퇴행 + box 오차맵 육안점검.

### Stage D — 인접 슬라이스 "제대로" (백본+로더 변경, 메모리 인지)
제약 B 준수: 인접은 전역/저주파·감도추정에만, 병변부 차단. 8GB 예산상 값싼 형태부터.
- **D1 (최저위험·추론비용 0): 인접 일관성 정규화.** center 1회 forward만 하고, 이웃 **target**의 z-방향
  기울기와 recon의 z-기울기를 **부드러운 영역(비-bbox·저gradient)에서만** 일치. 추가 forward 없음
  → 메모리·추론 무해. SSIM_full 일관성↑, 병변 blur 없음.
- **D2 (중간): 감도추정/첫 cascade에만 게이트 cross-slice 융합.** 이웃 저해상 정보를 CSME 또는 첫
  cascade에 학습된 게이트로 residual 주입(불일치=병변부 자동 차단). 이웃은 값싼 단계에만.
- **D3 (스트레치): 저해상도 latent cross-slice 어텐션.** Restormer 최저해상도 레벨에서만 볼륨 어텐션
  (구조적·저주파, 병변 고주파 미표현) → blur 없이 볼륨 일관성.
- **게이트**: SSIM_full↑ & SSIM_bbox **하드 비퇴행** & 추론시간 예산 내. D1→D2→D3 순.

---

## 6. 핵심 결정 (구현 전 확정)

1. **Stage C saliency 지도 신호**: 오차맵 primary(권장) vs bbox마스크 primary vs 결합.
2. **로드맵 폭**: 3 Stage를 하나의 통합 모델로 vs 순차 A/B(권장 — 재현성·게이트에 유리).
3. **Stage D 착수 형태**: D1(안전)부터 vs D2 동시.

## 7. 재사용할 기존 자산

- `SSIMLoss`(loss_function.py) — bbox 항의 conv 기계·win/k 파라미터 그대로.
- `--fg-loss` 경로 — 손실 항 추가·tuple 언패킹·validate 정합의 **정확한 템플릿**.
- `return_bbox`·`_bbox_collate`·`_parse_bbox`·flip-sync — bbox 배관 완비.
- `foreground_mask`(metrics.py) — smooth/비-병변 영역 마스킹(D1)·fg 손실 재사용.
- SDUM SWDC(`dc_weight_map`)·CSME — DC 보존·감도추정 융합 지점.
- `recon_eval.py`(고정) — 모든 Stage의 A/B 채점기(3축).

## 8. 검증 (end-to-end)

- **끄기 동등성**: 각 플래그 off로 짧은 학습 → 기존 run과 비트 동일(fg-loss 검증 방식).
- **손실 정합**: bbox 손실이 `metrics.ssim_bbox`와 동일 window/좌표계로 계산되는지 `tests/` 단위 테스트.
- **A/B 채점**: 각 Stage 후 `python recon_eval.py -n <net> -p /root/Data/leaderboard`로 3축 산출,
  하드 게이트(SSIM_bbox 비퇴행) 통과 시에만 채택. off-팔은 현행 run 재활용(worklog A/B 방식).
- **재현성**: seed 고정 재학습 동일 결과 확인.

## 9. 후속

승인 후: 각 Stage를 개별 spec으로 분화 → writing-plans로 구현 계획화. **Stage 0·A부터 착수**
(최저위험·기반). C·D는 A 채택 후 백본 변경으로 진행. 관련 전략 문서(`candidates.md` bbox 가중 손실,
`model_strategy.md §2·§3`)와 상호 링크.
