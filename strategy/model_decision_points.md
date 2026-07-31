# 모델 결정 사안 (Model Decision Points) — SDUM 백본 이식

> ⚠️ **SUPERSEDED (2026-07-19): 백본이 SDUM에서 PromptMR+로 교체됨. 현재 SPEC = [`promptmr_plus.md`](promptmr_plus.md).**
> 이 문서는 **SDUM 이식 시기의 결정 노브 기록**으로 보존한다. 아래 SDUM 5원리(Restormer·per-cascade CSME·
> SWDC·UC·progressive expansion) 논의는 더 이상 현행 아님. 다만 **모델 무관 결정**(A11 SVD coil compression,
> B9 손실 방향, 8GB·FP32·batch1 제약)은 PromptMR+에도 유효하다. 현재 백본의 확정 구조·레버는 `promptmr_plus.md`.

> 상태: **초안 (2026-07-06)** · 검토 대기. 아직 SPEC 아님(확정은 [`confirmed_model.md`](confirmed_model.md)).
> 배경: 기존 **E2E-VarNet 백본을 완전히 제거하고 SDUM으로 교체**하기로 결정. 이 문서는 SDUM을 우리 대회 환경(knee 전용·8GB·FP32)에 이식할 때 **우리가 직접 정해야 하는 결정 노브**를 정리한다.
> 논문: SDUM — *A Scalable Deep Unrolled Model for Universal MRI Reconstruction* (arXiv:2512.17137, JHU+NVIDIA).
> 공식 코드: <https://github.com/NVIDIA-Medtech/NV-Raw2insights-MRI> (Apache 2.0, PyTorch). 가중치: HuggingFace (Small 230M / Base 760M / Large 1.4B).

---

## 0. 전제 — "논문에 충실"의 의미 & 규모 현실

논문 그대로는 우리 환경에 **절대 안 들어간다.**

| 항목 | 논문(SDUM) | 우리 환경 |
|---|---|---|
| 파라미터 | T18 = **759M** (최소 변형도 230M) | 1080 8GB에 맞춘 수 M~수십 M |
| 학습 인프라 | **64× H100 (8노드)** | GTX **1080 8GB 1장** |
| 정밀도 | BF16 + gradient checkpointing | 1080 = **FP32 전용**(Pascal, BF16 이득 없음) |
| Restormer 폭 | {256, 512} 채널 | 이대로면 단일 cascade도 OOM |
| 추론 | H100 256²에 1.0s/6GB | 1080 8GB · 추론시간 채점 |

→ **"논문에 충실" = 759M 규모 재현이 아니라, SDUM의 5개 설계 원리(Restormer 백본 · per-cascade CSME · SWDC · UC · progressive expansion)를 8GB FP32에 맞게 축소 이식.** SDUM은 애초에 "scalable"이 셀링포인트라(논문 T1=42M도 동작) **작게 줄여도 구조가 성립**한다.

⚠️ **가중치 사용 금지**: 대회 규정이 "사전학습 가중치·외부 데이터 금지" + NVIDIA 가중치는 cardiac라 knee와 무관 → **코드(아키텍처)만 가져와 우리 knee 데이터로 처음부터 학습**하고, repo의 HuggingFace 자동 다운로드는 끈다.

### 목표 워크플로 — 8GB에 "간당간당" 붙이기
학습 중 OOM을 런타임에 우아하게 회복하는 건 불가능(backward 중 터지면 프로세스 오염). 대신:
- **탐침(probe) 스윕**: 각 하이퍼파라미터 조합으로 **1~2 iteration만** 돌려 OOM 여부 판별(OOM은 보통 첫 batch에서 바로 남) → **살아남는 가장 큰 조합** 선택 → 본 학습.
- 또는 `torch.cuda.OutOfMemoryError`를 catch → `empty_cache()` → **한 단계 작은 config로 재시작**하는 오토튜너.
- **스윕 대상 = 아래 그룹 A의 VRAM 레버들.**

---

## 일단 최현수가 결정한 것 (2026-07-06, 진행 중)

> 아래는 위 선택지 중 **최현수가 직접 고른 것**만 기록(어시스턴트의 구현 조언과 구분). 전부 **변경 가능**.

- **Gradient checkpointing: 채택 (ON).** VRAM 대폭 절감 우선, 학습 속도 저하는 감수. 논문도 사용(BF16+checkpointing) → repo 코드에서 정확한 배선 방식 확인 후, 그대로 못 쓰면 cascade 단위로 `torch.utils.checkpoint` 직접 적용. (레버 A1/C8)
- **A3 CSME: 옵션 (c) — 몇 cascade마다 1회 감도맵 재추정.** per-cascade(무겁고 품질↑)와 1회-공유(가벼움)의 절충. 나중에 (a)/(b)로 변경 가능.
- **B4 SWDC: 유지.** "유지 권장"으로 표시된 항목 유지 방침.
- **B5 UC: 제거(제외) 확정.** MoE로도 바꾸지 않음. UC 조건화 메커니즘을 아예 빼고 간다. (단일 modality·단일 trajectory라 UC 이득이 작고, MoE는 저데이터 희석 위험이라 둘 다 제외.)
- **B10 GRAPPA prior + residual: 채택 확정.** 백본에 GRAPPA 영상 prior를 얹는다(다른 팀원이 진행 중).
- **배치 사이즈 = 1 확정.** (acc4 368 / acc8 372 폭 상이 + VarNet류 메모리 큼.)
- **입력 해상도: 유지(안 줄임) 확정.** readout crop 640→384(무손실)는 유지하되 그 이상 추가 축소는 안 함.
- **A11 coil compression: 일단 보류(당장 미적용).** 메모리는 checkpointing + 작은 폭/T로 확보 우선. 필요 시 추후 재검토.

**아직 미결정(스윕/추후)**: A1 cascade T · A2 Restormer 폭 (→ 둘 다 8GB 탐침 스윕으로 결정) · B9 손실(초기 SSIM, bbox 가중은 추후).

---

## 그룹 A — VRAM 지배 레버 (스윕 대상, 8GB에 붙이는 핵심 노브)

### A1. cascade 수 T (unrolled 깊이)
- **뭔가**: unrolled 모델은 "① data consistency(측정값에 맞추기) → ② 신경망으로 영상 정제"를 **T번 반복**한다. 이 반복 1회 = cascade 1개. VarNet의 cascade와 동일 개념.
- **효과**: T↑ → 품질↑ (논문: PSNR ∝ log(파라미터), T18까지 계속 상승). 대신 **학습 메모리 ≈ T에 비례**(backprop 위해 T개 cascade 중간값 보관), 추론시간도 T에 비례.
- **논문**: 18(=759M). **우리**: 4~8부터. **가장 직접적인 메모리·시간·품질 트레이드오프 → 스윕 1순위.**
- 💡 **gradient checkpointing**을 켜면 T개를 다 저장하는 대신 재계산 → 메모리를 T에 거의 무관하게 낮춤(학습 속도↓ 대가). 논문도 사용.

### A2. Restormer 백본 폭/깊이
- **뭔가**: 각 cascade 안의 "영상 정제 신경망". SDUM은 U-Net 대신 **Restormer**(MDTA 어텐션 + GDFN)를 쓴다. **폭(채널 수)**·**깊이(블록 수)**가 크기를 결정.
- **효과**: **VRAM을 가장 크게 잡아먹는 부분.** 활성값 메모리가 채널 수의 제곱 수준으로 증가. 논문 {256,512}는 우리 기준 단일 cascade도 못 올림.
- **논문**: 2-level, 채널 {256,512}, heads {1,2}, block depth {3,6}, GDFN 확장 3. **우리**: 채널 {32,64}급 이하로 **대폭 축소** → T와 함께 2대 레버.
- 💡 논문 실측: **폭보다 깊이(T)가 효율적**(width는 42M에서 포화). → **폭은 작게, cascade를 여러 개**가 유리할 수 있음(스윕에서 확인).

### A3. CSME (coil sensitivity map estimator) — per-cascade vs 공유, 폭
- **감도맵(CSM)이란**: MRI는 여러 수신 코일(knee=15개)로 동시에 측정. 각 코일은 공간의 특정 영역을 더 민감하게 잡음. **감도맵 = "각 코일이 어디를 얼마나 민감하게 보는지"의 지도** → 15코일 영상 합성에 필수.
- **CSME**: 감도맵을 신경망(complex U-Net)으로 추정하는 모듈. VarNet에도 `SensitivityModel`로 하나 있음.
- **SDUM 차별점**: 감도맵을 **cascade마다 재추정**(per-cascade CSME) → 움직임·잡음 보정. 논문 ablation **+0.51 dB**.
- **효과**: per-cascade면 U-Net을 T개 → **파라미터·메모리 T배**. 8GB엔 부담.
- **우리 결정**: (a) **맨 앞 1회 추정 후 공유**(VarNet 방식, 가벼움) / (b) per-cascade(무겁지만 품질↑) / (c) 몇 cascade마다 1회. + CSME 폭 축소. → 메모리 스윕 대상.

### A11. SVD coil compression (15 → N 가상코일)
- **뭔가**: 15개 물리 코일을 **정보 손실 거의 없이 N개(예: 8) 가상 코일로 압축**(SVD). 이미 repo에 플래그로 구현됨.
- **효과**: 코일 축 15→N → 코일을 다루는 모든 연산(감도맵·DC) 메모리·연산 **약 절반**. 화질 중립(속도/VRAM 레버).
- **우리 결정**: 켤지 + N(8/10/12) 스윕. **거의 공짜 메모리 절감**이라 유력. SDUM 논문도 "coil-reduced images에서 동작"이라 호환.

### (그룹 A 밖이지만 스윕 필수)
- **batch size**: 우리는 사실상 **1 고정** 권장(VarNet류 메모리 큼). acc4(368)/acc8(372) 폭이 달라 batch>1은 별도 pad 필요.
- **입력 해상도**: readout crop 640→384로 이미 ~40% 절감(무손실). 추가 축소는 화질 손해라 지양.
- **gradient checkpointing**: on/off 자체가 큰 메모리 레버(A1 참조).

---

## 그룹 B — 구조/품질 결정 (메모리 영향 작음, 화질·순위축이 목적)

### B4. SWDC (sampling-aware weighted data consistency)
- **data consistency(DC)란**: unrolled의 "① 측정값에 맞추기" 단계. "복원 영상을 다시 k-space로 바꿨을 때 **측정한 라인은 측정값과 일치**해야 한다"를 강제. VarNet은 **스칼라 가중치 하나**(`dc_weight`)로 조절.
- **SWDC**: 그 스칼라를 **k-space 위치마다 다른 weight map**으로 교체(패턴별 학습). 논문 **+0.43 dB**.
- **효과**: 가볍고 효과 좋음. **우리 결정**: 유지 권장. 단 논문은 다양한 패턴(uniform/gaussian/radial)이라 패턴별 map이 필요했으나, **우리는 acc4/acc8 uniform Cartesian 2종뿐** → weight map 2개(또는 acc 조건부 1개)로 단순화.

### B5. UC (universal conditioning)
- **뭔가**: "몇 번째 cascade인지 + 어떤 프로토콜인지(mask/가속/모달리티)"를 **숫자 임베딩으로 만들어 신경망 매 블록에 주입**(FiLM/AdaLN 계열). 한 모델이 상황을 알고 행동을 바꾸게 함.
- **논문에서 중요했던 이유**: cardiac 12종 contrast + 여러 trajectory + 4~24× 가속을 **단일 모델**로 처리하는 게 목표라 "상황 인지"가 핵심.
- **우리 맥락**: **단일 modality(knee) · 단일 trajectory(Cartesian) · 가속 2종(acc4/acc8)뿐** → UC 대부분 목적 소멸, 실질 조건은 **{cascade index, acc4/acc8}**만 남음.
- **우리 결정**: (a) cascade index만 최소 유지 / (b) acc4/acc8까지 조건화([`candidates.md`](candidates.md) acc-conditioning과 연결) / (c) 생략. 메모리 거의 안 듦 → 품질 관점 결정.

### B9. Loss (손실 함수)
- **논문**: SSIM loss만.
- **우리**: 순위 결정축이 **SSIM_bbox(병변 박스)** → `L = (1−SSIM_full) + λ·(1−SSIM_bbox)`처럼 **병변 영역 가중** 추가가 전략 방향([`model_strategy.md`](model_strategy.md) §2.1, [`candidates.md`](candidates.md)). SDUM 백본 위에 **손실만 우리 대회용으로** 교체.

### B10. GRAPPA prior + residual
- **뭔가**: 대회가 **GRAPPA로 미리 복원한 영상**(alias 상당히 제거된 초기해)을 제공. 이를 **보조 채널 + residual**(`출력 = GRAPPA + f(...)`)로 넣으면 "밑그림 다시 그리기" 대신 "틀린 부분만 고치기" → 같은 크기로 품질↑.
- **논문엔 없음**: SDUM은 cardiac라 GRAPPA 미제공 → 이 레버 부재. **우리만의 확정 자산**([`confirmed_model.md`](confirmed_model.md) §5, [`model_strategy.md`](model_strategy.md) §1).
- **우리 결정**: SDUM 백본에 **GRAPPA prior를 얹을지**(논문 밖 확장). confirmed_model이 강권하는 자산이라 유력하나, SDUM 구조 접붙이기 설계 필요.

### B6. 인접 슬라이스 입력
- **논문**: 동적 cardiac/3D용으로 인접 프레임 함께 입력.
- **우리**: 현재 **단일 슬라이스 고정**(코드 미배선). 단 🔁 **2026-08-01 기각 근거 반증** — 옛 사유 "bbox blur → SSIM_bbox 악화"는 대조군 없는 파일 2개 표본이었고, 재측정에서 병변 상관이 대조군보다 높았다([`model_strategy.md`](model_strategy.md) §3). **a=1 한정 재개 후보.**
- (단 "인접 슬라이스로 감도맵만 배치추정"하는 메모리 최적화는 별개로 채택 가능 — [`model_strategy.md`](model_strategy.md) §3.2.)

---

## 그룹 C — 학습 커리큘럼/설정

### C7. Progressive cascade expansion
- **뭔가**: 처음부터 T=18 학습은 불안정 → **6→10→18로 단계적으로 cascade를 늘리며**(양끝 고정, 내부 복제, warm-start) 학습하는 커리큘럼.
- **우리**: T를 작게(4~8) 쓸 거라 **과할 수 있음** → 채택 여부 결정. 작은 T면 처음부터 학습해도 무방.

### C8. 옵티마이저 / 정밀도
- **논문**: **Muon** 옵티마이저 + **BF16** 혼합정밀 + gradient checkpointing, LR 2.4e-4 cosine, wd 1e-3, SSIM loss, 스테이지당 ~20k step.
- **우리**: 1080 = **FP32 전용**(BF16 이득 없음), Muon은 실험적 → **AdamW FP32**(repo baseline과 동일)가 안전. **gradient checkpointing은 채택**(A1 메모리 레버).

### C12. 출력/전처리 (결정이라기보단 배선)
- repo 데이터 인프라(readout crop 640→384, 전역 정규화 ÷s/×s, 384² center crop)는 이미 구현됨. SDUM 백본 입출력을 **통합 계약** `forward(masked_kspace, mask) → (B,384,384)`에 맞추는 문제.
- ⚠️ 정규화·GRAPPA·bbox 플래그를 켜면 dataloader 반환 튜플이 6→7~9개로 늘어 **소비부(train_part/test_part) 언패킹이 깨짐** → 확장 필요([`implementation.md`](implementation.md) §5b).

---

## 요약 — 스윕할 것 vs 정할 것

| 분류 | 항목 | 성격 |
|---|---|---|
| **VRAM 레버 (스윕)** | A1 cascade T · A2 Restormer 폭 · A3 CSME 공유/폭 · A11 coil compression N (+ batch=1, gradient checkpointing) | 8GB에 붙이는 노브 |
| **구조/품질 (한 번 결정)** | B4 SWDC(유지·단순화) · B5 UC(축소) · B9 bbox 손실(추가) · B10 GRAPPA(얹을지) · B6 인접슬라이스(끔) | 화질·순위축 |
| **학습 설정** | C7 progressive(아마 생략) · C8 AdamW+FP32+checkpointing · C12 배선 | 학습 절차 |

---

## 다음 단계 (제안)
1. 공식 repo clone → 모델 코드(Restormer/CSME/SWDC/UC) 정독 → 위 레버가 **코드 어느 config·클래스인지** 매핑.
2. Small(230M) config를 축소 출발점으로 **8GB 축소 config 초안** + **OOM 탐침 스윕 스크립트** 작성.
3. 각 결정 노브를 하나씩 확정 → [`confirmed_model.md`](confirmed_model.md) 갱신.
