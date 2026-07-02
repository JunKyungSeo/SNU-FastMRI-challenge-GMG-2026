# 모델·메모리·구조 전략 (non-augmentation) — FastMRI 2026

> augmentation이 아닌, 논문 재검토에서 나온 **아키텍처·입력 설계·메모리·효율 전략**을 모으는 문서.
> (augmentation 기법은 [`augmentation.md`](augmentation.md)에 별도 정리.)
> 대상 제약: knee 전용 · SSIM_full + **SSIM_bbox(순위축)** + 추론시간 채점 · GTX 1080 8GB(FP32) · external/pretrained 금지.

---

## 0. TL;DR — 북극성 (3채점축 ↔ 모델 레버 매핑)

이 문서의 모든 결정은 **SSIM_full · SSIM_bbox(순위축) · 추론시간**을 **8GB·FP32**에서 동시에 미는 것을 목표로 한다. (별점 = 검증 전 상대 가이드)

| 모델 레버 | SSIM_full | SSIM_bbox | 추론시간 | 8GB | 근거 |
|---|---|---|---|---|---|
| **경량 E2E-VarNet 베이스**(cascade 6~8·채널 축소·FP32) | ⭐⭐⭐ | — | ⭐⭐⭐ | ⭐⭐⭐ | 백본 |
| **GRAPPA 초기화**(채널 주입 권장 / x0대체는 주의) | ⭐⭐⭐ | ⚠ neutral~risk | ⭐⭐⭐ | ⭐⭐ | §1 |
| **bbox 가중 손실**(MOST 프레이밍) | — | ⭐⭐⭐ | — | ⭐⭐⭐ | §2, REVIEW #4 |
| **배치 SME**(인접슬라이스, 채택) | — | — | △ | ⭐⭐⭐ | §3.2, PromptMR+ |
| **sens 효율화 택1**(UEPS 제거 ↔ PromptMR+ 개선) | ⭐⭐ | — | ⭐⭐ | ⭐⭐⭐ | §4.1 |
| **FLAT 중간감독**(cascade 다이어트) | ⭐⭐ | — | ⭐⭐⭐ | ⭐⭐ | §4.2, REVIEW #1 |

**한 줄 전략**: 경량 VarNet에 **① GRAPPA 초기화(§1) + ② bbox 가중 손실(§2)**을 먼저 얹어 3축 동시 상승을 노리고, 남는 메모리/시간은 **③ 배치 SME(§3.2) + sens 효율화 택1(§4.1) + FLAT cascade 다이어트(§4.2)**로 쥐어짠다. 회전·스케일 등 비물리 변형은 [`augmentation.md`](augmentation.md) Tier 3에서 이미 금지.

> ⚠️ **선결 과제(파이프라인 갭, §2 bbox 레버 공통)**: 현재 `utils/data/transforms.py`의 `DataTransform`은 `attrs`에서 `maximum`만 반환하고 **annotation을 버린다**(확인함). bbox 가중 손실을 쓰려면 **transform이 `json.loads(attrs.get("annotations","{}"))`를 파싱**해 slice별 bbox를 emit하고 **가변 길이 bbox용 custom `collate_fn`**(또는 batch_size=1)이 필요. (bbox 좌표는 384², 모델 최종 `center_crop(384,384)` 출력과 좌표 정합됨 → 출력단 손실은 좌표 일치.)

---

## 1. GRAPPA 영상 초기화 — 유력 지렛대 (SSIM_full + 추론시간)

> REVIEW.md 갱신권고 §2에서 "논문 6편 어디도 안 다루는 최대 지렛대"로 지목. 단 아래 **주입 방식의 물리적 정합**을 지켜야 한다.

**배경**: 대회는 acc4/acc8 각각의 **GRAPPA 복원 영상(`image_grappa`, 384×384, magnitude float32)** 을 제공한다. 이미 alias가 상당히 줄어든 초기해라, VarNet이 zero-filled(`Aᴴy`)보다 좋은 출발점을 얻으면 **같은 cascade로 더 좋은 최종해(SSIM↑)** 또는 **더 적은 cascade로 동품질(추론시간↓)**.

**주입 방식**:
- ✅ **(b) 권장 — 추가 입력 채널**: GRAPPA를 네트워크 이미지 그리드로 **resample 후 regularizer U-Net에 real-valued prior 채널로 concat**. magnitude prior가 well-defined하고 **DC(data consistency)는 원 언더샘플 k-space 기준으로 유지** → GRAPPA 잔여 아티팩트를 물리로 교정.
- ⚠️ **(a) 주의(비자명, drop-in 아님) — x0 대체**: GRAPPA는 **magnitude-only 384²**, VarNet의 x0는 **complex multi-coil k-space(~640×368)**. 그대로 대체하려면 **위상(phase) 합성 + coil 확장(sens) + 해상도 재샘플**이 필요 → ill-posed·**fabrication(hallucination) 위험**. 쓰려면 image 도메인 prior로만, 위상 날조 금지.

**채점축 효과**: SSIM_full 직접 상승 / 추론시간(좋은 초기해 → cascade 6→4로도 품질 유지, §4.2 FLAT와 시너지). **SSIM_bbox는 neutral-to-risk**: GRAPPA 자체의 잔여 alias/아티팩트가 **저SNR 병변 box에 새어들 수 있음** → **§2·측정 프로토콜의 bbox 비퇴행 게이트 필수**, box 내부 아티팩트 육안 점검.

**제약·구현**:
- GRAPPA는 **제공 입력**이라 external/pretrained 규칙 무관, 물리적 복원(합성 아님).
- acc4/acc8 GRAPPA 품질이 달라 **가속률별 스케일/정규화** 확인.
- **구현 지점(코드 점검 완료)**: `utils/data/load_data.py`는 `image_grappa`를 **미로드**(input=kspace, target=label만), `utils/model/varnet.py`의 `VarNet.forward`는 `masked_kspace`에서 출발 → **loader와 forward 둘 다 수정** 필요.

### 1.1 GRAPPA 보조 입력의 augmentation 정렬 규칙 (필수)
> GRAPPA를 보조 입력 채널로 쓰면, 메인 입력 k-space와 **공간적으로 정확히 정렬**돼야 한다. 메인만 증강하고 GRAPPA를 안 하면(또는 반대) 두 채널이 어긋나 학습이 오히려 망가진다. augmentation 종류별로 처리가 다르다.

| 증강 종류 | 메인 k-space | GRAPPA 채널 | 이유 |
|---|---|---|---|
| **좌우 플립 / 정수 translation** (기하) | 적용 | ✅ **동일 적용 필수** | 같은 해부구조를 가리켜야 정렬 유지. target·bbox도 동일 g로. |
| **k-space 랜덤 위상** | 적용 | 무방(magnitude 불변) | 위상은 magnitude에 영향 없음 |
| **마스크 증강(추가 언더샘플)** | 적용 | ❌ **미적용** | GRAPPA는 이미 복원된 깨끗한 prior — 라인 더 지우면 prior를 스스로 훼손 |
| **coil 노이즈** | 적용 | ❌ (GRAPPA는 coil 없음) | GRAPPA는 coil-combined magnitude |

- ⚠️ **"더 언더샘플된 GRAPPA"는 만들 수 없고 만들 이유도 없음**: GRAPPA는 미리 계산돼 제공된 것이라 재생성 불가(fully-sampled 없음). 마스크 증강으로 입력만 어려워지고 **GRAPPA prior는 그대로** 두는 게 맞다(입력보다 깨끗한 사전정보를 주는 게 목적).
- **구현 권장**: 기하 변형은 k-space에서 각자 뒤집기보다 **이미지 도메인 변환 g를 한 곳에서 정의 → 모든 짝(메인 k-space·GRAPPA·target·bbox)에 동일 g 적용**. 두 k-space는 격자·물리량이 다르므로(메인 640×368×15coil complex vs GRAPPA 384² magnitude) 이 방식이 안전.
- ※ 이 정렬은 엄밀히 augmentation이 아니라 **GRAPPA 보조 입력을 쓸 때의 정렬 규칙**(좌표 동기 원칙의 확장). augmentation.md의 좌표 동기(§3)와 같은 계열.

#### 격자 정렬 — 384² ↔ 640×368은 리스케일 없는 crop+pad (2026-07-03 데이터 검증)
> 앞서 "두 격자가 달라 재샘플 필요"라던 우려는 **해소됨**. 실측으로 **384²는 640×368의 순수 crop+pad(공간 리스케일 없음)**임을 확인 → 기존 `center_crop` 유틸 하나로 픽셀 정확 정렬 가능.

- **검증**: 제공 undersampled k-space를 `ifft2c → RSS → center_crop(384,384)` 한 결과가 제공 `image_input`과 **픽셀 상관 0.975**(스케일 0.94는 밝기 정규화일 뿐 공간 변형 아님). 리스케일이 있었다면 crop 후 정렬이 깨져 상관이 급락함 → **없음** 확인. (`image_grappa`는 alias를 지운 만큼만 차이, 격자·스케일 동일.)
- **관계**: `varnet.py`의 출력 파이프라인이 정확히 `ifft2c→complex_abs→rss→center_crop(result,384,384)`(varnet.py:244). 즉 **384² = 640×368의 "높이 crop(640→384) + 폭 pad(368→384)"**, 픽셀 간격 보존.
- **GRAPPA를 U-Net 격자(640×368)에 올리는 법**: U-Net은 각 cascade 안에서 640×368로 동작(출력 직전에만 384² crop). 역연산 `center_crop(grappa_384, 640, 368)` → **높이 384→640 pad + 폭 384→368 crop**(center_crop이 pad·crop 둘 다 처리) → 픽셀 정확 정렬.
  - ⚠️ 뉘앙스: 높이 pad로 **위/아래 128행은 0으로 채워짐**(GRAPPA엔 바깥 데이터 없음). GRAPPA prior는 중앙 384행만 실내용, 주변부 0 — 보조 채널로는 무해(메인 k-space·DC는 640 전체 유지).
- **더 깔끔한 대안**: GRAPPA·최종출력 둘 다 384²이므로, cascade 안 대신 **출력단에 384² refinement 단계**를 두면 zero-pad 자체가 불필요(네이티브 동일 격자). 아키텍처 변경폭은 조금 더 큼.
- **주의(별개 이슈)**: batch_size>1로 올리면 acc4(368) vs acc8(372) width가 못 쌓여 깨짐 → 그때 입력 k-space width 통일(372 pad 등) 필요. GRAPPA 정렬과는 무관(샘플 단위).

---

## 2. 병변 충실도 (SSIM_bbox) — 모델측 레버 (순위 결정축)

> [`augmentation.md`](augmentation.md)가 "데이터로" bbox를 지키는 문서라면, 여기는 "손실·구조로" bbox를 직공략한다. 순위축이라 최우선.
> ⚠️ **선결**: §0의 파이프라인 갭(annotation 미전달) 해결이 아래 모든 레버의 전제.

### 2.1 bbox 가중 손실 (MOST 프레이밍) — ✅ 1순위 도입
- **근거**: MOST(REVIEW #4)는 recon을 downstream task 성능까지 최적화. downstream 망이 없으므로 **"병변 영역을 downstream처럼 취급" = bbox 영역 손실 가중**으로 치환.
- **형태**: `L = (1−SSIM_full) + λ·(1−SSIM_bbox)` (병변 슬라이스의 bbox 영역 SSIM/L1 별도 가중, λ 스윕).
  - bbox 좌표는 image 파일 `attrs['annotations']`(384² 기준) 그대로 사용 — 모델 최종 출력(center_crop 384²)과 좌표 정합.
  - annotation 없는 슬라이스는 SSIM_full 항만 → 자연 마스킹.
- **주의**: bbox 과가중 시 전역 SSIM 저하 → λ는 검증셋에서 두 SSIM 균형점으로 스윕.

### 2.2 중간 감독(IG loss) 결합 — 보조
- MOST의 **image-guided loss**(중간 recon을 alias-free 기준과 비교)와 FLAT(REVIEW #1) velocity-aligned 중간감독은 결이 같다 → cascade 중간 출력에도 bbox 인지 감독을 얹으면 병변 수렴 안정.
- **추가 파라미터·추론비용 0**(학습시 손실 항만) → 추론시간 무해.

### 2.3 hallucination 금지선 (하드 제약 재확인)
- SSIM_bbox는 **왜곡·blur 금지가 절대선**. 그래서 **2.5D 인접슬라이스 입력은 이미 제외**(§3.1). 손실로 bbox를 강조하되 **없는 신호를 만들지 않도록 DC를 항상 유지**.
- ⚠️ SSIM은 blur와 위조를 대칭 페널티하므로, 여력 시 **bbox 육안 점검(artifact leakage)**을 보조 게이트로 병행.

### 2.4 고주파 집중 손실 (k-space 저/고주파 불균형 활용) — 실험 후보
> **동기(2026-07-03 마스크 실측)**: 제공 언더샘플은 **중앙 ~8% ACS 완전샘플 + 주변부만 균일 1/acc**(acc4: ACS 29줄+1/4, acc8: ACS 31줄+1/8), 위상인코딩 1축만. 즉 **저주파는 사실상 given(ACS+DC가 못박음), 네트워크가 실제로 예측할 건 미획득 고주파**뿐.

- **아이디어**: 일반 image L1/SSIM 손실은 에너지의 대부분을 "이미 주어진 저주파 구조"에 쓴다. 손실을 **미획득 고주파 대역(주변부 PE 라인)에 가중**하면 학습을 실제 병목에 집중.
- **왜 순위축에 직결**: **병변의 미세 구조 = 고주파.** 고주파 복원 정확도 = SSIM_bbox 그 자체 → 순위축 직공략.
- **구현 형태(택1/결합, A/B)**:
  - k-space 대역 가중 L1: 예측 영상을 FFT해 **미획득 PE 라인**에서 GT(=`image_label` FFT)와 L1. 단 `image_label`은 combined-magnitude라 coil/phase 정보 손실 → **combined-image k-space 기준의 보조 손실**로만.
  - image 도메인 고역통과(high-pass) 성분에 SSIM/L1 가중.
- ⚠️ **양날의 칼(hallucination)**: "고주파를 채워라"를 강하게 밀면 **위조 위험** 그 자체. 반드시 **DC 강하게 유지 + SSIM_bbox 비퇴행 게이트(augmentation.md §4.2)** 하에서만. 가중치 λ_hf는 bbox 비퇴행 조건으로 스윕.
- 우선순위: **§2.1 bbox 가중 손실 다음**의 실험 후보(둘 다 "고주파/병변에 손실 집중"이라 상보적).

### 2.5 병변 bbox 좌표 활용 총정리 (제공 annotation을 쓰는 모든 레버)
> input `.h5`의 `attrs['annotations']`에 병변 슬라이스의 bbox 좌표(384² 기준)가 제공된다. 이를 학습·평가에 쓰는 방법을 한곳에 정리(상세는 각 소절/문서 링크).

1. **bbox 가중 손실**: 손실에서 병변 영역 SSIM/L1에 λ 가중 → 순위축 SSIM_bbox 직접 최적화. (상세 §2.1)
2. **병변 슬라이스 오버샘플링**: 병변 슬라이스는 전체의 일부라 희소 → 매 epoch 샘플링 확률을 높여 병변을 더 자주 학습(픽셀 미변형이라 hallucination 위험 0). **신규·저위험 권장.**
3. **병변 영역 보호**: coil 노이즈·mask 증강 등 교란 시 bbox 내부는 강도↓ 또는 제외 → 미세 병변 신호가 증강에 덮이는 것 방지. (augmentation.md coil 노이즈 항과 연동)
4. **위험 증강 회피 로직**: MixUp/CutMix 등 위험 기법 실험 시 bbox 걸린 슬라이스만 대상 제외. (augmentation.md Tier 3)
5. **좌표 동기 변환**: flip·translation 증강 시 bbox 좌표도 함께 변환 — 위 모든 기법의 전제. (augmentation.md §3)
6. **중간 감독(deep supervision)**: cascade 중간 출력에도 bbox 영역 손실 → 병변 조기 수렴. (상세 §2.2)
7. **검증 지표 분리**: bbox로 SSIM_bbox를 per-lesion 단위 별도 측정 → "full↑ bbox↓" 함정 조기 차단. (augmentation.md §4.2)
8. **hallucination 진단**: bbox 영역 error map 시각화로 없는 구조 생성 여부를 육안·정량 점검하는 게이트. (상세 §2.3)
9. ⚠️ **선결 조건**: 현재 `DataTransform`은 annotation을 버림 → `attrs['annotations']` 파싱 + slice별 bbox 전달 + 가변 길이용 `collate_fn`(또는 batch_size=1) 필요. (§0)
10. **권장 출발점**: **bbox 가중 손실(1) + 병변 슬라이스 오버샘플링(2)** — hallucination 위험 없이 SSIM_bbox를 직접 미는 가장 확실한 조합, 나머지는 그 위 보조.

---

## 3. 인접 슬라이스 활용 (PromptMR / PromptMR+)

관련 논문: `.claude/paper/01_candidates/PromptMR_prompting_2023.pdf`, `PromptMRplus_rethink-dum_2024.pdf`

### 근거: 인접 슬라이스 상관 측정 (image_label, 제공 파일 1개씩 — 표본 작음)
- 전역 인접 Pearson: **acc4 gap1 0.87**(gap2 0.74, gap3 0.66) / **acc8 gap1 0.75**(gap2 0.55, gap3 0.44).
- **bbox 내부(병변) 인접 상관은 훨씬 낮고 편차 큼**: acc4 MCL sprain 0.36~0.66, acc8 MCL 0.08~0.59·낭종 0.16~0.31.
- 해석: 전역 구조는 공유되나, **작고 슬라이스마다 바뀌는 병변(인대·낭종)은 인접 슬라이스로 뭉개질 위험** → SSIM_bbox 손해 우려.
- 측정 스크립트: scratchpad `adjacent_slice_corr.py` (임시). 데이터 더 확보되면 재측정 권장.

### 3.1 인접 슬라이스 (2.5D) 입력 스태킹 — ❌ **채택 안 함** (2026-07-02 결정)
- **무엇**: 인접 슬라이스 ±a를 함께 입력(concat)하고 중앙만 복원. PromptMR ablation상 a=1(3슬라이스) 최적.
- **왜 제외**: SSIM_full은 오를 수 있으나 위 측정에서 **bbox 내부 상관이 낮아(0.08~0.59) 병변 blur → SSIM_bbox 악화 위험**이 실증됨.
  순위축이 bbox라 리스크 대비 이득 불확실. (추후 여력 시 병변 슬라이스 bbox A/B로 재검토 가능.)

### 3.2 인접 슬라이스 배치 sensitivity map 추정 (메모리효율) — ✅ **채택** (2026-07-02 결정)
- **무엇**: 각 코일 sens-map을 **2a+1 인접 슬라이스에 대해 한 번에 배치 추정**(PromptMR+ Strategy C)
  → SME forward를 (2a+1)N→N, **GPU 메모리 ≈55% 절감** + 정확도↑(품질 저하 아님).
- **우리 적용**: VarNet 계열 SME에서 이웃 슬라이스 맵을 한 배치로 추정 → **1080 8GB 직접 이득**.
  복원 입력/bbox 내용은 안 건드림 → SSIM 위험 없음.
- **3.1과 구분(중요)**: 이건 **메모리 최적화**일 뿐 2.5D 입력(3.1)과 무관 — SSIM을 낮추지 않음.
- **구현 지점 확인 필요**: baby_varnet의 `SensitivityModel`이 슬라이스 단위 처리인지, dataloader가 볼륨 단위로 이웃을 넘기는지 점검 후 이식.

---

## 4. 메모리·추론시간 효율화 후보 (택1 실험 + 순서)

> 상세 리뷰는 `.claude/paper/REVIEW.md`. 여기서는 **어느 순서로, 무엇과 무엇을 배타 비교할지**를 정한다.

### 4.1 sens-map 효율화 — UEPS vs PromptMR+ (택1 A/B, 접근 상반)
같은 병목(coil sensitivity map)을 정반대로 푼다 → **동시 적용 말고 택1 비교**.

| | UEPS (REVIEW #2) | PromptMR+ (REVIEW 요약) |
|---|---|---|
| 접근 | **sens-net 제거**(코일을 배치축, CSM 미사용) + k-space progressive resolution | sens-net **유지·개선** + 인접정보 기반 효율 추정(**GPU ≈55%**) |
| 메모리 이득 | progressive(저해상도 시작)로 FLOPs·VRAM↓ | sens-map 추정 메모리 자체를 절반 |
| 위험 | 코일 배치화 → **15코일 knee에서 메모리 되레↑** 가능, 원 120M → 채널 대폭 축소 필수 | DUM 재설계 이식 난이도↑, 구조 침습 큼 |
| 우리 채택폭 | **UE(sens-net 제거) 또는 P(progressive)** 조각 이식 | **배치 SME(Strategy C)는 §3.2에서 이미 채택**, 나머지는 선택 |

- **권고**: §3.2 배치 SME(무위험 메모리 이득)를 기본으로 두고, 추가 여유 필요 시 **UEPS의 P(progressive resolution)를 먼저** 시도(sens-net 제거는 코일 배치화 리스크가 커 후순위). PromptMR+ 전면 이식은 침습이 커 최후 카드.
- **progressive가 이 데이터에서 특히 통하는 물리적 근거(2026-07-03 실측)**: 마스크가 **중앙 ~8% ACS 완전샘플 + 주변부 균일 1/acc** 구조라, **ACS 블록만 IFFT하면 alias 없는 저해상도 영상이 공짜로** 나온다. 이 신뢰 가능한 저해상도 앵커에서 출발해 고주파를 점진 복원 = progressive resolution의 이상적 조건. 덤: 완전샘플 ACS는 **sensitivity map / GRAPPA 커널 캘리브레이션의 표준 소스**(제공 GRAPPA도 이 중앙으로 생성) → §3.2 SME도 **ACS 위주 추정**이 정석. (고주파 손실 집중은 §2.4와 상보적.)

### 4.2 cascade 다이어트 — FLAT (REVIEW #1)
- velocity-aligned 중간감독 + ODE 일관 초기화로 **적은 cascade에서도 중간 출력이 단조 개선** → cascade 6→4로 줄여도 품질 유지.
- **추가 파라미터·추론비용 0**(loss 항만) → 추론시간 채점에 직접 유리. §2.2 bbox 중간감독과 결합 가능.
- 주의: fastMRI multi-coil에서 SSIM 소폭 하락 사례 → **w_v 튜닝** 및 SSIM판 손실로 개조(REVIEW `FLAT_정리.md`).

### 4.3 확정 적용 순서 (모델 축)
1. **베이스 고정**: 경량 E2E-VarNet(cascade 6~8·채널 축소·FP32) + **§1 GRAPPA 초기화(채널 주입)**.
2. **§2 bbox 가중 손실** 도입(순위축 직공략) — 단 §0 파이프라인 갭 해결 후.
3. **§3.2 배치 SME** 이식(무위험 메모리 확보) → 확보분을 채널/cascade에 재투자.
4. **추론시간 부족 시 §4.2 FLAT**로 cascade 축소(추론시간 측정은 augmentation.md §4.3 프로토콜).
5. **VRAM 여전히 부족 시 §4.1 택1 실험**(UEPS-P 우선).
6. TGVN은 **PD/PDFS 쌍 데이터 확인 시에만**(CLAUDE.md 미확정) — 없으면 제외, CG 루프는 추론시간 감점.

---

## 5. 재현성 & FP32(1080) 프레이밍

> 최종 모델은 주최 VESSL 서버에서 재학습하고 **고정 seed 재현을 검증**한다(CLAUDE.md). 재현성은 선택이 아니라 통과 조건.

- **FP32 고정**: 1080은 Pascal·Tensor Core 없음 → AMP/FP16 속도 이득 없음 → **FP32 학습**. FP32는 활성값 메모리가 FP16 대비 2배 → **§3.2/§4 메모리 최적화가 '있으면 좋은'이 아니라 필수**인 이유.
- **seed 고정 범위**: 모델 init·dataloader shuffle·augmentation RNG(augmentation.md §3와 동일 seed) 전부 고정. `torch.manual_seed`/`cudnn.deterministic=True`/`benchmark=False`까지 걸어 **비트 재현** 확보.
- **결정성 vs 속도**: `deterministic=True`가 일부 커널을 느리게 하나, 재현 검증이 있는 대회에선 안전 우선.
- **개인 GPU=실험, 서버=최종 재학습**: 아키텍처/하이퍼파라미터만 이식하고 **최종 수치는 서버 재현본 기준**으로 보고.
