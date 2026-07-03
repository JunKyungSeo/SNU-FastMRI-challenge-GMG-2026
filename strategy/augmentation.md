# Data Augmentation 전략 — FastMRI 2026 (Annotation-guided Knee Reconstruction)

> 📚 **근거(rationale) 문서.** 인덱스 [`README.md`](README.md) · 확정 SPEC [`confirmed_model.md`](confirmed_model.md) · 후보 [`candidates.md`](candidates.md) · 제외 [`rejected.md`](rejected.md).

> # ✅ 구현 완료 (2026-07-03)
> 이 문서의 채택 증강은 **`utils/data/transforms.py`(`MRIAugmentor`/`AugmentConfig`/`DataTransform`) · `utils/data/load_data.py`에 구현·검증 완료**.
> 실측 결과 요약은 아래 **[§ 증강 결과 (실측 요약)](#증강-결과-실측-요약)**. (시각 확인용 노트북은 검증 후 삭제 — 전략 검증용 임시 코드.)
> 기본값 전부 off라 켜기 전엔 기존 학습과 100% 동일(스위치 배선은 [`todo.md`](todo.md) 3·4번).

> **제1원칙 (이 문서의 조직 기준): hallucination-free lesion fidelity.**
> 대회의 신설 novelty는 "병변 bbox 위치를 위조(hallucination) 없이 정확히 복원"이며 **SSIM_bbox가 순위 결정축**이다.
> 따라서 모든 aug는 **① 병변 bbox 내용을 왜곡/흐림/위조하지 않는가(=SSIM_bbox 안전)** 를 1차 관문으로,
> ② SSIM_full 이득, ③ 물리적 유효성 순으로 판정한다. **화질(SSIM_full)만 올리고 bbox를 해치는 aug는 채택하지 않는다.**
> 추론시간 축에는 영향이 없으므로(학습시 dataloader에서만 동작) 화질·robustness는 사실상 free lunch
> — **단 예외**: consistency 손실(학습 2× forward)·TTA(추론 N×)는 시간 비용이 있어 별도 취급(§2 보조·§3.5).
>
> 📌 **역대 수상팀 근거**: 이전 SNU FastMRI 수상팀 기법 총정리는
> [`../.claude/paper/past_winners_augmentation.md`](../.claude/paper/past_winners_augmentation.md).
> 특히 2025 대회에서 **"knee는 강한 공간 변형에 취약, 약한 변화(마스크 위주)만 유효"** 라는 교훈이
> 아래 보수적 Tier 설계(flip+마스크 중심)를 직접 뒷받침한다.

---

## TL;DR — 한눈에 (ON / OFF / 스케줄 / 우선순위)

> **조직 기준: 모든 aug는 "SSIM_bbox(순위 결정축)를 해치지 않는가"로 먼저 거른다.** 전체 화질(SSIM_full)만 올리고 병변 bbox를 흐리거나 위조(hallucination)하면 순위가 떨어진다(3등→1등 반전이 bbox 채점에서 발생).

**ON (켠다)**
- ✅ **좌-우(phase-encode) 플립만**(마스크 동기 플립) — 픽셀 보존 → bbox 안전 + 해부학적으로도 유효(반대쪽 무릎과 동일)
  - ❌ 위-아래(readout) 플립은 **제외**(2026-07-02): 넙다리뼈↔정강이뼈가 뒤바뀌는 비현실적 무릎 형태 → Tier 3
- ✅ Mask augmentation(추가 언더샘플, 5~16× 랜덤 + center 0.08 + equispaced/오프셋랜덤) — 우리 데이터 유일 정석 물리 aug
- ✅ **원본(무증강) 고정마스크 입력 반드시 혼합** — test 분포=고정마스크, OOD 방지(= 실제 bbox 채점 조건 정합)
- △ **GRAPPA 영상 도메인 prior 채널 + residual**(fft 안 함, [`confirmed_model.md`](confirmed_model.md)) — 제공된 physics 기반 초기해를 영상 그대로 regularizer U-Net에 real-valued prior로 concat. 단 image size/슬라이스/스케일 정규화 필수, A/B 후 채택
- △ k-space 랜덤 위상(**image-domain smooth phase 한정**) — magnitude/RSS 불변이라 안전하나 **이득 marginal → 저우선**
- △ coil별 측정 노이즈(강도 보수적, 검토)
- ❌ 정수 픽셀 translation은 **미채택**(target=384 crop vs 모델 640 crop 정렬 불가, 아래 §Tier2 참고)

**OFF (끈다, Tier 3)**
- ❌ 밝기/gamma/image-domain 노이즈, elastic, cutout/erasing, MixUp/CutMix, 임의각 회전·스케일·shear, naive 합성병변 삽입 — 전부 **SSIM_bbox 직접 파괴/hallucination 위험**

**스케줄**
- 첫 10~15 epoch aug off(기본기) → 지수형으로 확률 p=0.5~0.6까지 점증
- 강도는 warmup(≈4ep)→막판 decay(막판엔 깨끗한 데이터로 consolidation)
- **모든 aug RNG는 고정 seed**(조직측 재현성 검증 대비)

**실험 우선순위(먼저 켤 것)**: ① mask-aug + 좌우 flip(**최대 이득으로 예상 — 검증 전 가설**) → ② 원본마스크 혼합비 튜닝 → ③ GRAPPA FFT prior 입력 → ④ acc4/acc8 커리큘럼 → ⑤ k-space 위상 → ⑥ translation/지터 → ⑦ consistency(후순위, **학습 2× 비용**). 상세 순서·측정법은 §4.1/§4.2.

> ✅ **구현 상태(2026-07-03)**: 위 증강은 `utils/data/transforms.py`(`MRIAugmentor`/`AugmentConfig`/`DataTransform`)와 `utils/data/load_data.py`에 **구현 완료**.
> **기본값 전부 off라 기존 학습과 100% 동일** — 켜려면 train.py `--aug-*` 인자 + train_part.py `set_epoch` 배선 필요([`todo.md`](todo.md) 3·4번).
> bbox는 `return_bbox=True`일 때 `DataTransform`이 `annotations`를 파싱해 flip과 동기 반환(가변 길이용 `_bbox_collate` 포함). bbox를 **손실에 쓰려면** train_part 소비부는 아직 배선 필요.

---

## 증강 결과 (실측 요약)

> `utils/data`의 실제 샘플(acc4: 23 slice, acc8: 31 slice)에 구현된 증강을 적용해 검증한 결과.
> 시각 비교 노트북(원본↔증강, 병변 슬라이스 + k-space/mask)은 검증용 임시라 삭제 — 아래 수치 요약으로 대체.
> ※ 아래는 눈으로 보기 위한 **고정 아티팩트**(파일당 flip·thinning 1회 적용) 기준. 실제 학습은 저장 없이 매 배치·슬라이스별 랜덤 확률로 on-the-fly 적용된다.

**증강하면 데이터가 어떻게 바뀌나 (3가지가 동시에·정렬 유지하며 일어남):**

1. **좌-우(width) flip — 픽셀 정확, 전부 동기**
   - `image_input`/`image_grappa`/`image_label`이 좌우 반전. `flip(원본 label) == 증강 label`이 **정확히 0**(오차 0.00e+00), 원본↔증강 상대차 0.40(acc4)/0.38(acc8) → 실제로 뒤집힘(대칭 아님).
   - k-space는 이미지 도메인 flip 후 재생성 → 재시뮬 없이 **relerr ≈ 1e-7**로 정렬, **샘플 라인 수 보존**.
   - mask도 even-length 정확 공식(`roll(flip(M),1)`)으로 동기.

2. **ACS 보존 마스크 thinning — 중앙 저주파는 그대로, 주변부 고주파만 추가 언더샘플**
   - 샘플 라인: acc4 **114→71** / 368, acc8 **74→52** / 372 (파일당 예시값; thin 배수는 슬라이스별 랜덤).
   - **중앙 ACS(≈8%) 블록은 100% 보존**(sensitivity map/GRAPPA 캘리브레이션 유지), 주변부만 성기게. 증강 마스크는 항상 **원본 마스크의 부분집합**(라인 추가 없음).
   - k-space 스펙트럼에서 중앙 밝은 저주파 코어는 원본과 동일, 바깥 고주파 줄무늬만 넓어짐(육안 확인함).

3. **bbox 좌표 flip 동기 — 병변을 계속 정확히 감쌈**
   - `x' = 384 − x − w`로 이동. 실측: acc4 `x 158→196`, acc8 `x 296→71` (기대값과 일치). `y`/`width`/`height`/`label`은 불변.
   - flip된 이미지 위에서 레드박스가 동일 해부 병변(연골/인대 등)을 그대로 둘러쌈 → **SSIM_bbox 채점 대상이 보존**.

**추가로 확인된 사실**
- 원본 `.h5`의 `kspace`는 **마스크 미적용(full)** 상태로 저장되어 있고, 언더샘플은 로드 시 `kspace * mask`로 적용된다(`DataTransform.__call__`과 동일). 시각화도 이를 반영해 양쪽 모두 마스크를 곱해 '모델이 실제 보는 입력'을 그린다.
- dtype은 complex64로 보존(FFT/exp가 complex128로 승격하는 것을 방지).
- 기본값(off)에서는 위 어떤 것도 일어나지 않아 기존 파이프라인과 비트 동일.
- **미채택 재확인**: 정수 translation은 target(384 crop)과 모델(640 crop) 정렬 불가(relerr≈0.44)로 제거, readout(위-아래) flip은 해부학적 비현실로 제외(Tier 3).

---

## 0. 파이프라인 전제

- 모델: VarNet 계열. **입력 = 언더샘플 multi-coil k-space + mask**, 타깃 = `image_label`(384×384).
- 관련 코드: `utils/data/transforms.py`의 `DataTransform.__call__`, `utils/data/load_data.py`.
- 병변 annotation: image 파일 `attrs['annotations']` (JSON, bbox 좌표는 384×384 image 공간 기준).
- GRAPPA 초기해: image 파일의 `image_grappa`는 제공 입력으로 사용 가능하나, 원본 multi-coil complex k-space와 동일한 물리량은 아님. **fft 하지 않고 영상 도메인 그대로**, multi-coil k-space 대체품이 아니라 **regularizer U-Net의 영상 prior 채널**로 취급한다(확정: [`confirmed_model.md`](confirmed_model.md)).
- ⚠️ **현재 파이프라인은 annotation을 미사용**: `DataTransform`은 `mask, kspace, target, maximum, fname, slice`만 반환(annotation drop). bbox 좌표 동기 aug / bbox 가중 손실을 쓰려면 **transform이 `json.loads(attrs.get("annotations","{}"))`를 파싱해 slice별 bbox를 emit**하고, **가변 길이 bbox 배치를 위한 custom `collate_fn`**(또는 batch_size=1)이 필요.

---

## 1. 우리 데이터가 augmentation에 거는 제약 (먼저 이해할 것)

1. **fully-sampled k-space가 없다.**
   - k-space 파일: 언더샘플 640×368×15coil. 라벨: 384×384 magnitude.
   - 해상도·도메인·coil 수가 달라 **라벨에서 멀티코일 k-space를 되만들 수 없음.**
   - → 전형적 MRAugment(타깃 변형→FFT→마스크→입력 재생성) 파이프라인을 **그대로 못 씀.**

2. **1D 위상인코딩(width축) 마스크가 고정.**
   - 90° 회전·전치 등 **축을 바꾸는 변형은 마스크의 물리적 의미를 깨뜨림.**
   - 임의 각도 회전/스케일은 재마스킹이 필요한데 fully-sampled가 없어 불가.

---

## 2. 채택할 augmentation (우선순위)

### 2.0 SSIM_bbox 렌즈 — 각 aug의 병변 채점 영향 (먼저 볼 표)

> Tier 1/2/3은 **물리적 유효성** 축 분류다. 아래 표는 같은 aug를 **채점축(특히 순위축 SSIM_bbox)** 기준으로 다시 본 것. `⚠`/`❌` 행이 함정(full↑인데 bbox 중립/하락).

| augmentation | SSIM_full | SSIM_bbox | bbox 리스크 플래그 |
|---|---|---|---|
| 좌우 플립(+마스크 동기) | ↑ | ↑ (동등) | 없음 — 픽셀 보존 + 해부학적 유효, **bbox 좌표 동기 변환만 필수** |
| ~~위아래(readout) 플립~~ | ↑(픽셀상) | ⚠ 비현실적 해부구조 | **제외**(Tier 3) — 뼈 위치가 뒤바뀜, hallucination 유사 위험 |
| Mask aug(추가 언더샘플) | ↑ | ↑ | 낮음 — 타깃 불변. 단 과도 배율은 저대비 병변 소실 → **원본마스크 혼합 필수** |
| 원본(무증강) 고정마스크 혼합 | ↑(분포정합) | ↑(분포정합) | 없음 — test 고정마스크 = 실제 bbox 채점 분포 |
| GRAPPA 영상 prior 채널(+residual) | ↑? | 중립~↑? | 낮음~중간 — 제공 physics prior라 안전 후보(fft 안 함). 단 크기/슬라이스/스케일 정규화 실패 시 OOD/shortcut 위험 |
| k-space 랜덤 위상(image-domain smooth) | ↑(robust, marginal) | **중립(불변)** | 없음 — magnitude/RSS 불변. 단 이득 marginal(불변이라 새 신호 없음) → 저우선 |
| ~~정수 translation~~ | — | — | ❌ **미채택** — target(384 crop) vs 모델(640 crop) 정렬 불가(relerr≈0.44) |
| 라인 지터(trajectory) | ↑(robust) | 중립 | 낮음 |
| coil별 노이즈 | ↑(robust) | ⚠ 중립~↓ | 저대비 병변 신호를 노이즈가 덮을 수 있음 → **강도 보수적** |
| consistency 정규화 | ↑(안정화) | 중립 | 없음(회전은 항 내부만) — 단 **학습 2× 비용** |
| 전역 스칼라 곱(Tier3 예외) | ?(분포차 있으면↑) | ⚠ 위험 | 병변 대비 왜곡 가능 → 2026 bbox 채점상 **기본 보류** |
| image노이즈/gamma/elastic/cutmix (Tier3) | 때때로↑ | ❌ 파괴 | hallucination / 병변 가림 |

**판정 규칙**: SSIM_bbox가 `중립` 이상이 아닌 행(⚠·❌)은 채택 근거가 약하다. 순위축이 bbox이므로 "full만 오르는" aug는 기본 탈락.

### Tier 1 — 안전 + 즉시 이득 (강력 추천)

- **Phase-encode(width) 축 플립(좌-우, 안쪽↔바깥쪽) + 마스크 동시 플립** ✅ 유일하게 채택하는 플립
  - 마스크도 같이 뒤집으면 물리적으로 유효(FFT와 commute).
  - **해부학적으로도 안전(2026-07-02 실측 확인)**: 관상면(coronal) 무릎에서 이 축은 안쪽(medial)↔바깥쪽(lateral) 방향.
    사람 무릎은 좌/우가 서로 거울상이라, 한쪽 무릎을 좌우로 뒤집으면 **실제 존재하는 반대쪽 무릎과 동일**한 형태 →
    비현실적 해부구조를 만들지 않음. 병변 라벨(내측/외측)이 뒤집혀도 순수 복원 과제라 무해(입력·GT·bbox를 함께 뒤집어 학습 신호는 일관).
- 데이터가 적을수록 효과가 크다.

### Tier 2 — 우리 데이터에 특히 잘 맞음

- **Mask augmentation (추가 언더샘플링)** ⭐ 우리 데이터셋에서 거의 유일한 "정석 물리 aug" — **역대 수상팀 다수 채택**
  - acc4 입력에 **랜덤 마스크를 더 씌워 더 어려운 입력**을 만들되 타깃은 그대로.
  - fully-sampled 불필요 → 물리적으로 완전 유효.
  - acc4↔acc8 사이 난이도를 채워 **robustness 향상.**
  - **가속률 랜덤화**: 재-언더샘플 비율을 매 배치 랜덤(acc4→ acc4~acc8 구간)으로 뽑으면
    두 가속률에 모두 강건해짐 (UNITS의 재샘플링 정당화 + K-Space Acquisition DG의 acquisition uncertainty).
  - **수상팀 구체 수치 (직접 채택 권장)**:
    - **5×~16×** 폭넓은 가속 배율을 임의 생성해 학습(222Hz, 재애애영) — acc4/acc8 밖까지 커버.
    - 제공 마스크 패턴 모방: **center fraction 0.08 / 0.083, equispaced** (맞는데왜틀려요, attention, sMaRtIfy).
    - equispaced뿐 아니라 **간격 무작위·오프셋 랜덤** 마스크도 섞음(attention, sMaRtIfy, EJ).
    - **커리큘럼**: 저배속(쉬움)→고배속(어려움) 순으로 마스크 난이도 점증.
    - **2단계 커리큘럼(UnifiedCS)**: 매 epoch 혼합 대신 **단일 배율 pretrain → 혼합(acc4/acc8) fine-tune** 도 유효 → A/B 가치.
  - ⚠️ **고정마스크 train/test 분포 규칙(UNITS 근거, 필수)**: 우리 test 마스크는 **고정**이므로
    추가 언더샘플만 학습하면 배포 분포와 **불일치(OOD)** 발생. → 증강 배치와 함께 **원본(무증강) 마스크 입력도 반드시 섞어** 학습.
  - ⚠️ **ACS(중앙 저주파) 보존 규칙 — 필수** (2026-07-03 실측 근거): 제공 마스크는 **중앙 ~8%가 통짜 완전샘플(ACS) + 주변부만 균일 1/acc**
    구조다(acc4: ACS 29줄 + 주변 1/4 / acc8: ACS 31줄 + 주변 1/8). 추가 언더샘플 증강 시 **중앙 ACS 라인은 절대 드롭 금지, 주변부 고주파 라인만 thinning**.
    - 이유: 실제 가속(acc4→acc8)이 곧 "**ACS 고정 + 주변부 1/4→1/8**"이라 이 형태만 물리적으로 정합. ACS를 지우면 sensitivity map/GRAPPA 캘리브레이션까지 깨져 복원 자체가 붕괴.
    - 즉 "acc4를 acc8처럼" 만드는 증강 = **8% 중앙 유지 + 주변부만 성기게** (center fraction 0.08 고정, 바깥 gap만 확대).
- **Instrumental noise 분포 보존 (SMPlayer 차용, 검토)**
  - 측정 고유 노이즈 분포를 유지하려 **채널별 노이즈에 sensitivity map을 곱해** 증강.
  - 우리는 멀티코일 k-space가 제공되므로 이식 가능 → coil별 노이즈 통계 존중한 노이즈 주입.
- **GRAPPA 영상 도메인 prior 채널 + residual** △ 제공 physics prior 활용 (확정: [`confirmed_model.md`](confirmed_model.md))
  - 아이디어: `image_grappa`는 주최측이 제공한 GRAPPA 초기 복원 영상이므로 external data가 아니라 **언더샘플 k-space에서 얻은 physics 기반 prior**다. **fft 하지 않고 영상 그대로** regularizer U-Net에 real-valued prior 채널로 concat한다.
  - ⚠️ **fft 안 함(2026-07-04 확정)**: GRAPPA는 magnitude·단일코일 384²라, fft하면 위상·코일이 날조된 가짜 k-space가 된다. VarNet cascade regularizer는 image 도메인에서 동작(`sens_reduce→U-Net→sens_expand`)하므로 **영상 채널로 넣는 게 정합**. data consistency는 제공 k-space+mask에만 건다.
  - **residual 출력**: `out = grappa + f([image_input, grappa, …])` — GRAPPA→label 잔차만 학습해 학습 난이도↓·SSIM↑([`model_strategy.md`](model_strategy.md) §1.2 ①).
  - **정규화 필수**: 파일별 image size, k-space size, slice 개수, acc4/acc8 페어 구성이 다를 수 있으므로 `(fname, acc, slice)` 기준으로 정확히 매칭한다. GRAPPA(384²)와 readout-crop된 k-space(384×폭) 격자 정합은 GRAPPA 폭 crop 또는 출력단 384² refinement head로.
  - **스케일 정규화(확정)**: 입력 유도 **전역 스칼라 ÷s**(zero-filled RSS robust percentile)를 `image_input`·`image_grappa`·target에 **동일 적용**, 출력단 `×s`. `maximum` attr은 label 통계라 입력 정규화엔 쓰지 않는다. acc 무관 단일 규칙.
  - **슬라이스 정규화**: image 파일과 k-space 파일의 slice 수가 다르면 더 짧은 쪽에 맞춰 자르거나 명시적으로 error를 낸다. silent broadcast/순서 mismatch는 bbox 학습을 망가뜨리므로 금지한다.
  - **동기 augmentation**: 좌우 flip을 적용하면 `kspace`, `mask`, `image_grappa`, `target`, bbox 좌표를 모두 같은 변환으로 맞춘다. **mask augmentation은 GRAPPA엔 미적용**(깨끗한 prior 보존).
  - 채택 기준: `baseline` vs `+GRAPPA prior`를 동일 split/seed에서 비교하고, **SSIM_bbox 비퇴행**을 먼저 확인한다. bbox가 유지되고 full이 오르며 1080 추론시간 증가가 작을 때만 유지한다.
- **Mask/라인 위치 지터(trajectory jitter)** ⭐ K-Space Acquisition DG 차용
  - **이미 샘플된 위상인코딩 라인의 인덱스를 소폭(정수 가우시안 오프셋 δ~N_int(0,τ²), 원논문 τ=10) 이동**시켜
    "샘플링 위치가 조금씩 달랐다면"을 시뮬레이션. **라인 개수 유지 + 이동 후 위치 중복 방지**(unique).
  - 원리: **k-space(획득) 도메인 교란이 image 도메인 노이즈보다 cross-domain 강건성에 유리**
    (해당 논문의 핵심 결론 — 실제로 image 노이즈는 U-Net에서 최대 −1.5dB로 **오히려 해로움**). acc4/acc8 간, 슬라이스/스캔 변동 robustness 확보.
  - 주의: 우리는 마스크가 고정·제공이라 **논문의 learned-trajectory / fully-sampled 재획득은 불가** →
    "제공된 샘플 라인을 지터/추가 언더샘플"하는 형태로만 개조 적용.
- **k-space 랜덤 위상(phase) augmentation** ⭐ 신규 (SSIBench VORTEX T1 차용)
  - 복소 k-space(또는 이미지)에 **랜덤 위상 오차(σ≈1)를 곱함**. MR 위상은 본질적으로 임의적 →
    **물리적으로 유효**하고 최종 **magnitude·bbox 채점에 영향 없음**(magnitude 불변).
  - image 도메인 노이즈/밝기 증강 대비 안전한 입력 robustness 증강. Tier 1과 함께 저위험.
- **작은 정수 픽셀 translation** ❌ **미채택으로 변경**(2026-07-03 구현 중 발견)
  - k-space는 선형 위상만 곱하면 마스크 보존되나, **target이 384로 center-crop돼 제공**되는데 모델은 **640×368에서 크롭**한다.
    640 도메인 shift는 384 target과 정렬 불가(높이 크롭이 순환 wrap과 불일치, 실측 relerr≈0.44). flip은 crop과 commute해 안전하지만 translation은 안 됨 → **제거**. (640 clean 영상이 없어 재정렬도 불가.)

### Tier 2 보조 — Equivariant consistency 정규화 (후순위)

- **개념 (SSIBench의 Equivariant Imaging 차용)**: augmentation을 단순 데이터 증식이 아니라
  **consistency 손실**로도 활용. 입력을 변환 g(플립/정수 translation)한 뒤 복원한 결과가
  원본 복원을 g로 변환한 것과 일치하도록 강제: `L_eq = ‖ f(g·y) − g·f(y) ‖`.
- **왜 우리 대회에 쓰나**: 대회는 **GT(image_label) 제공 → self-supervised(SSL) 자체는 불필요**
  (supervised가 곧 oracle 상한). 다만 EI 계열 consistency는 **GT가 있어도 보조 정규화**로 얹어
  SSIM_full 안정화·과적합 완화에 소폭 기여 가능.
- **제약**: 여기서 쓰는 g는 반드시 **Tier 1 물리 유효 변환(좌우 플립, 소폭 정수 translation)**으로 한정
  (마스크 축을 깨는 회전 등은 금지). bbox도 g로 동기 변환.
- **소각도 회전의 안전경계 (SSIBench VORTEX 실측)**: consistency 손실 **안에서만**이면
  **±15° 회전 + ±10% shift**는 이득. 단 **±360°/scale/shear는 성능 저하** 확인됨. 회전이
  **GT/bbox 채점 타깃엔 절대 미적용**(consistency 항 내부에서만 순환) → bbox 평가 왜곡 없음.
  ⚠️ diffeomorphic(CPAB) 변형은 EI에서 회전보다 우수하나 **elastic류라 병변 왜곡 위험 → 제외**.
- ⚠️ **학습시간 비용(VESSL 서버 관련)**: consistency 항은 `f(g·y)`와 `f(y)` **두 번의 forward가 필요 → 스텝당 ~2× 순전파**.
  추론시간 채점축엔 영향 없지만 **학습 시간·서버 점유를 늘림**(최종 모델은 VESSL 재학습이라 wall-clock 부담). SSIM 이득 대비 시간비용을 함께 측정할 것.
- 우선순위: **후순위**. supervised 기본 손실 + bbox 가중이 먼저, 여력 있을 때 실험.
- 참고: SSIBench는 "GT 없을 때 어떤 SSL이 쓸 만한가"의 벤치마크 → 우리에겐 **SSL 채택 안 함이 정답**
  이라는 판단 근거 + EI consistency 재료로만 활용.

### Tier 3 — 금지/고위험 (SSIM_bbox를 직접 파괴)

- **Readout(height) 축 플립(위-아래, 넙다리뼈↔정강이뼈)** ❌ (2026-07-02 실측 후 제외로 변경)
  - 물리적으론 마스크를 안 건드려 "안전"해 보이지만, **해부학적으로 무효**: 관상면 무릎에서 이 축은
    넙다리뼈(위)↔정강이뼈/종아리뼈(아래) 방향인데 **둘은 형태가 완전히 다른 뼈**라, 뒤집으면 실존하지 않는
    무릎 형태(뼈 위치가 뒤바뀜)가 됨. Foundation 교차검증(§ 아래 U-Net 절)의 "존재하는 해부구조 그대로 복원"
    원칙에 위배 → 픽셀 보존이라도 **hallucination 유사 위험**으로 제외. (좌-우 플립만 Tier 1로 채택.)
- 밝기/대비 jitter, gamma, **image 도메인 노이즈**: MRI 대비는 물리적 의미 → 왜곡 시 병변 신호 오학습 → hallucination.
  - ⚠️ **정량 증거(K-Space Acquisition DG)**: image 도메인 노이즈는 일반화를 **오히려 악화**(U-Net −1.5dB)시키는 반면
    k-space 도메인 교란은 이롭다 → **image 도메인 노이즈/밝기 증강은 기피, k-space 증강 우선**.
  - ⚠️ **단, 예외 검토**: 수상팀(STEM AI Lab, 샤샤)은 EDA로 **train vs leaderboard 픽셀 분포 차**를 발견해
    **전체 픽셀에 임의 스칼라(예: 1.0~2.0)를 곱하는 전역 스케일**을 사용했음. gamma/per-pixel jitter보다 온건하고
    분포 시프트가 실제라면 이득 → **약하게·전역 곱셈만 실험적으로**. (2026 knee+bbox 채점상 기본은 보류.)
- **Synthetic lesion injection** (PixelUncertainty 발상, 고위험·실험적): SSIM_bbox 병변 다양성 확보용으로
  **합성 고신호 병변을 삽입**하는 아이디어. 원논문은 **학습이 아니라 test-time OOD 프로브**로만 사용(파라미터 미제공).
  - ⚠️ **naive 이미지 삽입은 hallucination 그 자체** → 절대 입력에만 삽입 금지.
  - **유일한 안전 형태 = k-space consistent paired 삽입**: 병변을 **GT(label)에 넣고** 그 파일의 고정 마스크+코일 모델로
    **입력을 재생성**해 (입력, GT) 쌍으로 학습. 단 **우리는 fully-sampled 멀티코일 k-space가 없어** 재생성이 어려움 →
    실현성 낮음. 합성 bbox는 실제 annotation과 구분·추적 필수. **보류/후순위 실험**.
- Elastic deformation, cutout/random erasing: **병변 위에 걸리면 채점 핵심인 SSIM_bbox 파괴.**
  절대 병변 영역을 가리거나 비틀지 말 것.
- **MixUp / CutMix**: 수상팀(GrilledMackerel, STEM AI Lab, 샤샤)이 일반화용으로 썼으나,
  2025는 bbox 채점 비중이 지금보다 작았음. **CutMix가 병변 bbox 위에 걸리면 SSIM_bbox 파괴** →
  병변 슬라이스 회피 로직 없이는 **금지**. MixUp도 병변 신호 혼합 위험 → 기본 제외.
- 임의 각도 회전·스케일: 보간 아티팩트 + 마스크 축 붕괴.

---

## 3. 대회 특화 주의사항

- **bbox 좌표 동기 변환**: 플립/translation 적용 시 `attrs['annotations']`의 x,y도 같이 변환해야
  lesion-aware loss / 평가 정합이 유지됨. 현재 `DataTransform`은 attrs를 안 건드리므로 여기서 같이 처리.
- **추론시간 축은 공짜 이득**: augmentation은 학습시 CPU dataloader에서만 동작 → GTX 1080 추론시간 점수에 영향 0.
  3축 중 하나를 안 건드리고 화질만 올리는 free lunch.
- **acc4/acc8 둘 다 채점 — 한 모델로 양쪽 강건하게**: mask-aug 배율을 acc4~acc8 구간(그 밖 5~16×까지) 랜덤화해 두 가속률을 하나의 모델이 커버(가속률 조건부 입력이 없다면).
  - ⚠️ **acc8은 언더샘플이 심해 저대비 bbox 병변이 더 쉽게 소실** → acc8의 SSIM_bbox를 별도 감시(acc4에서 안전한 강도가 acc8에선 병변을 지울 수 있음).
  - 파일이 acc4/acc8로 분리 제공되므로 **배치에 두 가속률을 균형 있게 섞어** 한쪽 편향 방지. A/B 리포트는 항상 acc4·acc8 분리(§4.2).
  - 커리큘럼(단일배율 pretrain → 혼합 fine-tune, §2 Mask aug의 UnifiedCS)도 acc4/acc8 양립 수단 중 하나.
- **강도는 보수적 + 지수형 스케줄링** ⭐ (상위권 수상팀 거의 필수): MRAugment ablation 교훈 — 데이터 적으면 크게 이득, 과하면 오히려 SSIM 하락.
  - 구체 레시피(다수 수상팀 공통): **첫 10~15 epoch은 증강 off**(원본으로 기본기) →
    이후 **지수 함수 형태로 aug 확률을 p=0.5~0.6까지 점증**(Digital Catalyst, SMPlayer, 222Hz, 2대1000, 샤샤 등).
  - 초기부터 강한 증강 시 **수렴 실패** → 반드시 warmup 후 annealing.
  - **두 스케줄 축 구분(중요)**: 위는 aug **확률(probability)** 스케줄. 이와 별개로 aug **강도(intensity)**를
    **warmup-then-decay**로 거는 방식도 유효(K-Space Acquisition DG: 강도를 t_warmup≈4ep까지 선형 상승 → 이후 0으로 감쇠해
    막판엔 깨끗한 데이터로 consolidation). 확률↑ + 강도는 막판↓ 조합이 안전.
- **재현성 (조직측 검증 대비)**: 고정 seed 재현 검증이 있으므로 **aug RNG를 seed로 고정** 필수.
  - dataloader **worker별 seed 파생**(`worker_init_fn`)으로 멀티프로세스에서도 결정적.
  - 가능하면 **(샘플 인덱스, epoch)에서 aug 파라미터를 파생**(전역 순차 RNG 대신)해, aug ON/OFF·조합 A/B가 **동일한 무작위 추출 위에서** 비교되도록(비교 오염 최소화 + 완전 재현).
  - 최종 VESSL 재학습에서 동일 결과가 나오도록 seed·라이브러리 버전·데이터 순서 의존성을 리포지토리에 고정·커밋.

---

## 3.5 Test-Time Augmentation / Self-Ensemble (추론시간 트레이드오프 주의)

- **기법 (수상팀 파란코끼리, 맞는데왜틀려요)**: 추론 시 원본 + flip + 180° 회전 + (flip 후 회전) 등
  여러 변환본을 통과시킨 뒤 **역변환해 평균** → 단일 모델로 앙상블 효과.
- ⚠️ **2026 핵심 주의 — 추론시간이 채점축**: TTA는 forward pass가 N배(4-view면 ~4×)로 늘어
  **추론시간 점수를 직접 깎는다.** 학습 aug와 달리 free lunch가 **아님.**
- 권고: SSIM 이득 vs 시간 감점을 **반드시 함께 측정**. 쓰더라도 **저비용 2-view(flip만)**로 제한.
  180° 회전은 위상인코딩 축 이슈 확인 필요. 추론시간 가중치(미공개) 확인 후 채택 결정.

---

## 4. 권장 출발 조합

**Tier 1(좌우 플립 + 마스크 동기) + Tier 2(mask augmentation, 5~16× 랜덤 + 0.08 center + 커리큘럼)
+ 지수형 확률 스케줄링(첫 10~15ep off → p0.5~0.6)** 조합이
데이터·채점·하드웨어 제약에 가장 잘 맞고 위험이 낮은 시작점 — **역대 수상팀 공통 레시피와 일치**.

- 구현 위치: `utils/data/transforms.py`의 `DataTransform` (seed 고정 · 확률 스케줄 · bbox 좌표 변환 포함).

### 4.1 실험 우선순위 (먼저 켤 것)

1. **mask-aug + 좌우 flip** — 수상팀 공통·물리 유효·bbox 안전·해부학적 유효라 **최대 이득으로 예상(검증 전 가설)**. 이후 실험의 baseline.
2. **원본(고정)마스크 혼합비** 튜닝(예: 50/50 → 70/30) — test 분포 정합·OOD 방지, bbox 채점에 직결.
3. **GRAPPA 영상 prior 채널 + residual**(fft 안 함): 크기/슬라이스/스케일 정규화를 고정한 뒤 영상 prior로 A/B. data consistency에는 직접 섞지 않음.
4. **acc4/acc8 커리큘럼**: 단일배율 pretrain → 혼합 fine-tune(UnifiedCS) A/B.
5. **k-space 랜덤 위상** — 저위험이나 magnitude/RSS 불변이라 이득 marginal. ON/OFF로 실이득 확인 후만 유지.
6. **정수 translation / 라인 지터** — 소폭.
7. **consistency 정규화** — 후순위(학습 2× 비용). 1~6으로 여력 남을 때만.

각 단계는 직전 best 조합 위에 **하나만 추가하는 incremental A/B**. **full이 올라도 bbox가 떨어지면 롤백.**

### 4.2 SSIM_bbox 측정 프로토콜 (A/B 표준)

- **고정 val split**: 재조합은 허용이나 **A/B 내내 동일 split·동일 seed로 고정**(비교 오염 방지). train에 절대 안 섞인 홀드아웃.
- **K-Fold Cross Validation 구현 예정**: 최종 조합 후보는 단일 split 운에 흔들리지 않도록 K-Fold로 재평가한다. 단, aug A/B 초기 탐색은 비용 절감을 위해 고정 val split으로 빠르게 거르고, 상위 후보만 K-Fold에 올린다.
- **두 지표 분리 리포트**: `SSIM_full`(배경 마스크 후 전체) / `SSIM_bbox`(**annotation 있는 슬라이스의 bbox 내부만**; 병변 없는 슬라이스는 bbox 집계 제외).
  - ⚠️ **SSIM_full 배경 마스크 정의 미확정**(eval code 7/8 공개 예정) → 로컬 SSIM_full은 **잠정 proxy**로만 취급, 공개 후 마스크 재동기화. 불확실성에 강건한 **bbox 비퇴행 게이트를 1차 기준**으로.
- **집계 단위**: **per-lesion-bbox 평균을 1차 지표**로(작은 병변이 파일 평균에 묻히는 것 방지), per-file 평균은 보조. 파일 수가 적어 bbox 수가 통계력을 좌우함.
- **acc4/acc8 분리**: 두 가속률 각각 full·bbox 4개 숫자를 항상 함께(한쪽만 오르는 aug 감지).
- **판정 규칙(가중치 미공개 대응)**: **SSIM_bbox 비퇴행(non-regression)을 하드 게이트**로 두고, 그 조건에서 SSIM_full 최대화. bbox가 유의하게 떨어지면 full 이득과 무관하게 탈락.
- **평가 입력은 항상 제공 고정마스크**(추가 언더샘플 aug는 train에만) — test 조건 일치시켜야 bbox 숫자가 실제 순위를 예측.
- seed·split·집계 코드는 리포지토리에 커밋(재현성 검증 대비).

### 4.3 추론시간 측정 프로토콜 (채점 3축 중 하나 — 반드시 함께 측정)

- **1080 기준으로 시간 측정**(개발 A100 수치 신뢰 금지). FP32.
- `torch.cuda.synchronize()` + **warmup 몇 회 후** 측정, **ms/slice·ms/volume** 병기.
- 동일 val split·고정마스크 입력에서 측정(SSIM 측정과 동일 조건).
- **결정 규칙**: cascade·모듈은 **ΔSSIM 대비 +Δms가 임계 이상 값어치 있을 때만** 채택(예: cascade 1개 추가가 bbox/full을 유의미하게 못 올리면 제거). cascade 수·GRAPPA/FLAT 선택을 이 측정치에 묶는다(주장 아닌 숫자로).

---

## MRAugment과 Tier 1의 관계 (혼동 주의)

**Tier 1 ≠ MRAugment.** Tier 1은 MRAugment의 **부분집합**이며, 그것도 우리 제약 때문에 변형된 형태다.

- MRAugment의 핵심은 변형 종류가 아니라 **파이프라인**이다:
  `fully-sampled 이미지 변형 → FFT로 k-space 재생성 → 새 마스크 적용 → 언더샘플 입력 재시뮬레이션`.
  이 재시뮬레이션 덕분에 90° 회전·스케일·shear까지 물리적으로 유효해진다(회전시켜도 맞는 방향의 새 마스크를 붙임).
- MRAugment 변형은 두 그룹: **pixel-preserving**(플립, 90° 회전, translation) + **interpolating**(임의각 회전/스케일/shear) + **확률 스케줄링**.

| MRAugment 구성 | 우리 상태 |
|---|---|
| pixel-preserving 중 **플립(좌-우만)** | ✅ = **Tier 1** — 해부학적으로도 유효(반대쪽 무릎과 동일) |
| pixel-preserving 중 **플립(위-아래)** | ❌ Tier 3 — 픽셀은 보존돼도 **뼈 위치가 뒤바뀐 비현실적 무릎** |
| pixel-preserving 중 **90° 회전** | ❌ Tier 3 (마스크 축 붕괴) |
| **interpolating**(임의각 회전/스케일/shear) | ❌ Tier 3 (보간 아티팩트 + 재시뮬 불가) |
| 확률 스케줄링 | ✅ 채택 (§3 "보수적+스케줄링") |

- **왜 부분집합만 쓰나**: 우리는 fully-sampled k-space가 없어 **재시뮬레이션이 불가**(§1). 그래서
  회전/스케일/shear는 물리적으로 무효 → 제외. **플립만 살아남는 이유**는 플립이 좌표 뒤집기라
  **FFT와 교환(commute)**되어 재시뮬 없이 k-space에 직접 적용해도 유효하기 때문(마스크만 같이 뒤집으면 됨).
- 즉 **Tier 1 = "MRAugment 중, 재시뮬레이션 없이도 물리적으로 유효한 유일한 조각"**.

---

## Foundation 논문 교차검증 — U-Net(2015)의 augmentation은 왜 안 쓰나

원조 U-Net(Ronneberger, 전자현미경 **세그멘테이션**)의 augmentation을 우리 과제에 대조하면,
**핵심 기법이 오히려 우리 Tier 3(금지)와 정확히 겹친다.** 이는 우리 보수적 설계가 옳다는 방증.

- U-Net이 쓴 것: ①**random elastic deformation**(3×3 coarse grid, 가우시안 σ=10px, bicubic 보간)
  — 논문이 "few-shot 세그멘테이션의 **key concept**"이라 강조, ②shift/rotation, ③gray value variation, ④drop-out.

| U-Net 기법 | 우리 적용 | 근거 |
|---|---|---|
| **Elastic deformation** | ❌ **금지 (= Tier 3)** | interpolating→재시뮬 불가 + 비물리적 해부구조 → 병변에 걸리면 **SSIM_bbox 파괴=hallucination** |
| Shift | △ 부분만 | 소폭 **정수** 이동만 유효(= Tier 2). subpixel/큰 이동은 보간 필요 → 제외 |
| Rotation | ❌ | 임의각=보간+재시뮬 불가, 90°=1D 마스크 축 붕괴 |
| Gray value variation | ❌ (= Tier 3) | MRI 대비는 물리적 의미 → 병변 신호 오학습 |
| Drop-out | △ 유일 이식 가능 | 아키텍처 정규화(도메인 무관). 단 복원은 충실도 과제라 잡음처럼 작동 가능 → 신중 |

- **왜 정반대인가 (핵심)**: U-Net은 "세포가 실제로 변형되니 **deformation invariance**가 필요"해 elastic을 쓴다.
  우리는 **존재하는 해부구조를 그대로 복원**해야 하므로 deformation invariance = hallucination → 원하는 것의 정반대.
- 결론: 가장 유명한 foundation augmentation(elastic deformation)이 우리 채점 핵심(SSIM_bbox)을 직접 파괴 →
  flip+마스크 중심의 보수적 Tier 설계 + Tier 3 금지가 정당함을 재확인. (수상팀 교훈 "knee=강한 공간변형 취약"과 일치)

---

## 참고

- **비-augmentation 전략(모델·메모리·구조)**: [`model_strategy.md`](model_strategy.md)
  — 인접슬라이스 2.5D 입력(제외)·메모리효율 배치 SME(채택) 등, 논문 재검토에서 나온 non-aug 기법.
- U-Net: Ronneberger, Fischer, Brox, "U-Net: Convolutional Networks for Biomedical Image
  Segmentation", MICCAI 2015. augmentation = elastic deformation(σ=10px, 3×3 grid) + shift/rotation +
  gray value + drop-out → **세그멘테이션용이라 우리 복원 과제엔 대부분 부적용/유해.**
  PDF: `.claude/paper/00_foundation/UNet_segmentation_2015.pdf`
- **역대 수상팀 augmentation 총정리**: [`../.claude/paper/past_winners_augmentation.md`](../.claude/paper/past_winners_augmentation.md)
  (MRAugment·마스크증강·스케줄링·부위별 전략·TTA 등, 2026 knee 적용 코멘트 포함).
- MRAugment: Fabian et al., "Data augmentation for deep learning based accelerated MRI
  reconstruction with limited data", NeurIPS 2021.
- K-Space Acquisition DG: Wattad, Shor, Bronstein, "On The Role of K-Space Acquisition in
  MRI Reconstruction Domain-Generalization", arXiv 2512.06530 (2025).
  → acquisition uncertainty(궤적/라인 지터)를 augmentation으로; **k-space 교란 > image 노이즈**.
  PDF: `.claude/paper/02_reference/KspaceAcq_domain-gen_2025.pdf`
- SSIBench: Wang, McDonagh, Davies, "Benchmarking Self-Supervised Learning Methods for
  Accelerated MRI Reconstruction", arXiv 2502.14009 (2025).
  → SSL 18종 벤치마크; 우리엔 EI consistency 손실 재료 + "SSL 불필요" 판단 근거.
  PDF: `.claude/paper/02_reference/SSLBench_ssibench_2025.pdf`
