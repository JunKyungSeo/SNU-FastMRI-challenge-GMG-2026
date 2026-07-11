# Preprocessing 전략 — FastMRI 2026

> 📚 **근거(rationale) 문서.** 인덱스 [`README.md`](README.md) · 확정 SPEC [`confirmed_model.md`](confirmed_model.md) · 후보 [`candidates.md`](candidates.md) · 제외 [`rejected.md`](rejected.md).
> 📌 **A1(readout crop)·B4(전역 스칼라 정규화)는 확정 채택 → [`confirmed_model.md`](confirmed_model.md).** A2·**B5는 구현됨**(플래그 off, 검증 전), B3·C6은 미구현 후보 → [`candidates.md`](candidates.md). 아래는 그 근거.
> augmentation([`augmentation.md`](augmentation.md))·모델/메모리([`model_strategy.md`](model_strategy.md))와 별개로,
> **입력 데이터를 네트워크에 넣기 전 결정적으로 가공하는 preprocessing**만 모은 문서.
> 여기 항목들은 **현재 코드·다른 전략 문서 어디에도 없는(미구현·미계획) 신규 후보**다.
> 대상 제약: knee 전용 · SSIM_full + **SSIM_bbox(순위축)** + 추론시간 채점 · GTX 1080 8GB(FP32) · external/pretrained 금지.

---

## 0. 현재 파이프라인의 preprocessing 실태 (출발점)

- `utils/data/transforms.py` `DataTransform.__call__`: 언더샘플 k-space에 **마스크만 곱해** `(real, imag)` 텐서로 변환. **명시적 스케일 정규화 없음**(정규화는 `varnet.py`의 `NormUnet.norm` group-norm이 cascade 내부에서만 흡수).
- coil(예 15개)·readout FOV(height 640)를 **원본 그대로** 모든 cascade·sens_net에 통과 → 이것이 8GB·추론시간의 실제 병목.
- `image_grappa`는 **항상 제공**(2026-07-03 확인) — §1(GRAPPA 주입, `model_strategy.md`)은 조건부가 아니라 **무조건 채택 전제**로 봐도 됨. 아래 C군은 그만큼 후순위.
- 출력단은 `ifft2c → complex_abs → rss → center_crop(640→384 height / 368→384 width)`(varnet.py:244-245).

**한 줄 요약**: 지금은 "마스크만 곱해서 640×368×15coil을 통째로" 넣는다 → **A군(무손실 다이어트)로 넣기 전에 줄일 여지가 크다.**

---

## TL;DR — 후보 ↔ 3채점축 매핑 (별점 = 검증 전 상대 가이드)

| preprocessing 후보 | SSIM_full | SSIM_bbox | 추론시간 | 8GB | 무손실? | 비고 |
|---|---|---|---|---|---|---|
| **A1. Readout oversampling 조기 제거** | — | — | ⭐⭐⭐ | ⭐⭐⭐ | ✅ 완전 무손실 | ✅**확정**, §A1 |
| **A2. Coil compression(SVD/GCC)** | △ | ⚠ 과압축시↓ | ⭐⭐⭐ | ⭐⭐⭐ | 준무손실 | §3.2 SME와 택1/AB, §A2 |
| B3. Coil noise pre-whitening | ⭐ | ⭐(acc8) | — | — | 근사 | §B3 |
| **B4. 전역 스칼라 intensity 정규화** | ⭐ | — | — | — | 무손실 | ✅**확정**, §B4 |
| **B5. eval 정합 foreground 마스크 손실·검증** | ⭐⭐ | ⭐ | — | — | — | ✅**구현됨**(off, 검증전), §B5 |
| C6. ACS-only 저해상도 앵커 채널 | △ | △ | ⚠(채널↑) | — | 무손실 | GRAPPA 상시제공으로 강등, §C6 |

**우선순위**: **A1 → (A2 ↔ §3.2 SME 택1) → B5 → B4·B3 → C6(여력시).**
A1은 무위험·무손실이라 즉시 후보, 나머지는 SSIM_bbox 비퇴행 게이트([`augmentation.md`](augmentation.md) §4.2) 하에서 A/B.

---

## A. 무손실 효율화 (8GB·추론시간 직격, 화질 손해 0)

### A1. Readout(주파수인코딩=height) oversampling 조기 제거 ✅ **확정 채택** (2026-07-04)

- **관찰**: k-space는 640×368인데 image는 384². [`model_strategy.md`](model_strategy.md) §1.1.1에서 **"384² = 640→384 height crop + 368→384 width pad"**로 실측 확인(`ifft2c→RSS→center_crop`이 `image_input`과 상관 0.975). 언더샘플은 **width(PE) 1축만**(마스크 shape `(368,)`/`(372,)`).
- **핵심 사실**: height(readout)는 **완전샘플**이고 640행 중 **256행은 해부구조 밖 oversampling FOV**라 어차피 출력단 `center_crop(640→384)`에서 버려진다(varnet.py:245).
- **아이디어**: 이 crop을 **네트워크 끝이 아니라 입력 preprocessing에서** 수행.
  - readout(height)축만 1D iFFT → FOV crop(640→384, 안전 여유 두려면 400) → 필요시 다시 1D FFT(하이브리드 유지).
  - 완전샘플 축이라 **정확 가역 = 완전 무손실**. PE 마스크·DC(원 언더샘플 라인)는 **그대로 유지**.
- **효과**: 모든 cascade·sens_net이 640이 아니라 384(또는 400) height에서 동작 → **연산·VRAM ≈40% 절감**. 15coil × cascade × `sens_expand`/`sens_reduce` 전부에 곱해지는 축을 줄이는 것 → 추론시간 직접 단축 + 8GB에서 **채널/cascade 재투자 여력** 확보(§3.2 배치 SME보다 크고 무위험).
- **주의(정합)**:
  - 출력단 `center_crop(640,384)`을 조기 crop과 **정합**시켜 이중 crop 방지(조기 crop 후엔 height가 이미 384/400).
  - width는 **건드리지 않음**(PE 언더샘플 축).
  - flip 증강([`augmentation.md`](augmentation.md) Tier 1)은 crop과 commute하므로 안전 — 조기 crop 후에도 논리 동일. mask 증강도 width축이라 무관.
  - GRAPPA 채널(§1)은 이미 384² → 조기 crop된 격자와 **오히려 정합이 쉬워짐**(640 pad 불필요, `model_strategy.md` §1.1.1의 zero-pad 뉘앙스 완화).
- **채점축**: 추론시간 ⭐⭐⭐ / 8GB ⭐⭐⭐ / SSIM 완전 중립(무손실). **문서에 없으면서 최대 지렛대.**

### A2. Coil compression (SVD / geometric coil compression, GCC) — ✅ **SVD 데이터측 구현됨** (2026-07-06)

- **아이디어**: 15코일 → 가상 8~10코일로 **선형 압축**. ACS(중앙 완전샘플) 블록에서 SVD 도출. sens_net·모든 cascade의 coil 차원이 그대로 축소.
- **출처(논문)**:
  - **SVD coil compression (SCC)**: Buehrer, Pruessmann, Boesiger, Kozerke, "Array compression for MRI with large coil arrays," *Magn. Reson. Med.* 57:1131, 2007. (원조; ACS/calibration에서 SVD 좌특이벡터로 코일 기저 압축.) 관련: Huang et al., "A software channel compression technique for faster reconstruction with many channels," *Magn. Reson. Imaging* 26:133, 2008.
  - **GCC (geometric-decomposition coil compression)**: Zhang, Pauly, Vasanawala, Lustig, "Coil compression for accelerated imaging with Cartesian sampling," *Magn. Reson. Med.* 69:571, 2013. (readout 방향 위치별 SVD + 정렬 → 가상코일 감도 smooth. SVD보다 재구성 오차 낮음: Wang et al., *CMMM* 2017.)
- **SSIM 실측 근거**: 딥러닝 복원에서 coil compression은 **SSIM "상승"이 아니라 "거의 동등(neutral)"** — 무압축 대비 SSIM 동등, PSNR만 소폭(≈−0.45 dB) 손실(Benchmarking 3D multi-coil NC-PDNet, arXiv:2411.05883, 2024). ⇒ **속도/VRAM 레버**로만 자리매김(화질 레버 아님). SVD의 SNR 집중이 고노이즈(acc8)에서 미미하게 도울 여지는 있으나 신뢰 상승 요인 아님.
- **효과**: coil은 메모리·연산의 곱셈 인자 → VRAM·추론시간 동시 절감. **A1과 곱셈적으로 결합**(height↓ × coil↓).
- **구현(2026-07-06)**: [`utils/data/transforms.py`](../utils/data/transforms.py) `coil_compress()`(SVD/SCC, per-slice, ACS 기반) + `DataTransform`의 `coil_compress`/`n_virtual_coils` 플래그(기본 off). `readout_crop` 직후·`normalize` 직전 삽입. **추론 안전성**: W를 **ACS에서만** 도출(추론 시 완전샘플은 ACS뿐 → train/test 일관). 미샘플 라인(0) 보존 → **DC 그대로 성립**(실측: 누설 0.00e+00). 실측(realistic 상관 코일): Nvc=8 에너지보존 99.999%·RSS corr 1.0. GCC는 추후 후보.
- **주의(SSIM_bbox)**: 과압축은 **저SNR 병변 신호를 소수 성분에 묻어 소실**시킬 수 있음(순위축 직접 손해). → 보수적으로(예: 8~10/15) 시작, **bbox 비퇴행 게이트로 배율 스윕**.
- **관계**: [`model_strategy.md`](model_strategy.md) §3.2 배치 SME와 **목적(SME 메모리 절감)이 겹침** → **동시 적용 말고 택1 또는 A/B**(코드도 상호배타 전제). §4.1 UEPS의 코일 배치화 리스크(15코일 knee에서 되레↑)와도 대비 — coil compression은 그 리스크를 **선형·사전계산으로 회피**하는 대안.
- **채점축**: 추론시간·8GB ⭐⭐⭐ / SSIM_full △(neutral) / SSIM_bbox ⚠(배율 의존).

---

## B. 정규화·측정품질 (화질·재현성)

### B3. Coil noise pre-whitening (디코릴레이션)

- **출처(논문)**: Roemer, Edelstein, Hayes, Souza, Mueller, "The NMR phased array," *Magn. Reson. Med.* 16:192, 1990 (노이즈 공분산 기반 최적 코일 결합의 원조). 실무 프로토콜: Kellman & McVeigh, "Image reconstruction in SNR units," *Magn. Reson. Med.* 54:1439, 2005 (noise-only 스캔으로 whitening → SNR 단위 복원).
- **아이디어**: 신호 없는 영역(영상 코너 / 외곽 k-space)에서 **coil 노이즈 공분산을 추정 → whitening 변환**을 coil에 적용. RSS·sensitivity 추정이 통계적으로 최적화됨.
- **왜 우리 데이터에**: acc8은 g-factor 노이즈 증폭이 심함([`model_strategy.md`](model_strategy.md) §1.2 ②에서 GRAPPA 약점으로 지적) → whitening이 **acc8 SSIM_bbox 안정화**에 특히 유효 가능.
- **제약**: 전용 noise-only scan은 없음 → **코너/배경 영역 추정으로 근사**(정확도 제한). 중간 난이도.
- **채점축**: SSIM_full ⭐ / SSIM_bbox ⭐(acc8) / 추론시간 무해(사전계산).

### B4. 전역 스칼라 intensity 정규화 (명시적 스케일 고정) ✅ **확정 채택** (2026-07-04)

- **현황**: 로더에 **k-space 스케일 정규화가 전혀 없음**. group-norm(varnet)이 cascade별로만 흡수 → 파일 간·acc4/acc8 간 스케일 편차가 DC 항·손실에 그대로 노출.
- **확정 방식([`confirmed_model.md`](confirmed_model.md))**: **입력 유도 전역 스칼라 1개**(zero-filled RSS 영상의 robust percentile, 예 p99)로 **k-space·GRAPPA·target 동일 ÷s**, 출력 `×s`. **가속률 무관 단일 규칙**. 코일별 아님(전역) → sens map 무해.
  - ⚠️ `maximum` attr은 **입력 정규화에 미사용**(label 통계라 추론 시 없음 → 학습·추론 불일치). SSIM data_range로만 유지하되 target ÷s에 맞춰 `maximum÷s` 정합.
- **효과**: DC·손실 스케일 안정 + acc4/acc8 스케일차 흡수 + GRAPPA↔입력 스케일 정합(residual 성립) + **FP32 재현성↑**. 저비용·무손실.
- **위치**: `DataTransform.__call__`(증강 후·`to_tensor` 전), 학습·추론 공통 경로. 스칼라 `s` 저장해 출력단 역정규화(384² 채점 crop과 혼동 금지).

### B5. eval 정합 foreground 마스크 손실·검증 ✅ **구현됨** (2026-07-11, @esel34)

- **동기**: SSIM_full은 **배경을 마스크로 날린 영상 안에서만** 채점. 기존 손실·검증은 전체 384²에 걸려 **채점과 불일치** → 배경(쉬운 영역)이 손실/지표를 희석, val이 실채점을 예측 못 함.
- **정의 확정(핵심 변화)**: 채점 코드가 고정·공개돼(`utils/common/metrics.py:48-56` `foreground_mask` = `target>2e-5` → erode1/dilate15/erode14) **근사 없이 그대로 재사용**. target 기반이라 학습 시 정확 계산 가능 → 예전의 "마스크 정의 미확정" 걸림돌 해소.
- **구현(2026-07-11)**: `--fg-loss` 플래그(기본 off). 데이터측 `DataTransform.return_fg_mask`가 **÷s 정규화 전 raw target**에서 `foreground_mask` 계산해 튜플로 전달; `SSIMLoss.forward(mask=...)`가 `ssim_full` 정합(입력 마스킹→win//2 valid 가중평균); `train_epoch`·`validate` 소비. **검증**: masked-loss == `1−ssim_full`(diff 2.4e-08), off=byte-identical. 상세: [`implementation.md`](implementation.md) §3c.
- **관계**: [`model_strategy.md`](model_strategy.md) §2.1 bbox 가중 손실과 자연 결합(`L = (1−SSIM_full_masked) + λ(1−SSIM_bbox)`).
- **남은 것(검증 전)**: 실제 화질 이득은 **matched-epoch A/B(bbox 비퇴행 게이트)** 로만 확정. normalize와 조합 미검증(현 regime은 정규화 off).
- **채점축**: SSIM_full ⭐⭐ / SSIM_bbox ⭐ / 추론시간 무해.

---

## C. bbox·저비용 prior (GRAPPA 상시 제공으로 후순위)

### C6. ACS-only 저해상도 앵커를 입력 채널로

- **아이디어**: 중앙 ACS 블록만 iFFT → **alias 없는 저해상도 clean 영상**을 보조 입력 채널로.
- **GRAPPA 상시 제공으로 강등**: GRAPPA가 ACS 캘리브레이션으로 만든 **상위호환 prior**(저해상도+고주파 교정)라 백업 명분은 사라짐. 남는 유일한 이유 = **성질이 다름**: GRAPPA는 g-factor 노이즈 증폭(acc8), ACS 앵커는 완전샘플 구간만 써 **노이즈 깨끗·저해상도** → [`model_strategy.md`](model_strategy.md) §1.2 ②의 상보 융합(`image_input ⊕ image_grappa`)에 **3번째 성질의 채널**로 추가하는 실험 카드.
- **주의**: 채널만 늘리는 marginal 이득 + 추론시간 소폭↑ → **여력 있을 때만**. §4.1 progressive(ACS에서 출발)와 물리적 근거는 공유하나 그건 아키텍처, 이건 입력 채널화로 구분.
- **채점축**: SSIM △ / 추론시간 ⚠(채널↑).

---

## 안 하는 게 나은 것 (제외)

- **Gibbs ringing 제거 / apodization / k-space 저역 필터**: 실제 edge(=병변 미세구조=고주파)를 뭉갬 → **SSIM_bbox 직접 손해**. [`model_strategy.md`](model_strategy.md) §2.4(고주파 집중 손실)와 정면충돌. 제외.
- **임의각 회전·스케일·elastic 등 기하 변형**: preprocessing이 아니라 aug 영역이며 이미 [`augmentation.md`](augmentation.md) Tier 3에서 금지(재시뮬 불가·병변 왜곡).

---

## 참고 (교차 링크)

- 증강(on-the-fly 데이터 변형): [`augmentation.md`](augmentation.md)
- 모델·메모리·구조(§1 GRAPPA 주입 / §3.2 배치 SME / §4.1 progressive·UEPS / §2.4 고주파 손실): [`model_strategy.md`](model_strategy.md)
- 병변 bbox 활용 총정리·측정 프로토콜: `model_strategy.md` §2.5 / `augmentation.md` §4.2·§4.3
- 구현 지점: `utils/data/transforms.py`(`DataTransform`) · `utils/data/load_data.py` · `utils/model/varnet.py`
