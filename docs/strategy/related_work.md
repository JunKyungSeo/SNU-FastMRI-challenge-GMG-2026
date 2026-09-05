# Related Work — 딥러닝 MRI 복원 + GRAPPA 논문 리서치 (2021–2026)

> 📚 **근거(rationale)·출처 아카이브.** 인덱스 [`README.md`](README.md) · 확정 SPEC [`confirmed_model.md`](confirmed_model.md).
> 목적: 우리 대회 세팅(**언더샘플 k-space + 제공된 GRAPPA 영상을 함께 딥모델 입력/prior로**)과
> 유사한 최근 5년(2021–2026) 논문을 조사해, 우리 전략([`model_strategy.md`](model_strategy.md) §1·§1.2,
> [`confirmed_model.md`](confirmed_model.md))의 선례·출처를 명확히 남긴다.
> 조사 방법: 다각도 웹 검색 → 원문 페치 → 주장별 3-vote 검증(2026-07-06 수행). 검증 로그는 세션 기록.

---

## 핵심 결론

**우리 대회와 완전히 동일한 패턴(언더샘플 k-space **AND** 미리 제공된 GRAPPA 영상을 함께 입력)은
논문상 매우 드물다.** 대부분은 둘 중 하나다 — ① GRAPPA 영상만 입력(k-space 없음), 또는
② GRAPPA를 네트워크 내부 레이어로 계산. 그럼에도 개념적 선례는 아래처럼 명확하며,
가장 실용적 선례는 **E2E-VarNet 출력에 GRAPPA 영상을 concat**하는 계열(1군)이다.

> ⚠️ 조사 한계: 검증 집합이 "GRAPPA를 안 쓰는 논문"으로 다소 편향(검색이 위양성 후보를 많이 반환).
> 2024–2026 최신 VarNet 변형 중 GRAPPA를 명시적 보조 채널로 받는 논문이 더 있을 수 있음(open question).

---

## 1군 — GRAPPA를 prior/초기해로 딥모델에 주입 (우리 세팅에 가장 근접)

| 논문 | 연도·venue | GRAPPA 사용 방식 | 데이터·가속 | 우리 매핑 |
|---|---|---|---|---|
| **Accelerated MRI reconstructions via variational network and feature domain learning** (Feature/FI-VarNet) — Nature *Scientific Reports* 14:8085 | 2024, Sci Rep | **E2E-VarNet 출력에 GRAPPA 영상을 concat**해 feature-domain 네트워크로 정제. k-space가 못 주는 정보를 GRAPPA가 보완 | fastMRI knee/brain | ⭐ **가장 유사** — [`confirmed_model.md`](confirmed_model.md) §5 GRAPPA prior concat와 동형 |
| **fastMRI 2020 Challenge 우승 (AIRS Medical)** — Muckley et al., "Results of the 2020 fastMRI Challenge…", *IEEE TMI* 40(9):2306, 2021 (PMC8428775 / arXiv:2012.06318) | 2021, IEEE TMI | **GRAPPA 추정치로 네트워크 초기화**(reconstruction + residual). "GRAPPA를 initialization으로 쓴 유일한 팀." **AIRS = 2026 대회 후원사** | fastMRI multicoil, 4X/8X | ⭐ [`model_strategy.md`](model_strategy.md) §1.3 우승 레시피 이식(§1·§1.2) |
| **GRAPPA-GANs for Parallel MRI Reconstruction** — Tavaf et al., arXiv:2101.03135 (ISBI 2021) | 2021 | GRAPPA RSS 영상을 conditional GAN(U-Net generator) 입력/prior로. k-space는 data-consistency loss로만 | fastMRI **brain**, R4/R8 | ⚠️ GAN=hallucination 위험 → 우리 채점(SSIM_bbox)엔 부적합, [`rejected.md`](rejected.md) 계열 |
| **DeepMRIRec** — Alam et al., arXiv:2311.13485 (St. Jude, github.com/stjude/DeepMRIRec) | 2023 | GRAPPA 복원 영상을 U-Net "enhancer"의 **유일 입력**(k-space 브랜치 없음, 순수 image-to-image) | 소아 **brain** 종양, 4x | fallback 후보(값싼 image-domain refiner). data-consistency 없음이 약점 |
| **2022 SNU FastMRI Challenge 제출작** — github.com/orange-fritters/fast-mri | 2022 | E2E-VarNet 출력 + **GRAPPA 영상 concat** + RCAN super-resolution. **바로 이 대회 계보에서 검증** | fastMRI knee | ⭐ 1군 concat 방식이 우리 대회에서 실동작한 실증 |

---

## 2군 — GRAPPA를 네트워크 내부 미분가능 연산으로 (외부 prior 아님)

| 논문 | 연도·venue | 방식 | 데이터·가속 | 우리 매핑 |
|---|---|---|---|---|
| **GrappaNet** — Sriram et al., "GrappaNet: Combining Parallel Imaging with Deep Learning for Multi-Coil MRI Reconstruction," *CVPR* 2020 (arXiv:1910.12325) | 2020, CVPR | 두 CNN 사이에 **미분가능 GRAPPA 연산자 G**를 끼워 end-to-end: `x = h(f2(G·f1(k)))`. GRAPPA를 내부에서 계산(외부 제공 아님) | fastMRI **knee**, 4x/8x | 해부학·가속 매치 최상. 아이디어(미분가능 GRAPPA)만 참고. 8GB 이식 유연성 낮음 |

---

## 3군 — VarNet 계열: GRAPPA를 의도적으로 안 씀 (우리 backbone, GRAPPA 수용하도록 개조)

| 논문 | 연도·venue | 방식 | 우리 매핑 |
|---|---|---|---|
| **E2E-VarNet** — Sriram et al., "End-to-End Variational Networks for Accelerated MRI Reconstruction," *MICCAI* 2020 (arXiv:2004.06688) | 2020, MICCAI | ESPIRiT/GRAPPA 대신 **SME 모듈로 감도맵 내부 추정**. 고가속·ACS 부족 시 사전계산 맵 열화가 동기 | ✅ 우리 백본. PDF: `.claude/paper/00_foundation/VarNet_e2e-recon_2020.pdf` |
| **FFVN** — "FFVN: An explicit feature fusion-based variational network…," *Magn. Reson. Imaging* 2022 (S0730725X22002338) | 2022 | CS+PI+DL feature fusion 변분망. GRAPPA 미사용 | feature fusion 참고 |
| **CAMP-Net** — "Consistency-Aware Multi-Prior Network…," *IEEE J-BHI* 2025 (arXiv:2306.11238) | 2025, J-BHI | ACS에서 GRAPPA류 calibration을 **암묵 학습**(명시적 GRAPPA 영상 입력 없음) | calibration prior 참고 |

---

## 제외(우리 세팅 불일치, 조사 중 위양성으로 확인)

- Continuous K-space Recovery (arXiv:2411.11282) — guidance가 zero-filled, GRAPPA는 related work만.
- Cardiac dual-domain recon (PLOS ONE 2025, pone.0313226) — GRAPPA 언급 없음.
- Multi-contrast low-field (Med Image Anal, S1361841525003792) — 학습 k-space 진행, GRAPPA 없음.
- PASS (arXiv:2604.06849) — GRAPPA는 비교 baseline으로만.
- CSGM / Robust CS-MRI (arXiv:2108.01368, NeurIPS 2021) — 생성 prior + Langevin posterior sampling, GRAPPA 없음.
- MA-RECON (S0169260723006089) — mask-aware k-space 보간, GRAPPA prior 없음.

---

## 시사점 (우리 전략에의 반영)

1. **1군 concat 방식(Sci Rep 2024 + 2022 SNU 제출작)이 가장 실용적 선례** → [`confirmed_model.md`](confirmed_model.md) §5
   "GRAPPA 영상 prior 채널 + residual"과 정확히 일치. 우리 방향이 검증된 계보임을 확인.
2. **AIRS(후원사) = GRAPPA 초기화 + residual** → [`model_strategy.md`](model_strategy.md) §1.3에 이미 이식 매핑.
3. **미개척 공백**: 서베이한 논문 중 **병변 bbox 영역 SSIM(우리 SSIM_bbox 순위축)이나 hallucination을
   직접 평가한 GRAPPA-prior 논문은 없음.** 이 지점이 대회의 open-question이자 우리의 차별화 여지.

## 기반 기법 출처 (본문 곳곳에서 참조되는 고전)

- **GRAPPA**: Griswold et al., "Generalized autocalibrating partially parallel acquisitions (GRAPPA)," *Magn. Reson. Med.* 47:1202, 2002.
- **ESPIRiT**: Uecker et al., "ESPIRiT—an eigenvalue approach to autocalibrating parallel MRI," *Magn. Reson. Med.* 71:990, 2014.
- **VarNet(원조)**: Hammernik et al., "Learning a variational network for reconstruction of accelerated MRI data," *Magn. Reson. Med.* 79:3055, 2018.
- **fastMRI 데이터셋**: Zbontar et al., "fastMRI: An Open Dataset and Benchmarks for Accelerated MRI," arXiv:1811.08839, 2018.
- **fastMRI+ (병변 bbox 주석)**: Zhao et al., "fastMRI+: Clinical pathology annotations…," *Scientific Data* 9:152, 2022.
- **SSIM(채점 지표)**: Wang et al., "Image quality assessment: from error visibility to structural similarity," *IEEE TIP* 13:600, 2004.
