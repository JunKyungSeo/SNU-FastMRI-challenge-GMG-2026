# 결정 로그 (Decisions) — 열린 후보 + 제외

> A/B 판정 원장의 앞뒷면을 한 파일에 모은다.
> - **[Part 1 — 후보 (Candidates)](#part-1--후보-candidates)**: [`pipeline.md`](pipeline.md) 확정 파이프라인 **위에** 얹을지 검증할 열린 항목. 아직 spec 아님.
> - **[Part 2 — 제외 (Rejected)](#part-2--제외-rejected)**: 안 쓰기로 한 것 + 이유. 재논의 방지용 로그.
> 근거 문서는 [`research/`](research/) (model_strategy·augmentation·preprocessing·related_work).

---

# Part 1 — 후보 (Candidates)

> [`pipeline.md`](pipeline.md)의 확정 파이프라인 **위에** 얹을지 A/B로 검증할 항목들. 아직 spec 아님.
> 판정 공통 규칙: **SSIM_bbox 비퇴행을 하드 게이트**로, 그 조건에서 SSIM_full·추론시간 최적화([`research/augmentation.md`](research/augmentation.md) §4.2·§4.3). 직전 best 위에 **하나만 추가하는 incremental A/B**.

## 즉시 A/B (확정 구조에서 바로)
| 항목 | 무엇 | 게이트 | 근거 |
|---|---|---|---|
| **입력crop vs 출력crop** | readout crop을 입력단(확정) vs 출력단(현행)에 둘 때 비선형 U-Net 문맥 차이 | SSIM 동등 확인(이론상 차 작음, readout 완전샘플이라 alias 유입 없음) | pipeline 실측 |
| **GRAPPA 상보 융합** | `[image_input, image_grappa]` 2채널 입력(noise-faithful ⊕ alias-free) | acc8 bbox 비퇴행 | model_strategy §1.2 ② |

### acc-conditioned GRAPPA·k-space 가중 (network 융합, 후보)
> **동기**: acc4 GRAPPA는 깨끗하나 acc8 GRAPPA는 g-factor 노이즈 증폭([model_strategy §1.2 ②](research/model_strategy.md)). → acc4는 GRAPPA를 더 믿고, acc8은 k-space·DC에 더 의존하도록 **가속률별로 GRAPPA·k-space 융합 가중을 다르게**. 특히 acc8 병변부(SSIM_bbox 순위축)에서 노이즈 GRAPPA 과신 방지.

- **acc 판단 = mask에서(파일명 X, 추론서도 robust)**: 파일마다 sample rate가 미세하게 달라 **딱 숫자 대신 임계값**으로 이진 판정.
  - **규칙**: `frac = sampled/total`. **`frac ≤ 0.25` → acc8(8배), 아니면 → acc4(4배).**
  - 근거(실측): acc4 frac≈0.310, acc8 frac≈0.199 → 임계 0.25는 양쪽 여유 ≈0.06(파일별 미세 변동 흡수).
  - ⚠️ **임계값 재조정 필요**: 위 0.25는 **샘플 2개 파일(acc4_1·acc8_1)** 실측 기반. **전체 데이터셋 접근 가능해지면 두 그룹 frac 분포를 재측정해 임계값을 다시 정할 것**(파일별 미세 변동이 크면 두 분포가 겹치는지 확인, 필요시 ACS 제외 주변부 factor로 판정 방식 자체 재검토).
- **가중 방식(권장)**: 하드코딩(acc4=0.8 등) 대신 **acc 신호를 conditioning으로 주고 network가 학습**(FiLM 또는 GRAPPA prior 채널 학습 게이트 `g=σ(MLP(acc))`, [§1.2 ③ anti-copy 게이팅](research/model_strategy.md#L85)). 한 모델로 acc4/acc8 둘 다 커버.
  - 출처(FiLM): **Perez, Strub, de Vries, Dumoulin, Courville, "FiLM: Visual Reasoning with a General Conditioning Layer," *AAAI* 2018 (arXiv:1709.07871).** feature를 조건 스칼라로 affine 변조(γ·x+β) → acc 조건화에 경량 적합.
- **데이터 측 지원(미구현)**: mask에서 acc 스칼라/라벨을 뽑아 transform 출력에 실어주면(`return_acc` 게이트 예정) 동료가 융합에서 바로 사용. 실제 가중 로직은 network(varnet.py) 쪽.
- **게이트**: acc8 SSIM_bbox 비퇴행 우선 확인.

## 손실 함수 (순위축 직공략 — 1순위 후보, 아직 미확정)
> ⚠️ 선결: `DataTransform`이 annotation을 emit해야 함(현재 bbox는 `return_bbox=True`로 파싱만, 손실 소비부 미배선).

| 항목 | 무엇 | 우선순위 | 근거 |
|---|---|---|---|
| **bbox 가중 손실** | `L=(1−SSIM_full)+λ(1−SSIM_bbox)`, 병변영역 가중 | **1순위** | model_strategy §2.1 (MOST) |
| **병변 슬라이스 오버샘플링** | 병변 슬라이스 샘플 확률↑ (픽셀 미변형, 위험 0) | 1순위(저위험) | model_strategy §2.5 |
| 중간 감독(deep supervision) | cascade 중간 출력에도 bbox 손실 | 보조 | §2.2 (출처: **Lee, Xie, Gallagher, Zhang, Tu, "Deeply-Supervised Nets," *AISTATS* 2015**) |
| 고주파 집중 손실 | 미획득 PE 라인/고역통과에 손실 가중 | 실험 후보 | §2.4 (⚠ hallucination 양날) |
| foreground 마스크 손실 | 배경 제외 영역에서만 SSIM_full (채점 정합) | **eval code 대기** | preprocessing B5 |

## 효율 레버 A군 (SME 비용 절감 — 택1/AB, 동시적용 금지)
> 아래 4개는 **목적이 겹쳐(sens/코일/해상도/cascade 비용 절감) 동시적용 금지**, 하나만 택해 A/B.

| 항목 | 무엇 | 상태 | 근거·출처 |
|---|---|---|---|
| **배치 SME** | 인접 2a+1 슬라이스 감도맵 배치 추정, GPU ≈55%↓ | 잠정채택(spec 미반영) | model_strategy §3.2 (PromptMR+: `.claude/paper/01_candidates/PromptMRplus_rethink-dum_2024.pdf`) |
| **coil compression (SVD)** | 15→8~10 가상코일 SVD 압축 | ✅ **데이터측 구현(2026-07-06)**, 배치SME와 택1 | preprocessing A2 · related_work. SVD: **Buehrer et al., MRM 2007**; GCC: **Zhang et al., MRM 2013**. SSIM neutral(arXiv:2411.05883). ⚠ 과압축시 bbox↓ |
| progressive resolution | ACS 저해상도 앵커에서 고주파 점진 복원 | VRAM 부족 시 | model_strategy §4.1 (UEPS-P: `.claude/paper/01_candidates/UEPS_efficient-recon_2026.pdf`) |
| FLAT cascade 다이어트 | 중간감독으로 cascade 6→4 축소, 추론시간↓ | 추론시간 부족 시 | §4.2 (FLAT: `.claude/paper/01_candidates/FLAT_prob-flow_2025.pdf`) |

## 효율 레버 B군 (A군과 직교 — 조합 가능, "SSIM 보존하며 속도↓")
> A군과 달리 **학습 전략(distillation)·배포 단계(TensorRT)**라 A군 어느 것과도 곱해 쓸 수 있다. frontier를 미는 레버(2026-07-06 논의).

| 항목 | 무엇 | 상태 | 근거·출처 |
|---|---|---|---|
| **Knowledge distillation** | 서버서 큰 teacher(cascade 多) 학습 → 작고 빠른 student가 teacher 출력까지 모방 → **작은 student가 큰 SSIM 흡수**. 최종 제출은 student(재현성: student만 서버 재학습) | 추론시간·SSIM 동시 개선 후보 | **Hinton, Vinyals, Dean, "Distilling the Knowledge in a Neural Network," arXiv:1503.02531 (NeurIPS 2014 DL Workshop).** MRI 복원 적용례 다수 |
| **TensorRT / 그래프 최적화** | 학습 후 추론엔진 재컴파일(layer fusion·커널 오토튜닝). **결과 수치 동일, 실행만↓** | 배포 마감 단계 | NVIDIA TensorRT SDK (라이브러리, 논문 아님). ⚠ **1080=Pascal**: Tensor Core 부재로 FP16/INT8 이득 없음 → **FP32 fusion 수준의 완만한 이득만** |

## 증강 추가 (저위험, marginal)
| 항목 | 무엇 | 우선순위 | 근거 |
|---|---|---|---|
| 원본마스크 혼합비 | 무증강 고정마스크 혼합 50/50→70/30 튜닝 | 2순위 | augmentation.md §4.1 |
| acc4/acc8 커리큘럼 | 단일배율 pretrain→혼합 fine-tune | 4순위 | §2 UnifiedCS |
| k-space 랜덤 위상 | magnitude 불변, robust(이득 marginal) | 5순위 | §2 (기본 off) |
| 라인 지터 | 샘플 라인 인덱스 소폭 이동 | 6순위 | K-Space DG |
| coil 노이즈 | 채널별 보수적 노이즈(강도 보수적) | 저순위 | §2 (⚠ 저대비 병변) |
| consistency 정규화 | `‖f(g·y)−g·f(y)‖`, 학습 2× 비용 | 후순위 | §2 보조 (EI) |

## 대기 (외부 조건 필요)
| 항목 | 무엇 | 필요 조건 |
|---|---|---|
| SSIM_full 배경 마스크 정의 | 채점 정합 마스크 확정 | eval code 7/8 공개 |
| TGVN (2nd contrast) | side-info 기반 재구성 | PD/PDFS 쌍 제공 여부 확인 (미확정) |
| coil noise pre-whitening | 노이즈 공분산 whitening (acc8 안정) | 코너/배경 추정 정확도 | preprocessing B3 |
| ACS-only 저해상도 앵커 채널 | 3번째 성질 prior 채널 | 여력 시(GRAPPA로 강등) | preprocessing C6 |

---

# Part 2 — 제외 (Rejected)

> 재논의 방지용 로그. **왜 제외했는지**를 함께 남겨, 나중에 같은 아이디어가 다시 나와도 근거로 바로 거른다.
> 대부분 **SSIM_bbox(순위축) 파괴 / hallucination 위험 / 물리적 무효 / 우리 데이터 제약** 중 하나.

## 기하 변형 증강
| 항목 | 제외 이유 | 출처 |
|---|---|---|
| **위-아래(readout) flip** | 관상면 무릎에서 넙다리뼈↔정강이뼈가 뒤바뀜 → 실존하지 않는 해부구조(hallucination 유사). 픽셀 보존이라도 무효 | augmentation.md Tier 3 |
| **정수 픽셀 translation** | target은 384 center-crop 제공인데 모델은 640×368에서 crop → 640 도메인 shift가 384 target과 정렬 불가(실측 relerr≈0.44). flip은 commute하나 translation은 안 됨 | augmentation.md §Tier2, transforms.py 주석 |
| **임의각 회전·스케일·shear** | 보간 아티팩트 + fully-sampled 없어 재시뮬 불가 → 물리적 무효 | augmentation.md Tier 3 |
| **90° 회전·전치** | 1D 위상인코딩 마스크의 물리적 의미(축) 붕괴 | augmentation.md §1 |
| **elastic deformation** | interpolating→재시뮬 불가 + 병변에 걸리면 SSIM_bbox 파괴=hallucination. (U-Net의 key aug지만 우리 복원 과제엔 정반대) | augmentation.md Tier 3, Foundation 교차검증 |

## 강도·노이즈 증강
| 항목 | 제외 이유 | 출처 |
|---|---|---|
| **image 도메인 노이즈** | K-Space DG 실측: image 노이즈는 일반화 **악화**(U-Net −1.5dB). k-space 교란이 우월 | augmentation.md Tier 3 |
| **밝기/대비 jitter·gamma** | MRI 대비는 물리적 의미 → 왜곡 시 병변 신호 오학습→hallucination | Tier 3 |
| **cutout / random erasing** | 병변 위에 걸리면 SSIM_bbox 파괴 | Tier 3 |
| **MixUp / CutMix** | 병변 bbox 위에 걸리면 SSIM_bbox 파괴(회피 로직 없이는 금지). 2025는 bbox 비중 작아 썼을 뿐 | Tier 3 |
| **naive 합성병변 삽입** | 입력에만 삽입 = hallucination 그 자체. (안전형=k-space consistent paired지만 fully-sampled 없어 실현성 낮음) | Tier 3 |
| **전역 스칼라 곱 aug** | 병변 대비 왜곡 가능 → 2026 bbox 채점상 보류(정규화 ÷s와 혼동 금지, 그건 확정) | Tier 3 예외 |

## 입력·구조
| 항목 | 제외 이유 | 출처 |
|---|---|---|
| **GRAPPA → fft → k-space** | GRAPPA는 magnitude·단일코일 384² → fft하면 위상·코일 날조된 가짜 k-space. cascade regularizer는 영상 도메인 동작이라 영상 prior가 정합 (2026-07-04 확정) | pipeline, model_strategy §1 |
| **완전 384²(폭까지 pad)** | 폭 368/372→384 image-pad 후 fft하면 k-space 1.35배 번짐(실측) → phase-encode mask·DC 오염. readout(높이)만 crop해야 함 | pipeline 실측 |
| **GRAPPA로 x0 대체** | magnitude-only 384² vs VarNet x0(complex multi-coil ~640×368). 위상 합성+coil 확장 필요→ill-posed·fabrication 위험 | model_strategy §1 (a) |
| **2.5D 인접슬라이스 입력 스태킹** | bbox 내부 인접 상관 낮음(0.08~0.59) → 병변 blur→SSIM_bbox 악화. (배치 SME §3.2와 구분: 그건 메모리 최적화라 채택) | model_strategy §3.1 |
| **coil 배치화(UEPS UE)** | 15코일 knee에서 메모리 되레↑ 가능 → progressive(P)만 조각 이식 | §4.1 |

## 손실·기타
| 항목 | 제외 이유 | 출처 |
|---|---|---|
| **Gibbs 제거·apodization·k-space 저역필터** | edge=병변 미세구조=고주파를 뭉갬 → SSIM_bbox 직접 손해(§2.4 고주파 집중과 정면충돌) | preprocessing 제외 |
| **SSL(self-supervised) 채택** | GT(image_label) 제공 → supervised가 oracle 상한. SSL 불필요(EI consistency 재료로만) | augmentation.md §2 보조 |
| **고비용 TTA(4-view+)** | 추론시간 채점축 직접 감점. 쓰더라도 저비용 2-view(flip)만 조건부 | augmentation.md §3.5 |

> 참고: **조건부 재검토 가능** 항목 — 정수 translation·2.5D는 데이터 더 확보 시 병변 bbox A/B로 재검토 여지(현재 근거로는 제외). 나머지는 원칙적 제외.
