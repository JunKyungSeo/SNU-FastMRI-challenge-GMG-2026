# 제외 (Rejected) — 안 쓰기로 한 것 + 이유

> 재논의 방지용 로그. **왜 제외했는지**를 함께 남겨, 나중에 같은 아이디어가 다시 나와도 근거로 바로 거른다.
> 대부분 **SSIM_bbox(순위축) 파괴 / hallucination 위험 / 물리적 무효 / 우리 데이터 제약** 중 하나.

---

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
| **GRAPPA → fft → k-space** | GRAPPA는 magnitude·단일코일 384² → fft하면 위상·코일 날조된 가짜 k-space. cascade regularizer는 영상 도메인 동작이라 영상 prior가 정합 (2026-07-04 확정) | confirmed_model, model_strategy §1 |
| **완전 384²(폭까지 pad)** | 폭 368/372→384 image-pad 후 fft하면 k-space 1.35배 번짐(실측) → phase-encode mask·DC 오염. readout(높이)만 crop해야 함 | confirmed_model 실측 |
| **GRAPPA로 x0 대체** | magnitude-only 384² vs VarNet x0(complex multi-coil ~640×368). 위상 합성+coil 확장 필요→ill-posed·fabrication 위험 | model_strategy §1 (a) |
| **2.5D 인접슬라이스 입력 스태킹** | 🔁 **2026-08-01 기각 철회(근거 반증)**. 옛 사유였던 "bbox 내부 상관 0.08~0.59 → 병변 blur"는 **대조군 없는 파일 2개 표본**이었다. 170파일·박스 7394개 재측정에서 병변 bbox 상관이 **같은 크기 대조군보다 +0.12 높고** gap1 SSIM은 동일 → 병변 특이 blur 근거 없음. 단 **a=1(3슬라이스) 한정**(gap2↑는 병변 SSIM이 대조군 아래로 뒤집힘). 이득은 **실측 결과 노이즈 수준**(shell30: ValLoss −0.0009, 부호 뒤집힘) 인데 비용은 학습·추론 1.9× → **채택 보류**. 배선은 유지(기본 1=비트 동일). (배치 SME §3.2와 구분: 그건 메모리 최적화라 별도 채택) | model_strategy §3·§3.1 |
| **coil 배치화(UEPS UE)** | 15코일 knee에서 메모리 되레↑ 가능 → progressive(P)만 조각 이식 | §4.1 |

## 손실·기타
| 항목 | 제외 이유 | 출처 |
|---|---|---|
| **Gibbs 제거·apodization·k-space 저역필터** | edge=병변 미세구조=고주파를 뭉갬 → SSIM_bbox 직접 손해(§2.4 고주파 집중과 정면충돌) | preprocessing 제외 |
| **SSL(self-supervised) 채택** | GT(image_label) 제공 → supervised가 oracle 상한. SSL 불필요(EI consistency 재료로만) | augmentation.md §2 보조 |
| **고비용 TTA(4-view+)** | 추론시간 채점축 직접 감점. 쓰더라도 저비용 2-view(flip)만 조건부 | augmentation.md §3.5 |

> 참고: **조건부 재검토 가능** 항목 — 정수 translation·2.5D는 데이터 더 확보 시 병변 bbox A/B로 재검토 여지(현재 근거로는 제외). 나머지는 원칙적 제외.
