# 갭 & 사각지대 (Gaps & Blind Spots) — FastMRI 2026

> 작성: 2026-07-07 · 코드(`utils/`)와 전략 문서 전체를 대조한 감사(audit) 결과.
> 목적: 지금까지의 전략이 **놓치고 있는 두 층위**를 한곳에 모은다.
> - **Part A — 구현 정합성 갭**: 문서엔 있으나 코드에 없거나, 코드가 문서와 어긋난 부분(사실 확인됨).
> - **Part B — 전략적 사각지대**: 우리 사고 틀 자체가 인지하지 못한, 대회의 본질·승부 아이디어(가설·검증 필요).
> 각 항목은 **무엇을 놓쳤나 · 왜 중요한가(이유) · 방향**으로 적는다.
> 관련: SPEC [`confirmed_model.md`](confirmed_model.md) · 후보 [`candidates.md`](candidates.md) · 배선 [`implementation.md`](implementation.md).

---

# Part A — 구현 정합성 갭 (코드로 확인됨)

> 전략 문서는 **입력 데이터 처리**엔 촘촘하나, **"우리가 이기려는 숫자(SSIM_bbox)를 실제로 계산·최적화·모니터링하는 루프"**와 **통합 실행**이 비어 있다. 아래는 코드 실측 근거와 함께.

## A1. 🔴 검증 루프가 실제 채점 지표를 안 쓴다 (순위축을 학습 내내 못 봄)

- **무엇**: [`train_part.py:74`](../utils/learning/train_part.py#L74) `validate()`는 `ssim_loss(targets, reconstructions)`([`utils.py:35`](../utils/common/utils.py#L35))로 채점한다 — **배경 포함 전체 384² 평범한 SSIM**. 그런데 [`metrics.py`](../utils/common/metrics.py)에는 **진짜 리더보드 채점 함수**(`foreground_mask`·`ssim_full`·`ssim_bbox`)가 이미 구현돼 있는데 **train_part.py가 import조차 안 한다.**
- **이유(왜 치명적)**:
  - `is_new_best` 모델 선택이 **틀린 지표**로 이뤄진다(배경 포함 full SSIM).
  - **SSIM_bbox(순위 결정축)를 학습 중 단 한 번도 계산하지 않는다.** 전략 전체가 이 값을 밀자는 것인데, 정작 볼 수가 없다 → "full↑ bbox↓" 함정을 조기에 못 잡는다([`augmentation.md`](augmentation.md) §4.2의 측정 프로토콜이 코드에 없음).
- **방향**: `validate()`를 `metrics.ssim_full`·`ssim_bbox`로 교체, best 모델 선택 기준을 **SSIM_bbox 비퇴행 게이트**로. acc4/acc8·full/bbox 4숫자 분리 리포트.

## A2. 🔴 "배경 마스크 정의 미확정"은 사실이 아님 — 이미 코드에 있음 (문서 stale)

- **무엇**: 여러 문서([`confirmed_model.md:72`](confirmed_model.md), [`augmentation.md`](augmentation.md) §4.2, [`README.md`](README.md), `.claude/CLAUDE.md`)가 *"SSIM_full 배경 마스크 정의 미확정, eval code 공개 후 재동기화"*로 적혀 있다. 그러나 [`metrics.py:48`](../utils/common/metrics.py#L48) `foreground_mask`에 정의가 **이미 있다**: `target > 2e-5` (절대 threshold) → `erode×1 → dilate×15 → erode×14`.
- **이유**: "잠정 proxy"라며 미뤄둔 로컬 SSIM_full을 **지금 확정**할 수 있다. 불확실성을 이유로 보류한 실험들이 실제로는 진행 가능.
- **방향**: 이 `metrics.py`가 **공식 배포본이 맞는지 확인**(맞으면 문서의 "미확정" 표기 일괄 정정) → 로컬 채점을 이 정의로 고정.

## A3. 🔴 확정 모델이 통합 실행된 적이 한 번도 없다

- **무엇**: 데이터측 플래그(GRAPPA·normalize·readout_crop·coil_compress·return_bbox)는 "구현·검증 완료"지만, **켜는 순간 소비부에서 터진다**.
  - [`train_part.py:24`](../utils/learning/train_part.py#L24): `mask, kspace, target, maximum, _, _ = data` — **6개 고정 언패킹**. 플래그 on이면 반환 튜플이 7~9개가 되어 즉시 에러.
  - [`varnet.py:238`](../utils/model/varnet.py#L238) `VarNet.forward(masked_kspace, mask)`만 받음 — GRAPPA·residual·역정규화(×s)·감도맵 재사용 전부 미배선([`implementation.md`](implementation.md) §4·§5·§5b).
- **이유**: 검증된 데이터측 조각들이 **단독으로도 못 켜지는 상태** → 확정 파이프라인이 end-to-end로 돈 적이 없다. 이게 실질적 병목이다.
- **방향**: 소비부 튜플 확장(6→9) + `VarNet.forward`에 GRAPPA/residual/×s 배선을 최우선 통합.

## A4. 🟠 전역정규화 ÷s ↔ 절대 threshold 마스크(2e-5)의 충돌

- **무엇**: 배경 마스크가 `target > 2e-5`라는 **절대값** 기준(A2). 확정된 전역정규화 ÷s가 밝기 스케일을 바꾸므로, **정규화 공간에서 채점/모델선택하면 마스크가 선택하는 전경이 달라진다.**
- **이유**: 출력을 `×s`로 역정규화한 **원본 스케일에서 채점**하지 않으면 로컬 SSIM_full이 리더보드를 예측하지 못한다. 문서엔 "maximum÷s 정합"만 있고 이 threshold 함정은 언급이 없다.
- **방향**: 채점·모델선택은 항상 **역정규화 후 원본 스케일**에서. `data_range`도 원본 volume max로 일관.

## A5. 🟠 학습 손실의 목적 ≠ 채점의 목적

- **무엇**: [`loss_function.py:12`](../utils/common/loss_function.py#L12) `SSIMLoss`는 **배경 포함 full SSIM**. 채점은 (a)전경마스크 SSIM_full + (b)bbox SSIM.
- **이유**: **손실이 최적화하는 대상과 채점 대상이 다르다.** 배경(공기)까지 SSIM에 포함되어 학습 신호가 희석되고, 순위축 bbox는 손실에 아예 없다.
- **방향**: 손실에 전경마스크 적용 + **bbox 가중 손실**([`candidates.md`](candidates.md) 1순위) 배선. bbox 파싱은 되나 손실 소비부 미배선.

## A6. 🟡 배치가 bs=1에 갇혀 있고 acc4/acc8 폭(368/372) pad 미구현

- **무엇**: [`train.py:20`](../train.py#L20) 기본 `batch_size=1`. 폭이 다른 acc4(368)/acc8(372)를 한 배치에 못 쌓아 batch>1이 막힘. 문서는 "mask=0 pad로 폭 통일"을 **언급만** 하고 구현·계획 없음([`model_strategy.md`](model_strategy.md) §1.1 주의).
- **이유**: 1080에서 학습 효율(수렴 속도)의 상한을 스스로 묶고 있다.
- **방향**: k-space 폭 pad(mask=0) 유틸 + 배치에 acc4/acc8 균형 혼합.

## ✅ 참고: 이미 잘 되어 있는 것 (헛다리 방지)
- **재현성 seed는 이미 걸려 있음** — [`utils.py:76`](../utils/common/utils.py#L76) `seed_fix`가 `cudnn.deterministic=True`까지 설정. 문서 강조 대비 이미 구현.
  - ⚠️ **단, blindspot이 하나 있었음(2026-07-19 수정)**: 증강 RNG(`MRIAugmentor._rng`)가 builtin `hash()`(문자열 salt, `PYTHONHASHSEED` 미설정)를 써서 **프로세스마다 달라졌다** — `seed_fix`로도 안 잡히는 구멍. `zlib.crc32(repr(...))`로 교체해 재실행 간 동일 추출 확인. 증강이 dead라 잠복 상태였고 이제 배선해도 재현성 보장.
- 데이터측 전처리 조각의 **수치 검증(무손실·DC 보존)**은 탄탄.

---

# Part B — 전략적 사각지대 (프레임 자체를 놓침 · 가설)

> Part A가 "적어둔 걸 안 했다"라면, 여기는 **아예 적지 않은(인지하지 못한) 것**이다. 우리 문서 전체가 **"어떻게 더 잘 복원하나(화질↑)"** 프레임에 갇혀 있어서 구조적으로 안 보이는 승부 아이디어들. 모두 **검증 필요한 가설**이나, 대회 특수성에 직결된다.

## B1. ⭐ 이건 "복원 문제"가 아니라 "환각 통제(불확실성) 문제"다 — 최상위 리프레이밍

- **무엇**: 대회가 밝힌 novelty는 *"레퍼런스 없는 open question"*이고 순위축 SSIM_bbox의 본질은 **"없는 병변을 지어내지 않는가"**. 즉 진짜 난제는 "더 선명하게"가 아니라 **"확신 없는 곳에서 함부로 그리지 않게"**. 그런데 우리의 환각 억제 도구는 **DC 하나뿐**이다.
- **이유**: 화질 극대화 프레임에선 고주파를 채우는 방향으로 압력이 걸려 **순위축과 정면충돌**한다. 프레임을 "환각 최소화 하에서 화질"로 바꾸면 아래 B2·B3·B5가 자연스럽게 도출된다.
- **방향(놓친 도구들)**: 불확실성 인지 복원(모델이 "여기는 모른다"를 알고 GRAPPA/DC로 후퇴) · **뷰/모델 간 불일치 = 환각 신호**로 억제.

## B2. ⭐ SSIM은 작은 병변 패치에서 "선명함"이 아니라 "보수성"을 보상한다 (우리 §2.4가 방향이 거꾸로일 수 있음)

- **무엇**: [`model_strategy.md`](model_strategy.md) §2.4는 *"미획득 고주파에 손실 집중해 병변 미세구조를 살리자"*를 민다. 그런데 [`metrics.py:73`](../utils/common/metrics.py#L73) `ssim_bbox`는 **16~수십 px 작은 패치**에서 SSIM을 잰다 — 이 스케일에선 국소 평균/분산 일치(luminance·contrast)가 지배적이고, SSIM은 노이즈 분산을 페널티한다.
- **이유(비대칭)**: **흐릿하지만 충실한 병변**은 구조 보존으로 SSIM이 괜찮고, **선명하지만 위조된 텍스처**는 없는 구조를 만들어 SSIM이 오히려 **더 떨어지고** 임상적으로도 위험. 저SNR 병변부에 "고주파를 채워라"를 강하게 밀면 곧 위조·노이즈 = 순위축 손해. (우리도 §2.4를 "양날의 칼"이라 했으나 결론이 "밀어보자"로 기울어 있음.)
- **방향 — 영역 적응(region-adaptive)**: 벌크 조직은 선명하게(SSIM_full), **병변 bbox 안은 의도적으로 보수적으로**(약한 스무딩·GRAPPA 근접). 전역 sharpen과 정반대.

## B3. ⭐ GRAPPA를 병변에서 "덜 베끼게"가 아니라 "더 믿게" (anti-copy의 역발상)

- **무엇**: [`model_strategy.md`](model_strategy.md) §1.2 ③ anti-copy 게이팅은 *"GRAPPA 상한에 갇히지 말고 넘어서라"*. 맞지만 **공간적으로 반대로 적용해야 할 지점**을 놓쳤다.
- **이유**: GRAPPA는 선형·물리 복원 = **절대 병변을 지어내지 않는 안전 앵커**. 환각 위험이 가장 높은 곳 = 병변부. 그렇다면 **병변 bbox 안에서는 GRAPPA를 강하게 신뢰**(trust gate↑)하고, 창의적 복원은 위험 낮은 벌크에서만 허용.
- **방향 — 병변 인지 신뢰 게이팅**: 게이트 `g`를 전역 학습이 아니라 **병변 saliency로 공간 변조**. (B2=손실 버전, B3=구조 버전, 같은 원리 "병변부=보수 영역".)

## B4. ⭐ VarNet은 "주어진 베이스라인"일 뿐 — 백본을 한 번도 의심하지 않았다

- **무엇**: VarNet은 팀이 A/B로 고른 게 아니라 **주최측 제공 베이스라인(`2026_baby_varnet`)이라 기본값으로 깔린 것**이다. 그런데 [`confirmed_model.md`](confirmed_model.md) §5·[`README.md`](README.md) 상태판이 이걸 "main: VarNet cascade+DC"로 적으면서, **고른 적 없는 상속된 기본값이 의심 없이 '확정'처럼 굳었다.** 백본 자체를 대안과 비교한 기록이 없다.
- **이유**: 이 대회엔 백본 선택을 다시 열어야 할 두 특수 조건이 있다 — ① **GRAPPA가 병렬영상의 무거운 일을 끝내 입력으로 제공**됨(VarNet의 가장 비싼 감도맵 SME+다단계 cascade가 하는 일을 상당 부분 대체), ② **추론시간 채점 + 1080 8GB**. 즉 **"GRAPPA 초기해 + 경량 image-domain refiner + 가벼운 k-space DC 한두 번"** 구조가 추론시간을 압도하면서 GRAPPA 출발이라 환각도 안전할 수 있다([`related_work.md`](related_work.md) DeepMRIRec/feature-domain 계열). 상속된 기본값을 그대로 밀면 이 가능성을 통째로 못 본다.
- **방향**: VarNet을 "기본값"이 아니라 **하나의 후보**로 내려놓고, **GRAPPA-centric 경량 refiner**와 동등하게 A/B(현재 비교 대상에서 빠져 있음). 백본은 A7 baseline 측정 위에서 3축(SSIM/시간/VRAM)으로 판정.

## B5. 앙상블/self-ensemble을 "화질 도구"로 보고 기각했다 — 실은 "환각 제거 도구"

- **무엇**: [`augmentation.md`](augmentation.md) §3.5는 TTA를 *"추론시간 N배 → 감점"*으로 사실상 접었다.
- **이유(저울이 틀림)**: 여러 복원본 평균은 **일관된 구조(진짜)는 살리고 뷰마다 갈리는 구조(환각)는 상쇄** → **순위축(hallucination-free bbox)을 직접 겨냥하는 억제 도구**. "저비용 2-view flip 앙상블"의 가치는 "SSIM 약간 상승"이 아니라 **"병변부 위조 확률 감소"**로 재평가해야 한다.
- **방향**: 2-view flip 앙상블을 **Δms vs Δ(bbox 위조율)**로 재측정. B1(환각 통제) 프레임의 실전 도구.

## B6. 미공개 가중치 대응이 "3축 균형" 구호에 그친다 (게임이론적 사각지대)

- **무엇**: 문서는 *"가중치 미공개니 3축 균형"*이라고만 한다.
- **이유**: 실제 필요한 건 **minimax-regret** — 어느 한 축에서도 크게 지지 않는(spiky하지 않은) 제출물이 가중치가 어떻게 나오든 상위. bbox 올인으로 full/time을 희생하면 가중치가 반대일 때 폭망.
- **방향**: 후보 모델들을 **가능한 가중치 벡터들에 대한 최악 regret**으로 평가. 파생: **acc4/acc8 통합 단일모델 vs 두 전문가 모델** — acc8은 환각 위험이 커 다른 보수성이 필요할 수 있어, "한 모델로 둘 다"가 양쪽을 어중간하게 만드는 함정일 수 있음.

## B7. 병변을 "슬라이스 유무"로만 보고 pathology 22종의 분포/불균형을 안 본다

- **무엇**: 데이터 활용이 "병변 슬라이스 오버샘플링" 수준. SSIM_bbox는 **per-lesion 평균**이 1차 지표([`augmentation.md`](augmentation.md) §4.2).
- **이유**: **희귀 pathology 카테고리 하나가 분산을 좌우**할 수 있다. 22종 분포·난이도 분석 없이 뭉뚱그리면 흔한 병변에 과적합되고 희귀 병변에서 무너져 순위가 흔들린다. **자유 재조합 허용**을 K-fold로만 쓰는데, 진짜 레버는 **카테고리 균형 학습 분포 큐레이션**.
- **방향(B2·B3의 실전화)**: 추론 시 bbox가 안 주어질 가능성이 높으니 **보조 lesion-saliency head**(annotation으로 학습)를 달아 모델이 스스로 병변 위치를 예측 → 그 saliency로 B2(보수적 손실)·B3(GRAPPA 신뢰)을 **추론 시점에도 공간 변조**. 문서에 없는, 이 대회에 딱 맞는 구조.

## ⚠️ 확인이 필요한 대회 근본 질문
- **"Annotation-guided" — bbox가 추론 시점에도 입력으로 주어지나?** 주어지면 bbox를 공간 prior/attention으로 넣는 강력한 레버가 통째로 비어 있는 것이고, 아니면 지금처럼 학습 전용(+B7의 saliency head). **아키텍처 한 부류를 가르는 질문인데 문서에 명시적 확인이 없다.** 주최측 확인 1순위.

---

# 통합 요약 — 사고 틀 전환표

| 지금 우리 프레임 | 놓친 프레임 | 근거 |
|---|---|---|
| 화질을 더 올린다 | **환각을 통제한다** | B1 |
| 병변부를 더 선명하게(§2.4) | 병변부는 **보수적으로** | B2·B3 |
| VarNet(상속된 기본값)을 안 의심 | 백본도 **후보**로 — GRAPPA-centric 경량과 A/B | B4 |
| 앙상블 = 시간 낭비 | 앙상블 = **환각 억제기** | B5 |
| 3축 균형(구호) | **최악 가중치 대비 regret 최소화** | B6 |
| 병변 슬라이스 오버샘플 | **pathology 균형 + saliency head** | B7 |
| (데이터 처리 중심) | **채점 루프를 실제 지표로** | A1·A2 |

# 권장 착수 순서

1. **A1 + A2**: `validate()`를 `metrics.py`(전경마스크 SSIM_full + SSIM_bbox)로 교체 → **순위축을 눈으로 보이게** 만든다. (모든 판단의 전제)
2. **A3**: 소비부 6→9 튜플 확장 + `VarNet.forward` GRAPPA/residual/×s 배선 → **확정 모델을 처음으로 end-to-end 실행.**
3. **A7**: 1080 baseline(ms/slice·VRAM·SSIM_full/bbox) 측정 → 모든 A/B의 기준점.
4. **B2 첫 실험(가장 반직관적 = 승부처)**: **"고주파 집중(§2.4) vs 병변부 보수화(B2)"**를 acc8 bbox에서 직접 A/B. 어느 쪽이 SSIM_bbox를 올리는지 데이터로 판정.
5. **B4 / B1**: GRAPPA-centric 경량 refiner를 VarNet과 A/B(시간축) · 불확실성/앙상블(B5)로 환각 억제.

> ⚠️ 이 문서의 Part A는 **사실(코드 확인)**, Part B는 **가설(검증 필요)**. Part B 항목은 [`candidates.md`](candidates.md)의 A/B 규칙(SSIM_bbox 비퇴행 하드 게이트, incremental) 하에서만 채택한다.
