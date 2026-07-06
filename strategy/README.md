# strategy/ — FastMRI 2026 전략 문서 인덱스

> 2026 SNU FastMRI Challenge (Annotation-guided Knee Reconstruction) 참가 전략 모음.
> 구조: **결정(SPEC) · 실행 · 후보 · 제외 · 근거**의 5역할로 분리. 결정은 `confirmed_model.md` 한 곳에서만 "정의", 나머지는 참조.

## 파일 역할 (읽는 순서)

| 파일 | 역할 | 언제 보나 |
|---|---|---|
| **[confirmed_model.md](confirmed_model.md)** | ✅ **SPEC — 확정 파이프라인 단일 기준** | 구현 시작·구조 논의 |
| **[implementation.md](implementation.md)** | 🔧 코드 배선 체크리스트(의존성 순서) | 배선하는 중 |
| **[candidates.md](candidates.md)** | 🔬 미결정 후보·열린 A/B | 다음에 뭘 실험할지 |
| **[rejected.md](rejected.md)** | ❌ 제외된 것 + 이유 | 재논의 방지 |
| [model_strategy.md](model_strategy.md) · [augmentation.md](augmentation.md) · [preprocessing.md](preprocessing.md) · [related_work.md](related_work.md) | 📚 근거(rationale)·출처 아카이브 | "왜 이렇게 정했지?"·"선례 논문?" |

> ⚠️ **confirmed_model = 재구성 forward 파이프라인 구조만 확정.** 손실 함수·효율 레버·추가 증강은 아직 **candidates.md**(미확정).

## 상태 대시보드

### ✅ 확정 (→ [confirmed_model.md](confirmed_model.md))
- 입력 2분기: **k-space readout(높이) 640→384 crop, 폭 native** + **GRAPPA 영상 도메인 유지(fft 안 함)**
- **전역 스칼라 정규화 ÷s** (입력유도, acc 무관 단일, k-space·GRAPPA·target 동일), 출력 ×s
- 증강: mask(k-space만)·flip(전부 동기), 지수 스케줄 (로직 구현됨, 배선 대기)
- 감도맵 **1회 추정 후 재사용** (mask=그대로 / flip=flip)
- main: VarNet cascade+DC, **GRAPPA 영상 prior 채널 + residual**

### 🔬 열린 후보 (→ [candidates.md](candidates.md))
- **1순위 손실**: bbox 가중 손실 · 병변 슬라이스 오버샘플링
- **효율 A군(택1 A/B, SME비용)**: 배치 SME(잠정채택) / **coil compression(SVD, 데이터측 구현됨)** / progressive / FLAT
- **효율 B군(직교·조합가능)**: knowledge distillation(teacher→student) / TensorRT(배포, Pascal 이득 제한)
- **즉시 A/B**: 입력crop vs 출력crop · GRAPPA 상보융합(input⊕grappa)
- **대기**: SSIM_full 배경 마스크(eval code 7/8) · TGVN(PD/PDFS 확인)

### ❌ 제외 (→ [rejected.md](rejected.md))
- 기하: 위-아래 flip · 정수 translation · 회전/스케일/shear/elastic
- 강도: image 노이즈/밝기/gamma · cutout · MixUp/CutMix · 합성병변 삽입
- 구조: GRAPPA fft · 완전 384²(폭 pad) · x0 대체 · 2.5D 인접슬라이스 입력

## 하드 제약 (전 문서 공통)
knee 전용 · SSIM_full + **SSIM_bbox(순위축)** + 추론시간 채점 · GTX 1080 8GB FP32 · external/pretrained 금지 · 세부는 [`.claude/CLAUDE.md`](../.claude/CLAUDE.md).
