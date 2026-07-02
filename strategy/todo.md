# 구현 TODO — Augmentation 파이프라인 배선 (train.py / train_part.py)

> 작성 2026-07-03 · 상태: **미구현 (설계만 기록)**
> 관련 전략: [`augmentation.md`](augmentation.md) · 코드 매핑 근거는 실제 파일 확인 완료.
> 이 문서는 **3번(train_part.py)·4번(train.py) 수정**이 무엇을·왜인지 맥락과 함께 남겨둔 것.
> ⚠️ 아직 구현하지 말 것. transforms.py의 실제 증강 로직이 먼저/함께 필요(아래 의존성 참고).

---

## 왜 이 두 파일을 고쳐야 하나 (맥락)

augmentation.md의 **증강 로직 자체는 [`utils/data/transforms.py`](../utils/data/transforms.py)의 `DataTransform.__call__`** 에 들어간다.
그런데 augmentation.md의 **스케줄 규칙**이 epoch에 의존한다:

> 첫 10~15 epoch은 증강 off → 이후 지수형으로 확률 p=0.5~0.6까지 점증 (강도는 warmup→막판 decay)

문제: **`transforms.py`는 "지금 몇 번째 epoch인지" 모른다.** 데이터 한 조각을 변형할 뿐이라 스케줄을 못 건다.
→ epoch 정보와 증강 설정을 **위에서(train.py args) 아래로(transform) 전달하는 통로**가 필요하고, 그 배선이 3·4번이다.

비유: **train.py = 볼륨 노브 설치(설정 정의)** / **train_part.py = 곡 진행(epoch)에 따라 볼륨 자동 조절(배선)** / **transforms.py = 실제 소리(증강)**.

---

## 4번: `train.py` — 증강 설정(다이얼) 정의

**위치**: [`train.py`](../train.py)의 `parse()` (현재 증강 관련 인자 **전무**).

**할 일**: 증강을 켜고/끄고/조절하는 argparse 인자 신설. 코드를 안 고치고 명령어로 A/B 하기 위함(augmentation.md의 실험 우선순위·A/B 전제).

**추가할 인자(예시)**:
- `--aug-on` (bool): 증강 전체 on/off
- `--aug-warmup-epochs` (int, 예 10): 이 epoch까지는 증강 off(기본기 학습)
- `--aug-max-prob` (float, 예 0.5): 스케줄 최종 도달 확률
- `--aug-schedule` (str, 예 'exp'): 확률 증가 형태(지수형)
- 어떤 증강을 켤지 플래그: `--aug-flip`(좌우만), `--aug-mask`(마스크 증강), `--aug-phase` 등
- (선택) 마스크 증강 세부: `--aug-acc-min/--aug-acc-max`(예 4~16), ACS 보존은 기본 강제

**주의**:
- 기본값은 **전부 off / 보수적**으로 (재현성·기존 동작 보존).
- 이 인자들은 `args`에 실려 `create_data_loaders → DataTransform`까지 전달되어야 함(현재 `DataTransform(isforward, max_key)`만 받으므로 시그니처 확장 필요 — transforms.py 작업과 연동).

---

## 3번: `train_part.py` — epoch 전달 + train 전용 배선

**위치**: [`utils/learning/train_part.py`](../utils/learning/train_part.py)의 `train()` (epoch 루프가 여기 있음).

**핵심 사실**: `for epoch in range(...)` 루프는 이 파일에만 있고, `train_epoch(args, epoch, ...)`에 epoch이 이미 넘어온다. 하지만 그 epoch이 **transform까지는 안 내려간다.**

**할 일 ①: 매 epoch 시작 시 transform에 epoch 알리기** (한 줄):
```python
for epoch in range(start_epoch, args.num_epochs):
    train_loader.dataset.transform.set_epoch(epoch)   # ← 추가 (DataTransform에 set_epoch 구현 전제)
    train_epoch(args, epoch, model, train_loader, optimizer, loss_type)
```
→ 이러면 transform이 epoch을 알고 스케줄대로 확률·강도를 조절.

**할 일 ②: 검증(val)에는 증강 절대 미적용**:
- `train_loader`와 `val_loader`는 지금 둘 다 `create_data_loaders`로 생성됨.
- **train_loader에만 증강 on**, val_loader는 원본 그대로여야 함(검증은 진짜 실력 측정).
- 구현 방법: `create_data_loaders`에 `is_train` 플래그를 넘겨 val 쪽 transform은 증강 off로 만들거나, val_loader의 transform엔 `set_epoch`를 호출하지 않음(또는 aug_on=False).

**주의**: `save_reconstructions`/`validate`는 건드리지 않음(부수 수정 범위). 여기선 순수 배선만.

---

## 의존성 / 구현 순서

1. **먼저 `transforms.py`의 인터페이스 확정**: `DataTransform`이 (a) 증강 설정(args)을 받고, (b) `set_epoch(epoch)` 메서드를 갖고, (c) epoch·확률 스케줄로 증강 적용하도록 설계. ← 실제 증강 로직의 몸통.
2. 그 인터페이스에 맞춰 **4번(args 신설)** → **3번(args 전달 + set_epoch + val 제외)** 배선.
3. 최소 시작 범위 권장: **bbox 안 쓰는 기하/마스크 증강(좌우 flip + 마스크 증강)만** 먼저. bbox 관련(collate·metrics)은 별도 단계(부수 수정, 지금 범위 아님).

---

## 체크리스트

- [ ] `transforms.py`: `DataTransform`에 args·`set_epoch`·확률 스케줄 반영 (몸통 — 이 문서 범위 밖이지만 선결)
- [ ] **4번** `train.py`: `--aug-*` 인자 신설, 기본 off, `args`로 전달
- [ ] **3번** `train_part.py`: 매 epoch `set_epoch(epoch)` 호출
- [ ] **3번** `train_part.py`: val_loader는 증강 off 보장
- [ ] 재현성: 기본 off일 때 기존과 **비트 동일** 확인 (seed 고정 회귀 테스트)

> 참고: **GRAPPA FFT 보조 입력**은 augmentation이 아님(모델 varnet.py 수정 필요 = 입력/구조 설계). 이 배선 범위에 포함하지 말 것. → model_strategy.md §1.

---

## (별건) GRAPPA 보조 입력 사용 시 — augmentation 정렬 규칙

> GRAPPA를 보조 입력 채널로 쓰기로 하면(model_strategy.md §1), 증강 시 GRAPPA 채널을 메인 입력과 **정렬**시켜야 함.
> 상세·근거는 [`model_strategy.md`](model_strategy.md) §1.1. 여기엔 구현 시 잊지 말 규칙만 요약.

- **기하 변형(좌우 플립·정수 translation)**: 메인 k-space에 적용한 g를 **GRAPPA 채널·target·bbox에도 동일 적용**(정렬 유지).
- **마스크 증강(추가 언더샘플)**: **GRAPPA엔 미적용** — GRAPPA는 이미 복원된 깨끗한 prior라 라인 더 지우면 스스로 훼손. "더 언더샘플된 GRAPPA"는 재생성 불가·불필요.
- **k-space 위상 / coil 노이즈**: GRAPPA(magnitude, coil 없음)엔 사실상 무의미 → 미적용.
- **구현 권장**: 기하 변형은 **이미지 도메인 g를 한 곳에서 정의 → 모든 짝에 동일 적용**(두 k-space는 격자·물리량이 달라 각자 뒤집기보다 안전).
- ⚠️ 이건 augmentation 배선(3·4번)이 아니라 **GRAPPA 입력 도입(varnet.py 등) 단계의 규칙** — 순서상 GRAPPA 채널을 실제로 넣을 때 함께 처리.
