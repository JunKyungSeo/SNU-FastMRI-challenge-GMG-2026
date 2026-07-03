# 구현 배선 체크리스트 (Implementation)

> [`confirmed_model.md`](confirmed_model.md) 확정 구조를 코드에 배선하는 순서. 의존성 기준 정렬.
> 상태 범례: ✅ 구현됨 · ⏳ 보류(외부조건) · ⬜ 미구현.
> **데이터 측(utils/data/)은 2026-07-04 구현·검증 완료**(플래그 전부 기본 off라 기존과 비트 동일). network·학습 측은 동료 담당.

## 의존성 순서 (권장)

### 1. ✅ GRAPPA 로딩 — [`utils/data/load_data.py`](../utils/data/load_data.py) **(train/val 구현됨)**
- ✅ **train/val 경로 구현됨**: `return_grappa=True`면 `SliceData.__getitem__`이 `image_grappa[dataslice]` 로드 → `DataTransform` 전달. flip 증강 시 GRAPPA 동기 정렬(`MRIAugmentor._flip_lr`)도 구현됨.
- ⏳ **보류(leaderboard 구조 미공개)**: **forward(추론) 경로의 GRAPPA 로딩은 미구현**. 현재 forward 로더는 `kspace/` 폴더만 읽음. leaderboard가 `image_grappa`를 어떤 폴더/키로 주는지 공개되면 배선할 것(코드에 TODO 주석 있음). GRAPPA를 쓰는 모델은 추론에도 GRAPPA 필요 → **제출 전 필수 확인 항목**.
- 근거: model_strategy §1, confirmed_model.

### 2. ✅ 전역 스칼라 정규화 ÷s / ×s — [`utils/data/transforms.py`](../utils/data/transforms.py) **(데이터측 구현됨)**
- ✅ `DataTransform.__call__`에 `normalize` 플래그: 입력유도 전역 스칼라 `global_scale()`(zero-filled RSS p99)로 `kspace/grappa/tgt/maximum` 동일 ÷s, `scale`을 튜플에 실어 반환(isforward 공통).
- ⬜ **출력단 `×s` 역정규화는 network·학습 측(train_part/test_part)**에서 `output*scale`로 처리 필요.
- ⚠️ `maximum` attr은 입력 스칼라로 미사용(label 통계·추론 시 없음) — 지킴.
- 근거: preprocessing B4, confirmed_model.

### 3. ✅ Readout crop 640→384 — [`transforms.py`](../utils/data/transforms.py) **(구현됨)**
- ✅ `readout_crop` 플래그: `readout_crop()` 함수가 `ifft_h → crop 640→crop_h → fft_h`(readout만, 폭 native). 실측 recon corr 1.0(무손실). flip/mask와 commute.
- ⚠️ network에서 출력단 `center_crop`(varnet.py:245)과 이중 crop 안 되게 정합(조기 crop이면 이후 no-op).
- 근거: preprocessing A1, confirmed_model. 실측 recon corr 1.0으로 무손실 검증 완료(검증 스크립트는 임시라 삭제).

### 4. ⬜ GRAPPA 영상 prior + residual — [`utils/model/varnet.py`](../utils/model/varnet.py) (network 측)
- `VarNet.forward`에 GRAPPA 영상 보조 채널 경로 + **residual 출력**(`out = grappa + f(...)`).
- GRAPPA(384²) ↔ k-space 격자(384×폭) 정합: 폭 crop 또는 출력단 384² refinement head.
- 근거: model_strategy §1.2, confirmed_model.

### 5. ⬜ 감도맵 재사용 — [`varnet.py`](../utils/model/varnet.py) (network 측)
- sens_net 추정을 per-view forward에서 분리: canonical source 1회 추정 → mask증강=맵 그대로 / flip=맵도 flip.
- 근거: confirmed_model, model_strategy §3.2(배치 SME와 결합 가능).

### 5b. ⬜ **소비부 확장** — [`train_part.py`](../utils/learning/train_part.py) · [`test_part.py`](../utils/learning/test_part.py) (network·학습 측)
> 데이터 측이 플래그 on일 때 반환 튜플이 6→7~9개로 늘어남. 소비부(배치를 받아 모델에 먹이고 loss 거는 코드)를 그에 맞춰 확장해야 실제로 GRAPPA·정규화가 동작한다. **현재 소비부는 6개 언패킹이라, 플래그 켜면 여기 수정 없이는 에러.**

- `train_epoch`(train_part.py:24)·`validate`(:57)·`test`(test_part.py:14)의 `... = data`를 확장 튜플에 맞게 수정.
- `model(kspace, mask, grappa)`로 GRAPPA 전달 + 출력 역정규화 `output * scale` + loss/SSIM은 역정규화 후.
- 반환 순서 규약은 `DataTransform` docstring: `(mask,kspace,target,maximum,fname,slice,[scale],[grappa],[bbox])`.
- network 확정 후 진행(동료 담당).

### 6. 증강 스케줄 배선 (mask/flip 로직은 이미 구현됨)
> 증강 몸통은 `transforms.py`(`MRIAugmentor`)에 있음. epoch 스케줄만 배선 필요.

- **6a. [`train.py`](../train.py) `parse()`**: `--aug-*` 인자 신설(`--aug-on`, `--aug-warmup-epochs`, `--aug-max-prob`, `--aug-schedule`, `--aug-flip`, `--aug-mask` 등). 기본 전부 off(기존 동작 보존). `args`로 `create_data_loaders`→`DataTransform` 전달.
- **6b. [`train_part.py`](../utils/learning/train_part.py) `train()`**: 매 epoch `train_loader.dataset.transform.set_epoch(epoch)` 호출. **val_loader는 증강 off 보장**(검증=진짜 실력).
- 근거: 상세 맥락은 이 파일 하단 "증강 스케줄 배선 상세" 참조.

### 7. bbox 손실 소비부 (손실 확정 후 — candidates)
- `DataTransform`은 `return_bbox=True`로 bbox 파싱·flip 동기까지 됨(`_bbox_collate` 포함). **손실에서 소비하는 부분만 미배선** → [`candidates.md`](candidates.md) 손실 확정 후.

## 재현성 (전 항목 공통)
- 기본 off일 때 기존과 **비트 동일** 회귀 테스트(seed 고정). 최종 VESSL 재학습 재현 검증 대비.
- seed·split·라이브러리 버전·데이터 순서 리포지토리 커밋.

---

## 증강 스케줄 배선 상세 (구 todo.md)

> 왜 train.py/train_part.py 두 곳인가: 증강 로직은 `transforms.py`에 있으나 **transform은 "지금 몇 epoch인지" 모른다**. epoch·설정을 위(train.py args)→아래(transform)로 내리는 통로가 6a·6b.
> 비유: train.py=볼륨 노브 설치 / train_part.py=곡 진행(epoch)에 따라 볼륨 조절 / transforms.py=실제 소리.

**6a train.py 주의**: 기본값 전부 off·보수적(재현성). 현재 `DataTransform(isforward, max_key)`만 받으므로 시그니처 확장은 transforms 작업(2·3)과 연동.

**6b train_part.py 주의**:
```python
for epoch in range(start_epoch, args.num_epochs):
    train_loader.dataset.transform.set_epoch(epoch)   # ← 추가
    train_epoch(args, epoch, model, train_loader, optimizer, loss_type)
```
- val_loader는 `set_epoch` 미호출 또는 `aug_on=False`. `save_reconstructions`/`validate`는 안 건드림.

**GRAPPA 정렬 규칙(4번 배선 시)**: 기하변환 g는 {k-space, GRAPPA, target, bbox}에 동일 적용. mask증강은 GRAPPA 미적용. 이미지 도메인 g 한 곳 정의 → 모든 짝에 적용. 상세: model_strategy §1.1.
