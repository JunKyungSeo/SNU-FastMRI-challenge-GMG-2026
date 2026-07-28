# 구현 배선 체크리스트 (Implementation)

> ⚠️ **일부 SUPERSEDED (2026-07-19): 백본이 SDUM→PromptMR+로 교체됨.** network·학습 측 배선(빌드 분기·손실·
> 옵티마이저·grad clip/accum·resume·coil compression)은 이미 PromptMR+ 기준으로 재배선됨 → 현행은
> [`promptmr_plus.md`](promptmr_plus.md). 아래 ⬜ 미구현 중 **GRAPPA prior→`varnet.py`** 항목은 옛 VarNet 백본 기준
> 이라 그대로 따르지 말 것(현재 백본은 promptmr). 데이터 측(readout crop·정규화·coil compression 등)은 백본 무관·유효.
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

### 3b. ✅ Coil compression (SVD) — [`transforms.py`](../utils/data/transforms.py) **(데이터측 구현됨, 2026-07-06)**
- ✅ `coil_compress` 플래그 + `n_virtual_coils`(기본 8): `coil_compress()`(SVD/SCC, ACS 기반 압축행렬). `readout_crop` 직후·`normalize` 직전. 실측: 미샘플 라인 누설 0.00e+00(DC 보존), realistic 코일 Nvc=8 에너지 99.999%.
- ⚠️ **network 측 영향**: 켜면 kspace coil축이 15→`n_virtual_coils`로 축소되어 전달됨. sens_net·cascade는 코일 수에 무관하게 동작하므로 **모델 코드 수정 불필요**, 단 소비부가 shape 가정을 하드코딩하지 않는지 확인.
- ⚠️ **상호배타**: §3.2 배치 SME와 **택1**(동시 적용 금지, 목적 중복). 하나만 켠다.
- ⚠️ **화질 neutral**: 속도/VRAM 레버(SSIM 상승 아님). 근거·출처: preprocessing A2, related_work, candidates 효율레버 A군.
- 근거: preprocessing A2, candidates.md. 출처: SVD=Buehrer et al. MRM 2007 / GCC=Zhang et al. MRM 2013.

### 3c. ✅ Foreground 마스크 손실·검증 정합 (B5) — [`transforms.py`](../utils/data/transforms.py) · [`loss_function.py`](../utils/common/loss_function.py) · [`train_part.py`](../utils/learning/train_part.py) **(구현·검증됨, 2026-07-11 @esel34)**
- ✅ `--fg-loss` 플래그(기본 off): 켜면 SSIM 손실·검증을 **채점(`metrics.foreground_mask`) 정합 foreground 영역 안에서만** 계산. off면 기존 전체 384² 그대로(byte-identical).
- ✅ 데이터측: `DataTransform(return_fg_mask=True)`가 **÷s 정규화 전 raw target**에서 `metrics.foreground_mask`(재사용, 중복정의 금지) 계산 → 튜플에 `fg_mask`(384², bbox보다 앞) 추가. `not isforward` 게이트라 추론 튜플은 6개 유지.
- ✅ 손실측: `SSIMLoss.forward(mask=...)`가 `metrics.ssim_full` 정합(입력 마스킹 후 win//2 valid 영역 가중평균). off 경로(`mask=None`)는 기존 정의 불변.
- ✅ 소비부: `train_epoch`·`validate` 튜플 언패킹 확장. **val을 masked SSIM(=1−SSIM_full)의 슬라이스 평균으로** 바꿔 val이 실채점 SSIM_full을 예측.
- ✅ 검증: masked-loss == `1 − metrics.ssim_full` (diff 2.4e-08) · off=6/on=7/forward=6 튜플 · import smoke(recon_eval 포함) 통과.
- ⚠️ **normalize와 조합은 미검증**(현 regime은 readout-crop+정규화 off). ⚠️ 실제 화질 이득은 **matched-epoch A/B(bbox 비퇴행 게이트)로 확인 필요** — 아직 정합만 완료, 이득 미확정.
- 근거: preprocessing B5, candidates.md.

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

### 6. ✅ 증강 스케줄 배선 (mask/flip 로직은 이미 구현됨)
> 증강 몸통은 `transforms.py`(`MRIAugmentor`)에 있음. epoch 스케줄도 배선 완료.

- **6a. ✅ [`train.py`](../train.py) `parse()`**: `--aug-*` 인자 신설(`--aug-on`, `--aug-schedule`, `--aug-max-prob`, `--aug-delay`, `--aug-exp-decay`) — **2026-07-28에 mask/flip선택/가중치 인자 6개 삭제**, 증강이 좌우 flip 하나로 정리됐다(augmentation.md §0). 기본 전부 off/no-op(기존 동작 보존). `args`로 `create_data_loaders`→`DataTransform` 전달.
- **6b. ✅ [`train_part.py`](../utils/learning/train_part.py) `train()`**: 매 epoch `train_loader.dataset.transform.set_epoch(epoch)` 호출. **val_loader는 증강 off 보장**(검증=진짜 실력).
- 근거: 상세 맥락은 이 파일 하단 "증강 스케줄 배선 상세" 참조.

> ✅ **구현 완료 (2026-07-26)**: `train.py` `parse()`(`--aug-*` 5개 플래그) · `utils/data/load_data.py`의 `AugmentConfig(...)` 전달 · `utils/data/transforms.py`의 `prob()`(MRAugment `schedule_p` 형태로 재구현) · `utils/learning/train_part.py`의 epoch 루프(매 epoch `set_epoch` 호출 + `[aug]` 로그 + `--aug-on`인데 미배선이면 `RuntimeError`) 모두 배선 완료. 테스트: `scripts/tests/test_augment.py`(14개). 서버 검증: `scripts/verify_aug.sh`(전항목 통과).

### 7. ✅ bbox 손실 소비부 (배선 완료 · run12 롤백)
- `DataTransform`은 `return_bbox=True`로 bbox 파싱·flip 동기까지 됨(`_bbox_collate` 포함). **손실 소비부 배선 완료**: `train.py:48 --bbox-loss-lambda` · `train_part.py:29,69,90` · `loss_function.py:74 bbox_ssim_loss`. run12에서 λ=0.5로 실학습 → 퇴행 확인 후 기본값 0.0(off)으로 롤백([`training_anal/2026-07-18-run12-bbox-uc-analysis.md`](../training_anal/2026-07-18-run12-bbox-uc-analysis.md)).

### 8. ⬜ 학습 기록 장치 (Claude 입력용)
> 학습 소요 시간·중단 지점 같은 걸 사람이 손으로 옮겨 적지 않아도, Claude가 파일로 읽을 수 있게 하는 최소 장치.

- **epoch 로그**: train 루프가 epoch마다 한 줄 append — epoch 번호·시작/끝 시각·loss·val SSIM. `../result/<net_name>/train_log.jsonl` 한 파일이면 충분.
- **중단 기록**: KeyboardInterrupt·예외 시 마지막 줄에 `interrupted @ epoch N + 시각` 기록. 로그가 끊긴 지점 자체가 단서가 되게.
- **런 메타**: 시작 시 args 전체·git commit을 같은 폴더에 저장.
- 원칙: 나중에 Claude에게 말로 설명하게 될 정보는 학습 중에 파일로 남긴다. 거창한 도구 말고 로그 몇 줄이면 된다.

## 재현성 (전 항목 공통)
- 기본 off일 때 기존과 **비트 동일** 회귀 테스트(seed 고정). 최종 VESSL 재학습 재현 검증 대비.
- seed·split·라이브러리 버전·데이터 순서 리포지토리 커밋.

---

## 증강 스케줄 배선 상세 (구 todo.md, 설계 근거로 보존 — 반영 완료)

> 왜 train.py/train_part.py 두 곳인가: 증강 로직은 `transforms.py`에 있으나 **transform은 "지금 몇 epoch인지" 모른다**. epoch·설정을 위(train.py args)→아래(transform)로 내리는 통로가 6a·6b.
> 비유: train.py=볼륨 노브 설치 / train_part.py=곡 진행(epoch)에 따라 볼륨 조절 / transforms.py=실제 소리.

**6a train.py 주의**: 기본값 전부 off·보수적(재현성). ✅ 반영됨 — `DataTransform`의 `__init__`은 이미 `augmentor`/`is_train`/`return_bbox`/`normalize`/`readout_crop` 등을 전부 받도록 확장돼 있다(§6은 완료 상태).

**6b train_part.py 주의**:
```python
for epoch in range(start_epoch, args.num_epochs):
    train_loader.dataset.transform.set_epoch(epoch)   # ✅ 반영됨: train_part.py:396
    train_epoch(args, epoch, model, train_loader, optimizer, loss_type)
```
- val_loader는 `set_epoch` 미호출 또는 `aug_on=False`. `save_reconstructions`/`validate`는 안 건드림.

**GRAPPA 정렬 규칙(4번 배선 시)**: 기하변환 g는 {k-space, GRAPPA, target, bbox}에 동일 적용. mask증강은 GRAPPA 미적용. 이미지 도메인 g 한 곳 정의 → 모든 짝에 적용. 상세: model_strategy §1.1.
