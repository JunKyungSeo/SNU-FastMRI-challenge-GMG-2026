# 설계 — 증강 활성화 배선(A) + 증강 ON/OFF A/B(B)

- 날짜: 2026-07-26
- 관련: [`strategy/augmentation.md`](../../../strategy/augmentation.md) · [`strategy/implementation.md`](../../../strategy/implementation.md) §6a·6b · [`CLAUDE.md`](../../../CLAUDE.md) §10
- 공식 참조: MRAugment (Fabian et al., NeurIPS 2021) — 논문 `papers/04_MRAugment.pdf`, 공식 repo `github.com/MathFLDS/MRAugment`

---

## 1. 배경 — 왜 지금 이걸 하나

**증강 코드는 있지만 한 번도 학습에 쓰인 적이 없다(dead).**
`MRIAugmentor`([transforms.py](../../../utils/data/transforms.py) L203-303)는 좌우 flip(k-space·mask·target·bbox·GRAPPA 동기) + ACS 보존 mask thinning까지 구현돼 있고 `load_data.py` L193-209까지 배선돼 있다. 그러나

1. **`train.py`에 `--aug-*` 인자가 없다** → `getattr(args,"aug_on",False)`가 항상 False → `augmentor=None`.
2. **`train_part.py`가 `set_epoch`을 호출하지 않는다** → 스케줄이 epoch을 모른다.
3. **스케줄 기본값이 런 길이와 어긋난다** — `_prob()`는 `warmup_epochs=10` 미만이면 0을 반환하는데 실제 런은 `-e 10`(epoch 0~9) → **전 구간 `p=0`**. 위 1·2를 고쳐도 증강은 0회 발생하고, `p`를 로깅하지 않아 **조용히** 아무 일도 안 일어난다.

그리고 **현재 백본(PromptMR+ nc10/nf48)은 10 epoch 안에 과적합이 관측됐다**(사용자 확인, `training_anal` 미기록 런). 이는 MRAugment의 핵심 실측(Fig 3a: no-DA는 정점 후 하락, DA는 4× 더 오래 상승)이 우리 사례에 그대로 해당한다는 뜻이고, 데이터 규모(fold2 train 4343 slice)도 논문이 "이득이 크다"고 명시한 low~moderate regime(≈4k) 하단이다. 증강을 켜 볼 근거가 두 다리로 선다:

- **과적합 = 4.3k 슬라이스 암기** → flip(유효 데이터 2배)이 직접 값어치.
- **acc8 약점**(run11 채점: acc4 full 0.9390/bbox 0.9361 vs acc8 0.8933/0.8828) → mask-aug(acc4를 acc8 난이도로)가 그 격차를 정면으로 겨냥.

> ⚠️ `training_anal`의 "오래 = 좋음, 학습량이 지배 레버"(run9·run10)는 **작은 VarNet [16,32]×4** 기준이다. 현재 백본에는 전이되지 않는다 → 문서 정정이 이 작업의 산출물에 포함된다.

## 2. 목표 · 비목표

**목표**: "증강 ON이 과적합을 늦추고 best 점수를 올리는가"를 **동일 fold·seed에서 판정 가능한 상태로 만들고, 판정한다.**

**비목표(이번에 안 한다)**: phase/noise/라인 지터 활성화 · translation · TTA · consistency 손실 · GRAPPA prior 배선 · bbox 가중 손실 · label→FFT 합성 k-space(별개 베팅, 논문 §3.1·§4.5가 naive 형태를 실험으로 기각) · 증강 강도(intensity) 스케줄.

---

## 3. A — 배선 (B의 선행조건)

A 없이는 두 arm이 비트 단위로 동일해지므로 실험이 성립하지 않는다. 범위는 **실험을 가능하게 하는 최소**로 한정한다.

### A1. `_prob()`를 공식 `schedule_p()` 형태로 교체

현재는 `1-exp(-3t/ramp_epochs)`로 **총 epoch과 무관**하다. 그래서 (a) 10ep 런에서 영구 0, (b) 같은 플래그라도 런 길이에 따라 실제 증강량이 말없이 달라져 **epoch 수가 다른 run 간 비교가 오염된다.**

공식 repo(`mraugment/data_augment.py`)와 동일한 형태로 바꾼다:

```python
D, T, t, p_max = cfg.delay, cfg.total_epochs, self.epoch, cfg.max_prob
if t < D:                      return 0.0
if cfg.schedule == 'constant': return p_max
if T - D <= 0:                 return p_max          # 가드(T<=D인 짧은 런)
if cfg.schedule == 'ramp':     return p_max * (t - D) / (T - D)
k = cfg.exp_decay / (T - D)                          # exp_decay 기본 5.0
return p_max / (1 - exp(-(T - D) * k)) * (1 - exp(-(t - D) * k))
```

- `AugmentConfig`: `warmup_epochs`·`ramp_epochs`를 **`delay`·`total_epochs`·`exp_decay`로 교체**. `schedule`은 `'exp'|'ramp'|'constant'`.
- `total_epochs`는 `load_data.py`에서 `args.num_epochs`로 채운다 → 런 길이에 자동 스케일.
- 상류와 동일하게 epoch은 `0..T-1`이므로 `p`가 정확히 `p_max`에 닿지는 않는다. **상류 동작이라 그대로 둔다**(임의 보정 금지).

### A2. 변환별 가중치 분리 (`p_i = p · w_i`)

현재 flip·mask가 같은 `p`를 공유하고, mask만 `rng.random()` 2회 draw로 실효 확률이 `p·(1-orig_mask_prob)`이라 A/B 해석이 꼬인다. 공식 repo의 `--aug_weight_*`에 대응하는 `weight_flip`·`weight_mask`(기본 1.0)를 두고, **draw를 하나로 합친다**:

```python
p_flip = clip(p * cfg.weight_flip, 0, 1)
p_mask = clip(p * cfg.weight_mask * (1 - cfg.orig_mask_prob), 0, 1)   # 원본 고정마스크 혼합 유지
```

`p_mask` 값은 기존 2회 draw와 **확률적으로 동일**하다(순수 정리 + 가중치 추가). RNG 소비 횟수가 2→1로 바뀌지만 증강이 실행된 적이 없어 재현성 부채가 없다.

### A3. `train.py` 인자 (implementation.md §6a)

`load_data.py`의 `getattr` 키와 1:1로 맞춰 추가 — 그래야 `load_data.py`는 `total_epochs` 전달과 키 2개 교체(`aug_warmup_epochs`·`aug_ramp_epochs` → `aug_delay`·`aug_exp_decay`) 외에 손댈 게 없다.

`--aug-on`(store_true) · `--aug-schedule`(exp) · `--aug-max-prob`(0.55) · `--aug-delay`(0) · `--aug-exp-decay`(5.0) · `--aug-flip`/`--aug-mask`(BooleanOptionalAction, 기본 True) · `--aug-weight-flip`/`--aug-weight-mask`(1.0) · `--aug-mask-thin-max`(4) · `--aug-orig-mask-prob`(0.5)

**기본 `--aug-on` off → 기존 학습과 100% 동일.** phase/noise는 비목표이므로 인자를 추가하지 않는다(`AugmentConfig` 기본 off 유지).
`AugmentConfig`의 `max_prob` 기본값도 0.5 → **0.55**(논문 기본)로 맞춰 CLI 기본값과 어긋나지 않게 한다.

### A4. `train_part.py` epoch 전달 (implementation.md §6b)

`train()`의 `for epoch in range(start_epoch, args.num_epochs)`(L389) 첫 줄에 `train_loader.dataset.transform.set_epoch(epoch)` 추가. `--resume`도 루프 안이라 자동으로 맞는다.

**val은 구조적으로 이미 안전하다** — `create_kfold_data_loaders`가 val에 `is_train=False`를 주고 `augmentor = MRIAugmentor(cfg) if (is_train and aug_on) else None`이므로 val 로더엔 augmentor가 붙지 않는다. 이 불변식을 테스트로 고정한다(A5).

### A5. 관측성 — "조용히 p=0"을 다시 못 하게

이번 사안의 근본 원인이 무증상 실패다. 두 겹으로 막는다.

- **메인 프로세스 epoch 로그**: 매 epoch 시작에 `aug: p=… (p_flip=… p_mask=…)` 1줄. `p`는 epoch만의 함수라 메인에서 정확히 계산된다.
- **CPU 단위 테스트** `tests/test_augment.py` (GPU·데이터파일 불필요, 합성 numpy 배열 사용):
  1. `_prob()` — `T=10, D=2`에서 `p(0)=p(1)=0`, **`p(2)=0`**(`t=D`는 지수식이 0), `t=3..9` 단조 증가하며 전부 > 0, `p(9) < p_max`. **기존 기본값(10ep 런에서 전 구간 0)이 재발하면 실패**하는 회귀 테스트.
  2. flip 정확성 — `flip(target) == aug_target` 오차 0, `recon(K_flip) ≈ flip(recon(K))` (~1e-7), 샘플 라인 수 보존, mask 공식 `roll(flip(M),1)`.
  3. mask thinning — 결과가 항상 원본 마스크의 **부분집합**, **ACS 라인 100% 보존**.
  4. bbox 동기 — `x' = W - x - width`, `y/width/height/label` 불변.
  5. 결정성 — 같은 `(seed, epoch, fname, slice)`면 동일 결과(프로세스 재실행 포함).
  6. 배선 불변식 — `is_train=False`(val) 및 `isforward=True`에는 augmentor가 붙지 않는다.
  7. `--aug-on` 없이 만든 `DataTransform`은 6-튜플을 그대로 반환(기존 언패킹 비트 동일).

> 워커 프로세스(`--num-workers 4`)에서는 augmentor가 복사되므로 **적용 횟수 카운터를 메인에서 집계할 수 없다.** 그래서 "실제 발생 여부"는 위 CPU 테스트로 보증하고, 학습 로그에서는 `p`만 찍는다.

---

## 4. B — A/B 실험

### B0. baseline arm 확보 — 재학습하지 않는다

과적합이 관측된 런이 **현재 HEAD와 동일 config·fold·seed**라면 그 런이 그대로 OFF arm이다. 그러면 새로 돌릴 건 ON arm 하나(≈21h)뿐이고 42h가 아니다.

1단계는 **대조**다: 그 런의 `net_name`·shell 슬롯·`val_loss_log`를 찾아 config를 HEAD와 비교한다. 어긋나면 baseline도 재학습해야 하므로 **예산이 2배가 되는 유일한 갈림길**이다.
같은 로그에서 **val이 꺾인 epoch**을 읽어 `--aug-delay = (꺾인 epoch − 1)`로 정한다 — `p`가 과적합 구간을 덮게 하는 게 목적이다. (10ep·`D=2`·decay 5면 ep5에 `p≈0.86·p_max`.)

### B1. 증강 config — 조합 하나로 시작

`--aug-on --aug-schedule exp --aug-max-prob 0.55 --aug-delay <B0> --aug-exp-decay 5.0` + flip·mask 둘 다 ON, 가중치 1.0/1.0, `orig-mask-prob 0.5`, `mask-thin-max 4`.

`p_max=0.55`는 논문 기본값(§4.1). flip·mask를 처음부터 분리 측정하면 3런(≈63h)이라 예산이 안 맞는다 → **조합 승패를 먼저 보고, 이기면 쪼갠다**(가중치로 분리 가능하게 배선하는 이유).

### B2. shell 슬롯 (GPU 실행은 사람이)

`shell_scripting/shell22.sh` = [shell21.sh](../../../shell_scripting/shell21.sh) 복제 + 증강 인자. `FOLD`·`KFOLD_SEED`·나머지 config는 baseline과 **완전 동일**(변경은 증강 인자 하나뿐).

**스모크 게이트**(`SMOKE=1`, 120 iter)의 목적을 재정의한다:
- `pytest tests/test_augment.py` 전부 통과 (CPU, 큐 밖에서 즉시 실행)
- 로그에 `p > 0`이 찍히는지
- **iter time을 baseline과 비교** — 증강은 샘플당 FFT 2회(15×640×368 complex)를 dataloader에 추가한다. `--num-workers 4`에서 CPU가 병목이 되면 21h가 늘어난다. 늘어난 폭을 여기서 수치로 확인하고, 크면 `--num-workers` 상향으로 대응한다.

### B3. 판정 — 과적합 서명 3개 + 채점 4숫자

⚠️ 제약: k-fold 서버엔 leaderboard 데이터가 없어 `recon_eval` 채점을 못 한다(shell21 주석). 판정을 2단으로 나눈다.

- **1차(이 서버, 필수)** — fg-masked hold-out val(run11에서 실채점 예측력 확인)로 논문 Fig 3a의 서명을 본다: ① **best-epoch val 상승** ② **val이 꺾이는 epoch 지연** ③ **best epoch index 후퇴**. ①이 없고 ②③만 있으면 "과적합은 늦췄으나 점수 이득 없음"으로 기록한다.
- **2차(채점 가능한 환경, 확정)** — `recon_eval`로 `acc4/acc8 × full/bbox` 4숫자. **bbox 비퇴행이 하드 게이트**(채점 0.5·full + 0.5·bbox).

### B4. 실패 시 다음 수 — 미리 하나만 정해둔다

| 결과 | 판정 · 다음 수 |
|---|---|
| bbox 유의 하락 | **즉시 탈락**(full 이득과 무관). 롤백. |
| acc8만 상승, acc4 하락 | mask 강도 문제 → `--aug-mask-thin-max` 하향(4→2) 1회 재시도 |
| 전체 패배 | `--aug-max-prob 0.3`으로 1회 재시도. 그 이상은 안 한다. |
| 승리 | flip·mask 분리 A/B → 그다음 **epoch 연장**(과적합 지점이 밀렸으므로 다시 레버) |

### B5. 산출물

- `training_anal/2026-…-aug-analysis.md` — `training_anal` 관례(직전 대비 diff 표 + 서명 3개 + 4숫자)
- `training_anal` / `strategy`의 "오래 = 좋음"(작은 VarNet 기준) 결론에 **현재 백본은 10ep 내 과적합** 정정 배너
- `strategy/implementation.md` §6a·6b를 완료 처리, `CLAUDE.md` §10에서 "증강 활성화 배선" 제거

---

## 5. 성공 기준 (검증 가능한 형태)

1. `pytest tests/test_augment.py` 전부 통과 — 특히 `_prob()` 회귀 테스트가 "10ep 런에서 전 구간 0"을 잡는다.
2. `--aug-on` 없이 돌린 학습이 변경 전과 **동일**(6-튜플·손실 경로 불변, 테스트 7).
3. `SMOKE=1 bash shell_scripting/shell22.sh`가 exit 0이고 로그에 `p > 0`이 찍히며, iter time 증가폭이 수치로 기록됨.
4. ON arm 완주 후 서명 3개가 baseline과 나란히 표로 정리되고, B4 표에 따라 **승/패가 한 줄로 판정**됨.

## 6. 열린 항목 (플랜 1단계에서 값으로 채운다)

- 과적합 런의 식별자(`net_name`/fold)와 `val_loss_log` 경로 → `--aug-delay` 숫자, baseline 재사용 가능 여부.
- 2차 판정(`recon_eval`)을 어느 환경·언제 돌릴지.
