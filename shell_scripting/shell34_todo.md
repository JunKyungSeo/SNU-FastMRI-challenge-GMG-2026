# shell34_todo.md — 학습 속도 레버 A·B (슬롯 미작성)

> 2026-08-06. `shell33.sh`(레버 C = `--sens-chunk`)의 형제 문서다. 여기 두 항목은 **아직 코드가
> 없다.** A 는 재보기 전에 이미 기각됐고, B 만 살아 있다. 슬롯을 만들 근거가 생기면 그때 만든다.
>
> 수치 출처를 필드로 구분한다 — `실측(GPU)`(과거 로그) / `실측(CPU)`(이 문서 작성 시 계측) /
> `추정`(계산). 추정은 행동의 근거로 쓰지 않는다.

---

## A. cascade checkpoint 만 따로 끄기 — **기각 (closed-measured)**

**아이디어**: `--use-checkpoint` 한 플래그가 (1) 코일별 감도맵 checkpoint 와 (2) cascade 루프
checkpoint 를 동시에 켠다([adapter.py:36](../utils/model/promptmr/adapter.py#L36) →
[promptmr_plus.py:631](../utils/model/promptmr/promptmr_plus.py#L631) ·
[:646](../utils/model/promptmr/promptmr_plus.py#L646)). cascade 쪽은 연산량이 훨씬 큰데
아끼는 메모리는 작으니, 쪼개서 cascade 만 끄면 공짜로 빨라질 것이다.

**기각 사유 — 전제가 100배 틀렸다.**

"cascade 당 activation ~15MB" 는 [`strategy/promptmr_plus.md`](../strategy/promptmr_plus.md) 의
표를 읽은 값인데, 그 표는 **64×64 · n_feat0≈6 · CPU** 짜리 장난감 측정이다. 문서 스스로
"이 95%를 CUDA peak 절감으로 읽지 말 것" 이라고 경고해 두었는데도 그 델타를 실 shape 에
그대로 옮긴 것이 이 계획의 오류였다.

실 shape(readout-crop 384 × 최악폭 480 · 코일 15 · n_feat0 48 · fp32)에서 다시 재면:

| | cascade 1개가 붙드는 activation |
|---|---|
| cascade checkpoint 켬 (현재) | 0.141 GB |
| cascade checkpoint 끔 | 1.536 GB |
| **차이** | **+1.395 GB / cascade → nc=10 이면 +13.9 GB** |

`실측(CPU)`: `torch.autograd.graph.saved_tensors_hooks` 로 역전파용으로 살아남는 텐서 바이트를
중복 제거하고 합산. 감도맵 checkpoint 는 켠 채로, cascade 루프만 통과시켜 측정.

GPU 로그 두 건이 같은 방향을 이미 가리키고 있었다:

- `실측(GPU)` [`logs/shell13_20260721_002541.log:12-13`](logs/shell13_20260721_002541.log) —
  `OOM cascades=4 ... ckpt=False coils=15 HxW=384x480`. 같은 조건 `ckpt=True` 는 `peak=2.03GB`.
  **nc=4 에서 이미 8GB 를 넘긴다.**
- `실측(GPU)` [`logs/shell18_20260723_153942.log:8,110`](logs/shell18_20260723_153942.log) —
  `--no-use-checkpoint --num-cascades 18` 이 19.67GiB 카드를 epoch 0 에서 터뜨림(19.28GiB 할당).

즉 cascade checkpointing 은 "메모리는 안 아끼면서 시간만 먹는 것" 이 아니라 **nc=10 이 8GB 에
들어가게 만드는 바로 그 장치**다. 8GB 에서도, 20GB 에서도 못 끈다.

**부수적으로 얻은 사실**: 같은 shell13 로그의 `ckpt=True` 기울기(nc 4→12 에서 2.03→2.73GB)는
**cascade 당 87MB**다. 문서 표의 0.6MB 가 아니다. 그 87MB 의 대부분은 checkpoint 를 켜도
경계에 남는 `history_feat` 이므로, 실질 메모리 레버는 cascade 가 아니라 `--n-history` 쪽이다.

**되살릴 조건**: VRAM 이 24GB 이상인 카드로 학습을 옮기고, 그 카드에서 `nc=10 · W=480 ·
ckpt=False` 가 실제로 fit 하는 것을 탐침으로 확인한 뒤. 그 전엔 다시 꺼내지 말 것.

---

## B. `UpBlock.up` 의 못 끄는 checkpoint — **open**

**무엇인가**: [promptmr_plus.py:177](../utils/model/promptmr/promptmr_plus.py#L177) 이
`self.up`(= `nn.Upsample(bilinear, ×2)` + 1×1 Conv)을 `ReentrantWrapper` 로 감싼다. 그런데
[reentrant_wrapper.py:28](../utils/model/promptmr/reentrant_wrapper.py#L28) 이
`use_checkpoint or DEFAULT_CHECKPOINT` 라, `False` 를 넘겨도 `False or True = True` 가 되어
**CLI 로는 절대 못 끈다.** `--no-use-checkpoint` 도 여기엔 안 닿는다.

호출 횟수: up block 3개 × (감도맵 코일 15 + cascade 10) = **iter 당 75회 재계산**.

| 항목 | 값 | 출처 |
|---|---|---|
| 끄면 빨라지는 폭 | +5.8% | `실측(CPU)`, nc4·nf16·C15 tiny. GPU 전이 미검증 |
| 출력·grad 차이 | 정확히 0.0 | `실측(CPU)`, 출력·전 파라미터 grad bitwise 동일 |
| 되돌려받는 peak | 약 56MB | `추정`. 아끼는 건 4배 크기 업샘플 텐서뿐 |
| 추론 시간 이득 | **없음** | `self.training` 게이트라 eval 에선 이미 안 걸린다 |

마지막 줄이 중요하다 — 이건 **학습 시간만** 줄인다. 대회 점수의 추론시간 항에는 1ms 도 안 준다.

### ⚠️ wrapper 를 **제거하지 말 것**

`ReentrantWrapper` 를 걷어내면 state_dict 키가 바뀐다:

```
지금 : ...dec_level3.up.module.module.1.weight  +  ...dec_level3.up._dummy_tensor
제거 : ...dec_level3.up.1.weight
```

`실측(CPU)`: tiny net 에서 `_dummy_tensor` 10개 확인. 실 config 에서는 unexpected 72 / missing 36
으로 `load_state_dict` 가 죽고, 파라미터 개수가 달라져 optimizer resume 까지 깨진다.
`../result/*/checkpoints/*.pt` 가 **전부** 못 쓰게 된다.

**안전한 방법은 하나뿐** — 모듈 구조는 그대로 두고 checkpoint 호출만 건너뛴다:

1. [reentrant_wrapper.py:28](../utils/model/promptmr/reentrant_wrapper.py#L28) 의 `or` 를 고친다.
   `self.use_checkpoint = self.DEFAULT_CHECKPOINT if use_checkpoint is None else use_checkpoint`
2. [promptmr_plus.py:177](../utils/model/promptmr/promptmr_plus.py#L177) 에서
   `ReentrantWrapper(self.up, use_checkpoint=False)` 로 넘긴다.

이러면 `_dummy_tensor` 도 `up.module.module.*` 키도 그대로 남아 기존 체크포인트가 100% 로드된다.
[promptmr_plus.py:570](../utils/model/promptmr/promptmr_plus.py#L570) 의 `sens_net_ckp` 는
`or` 를 우회해 직접 대입하므로([:630](../utils/model/promptmr/promptmr_plus.py#L630)) 영향 없다.

### 슬롯이 할 일

1. 위 2줄 수정 → `pytest scripts/tests/test_checkpoint_equivalence.py` (등가성이 이미 강제됨)
2. `--max-iters 60` 짧은 학습 두 판(수정 전/후)으로 s/iter · `[mem]` peak 비교
3. GPU 에서 +5% 가 안 나오면 그냥 되돌린다. 잃을 게 없다

**측정 전 주의**: GPU 에서는 "비트 동일" 을 검증 기준으로 삼지 말 것.
`nn.Upsample(bilinear)` 의 backward 가 atomicAdd 기반이라 CUDA 에선 원래 실행마다 비트가 흔들리고,
[utils.py:80-81](../utils/common/utils.py#L80-L81) 의 `cudnn.deterministic` 은 conv 만 덮는다.
방어 가능한 주장은 **수학적 항등**까지다(CPU 에서 bitwise 확인 완료).

---

## 곁가지 — 추론의 감도맵 코일 루프 (미착수)

[adapter.py:37](../utils/model/promptmr/adapter.py#L37) 이 `compute_sens_per_coil=True` 를
하드코딩한다. 이 루프의 명분은 **학습 때** 코일별 checkpoint 로 peak 을 낮추는 것인데,
추론은 `eval` + `no_grad` 라 [promptmr_plus.py:734](../utils/model/promptmr/promptmr_plus.py#L734)
의 `use_checkpoint and self.training` 이 항상 거짓이다 — 즉 **추론에서는 15번 순차 호출만 남고
얻는 게 없다.**

추론시간은 점수 항목이므로 잴 가치가 있다: `test_part.py` 가 `model.use_checkpoint = False` 를
심는 자리([:39](../utils/learning/test_part.py#L39) · [:136](../utils/learning/test_part.py#L136))
에서 `model.sens_chunk` 를 코일 수로 올려보고 `recon_eval.py` 의 시간 항을 비교하면 된다.
`--sens-chunk` 배선은 이미 있다(`shell33.sh` 참고).
