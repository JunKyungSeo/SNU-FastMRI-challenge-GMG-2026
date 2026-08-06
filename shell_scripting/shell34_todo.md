# shell34_todo.md — 학습 속도 레버 A·B (슬롯 미작성)

> 2026-08-06. `shell33.sh`(레버 C = `--sens-chunk`)의 형제 문서다. 여기 두 항목은 **아직 코드가
> 없다.** A 는 재보기 전에 이미 기각됐고, B 만 살아 있다. 슬롯을 만들 근거가 생기면 그때 만든다.
>
> 수치 출처를 필드로 구분한다 — `실측(GPU)`(과거 로그) / `실측(CPU)`(이 문서 작성 시 계측) /
> `추정`(계산). 추정은 행동의 근거로 쓰지 않는다.
>
> **형제 항목 C 도 기각됐다** — 아래 「C 측정 결과」 참고. A·C 가 닫힌 지금 이 파일에 남은
> 살아 있는 후보는 B 하나뿐이다.

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

**모듈 구조는 그대로 두고 checkpoint 호출만 건너뛴다.**

### 배선

`use_checkpoint` 는 그냥 인스턴스 속성이고
[reentrant_wrapper.py:34](../utils/model/promptmr/reentrant_wrapper.py#L34) 가 매 forward 에 그걸
읽는다 — 즉 **생성 뒤에 대입하면 `or` 버그를 우회한다.** 그래서 생성자도 모듈 트리도 안 건드리고
어댑터에서 한 번 훑어 끄는 게 가장 짧다. state_dict 키·파라미터 개수가 그대로 남는다.

**1)** [train.py:40](../train.py#L40) 아래 — 레버를 CLI 로 뺀다(기본 on = 지금과 동일).

```python
parser.add_argument('--up-checkpoint', action=argparse.BooleanOptionalAction, default=True,
                    help='UpBlock 업샘플 checkpoint (기본 on = 기존과 동일). --no-up-checkpoint 로 '
                         '끄면 iter 당 75회 재계산이 사라진다. 값은 안 바뀌고 peak 만 ~56MB 는다.')
```

**2)** [adapter.py:26](../utils/model/promptmr/adapter.py#L26) 다음 — 이게 전부다.

```python
if not getattr(args, "up_checkpoint", True):
    from utils.model.promptmr.promptmr_plus import UpBlock
    for m in self.net.modules():
        if isinstance(m, UpBlock):
            m.up.use_checkpoint = False
```

`getattr` 기본값이 True 라, 옛 체크포인트의 args 로 `_build_expert`
([test_part.py:36](../utils/learning/test_part.py#L36))를 돌려도 지금과 같은 동작이 나온다.

**3) 선택** — 생성자가 거짓말을 멈추게. 레버에 필요하진 않지만 다음 사람이 또 속지 않게 하려면
[reentrant_wrapper.py:28](../utils/model/promptmr/reentrant_wrapper.py#L28) 을 고친다:
`self.use_checkpoint = self.DEFAULT_CHECKPOINT if use_checkpoint is None else use_checkpoint`.
[:570](../utils/model/promptmr/promptmr_plus.py#L570) 의 `sens_net_ckp` 는 어차피 직접 대입하므로
([:630](../utils/model/promptmr/promptmr_plus.py#L630)) 영향 없다.

### 건드리지 말 것

- `check_resume_compat` 의 `fields`([train_part.py:249](../utils/learning/train_part.py#L249)) —
  학습 결과를 안 바꾸는 값이라 넣으면 기존 ckpt 재개만 막힌다(`early_stop_patience` 와 같은 부류).
- `test_part.py` — 추론은 `eval()` 이라
  [reentrant_wrapper.py:34](../utils/model/promptmr/reentrant_wrapper.py#L34) 의 `and self.training`
  이 이미 거짓이다. 여기 배선을 더해도 장식일 뿐이다.

### 검증

1. `pytest scripts/tests/test_checkpoint_equivalence.py` 는 **B 를 안 덮는다** — 그 테스트는
   UpBlock wrapper 를 건드리지 않는다. 한 판을 추가할 것: `m.up.use_checkpoint` 를 뒤집어
   출력·전 파라미터 grad 를 float64 로 비교(`실측(CPU)` 로는 이미 bitwise 동일).
2. 기존 `best_model.pt` 하나로 `load_state_dict` — missing/unexpected 가 0 이어야 한다.

### 측정 (shell34.sh)

`shell33.sh` 의 `STAGE=train` 을 베끼되 스윕 축만 `--up-checkpoint` / `--no-up-checkpoint` 로
바꾼다. 같은 fold·seed 로 `--max-iters 60` 두 팔, s/iter 와 `[mem]` peak 비교.

- 같은 seed 라 Loss 앞자리가 같아야 정상이다. **비트 일치를 기준으로 삼지 말 것** —
  `nn.Upsample(bilinear)` 의 backward 가 CUDA 에서 atomicAdd 라 원래 실행마다 흔들린다.
- +3% 미만이면 되돌린다. C 가 compute-bound 에 막혔으니 여기도 같은 벽일 공산이 크다.
  다만 B 가 지우는 건 런치가 아니라 **재계산(FLOP)** 이라 결이 다르다 — 그래서 재볼 값은 있다.

**측정 전 주의**: GPU 에서는 "비트 동일" 을 검증 기준으로 삼지 말 것.
`nn.Upsample(bilinear)` 의 backward 가 atomicAdd 기반이라 CUDA 에선 원래 실행마다 비트가 흔들리고,
[utils.py:80-81](../utils/common/utils.py#L80-L81) 의 `cudnn.deterministic` 은 conv 만 덮는다.
방어 가능한 주장은 **수학적 항등**까지다(CPU 에서 bitwise 확인 완료).

---

## C 측정 결과 — `--sens-chunk` **기각 (closed-measured)**

`실측(GPU)` 2026-08-06, `logs/shell33_probe_20260806_065606.log` (GTX 1080 8GB, W=480, reps=3):

| chunk | fwdbwd | peak | chunk=1 대비 |
|---|---|---|---|
| 1 | 4.19s | 3.07GB | — |
| 3 | 4.11s | 3.07GB | −1.9% |
| 5 | 4.12s | 4.32GB | −1.7% |
| 8 | 4.07s | 6.23GB | −2.9% |
| 15 | OOM | — | — |

**시간이 안 준다.** chunk 3·5·8 이 4.11 / 4.12 / 4.07 로 비단조이므로 그 0.05s 폭이 reps=3 의
노이즈 바닥이고, chunk=1 과의 격차 0.08s 는 그 바로 위다. 반면 메모리는 chunk=5 부터 급증해
8 에서 6.23GB, 15 에서 터진다.

**왜**: 이 GPU 에서 감도맵 U-Net 은 launch-bound 가 아니라 **compute-bound** 였다. 코일 1개짜리
384×480 conv 가 이미 1080 을 채우고 있어 묶어도 벌 게 없다. 배선 근거로 삼았던 CPU 측정
(+39%)은 전이되지 않았다 — CPU 에서 배치 묶기가 이득인 건 BLAS 스레딩 효과였고 GPU 와 무관했다.

**남긴 것**: `--sens-chunk` 배선 자체는 유지한다(기본 1 = 비트 동일이라 비용 0). 아래 곁가지
항목과 다른 GPU 로의 이전에서 다시 쓸 수 있다. 학습에는 `--sens-chunk 1` 이 정답이다.

**되살릴 조건**: launch-bound 인 GPU(더 빠른 SM · 같은 shape) 로 옮기거나, 폭이 크게 작아져
코일당 conv 가 GPU 를 못 채우게 될 때. 1080 에서는 다시 꺼내지 말 것.

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

⚠️ **기대치를 낮출 것.** 위 C 측정이 이 GPU 는 compute-bound 라고 말한다 — 학습에서 안 나온
이득이 추론에서 나올 이유는 약하다. 다른 점은 역전파·checkpoint 가 없어 작업 단위가 작고
런치 비중이 상대적으로 크다는 것뿐이다. 재는 건 싸니(채점 1회) 해볼 수는 있으나,
0 이 나와도 놀랄 일이 아니다.
