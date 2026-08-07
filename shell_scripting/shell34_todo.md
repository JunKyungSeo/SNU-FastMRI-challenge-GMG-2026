# shell34_todo.md — `UpBlock.up` 의 못 끄는 checkpoint 끄기

> 슬롯 미작성.

## 무엇인가

[promptmr_plus.py:177](../utils/model/promptmr/promptmr_plus.py#L177) 이
`self.up`(= `nn.Upsample(bilinear, ×2)` + 1×1 Conv)을 `ReentrantWrapper` 로 감싼다. 그런데
[reentrant_wrapper.py:28](../utils/model/promptmr/reentrant_wrapper.py#L28) 이
`use_checkpoint or DEFAULT_CHECKPOINT` 라, `False` 를 넘겨도 `False or True = True` 가 되어
**CLI 로는 절대 못 끈다.** `--no-use-checkpoint` 도 여기엔 안 닿는다.

호출 횟수: up block 3개 × (감도맵 코일 15 + cascade 10) = **iter 당 75회 재계산.**

| 항목 | 값 | 출처 |
|---|---|---|
| 끄면 빨라지는 폭 | +5.8% | 실측(CPU), nc4·nf16·C15 tiny. GPU 전이 미검증 |
| 출력·grad 차이 | 정확히 0.0 | 실측(CPU), 출력·전 파라미터 grad bitwise 동일 |
| 되돌려받는 peak | 약 56MB | 추정. 아끼는 건 4배 크기 업샘플 텐서뿐 |
| 추론 시간 이득 | **없음** | `self.training` 게이트라 eval 에선 이미 안 걸린다 |

마지막 줄이 중요하다 — 이건 **학습 시간만** 줄인다. 대회 점수의 추론시간 항에는 1ms 도 안 준다.

## ⚠️ wrapper 를 제거하지 말 것

걷어내면 state_dict 키가 바뀐다. `up.module.module.1.weight` 가 `up.1.weight` 로 이름이 바뀌고
`up._dummy_tensor` 는 통째로 사라지는데, 실측(CPU) 로 실 config(nc=10)에서 세어 보면 UpBlock
33개에 대해 각각 36개씩 걸려 있다(감도맵 3개는 `sens_net_ckp` 별칭으로 두 번 잡힌다).
파라미터 개수까지 달라져 `load_state_dict` 도 optimizer resume 도 깨진다 —
`../result/*/checkpoints/*.pt` 가 **전부** 못 쓰게 된다.

## 전략

건드릴 것은 **동작 하나**지 구조가 아니다. 그래서 생성자를 고치거나 모듈을 갈아끼우는 방향으로
가면 안 된다 — 위의 state_dict 문제가 그 길 끝에 있다.

빠져나갈 구멍은 `use_checkpoint` 가 평범한 인스턴스 속성이고
[reentrant_wrapper.py:34](../utils/model/promptmr/reentrant_wrapper.py#L34) 가 매 forward 에 그걸
다시 읽는다는 점이다. **모델을 다 만든 뒤에 그 속성만 덮어쓰면 생성자의 `or` 를 아예 안 거친다.**
모듈 트리도 파라미터도 그대로 남으므로 체크포인트 호환이 공짜로 보존된다.

그 덮어쓰기를 어디서 할지는 어댑터([adapter.py](../utils/model/promptmr/adapter.py))가 자연스럽다.
백본을 조립하는 유일한 지점이고, 이미 `use_checkpoint` · `sens_chunk` 같은 실행 옵션을 args 에서
받아 모델에 심는 자리다. 켜고 끄는 손잡이는 `--use-checkpoint` · `--sens-chunk` 와 같은 결로
CLI 플래그 하나를 새로 내고, **기본값을 지금 동작(on)으로** 두어 안 붙이면 아무것도 안 바뀌게 한다.
옛 체크포인트의 args 에는 그 플래그가 없으므로 기본값이 그대로 먹어 추론 경로도 안전하다.

## 건드리지 말 것

- `check_resume_compat` 의 `fields`([train_part.py:249](../utils/learning/train_part.py#L249)) —
  학습 결과를 안 바꾸는 값이라 넣으면 기존 ckpt 재개만 막힌다(`early_stop_patience` 와 같은 부류).
- `test_part.py` — 추론은 `eval()` 이라
  [reentrant_wrapper.py:34](../utils/model/promptmr/reentrant_wrapper.py#L34) 의 `and self.training`
  이 이미 거짓이다. 여기 배선을 더해도 장식일 뿐이다.

## 검증

1. `pytest scripts/tests/test_checkpoint_equivalence.py` 는 **이 레버를 안 덮는다** — 그 테스트는
   UpBlock wrapper 를 건드리지 않는다. 끈 상태와 켠 상태의 출력·전 파라미터 grad 를 float64 로
   비교하는 한 판을 추가할 것(실측(CPU) 로는 이미 bitwise 동일).
2. 기존 `best_model.pt` 하나로 `load_state_dict` — missing/unexpected 가 0 이어야 한다.

## 측정 (shell34.sh)

`shell33.sh` 의 `STAGE=train` 을 베끼되 스윕 축만 새 플래그의 on/off 로 바꾼다.
같은 fold·seed 로 `--max-iters 60` 두 팔, s/iter 와 `[mem]` peak 비교.

- 같은 seed 라 Loss 앞자리가 같아야 정상이다. **비트 일치를 기준으로 삼지 말 것** —
  `nn.Upsample(bilinear)` 의 backward 가 CUDA 에서 atomicAdd 라 원래 실행마다 흔들린다.
- +3% 미만이면 되돌린다.

## 곁가지 — 추론의 감도맵 코일 루프

[adapter.py:37](../utils/model/promptmr/adapter.py#L37) 이 `compute_sens_per_coil=True` 를
하드코딩한다. 이 루프의 명분은 **학습 때** 코일별 checkpoint 로 peak 을 낮추는 것인데,
추론은 `eval` + `no_grad` 라 [promptmr_plus.py:734](../utils/model/promptmr/promptmr_plus.py#L734)
의 `use_checkpoint and self.training` 이 항상 거짓이다 — 추론에서는 15번 순차 호출만 남고
메모리 명분이 0 이다.

추론시간은 점수 항목이니 잴 값은 있다. `test_part.py` 가 `model.use_checkpoint = False` 를 심는
자리([:39](../utils/learning/test_part.py#L39) · [:136](../utils/learning/test_part.py#L136))에서
`sens_chunk` 를 코일 수까지 올리고 `recon_eval.py` 의 시간 항을 비교하면 된다.
`--sens-chunk` 배선은 이미 있다.
