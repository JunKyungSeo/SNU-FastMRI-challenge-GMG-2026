# shell34_todo.md — `UpBlock.up` 의 못 끄는 checkpoint

[promptmr_plus.py:177](../utils/model/promptmr/promptmr_plus.py#L177) 이
`self.up`(= `nn.Upsample(bilinear, ×2)` + 1×1 Conv)을 `ReentrantWrapper` 로 감싼다.
[reentrant_wrapper.py:28](../utils/model/promptmr/reentrant_wrapper.py#L28) 의
`use_checkpoint or DEFAULT_CHECKPOINT` 때문에 `False` 를 넘겨도 `True` 가 되어 **CLI 로는 못 끈다.**
`--no-use-checkpoint` 도 여기엔 안 닿는다.

iter 당 **75회 재계산** — up block 3개 × (감도맵 코일 15 + cascade 10).

| 끄면 빨라지는 폭 | +5.8% | 실측(CPU), nc4·nf16·C15 tiny. GPU 전이 미검증 |
|---|---|---|
| 출력·grad 차이 | 0.0 | 실측(CPU), 전 파라미터 bitwise 동일 |
| 추론 시간 이득 | 없음 | `self.training` 게이트라 eval 에선 이미 안 걸린다 |

## shell34.sh

[shell33.sh](shell33.sh) 의 `STAGE=train` 을 베끼되 스윕 축만 on/off 로 바꾼다.
같은 fold·seed 로 `--max-iters 60` 두 팔, s/iter 와 `[mem]` peak 비교. **+3% 미만이면 되돌린다.**

[test_checkpoint_equivalence.py](../scripts/tests/test_checkpoint_equivalence.py) 는 UpBlock
wrapper 를 안 건드리므로 이 레버를 못 덮는다. 등가성을 걸려면 한 판을 새로 짜야 한다.

Loss 는 같은 seed 라 앞자리가 같아야 정상이다. 비트 일치를 기준 삼지 말 것 —
`nn.Upsample(bilinear)` 의 backward 가 CUDA 에서 atomicAdd 라 원래 실행마다 흔들린다.

## 곁가지 — 추론의 감도맵 코일 루프

[adapter.py:37](../utils/model/promptmr/adapter.py#L37) 이 `compute_sens_per_coil=True` 를
하드코딩하는데, 추론은 `eval` + `no_grad` 라
[promptmr_plus.py:734](../utils/model/promptmr/promptmr_plus.py#L734) 의
`use_checkpoint and self.training` 이 항상 거짓이다 — 코일 15번 순차 호출만 남고 메모리 명분이 0 이다.
추론시간은 점수 항목이니 `sens_chunk` 를 코일 수까지 올려 `recon_eval.py` 시간 항을 비교할 것.
