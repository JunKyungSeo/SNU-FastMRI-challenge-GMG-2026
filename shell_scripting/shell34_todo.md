# shell34_todo.md — 슬롯 34 할 일

## 시간 절약 전략

### `UpBlock.up` 의 못 끄는 checkpoint

**현상.** [promptmr_plus.py:177](../utils/model/promptmr/promptmr_plus.py#L177) 이
`self.up`(= `nn.Upsample(bilinear, ×2)` + 1×1 Conv)을 `ReentrantWrapper` 로 감싼다.
[reentrant_wrapper.py:28](../utils/model/promptmr/reentrant_wrapper.py#L28) 의
`use_checkpoint or DEFAULT_CHECKPOINT` 때문에 `False` 를 넘겨도 `True` 가 되어 **CLI 로는 못 끈다.**
`--no-use-checkpoint` 도 여기엔 안 닿는다. 그래서 up block 3개 × (감도맵 코일 15 + cascade 10) =
**iter 당 75회**가 매번 재계산된다.

**계획.** 이 checkpoint 를 끌 수 있게 만들고 GPU 에서 실제로 빨라지는지 잰다. 아끼는 메모리가
업샘플 텐서 한 장뿐이라 8GB 여유(현재 peak 3.3GB) 안에서 감당된다는 게 전제다.

**기대치.** CPU tiny(nc4·nf16·C15)에서 +5.8%. GPU 전이는 미검증이다 — 같은 세션에서 잰
`--sens-chunk` 는 CPU +39% 가 GPU 에서 0 이 됐으니 그대로 믿지 말 것. 다만 이쪽이 지우는 건
런치가 아니라 재계산(FLOP)이라 결이 다르다.

**안전.** 출력·전 파라미터 grad 가 실측(CPU) 로 bitwise 동일하다.

**범위.** 학습 시간만 준다. `self.training` 게이트라 추론(`eval`)에선 이미 안 걸리므로
대회 점수의 추론시간 항에는 영향이 없다.

**걸리는 것 둘.**
[test_checkpoint_equivalence.py](../scripts/tests/test_checkpoint_equivalence.py) 는 UpBlock
wrapper 를 안 건드려 이 레버를 못 덮는다. 그리고 GPU 에서는 비트 일치를 판정 기준으로 쓸 수 없다 —
`nn.Upsample(bilinear)` 의 backward 가 atomicAdd 라 원래 실행마다 흔들린다.
