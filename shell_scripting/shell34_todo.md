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

### cascade checkpoint 일부만 끄기 (예비용)

**계획.** cascade 10개 중 2개만 checkpoint 를 끈다. 그 2개의 재계산이 사라져 약 3% 준다.

**메모리.** cascade 1개를 안 감싸면 +1.395GB(실측(CPU), 384×480·C15·nf48). 전부 끄면 +13.9GB 라
8GB 에 안 들어가지만, 2개면 +2.8GB 로 현재 peak 3.3GB 에서 6.1GB 가 된다. 남는 여유가 1.9GB 뿐이라
W=480 슬라이스에서 단편화가 겹치면 터질 수 있다 — 예비용인 이유다.

**안전.** 값은 비트 동일이라 성능 확인이 필요 없다.

## 점수 개선

### EMA

**현상.** PromptMR+ 원 논문의 EMA 가 우리 코드에 없다. `ema`·moving average·SWA 어느 이름으로도
구현이 없고, 벤더 출처인 `sMartify-fastmri/2025_final` 에도 없다.

**계획.** [train_part.py](../utils/learning/train_part.py) 에 파라미터 지수이동평균 사본을 두고
매 step 갱신한다. validate 와 `save_model` 은 원본 대신 이 사본을 쓴다.

**기대치.** +0.000x~+0.001. 0 일 수도 있다.

**범위.** 추론에는 비용이 안 붙는다. 모델 크기·속도가 그대로라 대회 점수의 추론시간 항이 안 움직인다.
학습 중에만 파라미터 사본 1개(nc10·nf48 기준 72.9M × 4B ≈ 292MB)를 더 쓴다.

**걸리는 것 셋.**
`save_model` 의 `'model'` 키에 EMA 가중치를 넣을지 원본 옆에 별도 키로 넣을지 정해야 한다 —
[test_part.py:145](../utils/learning/test_part.py#L145) 가 `checkpoint['model']` 만 읽으므로
별도 키로 가면 recon 쪽도 같이 고쳐야 한다. `--resume` 은 EMA 상태까지 복원해야 이어학습이 맞는다.
그리고 decay 값과 초반 warmup 을 정해야 한다.
