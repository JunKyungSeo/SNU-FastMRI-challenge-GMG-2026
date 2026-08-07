## 시간 절약 전략 (학습시)

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

## 시간 절약 전략 (reconstruct시)

전제 둘: **모델은 안 건드린다**(cascade 수·폭·`n_history` 축소는 여기서 제외), **SSIM 은 유지한다**.
현재 747.3 ms/slice(`promptmrp_0801_moe`). 채점 시간은 `recon_slice` 호출 전체를 재므로
CPU 전처리와 H2D 전송도 안에 들어간다.

⚠️ 전처리를 `prep_volume` 으로 **옮기는** 건 `next_levers.md` L19 에서 이미 기각됐다 —
소유자가 전처리도 보고 시간에 포함돼야 한다고 정했다. 아래 항목은 옮기는 게 아니라
같은 자리에서 싸게 하는 것이라 그 결정과 무관하다.

### 프로파일링 (선행)

**현상.** 747ms 가 감도맵 / cascade U-Net / FFT / CPU 전처리 / H2D 중 어디에 얼마나 쓰이는지
잰 기록이 없다. 아래 항목의 기대치가 전부 추정이다.

**계획.** [probe_recon_vram.py](../scripts/probe_recon_vram.py) 가 이미
`load_model → prep_volume → recon_slice` 를 그대로 부르므로 여기에 `torch.profiler` 를 붙인다.

**범위.** 모델도 학습도 안 건드린다. 이걸 먼저 해야 나머지 순서가 정해진다.

**참고.** cuFFT 는 크기가 2^a·3^b·5^c·7^d 일 때 빠르다. 우리 W 는 368(=2⁴×23)·372(=2²×3×31)
가 대부분이라 큰 소인수가 끼어 있고 480(=2⁵×3×5)만 좋은 크기다. 패딩하면 출력이 바뀌니 고칠
수는 없고, FFT 비중을 볼 때 참고만 한다.

### 감도맵 배치화

**현상.** [adapter.py:39](../utils/model/promptmr/adapter.py#L39) 가 `compute_sens_per_coil=True`
를 무조건 넘겨 감도 U-Net 이 코일마다 순차로 15번 호출된다. 추론은 checkpointing 이 이미 꺼져
있어 메모리 명분이 없다. `next_levers.md` L11 로 등록돼 있다.

**계획.** 추론에서만 `compute_sens_per_coil=False` 또는 `sens_chunk=15`.

**기대치.** 3% 안팎. shell33 에서 학습 기준 chunk 8 이 4.19→4.07s(−2.9%)였다.
학습에서 chunk 15 는 OOM 이었지만 추론은 메모리가 남는다.

**걸리는 것.** 수치 동일 주장(코일이 독립이고 배치를 섞는 연산이 없다)이
L11 기록상 2026-07-31 이후 재검증되지 않았다. 가정으로 취급할 것.

### `recon_slice` CPU 전처리

**계획.** 비트 동일하게 줄일 수 있는 게 넷이다.

- [test_part.py:86](../utils/learning/test_part.py#L86) 의 마스크 곱을 `readout_crop` 뒤로.
  마스크는 W 축이라 H crop 과 교환 가능 → H 640 대신 384 에 곱하니 곱셈량 −40%.
- `readout_crop` 이 complex128 로 승격시켜 [99행](../utils/learning/test_part.py#L99) 의
  `astype` 으로 되돌린다 — 전체 복사 1회.
- `torch.stack((real, imag), -1)` 복사. complex64 는 실/허가 메모리상 인접이라
  `.view(np.float32)` 로 복사 없이 `(…,2)` 를 얻을 수 있다.
- `mask_t` 를 슬라이스마다 재생성한다. 볼륨 안에서 안 변한다.

**기대치.** 미측정. 프로파일링 뒤에 판단.

### cudnn.benchmark

**계획.** `load_model` 에서 켠다. 볼륨 안에서 shape 이 고정이고 하네스가 볼륨마다 5슬라이스를
warmup 으로 버리므로 autotune 비용이 거기 흡수된다.

**기대치.** 미측정. 어느 문서에도 없던 항목이다.

### CUDA Graphs

**계획.** 수천 개의 커널 호출을 warmup 구간에서 한 번 캡처해 통째로 재생한다.
batch=1 에서 커널 발사 오버헤드를 없애는 표준 해법이다.

**기대치.** 한 자릿수 %. 발사 오버헤드가 병목이 아니라는 신호가 우리 측정에 있다 —
shell33 에서 발사 횟수를 15회→2회로 줄였는데 −2.9% 였다. 747ms 면 연산 바운드일 공산이 크다.

**걸리는 것.** 캡처 중 host 동기화가 있으면 안 된다.
[promptmr_plus.py:233-247](../utils/model/promptmr/promptmr_plus.py#L233-L247) 의 argmin 결과를
인덱스로 쓰는 부분과 MoE 라우팅이 둘 다 동기화를 건다. 둘 다 볼륨 안에서 상수라 미리 뽑아두면 풀린다.

### 배제한 것

| 후보 | 사유 |
|---|---|
| Torch-TensorRT | FP32 유지 시 이득이 없다. 1080 은 FP16 이 FP32 와 같은 속도, INT8 은 느린 폴백. |
| cuFFT plan cache 확대 | 기본 용량이 CUDA 10 이상에서 4096. 우리 shape 종류는 몇 개뿐이라 thrashing 이 안 난다. |
| channels_last | 이득이 tensor core 와 묶여 있어 Pascal 에서 근거가 없다. 한 줄이라 재보는 건 싸다. |
| `--coil-compress` | 재학습이 필요하고, SSIM 중립 근거가 논문 인용이지 우리 실측이 아니다. |

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
