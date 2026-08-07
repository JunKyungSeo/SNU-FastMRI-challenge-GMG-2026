## 최종 run 이름

`promptmrp_GMG_final_0807` — 체크포인트는 `../result/promptmrp_GMG_final_0807_{shared,acc4,acc8}/`.

## Learning Rate 스케줄

세 단계 모두 base **2e-4**, 하한 **1.8e-5**(= base × 0.09).
[build_scheduler](../utils/learning/train_part.py#L243) 가 이미 이 모양을 그린다 —
`[0, warmup)` linear → `[warmup, decay_start)` base 유지 → 거기서 끝까지 cosine.

### shared — 30 epoch

- **0~5 epoch**: 2e-4 고정 (warmup 없음)
- **5~30 epoch**: cosine decay, 하한 1.8e-5

`-e 30 -l 2e-4 --lr-decay-start-epoch 5 --lr-min-factor 0.09`

### expert acc8 — 20 epoch (전역 30~50)

⚠️ **epoch 번호가 두 가지로 돈다.** 사람은 전역(shared 0~30 → expert 30~50)으로 세고,
`train.py` 의 `--lr-decay-start-epoch` 은 그 단계 안에서 0부터 세는 **로컬** 값을 받는다.
전역에서 2 빼려면 로컬에서도 2를 뺀다 — 아래 표의 두 열이 같은 지점이다.

| 구간 | 전역 | 로컬 | LR |
|---|---|---|---|
| linear warm-up | 30~33 | 0~3 | → 2e-4 |
| 유지 | 33~**38** | 3~**8** | 2e-4 (5 epoch) |
| cosine decay | **38**~50 | **8**~20 | → 1.8e-5 (12 epoch) |

`-e 20 -l 2e-4 --warmup-iters 3270 --lr-decay-start-epoch 8 --lr-min-factor 0.09`

**변경 이력.** 원안은 decay 시작이 전역 40(로컬 10)이었고 2026-08-07 에 **38** 로 앞당겼다.
그만큼 상수 구간이 줄어 사진의 "10 epoch 유지" 가 **5 epoch 유지**가 된다 — 줄어든 5 epoch 이
decay 쪽으로 넘어가 감쇠가 7 → 12 epoch 으로 길어진다. 총 20 epoch 은 그대로다.

⚠️ `--warmup-iters` 는 epoch 도 데이터로더 iter 도 아닌 **optimizer-step 단위**다
([train_part.py:248](../utils/learning/train_part.py#L248)). epoch 당 optimizer step =
`ceil(min(--max-iters, 4358) / --accum-steps)` = `ceil(2179/2)` = **1090** 이므로
3 epoch = 3 × 1090 = **3270**. `--max-iters` 나 `--accum-steps` 를 바꾸면 같이 바꿀 것.

(2026-08-07 정정 — 이 자리에 `--accum-steps 1` 로 계산한 6537 이 적혀 있었다. 실제 BASE_ARGS 는
`--accum-steps 2` 라 6537 이면 warm-up 이 3 epoch 이 아니라 **6 epoch** 으로 늘어난다.)

### expert acc4 — 10 epoch (전역 30~40)

acc8 의 비율(warm-up 15% · 유지 25% · decay 60%)을 10 epoch 에 맞춰 축소한다.

| 구간 | 전역 | 로컬 | LR |
|---|---|---|---|
| linear warm-up | 30~32 | 0~2 | → 2e-4 |
| 유지 | 32~34 | 2~4 | 2e-4 |
| cosine decay | 34~40 | 4~10 | → 1.8e-5 |

`-e 10 -l 2e-4 --warmup-iters 2180 --lr-decay-start-epoch 4 --lr-min-factor 0.09`

⚠️ **acc8 의 "전역 38" 을 그대로 가져오면 안 된다.** acc4 는 전역 40 에서 끝나므로 38 부터
감쇠하면 decay 가 2 epoch 뿐이다. 그래서 절대 지점이 아니라 비율로 옮겼다.

### early stopping — shared 8 · 전문가 6 · tail 0

| 런 | `--early-stop-patience` | 근거 |
|---|---|---|
| shared (30 ep) | **8** | 상수 LR 구간 5 + 여유 3 |
| expert acc4 · acc8 본체 | **6** | 상수 LR 구간 5 + 작은 val(20 vol · 540 slice) 몫 |
| hold-out tail (2 ep) | **0** | val 이 계기판 역할을 못 한다 |

**끄지 않는 이유.** 12.9일짜리 런에 자동 중단 장치가 하나도 없는 상태는 위험하다. `0=off`
([train.py:77](../train.py#L77))이고, 발동해도 `best_model.pt` 는 이미 저장돼 있어 잃는 게 없다.

**shell31 값(shared 6 · 전문가 3)을 그대로 쓰면 안 된다.** 특히 전문가의 3은 상수 LR 구간
5 epoch(로컬 3~8) **안에서 발동할 수 있다.** 이 스케줄은 상수 구간에서 val 이 몇 epoch
평평하다가 cosine 이 시작되면서 떨어지는 게 설계 의도라, patience 가 그 구간보다 짧으면
정상 동작 중인 런을 죽인다.

**과적합 보험이 아니다.** 12 epoch shared 런은 12 epoch 전부가 NewRecord 였고 val 이 한 번도
반등하지 않았다 — 우리 조건에서 과적합은 아직 관측된 적이 없다. 위 값들은 "발동 안 하는 게
정상, 8번 연속 안 좋아지면 진짜 사고" 라는 뜻의 바닥이다.

**tail 은 반드시 0.** 그 구간의 val loss 는 방금 학습한 데이터를 재는 값이라 개선/비개선
판정 자체가 성립하지 않는다(아래 "hold-out 을 막판에 학습에 합치기" 의 체크포인트 선택과 같은 이유).

### 학습 스크립트 — 완료

단계마다 파일 하나. 위의 LR·early stop·증강 값이 전부 여기 박혀 있다.

| 파일 | 단계 |
|---|---|
| [train_final_shared.sh](train_final_shared.sh) | shared 30 epoch |
| [train_final_acc8.sh](train_final_acc8.sh) | acc8 전문가 20 epoch |
| [train_final_acc4.sh](train_final_acc4.sh) | acc4 전문가 10 epoch |
| [train_final_tail_acc8.sh](train_final_tail_acc8.sh) | acc8 tail 2 epoch (hold-out 합침) |
| [train_final_tail_acc4.sh](train_final_tail_acc4.sh) | acc4 tail 2 epoch (hold-out 합침) |
| [_final_base.sh](_final_base.sh) | 공통 인자 + 실행 함수 + `promote_last_epoch` (슬롯 아님) |

실행 순서: `shared` → (`acc8`, `acc4`) → (`tail_acc8`, `tail_acc4`). tail 은 본 런의
`best_model.pt` 에서 `--init-from` 으로 출발하는 **별도 net** 이라 본 런 가중치를 안 건드린다.

다섯 스크립트 전부 `train.py` 의 실제 파서로 파싱해 값을 확인했다(스텁 python 으로 argv 를 떠서
`parse()` 에 그대로 먹임). tail 은 `-l` 이 두 번 들어가는데(BASE_ARGS 2e-4 + 델타) argparse 가
뒤를 채택하는 것까지 실측했다 — 앞이 이겼으면 tail 이 2e-4 로 도는 참사다.

공통 인자를 `_final_base.sh` 한 곳에 둔 이유는 **세 단계의 모델 인자가 같아야 `--init-from`
이 성립**하기 때문이다. 세 파일에 복붙했다가 한쪽만 고치면 shape mismatch 로 죽는다.

~~`train_expert_acc4.py` / `train_expert_acc8.py`~~ — 위 `.sh` 가 같은 역할을 이미 해서
2026-08-07 에 안 만들기로 했다. Python 쪽은 `train.py` 하나를 계속 쓴다.

### augmentation 코드도 단계별로 분리 — 완료

단계마다 파일 하나. `_final_base.sh` 의 `run_final` 이 `"${AUG_ARGS[@]}"` 로 끼워 넣는다.

| 파일 | 값 |
|---|---|
| [_aug_shared.sh](_aug_shared.sh) | flip 공통 + `--mask-aug full` |
| [_aug_acc4.sh](_aug_acc4.sh) | flip 공통 + `--remask 4` + `--mask-aug offset` |
| [_aug_acc8.sh](_aug_acc8.sh) | flip 공통 + `--remask 8` + `--mask-aug offset` |

flip 공통 = `--aug-on --aug-schedule constant --aug-max-prob 0.5 --aug-delay 0` (세 파일에 복붙.
빼내면 파일이 4개가 되는데 아낀 건 4토큰뿐이다).

**Python 은 안 나눴다.** 세 단계의 증강 *로직* 은 완전히 같고 값 6개만 다르다 →
[transforms.py](../utils/data/transforms.py) 의 `MRIAugmentor`·마스크 재생성 블록과
[load_data.py:266-368](../utils/data/load_data.py#L266) 은 그대로 두고, 갈라지는 **값 선언만**
셋으로 뺐다. Python 모듈 셋으로 만들면 CLI 와 preset 이 같은 값을 두 곳에서 정하게 되거나
`MRIAugmentor` 껍데기 셋이 남는다. `.sh` 로 나눈 건 `shell26.sh:113` 의 `AUG_COMMON=(...)`
과 같은 관용구다.

호출부가 `source` 를 빼먹으면 `run_final` 이 `[STOP] AUG_ARGS 없음` 으로 즉사한다(실측).
`set -u` 에 기대지 않고 명시 가드(`[ -n "${AUG_ARGS+x}" ]`)를 둔 이유는 **bash 4.4 부터
빈 배열의 `"${a[@]}"` 가 `set -u` 위반에서 빠졌기** 때문이다 — 여기(macOS 3.2)와
서버(Linux 5.x)의 동작이 같다는 근거가 없는데, 증강이 조용히 빠진 채 12일을 도는 사고는
로그로도 안 잡힌다.

동결 테스트: `python3 -m pytest scripts/tests/test_final_stage_args.py -q`
— 스텁 `python` 을 PATH 앞에 두고 세 스크립트를 실제로 돌려 증강 인자를 잡아 위 표와 대조한다.

**걸리는 것 둘.**

- `mask_aug` 는 resume 시 **일치를 강제하는 필드**다
  ([train_part.py:305](../utils/learning/train_part.py#L305)). 나누면서 저장되는 args 의
  **필드 이름이 바뀌면 기존 체크포인트에서 `--resume`·`--init-from` 이 막힌다.**
  값도 이름도 그대로 두고 코드 위치만 옮길 것.
- 전문가의 `--mask-aug offset` 은 **지금 배선에서 no-op** 이다 —
  [479행](../utils/data/transforms.py#L479)이 `--remask-offset` 기본값 `uniform` 아래에서는
  `--mask-aug` 유무와 무관하게 같은 `remask_offset(...)` 로 떨어진다(offset 재추첨은
  `--remask` 가 이미 하고 있다). 나누면서 "이건 안 쓰네" 하고 지우면 위의 resume 이 깨진다.
  이름값을 못 해도 남겨둘 것.

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

## 시간 절약 전략 (추론시)

전제 둘: **모델은 안 건드린다**, **SSIM 은 유지한다**. 현재 747.3 ms/slice(`promptmrp_0801_moe`),
VRAM 여유 약 3.5GB. 채점 시간은 `recon_slice` 호출 전체를 재므로 CPU 전처리와 H2D 도 안에 들어간다.

⚠️ 전처리를 `prep_volume` 으로 **옮기는** 건 `next_levers.md` L19 에서 기각됐다 — 전처리도 보고
시간에 포함돼야 한다는 소유자 결정. 아래는 옮기는 게 아니라 같은 자리에서 싸게 하는 것이다.

### 0. 프로파일링 (선행)

747ms 가 감도맵 / cascade U-Net / FFT / CPU 전처리 / H2D 중 어디에 쓰이는지 잰 적이 없다.
아래 기대치는 전부 추정이다. [probe_recon_vram.py](../scripts/probe_recon_vram.py) 가 이미
`load_model → prep_volume → recon_slice` 를 그대로 부르니 `torch.profiler` 만 붙이면 된다.
이걸 먼저 해야 나머지 순서가 정해진다.

### 1. cudnn.benchmark — 남는 VRAM 이 실제로 값을 하는 지점

conv 계산법은 GEMM·FFT·Winograd 등 여러 가지고 빠를수록 workspace 를 크게 먹는다. 메모리가
빠듯하면 그 할당이 실패해 빠른 후보가 목록에서 빠지고 느린 게 선택된다. **즉 benchmark 를 켜는
것과 VRAM 을 비워두는 건 한 세트다.** `load_model` 에서 켠다 — 볼륨 안 shape 이 고정이고
하네스가 볼륨마다 5슬라이스를 warmup 으로 버려 autotune 비용이 거기 흡수된다.

- **VRAM 증가**: workspace 는 conv 하나 도는 동안만 사는 임시 버퍼라 peak 증가 ≈ 가장 큰 것 하나.
  최대 중간 텐서가 72ch × 384×480 fp32 ≈ 53MB 이니 **수백 MB, 많아야 0.5GB** 로 본다(추정).
- **위험 없음**: autotune 은 할당 실패하면 그 알고리즘을 후보에서 뺄 뿐이라 OOM 으로 안 죽는다.
  안 되면 지금과 같은 선택으로 돌아간다.
- `torch.backends.cudnn.benchmark_limit` 로 시도 개수도 올릴 수 있다.

### 2. 감도맵 배치화

[adapter.py:39](../utils/model/promptmr/adapter.py#L39) 가 `compute_sens_per_coil=True` 를 무조건
넘겨 감도 U-Net 이 코일마다 15번 순차 호출된다. 추론은 checkpointing 이 꺼져 있어 메모리 명분이
없다(`next_levers.md` L11). 추론에서만 `sens_chunk=15`.

- **기대치**: 3% 안팎. shell33 에서 학습 기준 chunk 8 이 4.19→4.07s(−2.9%).
- **VRAM 증가**: sens U-Net 중간 텐서가 전부 15배 → **1GB 안팎**(추정). 1번과 합치면 1.5GB 근처.
- **걸리는 것**: 수치 동일 주장이 L11 기록상 2026-07-31 이후 재검증되지 않았다. 가정으로 취급.

### 3. `recon_slice` CPU 전처리 — 비트 동일하게 줄일 수 있는 넷

- [test_part.py:86](../utils/learning/test_part.py#L86) 마스크 곱을 `readout_crop` 뒤로.
  마스크는 W 축이라 H crop 과 교환 가능 → H 640 대신 384 에 곱하니 −40%.
- `readout_crop` 의 complex128 승격 → [99행](../utils/learning/test_part.py#L99) `astype` 복사 1회.
- `torch.stack((real, imag), -1)` 복사. complex64 는 실/허가 인접이라 `.view(np.float32)` 로 무복사.
- `mask_t` 를 슬라이스마다 재생성한다. 볼륨 안에서 안 변한다.

### 4. 할당자 — 측정 구간 안에서 `cudaMalloc` 이 안 터지게

볼륨마다 W 가 바뀌어 블록 크기가 달라지면 새 할당이 생길 수 있고 `cudaMalloc` 은 동기화를
동반한다. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 면 세그먼트가 하나의 연속 가상
주소로 자라 블록이 이웃과 합쳐진다. 메모리가 남을 때 특히 안전하다.

### 5. CUDA Graphs

수천 개 커널 호출을 warmup 에서 한 번 캡처해 통째로 재생. batch=1 의 표준 해법이고, 전용 정적
메모리 풀이 필요한데 지금은 VRAM 이 남아 감당된다.

- **기대치**: 한 자릿수 %. 발사 오버헤드가 병목이 아니라는 신호가 있다 — shell33 에서 발사를
  15회→2회로 줄였는데 −2.9%. 747ms 면 연산 바운드일 공산이 크다.
- **걸리는 것**: 캡처 중 host 동기화 금지.
  [promptmr_plus.py:233-247](../utils/model/promptmr/promptmr_plus.py#L233-L247) 의 argmin 결과
  인덱싱과 MoE 라우팅이 동기화를 건다. 둘 다 볼륨 안 상수라 미리 뽑아두면 풀린다.

### 배제한 것

| 후보 | 사유 |
|---|---|
| Torch-TensorRT | FP32 유지 시 이득 없음. 1080 은 FP16 이 FP32 와 동속, INT8 은 느린 폴백. |
| cuFFT plan cache 확대 | 기본 4096(CUDA 10+). 우리 shape 종류는 몇 개뿐이라 thrashing 이 안 난다. |
| channels_last | 이득이 tensor core 와 묶여 Pascal 에선 근거 없음. 한 줄이라 재보는 건 싸다. |
| 슬라이스 배치화 | 한 호출에서 여러 장 계산해 캐시하면 L19 와 같은 측정 회피. |
| `--coil-compress` | 재학습 필요, SSIM 중립 근거가 논문 인용이지 우리 실측이 아님. |
| k-space W 패딩 | cuFFT 는 2^a·3^b·5^c·7^d 가 빠른데 우리 W 는 368(2⁴×23)·372(2²×3×31)가 대부분. 패딩하면 출력이 바뀐다. FFT 비중 볼 때 참고만. |

## 점수 개선

### 용량 상향 — cascade 10→12 · n_history 3→11

**결정.** 최종 run 은 `--num-cascades 12 --n-history 11` 로 간다. 둘 다 PromptMR+ 원본
기본값([promptmr_plus.py:561](../utils/model/promptmr/promptmr_plus.py#L561))이고, 우리가
8GB 때문에 10·3 으로 줄여 쓰던 것을 되돌리는 방향이다. 작년 수상 두 팀도 원본 config 를 썼다.

2026-08-07 **재확인** — "아키텍처 무변경" 안을 검토했다가 기각했다. 바꾸는 건 이 둘뿐이고
**폭 48 · 감도맵 24 · adj 1 은 그대로** 다.

**VRAM.** [shell36](shell36.sh) 실측(crop-h 384 · W 368 · C15 · `--with-optimizer`):

| cascade | n_history | param | fwd+bwd | probe peak |
|---|---|---|---|---|
| 10 | 3 (현재) | 76.0M | 3.54s | 2.65GB |
| 12 | 3 | 90.8M | 4.05s | 2.91GB |
| 10 | 11 | 78.3M | 3.76s | 3.86GB |
| **12** | **11** | **93.6M** | **4.34s** | **4.34GB** |

두 레버의 비용이 반대다 — `n_history` 는 VRAM 을 +1.21GB 먹고 시간은 +0.22s 뿐인데,
cascade 는 VRAM +0.26GB 에 시간 +0.51s 다. history 버퍼가 cascade 를 가로질러 사는
상주 텐서라 gradient checkpointing 으로 안 줄어드는 게 이유다.

**8GB 여유.** probe peak 2.65GB 와 실학습 peak 4.87GB 의 차 2.2GB 가 로더·단편화 몫이다.
가산으로 보면 12:11 의 실학습 peak 는 **6.5GB 근처**. 학습셋 최대 폭 480 짜리 파일에서
더 붙으므로 첫 epoch `[mem]` 줄을 반드시 확인할 것.

**시간.** slice 당 4.71s → 5.51s(×1.17). 확정된 30/20/10 배분에 곱하면 이렇게 된다
(무변경 열은 shell31 실측 — shared 1 epoch 5.89h · expert 1 epoch 2.93h):

| 단계 | epoch | 10:3 (무변경) | **12:11** |
|---|---|---|---|
| shared | 30 | 176.7h | **207h** |
| expert acc8 | 20 | 58.6h | **68.6h** |
| expert acc4 | 10 | 29.3h | **34.3h** |
| **합** | 60 | 264.6h (11.0일) | **310h (12.9일)** |

용량 상향의 값은 **+45h ≈ 1.9일**이다. 8/7 에 시작하면 종료가 8/18 이 아니라 **8/20** 이므로
남은 대회 일정이 이걸 감당하는지 먼저 확인할 것.

**⚠️ 반대 증거 — 같은 시간을 주면 작은 arm 이 이겼다.**
[shell32](shell32.sh) 가 벽시계 시간을 맞추고(train 약 2820s) 잰 값이다.

| arm | iter | ValLoss |
|---|---|---|
| **10:3:48** | 600 | **0.132 / 0.135** |
| 12:11:48 | 481 | 0.1377 |
| 10:11:48 | 568 | 0.145 |

같은 시간이면 작은 모델이 iter 를 더 밟아 이긴다는 뜻이다. 다만 **1 epoch·600 iter·scratch**
라 전체 2179 iter 의 27% 이고, 용량의 이득은 늦게 나오는 성질이라 30 epoch 에서 뒤집힐
여지가 있다. 노이즈 바닥도 안 쟀다. **그래도 이 축의 유일한 직접 측정은 "작은 쪽이 이겼다"
이므로, 용량을 올리는 건 그 측정을 뒤집는 데 30 epoch 을 거는 선택이라는 걸 알고 갈 것.**

**첫 epoch 에서 반드시 볼 것 둘.** ① `[mem]` 줄 — 예상 6.5GB 가 8GB 안에 실제로 들어가는지.
② 1 epoch 소요 — 위 표의 6.9h 가 맞는지. 둘 중 하나라도 어긋나면 12일짜리 런이 시작부터
어긋난 상태로 도는 것이라, 며칠 뒤가 아니라 **첫 epoch 끝에서** 판단해야 한다.

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

→ b27c2d8 로 구현됨. 그런데 원복이 깨진다. 아래.

### EMA swap 이 별칭 파라미터에서 원복 실패 — 최종 run 전 필수

**현상.** [train_part.py:44](../utils/learning/train_part.py#L44) 의 `swap` 이 `state_dict()` 를
순회하며 제자리 교환을 한다. `state_dict()` 는 `named_parameters` 와 달리 **중복 제거를 안 한다.**
같은 텐서가 m개 키로 나오면 m번 교환되고, m≥2 면 swap 은 자기 역원이 아니라 위수 3 순열
(모델, shadow₁, shadow₂) → (shadow₂, 모델, shadow₁) 이 된다. 46행 docstring 의 "자기 역원" 이
거짓이다. 결과로 검증 뒤 학습이 θ 가 아니라 **θ_EMA 에서 재개**되고 shadow 두 벌이 epoch 마다 벌어진다.

별칭은 둘, 둘 다 [promptmr_plus.py](../utils/model/promptmr/promptmr_plus.py) 안이다. 프로덕션
config 에서 state_dict 1979키 중 431키가 중복이다.

- [555 / 570행](../utils/model/promptmr/promptmr_plus.py#L570) `self.sens_net_ckp = ReentrantWrapper(self.sens_net)`
  — 같은 모듈을 두 이름으로 등록. 126 param × 2키.
- [277행](../utils/model/promptmr/promptmr_plus.py#L277) `act=nn.PReLU()` mutable default argument
  — import 때 만들어진 PReLU 하나를 전 모델이 공유. 스칼라 1개가 306 경로. cascades 까지 걸친다.

177행 `self.up = ReentrantWrapper(self.up)` 는 같은 이름 재대입이라 별칭이 아니다(대조군).

**결과 2.** `save_model` 이 [385행](../utils/learning/train_part.py#L385)에서 desync 된 shadow 를
그대로 넣고, `load_state_dict` 는 등록 순서상 뒤쪽 키가 이긴다 → `best_model.pt` 는 **감도맵만 θ,
나머지는 θ_EMA 인 하이브리드**다. 커밋이 약속한 "best 는 θ_EMA" 가 감도맵에 한해 거짓이다.

**계획.** 570행 삭제는 안 된다 — 기존 ckpt 가 `strict=True` 에서 깨진다. 소비자 쪽에서 접는다.
`EMA.__init__` 이 `data_ptr` 로 대표 키를 뽑아 두고 `update`·`swap`·`load_state_dict` 가 대표 키만
돌게 한다. 저장물 키 집합은 그대로라 체크포인트·resume 호환이 안 변한다.

**걸리는 것.** [test_ema.py:26](../scripts/tests/test_ema.py#L26) `test_swap_is_involution` 은
`nn.Sequential` 이라 별칭이 없어 이 버그를 **원리적으로** 못 잡는다. 별칭 있는 케이스를 더해야 한다.

**영향 범위.** EMA 커밋은 08-07 03:42 UTC, 이 체크아웃 최신 로그는 08-06 17:47 UTC → 저장소
기록상 EMA 시대에 완주한 run 이 없다. 0.9336 포함 기존 체크포인트는 깨끗하다. 서버는 확인 못 했다 —
`../result/*/checkpoints/model.pt` 에 `'ema'` 키가 non-None 인 게 있는지만 사람이 보면 결판난다.

### MoE 추론에서 acc4 전문가가 acc8 의 PReLU 로 돈다 — 지금 점수에 이미 먹혔다

**현상.** 위 공유 PReLU 가 **인스턴스 경계도 넘는다**(실측: 모델 둘을 따로 만들어도 같은 저장소.
모델2 를 0.99 로 덮으면 모델1 도 0.99, 독립이면 0.25). [test_part.py:58-59](../utils/learning/test_part.py#L58)
가 한 프로세스에서 전문가 둘을 만들고 [46행](../utils/learning/test_part.py#L46)이 각각 로드하므로
**나중에 로드한 acc8 이 이긴다.**

**계획.** `import copy` 후 전문가 하나를 `copy.deepcopy`. deepcopy 는 모델 **내부** 공유는 보존하면서
바깥 공유만 끊는다.

**범위.** `promptmrp_0801_moe`(0.9336) 는 이 상태로 채점된 값이다. 고치면 모델이 실제로 달라지므로
**재채점이 필요하다.** 크기는 두 체크포인트의 slope 차이에 달려 미지수, 방향은 train/test 정합 쪽.

**하지 말 것.** 277행을 `act=None` 으로 고치는 근본 수정. 키 이름이 안 바뀌어 로드는 호환이지만
**학습 의미가 바뀐다**(전역 1개 slope → CAB 마다 1개). 최종 run 앞두고 0.9336 과의 비교가능성이
깨진다. 대회 끝나고.

### hold-out 을 막판에 학습에 합치기

**현상.** 지금은 fold 하나(200 volume 중 40)를 끝까지 빼둔다. val 곡선을 보려면 그래야 하지만,
제출 가중치는 결국 데이터의 80% 만 본 물건이 된다. 교과서 해법은 "하이퍼파라미터 확정 후 100%
데이터로 재학습"인데 shared 30 epoch 이 200시간대라 재학습은 선택지가 아니다. 그래서 후반부만
갈아타는 값싼 버전을 쓴다 — 그 전제를 알고 쓸 것.

**언제 갈아타나.** 스케줄은 그대로 두고 **마지막 2 epoch** 만 hold-out 을 합쳐 돈다 —
acc8 은 18 + **2**, acc4 는 8 + **2**.

**무엇을 얼마나 넣나.** 한 epoch 의 총 슬라이스는 지금과 같은 **2179 장**(`--max-iters 2179`)
으로 고정한다. 그 안에 **hold-out 은 전부** 넣고, 모자란 만큼만 기존 train 풀에서 뽑는다.

| | hold-out (전부) | train 에서 추출 | 합 |
|---|---|---|---|
| expert acc4 | 548 (20 vol) | 1631 / 4358 | 2179 |
| expert acc8 | 536 (20 vol) | 1643 / 4358 | 2179 |

슬라이스 수는 shell31 로그의 `[K-Fold]` 줄 실측이다 — train 160 vol · 4358 slice, val fold
40 vol · 1084 slice(acc4 20 vol 548 · acc8 20 vol 536)
([ACC4](logs/shell31_ACC4_20260804_024838.log#L8) · [ACC8](logs/shell31_ACC8_20260805_022840.log#L8)).

**추출은 epoch 마다 다시 뽑는다.** 두 epoch 이 같은 1631 장을 보면 train 풀의 나머지 2727 장은
tail 동안 아예 안 쓰인다. 매번 다시 뽑으면 tail 이 훑는 train 커버리지가 37% → **61%** 가 된다.

⚠️ **hold-out 이 볼륨 지분(20/180 = 11%)의 2.3배인 25% 를 차지한다.** "전부 넣는다" 는 결정에서
자동으로 따라오는 결과다 — 이 2 epoch 동안 hold-out 슬라이스는 정확히 **2회**, train 슬라이스는
평균 **0.75회** 지나간다. 새 데이터를 확실히 보여주는 게 목적이니 의도한 대로지만, 이 구간은
hold-out 쪽으로 기운 분포에서 도는 학습이라는 걸 알고 쓸 것.

**어느 단계에.** acc8·acc4 전문가에만. shared 에 붙이면 그 뒤 전문가 단계가 다시 hold-out 뺀
데이터로 돌면서 효과가 씻긴다. shared 는 초기값일 뿐이라 정직한 val 곡선을 남겨두는 쪽이 낫다.

**LR.** 갈아타는 지점의 LR 을 그대로 이어받아 계속 내린다. 데이터가 바뀌었다고 다시 올리면 앞에서
쌓은 걸 흔든다. acc8 스케줄(로컬 8→20 cosine · 하한 0.09×base)의 로컬 18(전역 48) 지점이면
**3.0e-5**, base 의 1/7 쯤 — 거기서 원래 하한 1.8e-5 까지 가면 원래 곡선의 꼬리와 그대로 겹친다.
LR 을 다시 올리자는 안은 기각했다: `--grad-clip 0.1` 이 스텝 대부분을 잘라 이동량이 LR 에 정비례하는데,
tail 2 epoch 은 LR 을 어떻게 잡아도 전체 이동량의 **2~4%** 라 손대봐야 위험만 는다.

**체크포인트 선택.** 갈아탄 순간부터 val loss 는 자기가 학습한 데이터를 재는 것이라 못 믿는다.
그 구간에선 best 자동 선택도 early stopping 도 끄고 **마지막 epoch 을 쓴다.** 애초에 이 짓의
전제가 "epoch 수는 앞의 hold-out 구간에서 이미 정했다" 이므로 마지막을 쓰는 게 맞다.

⚠️ **그런데 "마지막 epoch 을 쓴다" 가 그냥은 안 된다.** 채점·병합부는 `best_model.pt` 만 읽는데
([test_part.py:52·143](../utils/learning/test_part.py#L52) · `recon_eval.py` 는 경로 플래그 자체가
없다 · [merge_moe.py:30](../scripts/merge_moe.py#L30)), 마지막 epoch 은 `model.pt` 에만 있다.
그리고 `cp model.pt best_model.pt` 는 **EMA 를 조용히 버린다** — `save_model` 이
`model.pt['model']` 에는 원본 θ 를, `best_model.pt['model']` 에는 θ_EMA 를 넣기 때문이다
([train_part.py:385-403](../utils/learning/train_part.py#L385)). 그대로 복사하면 0.9336 을 포함한
기존 점수가 전부 θ_EMA 기준이었는데 **다른 물건**을 제출하게 되고, 로그 어디에도 안 나타난다.
→ `_final_base.sh` 의 `promote_last_epoch` 이 `model.pt['ema']['shadow']` 를 `best_model.pt` 로
옮긴다. tail 슬롯이 학습 성공 시 자동으로 부른다.

**판정.** leaderboard 로만 — 학습에 안 넣고 남겨둔 유일한 정직한 자다. 갈아타기 **전** 가중치를
지우지 말고 따로 남겨 둘 다 채점하고 이긴 쪽을 낸다. 데이터 20% 를 더 보는 게 항상 이득은 아니고,
이 구간엔 그걸 판별할 계기판이 없으니 백업이 판별 장치를 대신한다.

**비용.** epoch 당 슬라이스를 2179 로 고정하므로 **학습 시간이 안 는다** — tail 2 epoch 은 지금
전문가 2 epoch 과 같은 시간이다. 값은 시간이 아니라 train 풀에서 1631~1643 장이 빠지고 그 자리를
hold-out 이 차지하는 것이다.

**배선 — 완료 (`--holdout-tail`).** `HoldoutTailSampler`
([load_data.py](../utils/data/load_data.py))가 hold-out 전량 + 풀에서 `n − hold` 무작위를 뽑아
섞는다. `n` 은 `--max-iters` 에서 **유도**한다(그 값을 절단 장치로 쓰지 않는다 —
`len(loader) ≤ max_iters` 라 `train_epoch` 의 break 가 안 걸려서, epoch 전체에 흩어진 hold-out 이
잘려나갈 일이 없다). epoch 마다 재추첨되는 건 샘플러가 `DataLoader.generator` 를 **공유**하고
`train()` 이 epoch 마다 `manual_seed(seed + epoch)` 를 부르기 때문 — 별도 epoch 배선이 0줄이다.

검증: `python3 -m pytest scripts/tests/test_holdout_tail.py -q` (6개, GPU·데이터 불필요).
플래그 off 면 `RandomSampler` 로 되돌아가 기존 경로와 동일한 것까지 본다.

⚠️ **`--max-iters` 를 바꿔서 재개하지 말 것.** `n` 이 샘플러가 섞는 **순열의 길이**라 1만 달라져도
epoch 구성이 통째로 바뀐다 — 중간 재개가 다른 순열에서 `skip_iters` 를 세어 hold-out 을 두 번 보거나
아예 안 본다. `--holdout-tail` 이면 `max_iters` 도 resume 일치 필드에 넣어 막아뒀다
([train_part.py](../utils/learning/train_part.py#L320)). 상수 LR 이라 원래는 가드 밖이었다.
