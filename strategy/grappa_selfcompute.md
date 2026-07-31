# GRAPPA 자체 계산 — 손익 판정과 배선 설계

작성 근거: 이 repo의 코드(읽은 파일은 각 절에 파일:줄로 표시), 세 갈래 사전조사, 그리고 사용자가 확정해 준 전제(리더보드엔 `image_grappa`가 없다 / 우리가 ACS로 직접 계산할 수 있다 / 계산 시간은 `recon_slice()` 안에서 정직하게 계상한다).

---

## 1. 결론

**한다. 다만 학습을 걸기 전에 CPU 두 판과 GPU 한 판으로 먼저 검증한다.** 시간은 이 결정의 제약이 아니다 — 제대로 짠 torch-CUDA 구현이면 슬라이스당 +10~25 ms 안팎이고(추정), 시간 항의 계수를 어떻게 잡아도 손익분기 ΔSSIM이 0.003을 넘지 않는다. 그러니 판정은 오직 "GRAPPA가 SSIM을 올리는가"에 달렸는데, 그 답을 우리는 아직 모른다. 사용자가 지목한 반전은 절반 맞다. 자체 계산하면 complex multi-coil k-space를 얻으므로 `strategy/rejected.md:30·32`가 적은 "위상 날조" 기각 사유는 실제로 사라지고 주입이 훨씬 깨끗해진다. 그러나 코드를 읽어보니 PromptMR+에서 `img_zf`는 x0이면서 동시에 **매 cascade의 DC 기준점**이라(`promptmr_plus.py:498-505`), 거기에 GRAPPA로 채운 k-space를 그대로 꽂으면 DC 잔차가 구조적으로 틀어진다. x0 대체가 가능해진 것이지 drop-in이 된 건 아니고, 두 역할을 분리해야 한다. 그래서 채택 형태는 **버퍼의 5번째 항으로 붙이고 `img_zf`는 손대지 않는 V2**다. 이 방식은 새 입력 채널 2개를 0으로 초기화하면 현재 best 체크포인트와 비트 동일하게 동작하므로, A/B가 2.5~3.5 GPU일짜리 재학습이 아니라 **미세조정**이 된다 — 이번 판정에서 비용을 가장 크게 깎는 사실이다. 그리고 반전이 뒤집지 못한 위험이 하나 남는다. GRAPPA는 모델이 이미 받는 (언더샘플 k-space, ACS)의 결정론적 함수이고 PromptMR+는 그 ACS에서 감도맵을 뽑아 10 cascade 내내 DC를 건다. 새 정보가 아니라 귀납편향일 뿐이라 이득이 0일 개연성이 실재한다. 다행히 이 가설은 **학습 없이 recon_eval 한 판으로 검증된다**(관문 3). 그 한 판을 먼저 돌린 뒤에 학습 예산을 태워라.

---

## 2. 손익 계산

### 2.1 시간 항의 계수를 우리는 모른다

정직하게 적어두면, `0.5·SSIM_full + 0.5·SSIM_bbox + 추론시간`이라는 식과 "~0.00n 단위"라는 서술은 이 repo 안에서 2026-07-09 커밋 하나에서 인용 없이 처음 나왔고 이후 문서 열여섯 곳이 그 한 줄을 복사했을 뿐이다. 조직측이 준 파일 — 초기 커밋의 baseline README, 고정 하네스 `recon_eval.py` — 어디에도 시간이 점수에 들어간다는 말이 없다. `recon_eval.py:137-145`는 세 축을 각각 print만 하고 합산식을 갖고 있지 않다. 공식 FAQ도 아직 "평가방식은 추후 안내"다.

검색으로 회수한 후보 환산식이 하나 있다. τ(T) = 0.001·(2000−T)/1920 (T는 ms/slice, 80ms 이하면 0.001, 2000ms 이상이면 0). 출처가 죽은 임시 터널이라 원문 확인이 안 되지만, 시간점수 0점 지점 2000 ms가 하네스 상수 `VOLUME_TIME_WARN_S = 60.0`(recon_eval.py:30)을 볼륨당 평균 27.7슬라이스로 나눈 2169 ms와 10% 안에서 맞물린다. 조직측 숫자 둘이 같은 지점을 가리키니 개연성은 있다. 다만 **사람이 대회 공지에서 직접 확인해야 한다.** 웹으로는 여기까지가 한계다.

그래서 아래 표는 계수 α(점수/(ms·slice⁻¹))를 세 가지로 놓고 손익분기를 병렬로 제시한다.

### 2.2 손익분기 표

기준선: MoE 734.3 ms/slice, 총 1624.91 s, 슬라이스 2213장(acc4 1140 + acc8 1073).

| GRAPPA가 붙이는 Δt | 총 채점시간 증가 | α=5.2e-7<br>(회수 환산식) | α=4.7e-6<br>("0.00n"을 0.009폭으로) | α=2.5e-5<br>(최악 가정) |
|---|---|---|---|---|
| +10 ms/slice | +22 s (+1.4%) | ΔSSIM > 0.0000052 | > 0.000047 | > 0.00025 |
| +25 ms/slice | +55 s (+3.4%) | > 0.000013 | > 0.00012 | > 0.00063 |
| +50 ms/slice | +111 s (+6.8%) | > 0.000026 | > 0.00023 | > 0.00125 |
| +100 ms/slice | +221 s (+14%) | > 0.000052 | > 0.00047 | > 0.0025 |
| +200 ms/slice | +443 s (+27%) | > 0.00010 | > 0.00094 | > 0.0050 |

읽는 법은 간단하다. 셋 중 가장 비관적인 α에서도 +50 ms의 손익분기가 0.00125다. 우리가 run 사이에서 실제로 관측해 온 SSIM 차이가 +0.004~+0.011이니 한 자릿수 아래다. **어떤 계수를 골라도 시간은 저울에 안 올라간다.**

회수된 환산식이 맞다면 한 가지가 더 따라온다. 현재 734.3 ms에서 τ = 0.000659를 이미 확보한 상태라, 무한히 느려져도 **잃을 수 있는 전부가 0.00066**이다. 반대로 속도를 아무리 최적화해도 더 얻을 수 있는 건 0.00034뿐이다. 속도 레버는 사실상 고갈됐다는 뜻이기도 하다.

### 2.3 그래서 강제할 예산은 얼마인가

세 층으로 나눈다.

- **점수 예산**: +1266 ms/slice (2000 도달점. 그 너머는 점수상 공짜). 사실상 무제한.
- **경고 안전 예산**: +700 ms/slice 정도. 다만 볼륨당 최대 슬라이스 수 S_max가 과거 관측으로 [40, 83]에만 묶여 있어 첫 경고 지점이 717~1500 ms/slice로 미확정이다. leaderboard kspace h5의 `shape[0]` 최대값을 읽으면 10초에 확정된다(GPU 불필요, 관문 1에 포함).
- **실제로 강제할 예산: +50 ms/slice.** 이건 점수 제약이 아니라 **버그 탐지선**이다. 제대로 짠 구현은 +6~25 ms에 안착하므로, 50을 넘는다는 건 numpy/complex128 승격, conv2d에 stride 누락(정확히 R배 헛계산), shift별 개별 solve, CPU 폴백 중 하나가 일어났다는 신호다. 예산 초과를 "느려도 괜찮으니까"로 넘기지 말고 구현 결함으로 읽어라.

### 2.4 비용 추정 (전부 추정 — 서버 실측 전)

| 시나리오 | 내용 | ms/slice |
|---|---|---|
| 낙관 | GRAPPA 전용 SCC 6~8코일, shift 공유 AᴴA, conv2d 한 방, Cholesky | +6 |
| 현실(권장) | 15코일 그대로, H=384, per-slice 캘리브레이션, acc4 n_b=3·acc8 n_b=2~3, k_x=5 | **+10~25** |
| 비관 | Pascal cuSOLVER complex64 batched solve 지연 + W=480 볼륨 + 파이썬 오버헤드 | +60 |
| 재앙 | pygrappa 또는 numpy pinv — complex128 승격에 Pascal FP64가 1/32 | +1500~5000 |

근거가 수렴한다는 점은 적어둘 만하다. 세 갈래 조사가 독립적으로 FLOPs를 셌고 acc4 6~9 GFLOP / acc8 6~7 GFLOP로 일치했으며, 두 조사가 같은 연산 그래프를 CPU에서 실측해 11.5~33.5 ms를 얻었다. GTX 1080이 이 형태(작고 홀쭉한 cgemm + 단일 cuDNN conv)에서 CPU보다 느릴 이유는 없다. 다만 GPU 수치는 전부 외삽이다 — 서버에서 한 번 재기 전에 "6~8 ms"를 확정으로 인용하지 마라.

### 2.5 기대 이득 (추정)

| | 기준선 | 예상 Δ | 근거 |
|---|---|---|---|
| acc4 | full 0.9483 / bbox 0.9490 | 0 ~ +0.004, 중심 +0.001~0.002 | ACS 29줄에 R_eff 3.23. 기하가 건강하다. 다만 남은 결손이 0.0517이고 앨리어싱 성분은 그중 일부다. 이 계열에서 구조 개선 한 건이 움직인 폭이 +0.004 수준(E2E-VarNet 논문 ablation) |
| acc8 | full 0.9122 / bbox 0.9068 | −0.004 ~ +0.002, 중심 ~0 | ACS 30줄로 R=8 커널이 굶는다(§4). 결과가 앨리어싱이 아니라 구조화된 g-factor 잡음으로 나오는데 SSIM의 contrast·structure 항이 여기 민감하고, 병변 bbox는 작고 고주파라 SSIM_bbox가 먼저 깎인다 |
| 하네스 평균 | 0.9321 / 0.9300 | **중심 +0.0005~+0.001, 범위 −0.002 ~ +0.003** | acc4/acc8 동일 가중, full/bbox 동일 가중 |

문헌 인용 하나를 저울에서 내려야 한다. GrappaNet이 보인 8× +0.036은 평범한 image U-Net 대비 수치이고, 같은 fastMRI knee에서 GRAPPA 없는 E2E-VarNet(30M)이 GRAPPA 내장 GrappaNet(480M)을 8×에서 앞섰다(0.893 vs 0.882). AIRS 우승은 brain 트랙이고 200M·dual-domain·ESPIRiT와 교락돼 GRAPPA 기여분 ablation이 보고된 적이 없다. 이 근거로 기대치를 부풀리지 마라.

---

## 3. 자체 계산이 왜 게임을 바꾸나

### 3.1 얻는 물건이 다르다

주최측이 image h5에 넣어준 `image_grappa`는 **이미 코일결합된 magnitude 384²**다(`utils/data/transforms.py:363-365`가 그렇게 다룬다). 우리가 ACS에서 직접 캘리브레이션하면 나오는 건 **채워진 complex multi-coil k-space (C, H, W)** 다. 여기서 갈라지는 게 셋이다.

- **위상이 있다.** magnitude에 위상을 합성해 k-space로 되돌리는 짓(=`rejected.md:30`이 "위상·코일 날조된 가짜 k-space"라 기각한 그것)을 할 필요가 없다.
- **코일이 살아 있다.** 모델 자신의 감도맵으로 `fastmri.sens_reduce`를 태우면 `img_zf`·`img_pred`와 **같은 도메인, 같은 격자**의 complex 2채널이 나온다. 재격자화도, crop/pad 규칙 맞추기도 없다.
- **마스크와 동기가 된다.** remask 증강으로 재생성한 마스크에서 GRAPPA를 다시 뽑으면 정보 누설이 원천적으로 없다.

### 3.2 옛 기각 사유의 사후 판정

| `strategy/rejected.md` | 옛 사유 | 지금 |
|---|---|---|
| :30 GRAPPA → fft → k-space | "magnitude·단일코일 384² → fft하면 위상·코일 날조" | **무효.** 애초에 k-space로 나온다. fft도 위상 합성도 없다 |
| :32 GRAPPA로 x0 대체 | "위상 합성 + coil 확장 필요 → ill-posed" | **사유는 무효, 결론은 유지.** 이유가 바뀌었다 — 아래 |
| `model_strategy.md:66` "GRAPPA 재생성 불가" | "fully-sampled 없음" | **거짓.** ACS에서 캘리브레이션한다 |
| `augmentation.md:109` remask×GRAPPA 배타 | 위 전제의 파생 | **무효.** 재생성한 마스크로 GRAPPA를 다시 뽑으면 된다 |

### 3.3 그런데 x0 대체는 여전히 drop-in이 아니다

`promptmr_plus.py:486-505`를 그대로 옮기면:

```python
ffx = fastmri.sens_reduce(torch.where(mask != 0, current_kspace, zero), sens_maps)
buffer  = torch.cat([ffx, latent, img_zf] + [ffx-img_zf]*(n_buffer-3), dim=1)
soft_dc = (ffx - img_zf) * self.dc_weight
img_pred = current_img - soft_dc - model_term
```

`img_zf`(promptmr_plus.py:621)는 x0이면서 동시에 **DC의 기준점**이다. `ffx`는 마스크를 곱한 뒤 축약한 값이라 측정된 위치의 정보만 담는다. 여기에 GRAPPA로 채워진 k-space를 `img_zf` 자리에 넣으면, 측정 안 된 위치에서 두 항이 서로 다른 부분공간에 살게 되어 매 cascade의 DC 잔차가 체계적으로 틀어진다. 이건 분포 이동이 아니라 버그다.

그래서 **두 역할을 분리한다.**

- **V1 (역할 분리 warm start)** — `img_pred`·`latent`의 초기값만 GRAPPA 영상으로 바꾸고 `img_zf`는 그대로. 파라미터가 하나도 안 변해 **지금 체크포인트로 바로 돌아간다.** 채택 후보는 아니고, 관문 3의 탐침으로 쓴다.
- **V2 (채택안)** — GRAPPA k-space를 `sens_reduce`로 축약해 buffer의 5번째 항으로 붙인다. `n_buffer` 4→5, `feat_extract` in_chans 10→12. `img_zf`는 안 건드린다.
- **residual (`out = grappa + f`)은 쓰지 마라.** prior의 잡음이 출력에 1:1로 실린다. acc8에서 특히 위험하다. V2는 망이 prior가 쓸모없다고 판단하면 그 가중치를 0 근처에 두면 그만이라 최악이 "이득 0 + Δt"로 막힌다.

---

## 4. 알고리즘 사양

### 4.1 우리 마스크가 만들어주는 편의

`utils/data/transforms.py:317-326`의 생성기가 `m[offset % acc :: acc] = 1` + 중앙 ACS라, 소스 라인이 **균등 격자**다. 덕분에 GRAPPA 적용이 "구멍마다 파이썬 루프"(pygrappa 방식)가 아니라 **shift별 conv2d 한 방**으로 닫힌다. ACS 안이든 밖이든 소스 기하가 안 바뀌기 때문에 패턴 열거도 필요 없다.

### 4.2 인덱스 정의

`k[c, x, y]`: c=0..C−1 코일, x=0..H−1 readout, y=0..W−1 PE. `R ∈ {4, 8}`, `off = (ACS 밖 첫 샘플) mod R`.
ACS = `[pad, pad+nlf)`, `nlf = round(0.08·W)`, `pad = (W−nlf+1)//2` (transforms.py:314-326).

커널 `w[r, j, l, p, b]` — r=shift(1..R−1), j=타깃코일, l=소스코일, p=readout 탭(0..k_x−1), b=블록(0..n_b−1), `b0 = ⌊n_b/2⌋−1`:

```
k̂[j, x, off+dR+r] = Σ_l Σ_p Σ_b  w[r,j,l,p,b] · k[l, x+p−⌊k_x/2⌋, off+(d+b−b0)R]
```

**캘리브레이션**: 커널 footprint 전체가 ACS 안에 들어오는 d만 행으로 쓴다.
- `N_src = C·k_x·n_b`, `N_eq = (H−k_x+1) × (사용 가능한 오프셋 수)`
- `span = (n_b−1)·R + 1`, 오프셋 수 `= nlf − span + 1`
- R−1개 shift가 설계행렬 A를 **공유**한다(같은 소스 윈도우가 사이의 R−1개 구멍을 전부 예측). AᴴA를 한 번만 만들고 RHS만 C·(R−1)열로 넓히면 된다. 실측으로 캘리브레이션 비용이 acc4 2.0배·acc8 5.0배 줄었다.
- 정칙화는 pygrappa 관례를 그대로: `λ₀ = λ·‖AᴴA‖_F / N_src`, `W = (AᴴA + λ₀I)⁻¹ AᴴT`. 스케일 불변이라 `--normalize` 여부와 무관하게 동작한다.
- **`torch.linalg.lstsq`를 쓰지 마라.** CUDA에선 driver='gels'(QR) 전용이라 full-rank를 전제하고 정칙화가 없다. acc8처럼 오프셋이 적은 조건에서 에러 없이 쓰레기를 낸다. 정규방정식 + Tikhonov + Cholesky로 간다.

**적용**: `X = k[:, :, off::R]`을 실수 2채널로 펴서
`F.conv2d(X, W, stride=(1, R), dilation=(1, R), padding=(k_x//2, 0))` 한 번 → scatter → 측정값 복원.
복소 가중치는 `[[Wre, −Wim], [Wim, Wre]]` 실수 패킹으로 정확히 등가 변환한다(conv2d가 complex dtype을 안 받는다).
⚠️ **stride를 빠뜨리고 dilation만 주면 이미 획득된 라인 위치까지 전부 계산해 FLOPs가 정확히 R배가 된다.** 결과는 똑같은데(버릴 값을 계산) acc8이면 8배다.

### 4.3 acc4 / acc8 기하 — 여기가 갈리는 지점

`CENTER_FRACTION = 0.08` (transforms.py:314) 기준, `k_x = 5`, `H = 384`(readout_crop 후), `C = 15`.

| W | nlf | acc4 n_b=4<br>(span 13) | acc4 n_b=3<br>(span 9) | acc8 n_b=4<br>(span 25) | acc8 n_b=3<br>(span 17) | acc8 n_b=2<br>(span 9) |
|---|---|---|---|---|---|---|
| 320 | 26 | 14 | 18 | **2** ✗ | 10 | 18 |
| 368 | 29 | 17 | 21 | 5 | 13 | 21 |
| 372 | 30 | 18 | 22 | 6 | 14 | 22 |
| 480 | 38 | 26 | 30 | 14 | 22 | 30 |

(숫자 = 사용 가능한 캘리브레이션 오프셋 수 `nlf − span + 1`)

W=372 기준 방정식/미지수 비율(`N_eq / N_src`, N_eq = 380 × 오프셋):

| | n_b=2 | n_b=3 | n_b=4 |
|---|---|---|---|
| acc4 | 8360/150 = 55.7 | 7980/225 = 35.5 | 6840/300 = 22.8 |
| acc8 | 8360/150 = 55.7 | 5320/225 = 23.6 | **2280/300 = 7.6** |

⚠️ 정직하게 적어두면, 두 갈래 조사가 이 오프셋 수를 서로 다르게 셌다(한쪽은 acc8 n_b=4에서 6, 다른 쪽은 3). 타깃 라인까지 ACS 안이어야 한다는 조건을 어떻게 거느냐, readout 방향 가장자리를 어떻게 자르냐에 따라 2~3배 차이가 난다. **어느 쪽으로 세든 결론은 같다** — acc8에서 n_b를 키우면 캘리브레이션이 굶는다. n_b=5(span 33)는 ACS 30보다 넓어 아예 불가능하다.

### 4.4 권장 시작점

| | n_b | k_x | λ | 격자 |
|---|---|---|---|---|
| acc4 | 3 | 5 | 1e-3 | H=384 (readout_crop 후) |
| acc8 | 2~3 | 5 | 1e-2 | H=384. 관문 2에서 640이 유의하게 나으면 재검토 |

acc8에서 native H=640이 유리하다는 신호가 합성 실험에 있다(방정식 1.67배, NRMSE 0.108→0.091). 적용 비용도 1.67배다. 캘리브레이션과 적용이 **반드시 같은 격자**여야 한다는 것만 확실하고, 어느 격자가 나은지는 실데이터로 정해야 한다.

---

## 5. 배선 설계

### 5.1 추론 경로 — `utils/learning/test_part.py`

현재 `recon_slice`(:77-93)의 순서는 마스크 곱(:81) → readout_crop(:84-85) → coil_compress(:86-87) → dtype 방어(:88) → to_tensor(:89-90) → model(:93).

GRAPPA는 **:85와 :86 사이**, 즉 readout_crop 직후·coil_compress 직전에 넣는다.

```
:85  ks = readout_crop(ks, crop_h)
     ── 여기 ── ks_grappa = grappa_fill(ks, mask1d, R, n_b, k_x, lam)   # (C, H, W) complex
:86  if coil_compress: ks = coil_compress(...)         # ks_grappa도 같은 W로 압축
:89  kspace = to_tensor(ks) ;  kspace_g = to_tensor(ks_grappa)
:93  return model(kspace, mask_t, grappa_kspace=kspace_g)[0]
```

압축 앞에 두는 이유는 GRAPPA가 코일 다양성으로 먹고살기 때문이다(R=8을 가상코일 8개로 푸는 건 인코딩 여유가 0이다). 현재 최고 런은 `--coil-compress`를 안 쓰므로(shell20.sh:52) 당장은 무의미한 분기지만, 켤 때를 대비해 순서를 못 박아둔다.

`grappa_fill`은 새 함수다. `utils/data/transforms.py`의 `readout_crop`(:74)·`coil_compress`(:122) 옆이 자연스럽다. `_detect_acs`(:97)를 재사용해 ACS 구간을 잡는다. torch·CUDA로 짜되 numpy 입력을 받아 numpy를 돌려주지 말고, **`recon_slice` 안에서 GPU 텐서로 바로 만들어 GPU에 두는 게** 낫다(:89-90의 `to_tensor` → `.to(device)`가 어차� 있으므로 그 앞으로 당긴다).

### 5.2 MoE와의 결합 — `utils/model/moe.py`

`load_model`(test_part.py:52-62)이 MoE 체크포인트를 만나면 `MoEWrapper`를 만들고 전처리 속성을 래퍼에 심는다(:60). 그래서 `recon_slice`는 래퍼 속성 하나만 읽는다 — **전문가마다 GRAPPA를 다르게 켤 수가 없다.**

acc4에만 켜려면 GRAPPA 호출을 `MoEWrapper.forward`(moe.py:42-45) 안으로 내려야 한다. 거기서 이미 `extract_acc_from_mask`로 가속률을 판별하고 있으니 자리는 맞다.

```python
def forward(self, masked_kspace, mask):
    acc = extract_acc_from_mask(mask, self.threshold)
    if acc == 8:
        return self.expert8(masked_kspace, mask)
    g = grappa_fill_t(masked_kspace, mask, R=4, ...) if self.expert4.use_grappa else None
    return self.expert4(masked_kspace, mask, grappa_kspace=g)
```

`merge_moe.py`의 `MUST_MATCH`에 `n_buffer`가 없어 병합 자체는 통과한다(전문가 config가 달라도 막히지 않는다). 30분 작업으로 보이지만 실제로 짜보기 전엔 단정 못 한다.

### 5.3 모델 쪽 — `utils/model/promptmr/`

바꿔야 하는 곳이 다섯 군데다.

| 파일:줄 | 지금 | V2에서 |
|---|---|---|
| `config.py:25` | `n_buffer=4` | `n_buffer=5` (또는 CLI 레버로 노출) |
| `promptmr_plus.py:500` | `cat([ffx, latent, img_zf] + [ffx-img_zf]*(n_buffer-3))` | `cat([ffx, latent, img_zf, ffx-img_zf, img_grappa])` |
| `promptmr_plus.py:486-495` | `PromptMRBlock.forward(current_img, img_zf, latent, mask, sens_maps, history_feat)` | `img_grappa` 인자 추가 |
| `promptmr_plus.py:621-636` | `img_zf = sens_reduce(masked_kspace, sens_maps)` 후 cascade 루프 | `img_grappa = sens_reduce(grappa_kspace, sens_maps)`를 같이 만들어 cascade에 넘김. `checkpoint(...)` 호출(:626-629)의 인자 목록도 함께 |
| `adapter.py:26-46` | `forward(masked_kspace, mask)` | `grappa_kspace=None` 선택 인자를 받아 `self.net(...)`에 전달 |

`sens_reduce`를 태우는 게 핵심이다. 그래야 GRAPPA prior가 `img_zf`·`img_pred`와 **같은 도메인, 같은 격자, 같은 스케일**로 들어간다. 별도의 crop/pad 규칙이 필요 없다.

**모양이 변하는 텐서는 딱 둘이다.** `feat_extract`(promptmr_plus.py:294): in_chans 10→12. `conv_last`(:320): out_chans 10→12. cascade의 `NormPromptUnet`만 `adaptive_input=True, n_buffer=n_buffer`를 받고(:578-580), 감도맵 쪽 `NormPromptUnet`(:665-666)은 기본값 `adaptive_input=False`라 **전혀 안 건드려진다.**

출력 쪽이 늘어나는 건 무해하다. `promptmr_plus.py:473`의 `torch.split(x, [cc, cc, cc, x.shape[1]-3*cc])`가 앞 3×cc만 쓰고 나머지를 latent 뒤로 흘려보내므로, 채널이 늘어도 `latent`의 위치가 안 바뀐다.

### 5.4 비트 동일 warm start

```python
sd = torch.load('.../best_model.pt')['model']
# feat_extract.weight: (n_feat0, 10, 3, 3) → (n_feat0, 12, 3, 3)
new_w[:, :10] = old_w ;  new_w[:, 10:] = 0
# conv_last.weight:   (10, n_feat0, 5, 5) → (12, n_feat0, 5, 5)
new_w[:10] = old_w   ;  new_w[10:] = 0    # 뒤 2채널은 split에서 버려지므로 값 무관
```

입력 쪽 새 2채널이 0이면 초기 모델은 현재 best와 **출력이 완전히 같다.** A/B가 재학습이 아니라 미세조정이 되는 근거가 이것이다.

### 5.5 학습 경로 정합

`utils/data/load_data.py:160-172`가 train/val에서 주최측 `image_grappa`를 읽어준다. **이 경로를 쓰면 안 된다.** 학습은 주최측 magnitude 384²로, 추론은 우리 complex k-space로 하면 물건 자체가 다르다. 학습 로더에서도 `grappa_fill`을 돌려 같은 complex k-space를 만들어야 한다.

부수 효과로 `load_data.py:306-310`의 `--remask`×GRAPPA 배타 가드가 **의미를 잃는다.** 재생성한 마스크로 곱한 k-space에서 GRAPPA를 뽑으니 "제거한 라인의 정보가 옆문으로 새는" 경로 자체가 없다. 가드는 그때 풀되, 문구를 "주최측 image_grappa를 쓸 때만 배타"로 좁혀서 남겨라.

`train.py`에는 `--return-grappa` 플래그가 없다. 새 레버는 `--grappa` 하나로 하고, 커널 하이퍼(n_b·k_x·λ)는 acc별 기본값을 코드에 두고 CLI로 노출하지 마라 — 레버가 늘면 단일 레버 A/B가 깨진다.

### 5.6 스케일 정합

`--normalize`를 켜면 `transforms.py:476-487`이 k-space·GRAPPA·target을 같은 스칼라 s로 나눈다. 자체 계산 GRAPPA는 입력 k-space에서 나오므로 s가 곱해지든 나뉘든 **선형**이라 원리적으로는 순서 무관이다. 그래도 학습과 추론에서 순서가 정확히 같아야 한다. Tikhonov λ₀가 `‖AᴴA‖_F` 상대값이라 스케일 불변인 건 다행이다. 현재 최고 런이 normalize를 안 쓰므로 이 경로는 아직 한 번도 검증된 적이 없다 — 켜는 순간 조용히 어긋날 수 있는 지점이다.

---

## 6. 검증 순서 (관문)

### 관문 1 — 우리 GRAPPA가 제공 `image_grappa`를 재현하는가 · **비-GPU, 바로 실행**

사용자가 첫 관문으로 지목한 그것이다. 같은 볼륨·같은 딸려온 마스크에서 우리 `grappa_fill` → 코일결합(RSS) → `center_crop(384,384)` 한 결과와 h5의 `image_grappa`를 비교한다. `utils/data/verify_readout_crop.py`가 쓰는 관례(픽셀 상관·스케일비)를 그대로 따르면 된다.

스윕: `n_b ∈ {2,3,4}` × `k_x ∈ {3,5,7}` × `λ ∈ {1e-4, 1e-3, 1e-2}` × 격자 ∈ {640, 384}.

같은 스크립트에서 공짜로 딸려오는 게 셋이다.
- **(b)** train/val 200볼륨의 `image_grappa` vs `image_label`을 **우리 채점 함수 그대로**(`utils/common/metrics.py`의 `ssim_full`, `foreground_mask`, `data_range = attrs['max']`) 재라. acc4/acc8을 나눠서, 비교군으로 `image_input`(zero-filled)도.
- **(c)** 우리 GRAPPA vs zero-fill의 **실데이터 NRMSE**를 acc4/acc8 각각. 지금까지 acc8 위험 근거는 전부 합성 팬텀이다. 이게 그걸 메운다.
- **(d)** leaderboard kspace h5의 `shape[0]` 최대값 → S_max 확정 → 60초 경고까지의 실제 여유 확정.

```bash
# 비-GPU. 옆 터미널에서 바로. device는 cpu로 고정할 것.
python3 scripts/grappa_gate1.py --data /root/Data --out training_anal/grappa_gate1.json
```

**통과 기준**
- 재현: 픽셀 상관 ≥ 0.98이고 스케일비가 볼륨 간에 안정적이면, 주최측 것과 우리 것이 같은 물건이라는 뜻 → 옛 로그·문서와의 연결이 유지된다. 못 맞춰도 치명적이진 않다(어차피 학습·추론 양쪽 다 우리 것을 쓴다). 다만 **못 맞추면 `image_grappa`로 학습하는 선택지는 그 자리에서 사라진다.**
- **중단 조건 ①**: acc8 `image_grappa`의 SSIM_full이 0.88 미만이면 acc8 arm은 여기서 접고 acc4 전용으로만 간다.
- **중단 조건 ②**: acc8에서 우리 GRAPPA의 NRMSE가 zero-fill보다 나쁘면 acc8은 폐기.

### 관문 2 — 격자·하이퍼 확정 · **비-GPU, 바로 실행**

관문 1의 스윕 결과로 acc4/acc8 각각의 `(n_b, k_x, λ, 격자)`를 하나씩 못 박는다. 이후 단계에서 이 값은 안 건드린다 — 단일 레버 A/B의 전제다.

**통과 기준**: acc4·acc8 각각 NRMSE 최소점이 하이퍼 격자의 경계가 아닌 내부에 있을 것(경계면 스윕을 넓혀라).

### 관문 3 — 중복성 가설 검증 + Δt 실측 · **GPU 큐 슬롯 필요 (shell28.sh), 학습 없음, ~30분**

이번 조사에서 가장 값나가는 한 판이다. 현재 best MoE 체크포인트를 **그대로 두고** V1(역할 분리 warm start — `img_pred`·`latent`만 GRAPPA 영상으로, `img_zf`는 그대로)만 켜서 `recon_eval`을 돌린다. 파라미터가 안 변하니 학습이 필요 없다.

```bash
# ⚠️ GPU 큐 슬롯. shell_scripting/shell28.sh 에 작성 → 사람이 `bash shell28.sh` 실행.
python recon_eval.py -n promptmr_moe_grappa_v1 -p /root/Data/leaderboard
```

한 번에 둘을 얻는다.

- **Δt 실측** — 하네스가 재는 방식 그대로(`recon_eval.py:78-86`). 이게 §2 표의 분모다.
- **중복성 가설 판정** — 훨씬 좋은 x0를 줬는데 출력이 안 움직이면 이 망의 고정점은 초기화에 둔감하다는 뜻이고, GRAPPA prior의 한계가치가 0에 가깝다는 가장 강한 증거다.

**통과 기준 / 해석**
- `|ΔSSIM| < 0.001` → **여기서 멈춘다.** 학습 예산을 걸 근거가 없다.
- 크게 흔들린다 → 망이 초기화에 민감하다는 것만 알 뿐, **이득의 증거는 아니다**(방향이 어느 쪽이든 관문 5로). 이 비대칭을 정직하게 읽어라.
- 부수 판정: `Δt > 50 ms/slice`면 §2.3의 버그 탐지선을 넘은 것이다. 관문 5 전에 고쳐라.

### 관문 4 — 기회비용 방어선 · **비-GPU 판단, 실행은 기존 큐**

GPU 2.5~3.5일을 GRAPPA에 걸기 전에, 더 싸고 이미 알려진 레버를 먼저 소진한다.

| 레버 | 근거 |
|---|---|
| MoE 미세조정 lr | 3 epoch 동안 val이 0.07694→0.07690→0.07694. 4째 자리만 움직였다 — 사실상 미세조정을 한 적이 없다 |
| `--warmup-iters 0` 상수 LR → cosine | 스케줄이 아예 안 걸려 있다 |
| `--remask`로 전문가당 볼륨 풀 80→160 | acc8을 정조준한 레버. GRAPPA를 자체 계산하면 `load_data.py:306`의 배타 가드도 함께 풀린다 |

**통과 기준**: 이 셋의 합이 GRAPPA 낙관 시나리오(+0.003)보다 작을 것. 크면 GRAPPA는 지금 우선순위가 아니다.

### 관문 5 — V2 미세조정 A/B · **GPU 큐 슬롯 (shell29.sh)**

V2를 **acc4 전문가에만** 켜고 현재 best에서 warm start(§5.4). 같은 fold·같은 seed·단일 레버. 학습·추론 **양쪽 다 우리 GRAPPA 구현**을 쓴다.

**통과 기준: SSIM_bbox 비퇴행.** bbox가 점수의 절반이라 full만 올라도 소용없다.

### 관문 6 — acc8은 관문 5가 이겼을 때만 · **GPU 큐 슬롯**

그때 GRAPPA arm과 함께 **ACS-only 저해상도 앵커**(ACS 블록만 iFFT+RSS, 캘리브레이션 없음 → g-factor 잡음도 없음, 비용 수 ms)를 제3의 arm으로 같이 돌려라. acc8에서는 이쪽이 이길 개연성이 실재한다. MoE 전문가가 이미 갈려 있어 "acc4는 GRAPPA, acc8은 앵커"라는 비대칭 설계에 추가 아키텍처 비용이 0이다.

### 관문 7 — 타이밍 무결성 · **전 구간 상시**

`recon_eval.py:73-74`가 볼륨마다 앞 5슬라이스를 측정 없이 warmup으로 돌린 뒤, **같은 s=0..4를 포함해** 전체를 다시 타이밍 안에서 돈다. 그래서:

- **금지**: 볼륨 단위 커널 캐시, 슬라이스 인덱스를 키로 한 캐시. 전자는 캘리브레이션 비용을 통째로 warmup에 숨기고, 후자는 전체 슬라이스의 18%를 공짜로 만든다. 둘 다 `prep_volume` 회피와 같은 종류다.
- **무해**: 마스크·shape에만 의존하는 것(ACS 인덱스, unfold/scatter 인덱스, 버퍼, eye). warmup이 존재하는 이유가 그것이다.
- k-space 데이터에 의존하는 커널을 캐시하는 건 알고리즘 선택이므로 **화질로 방어할 수 있어야** 한다. 덧붙여 per-slice 캘리브레이션이 정직하면서 동시에 화질도 낫다 — 슬라이스마다 코일 감도가 다르다.

---

## 7. A/B 설계

기준선: **MoE (acc4/acc8 전문가, 증강 없음) — SSIM_full 0.9321 / SSIM_bbox 0.9300 / 734.3 ms/slice**

| | arm A (기준선) | arm B (GRAPPA) |
|---|---|---|
| 체크포인트 출발점 | 현재 best | 현재 best + 채널 확장(0 초기화) |
| 레버 | 없음 | `--grappa` 하나 |
| 적용 범위 | — | acc4 전문가만 (관문 5). acc8은 관문 6에서 별도 |
| fold / seed | 동일 | 동일 |
| epoch / lr / 스케줄 | 동일 | 동일 |
| GRAPPA 하이퍼 | — | 관문 2에서 고정, 이후 불변 |
| 학습 GRAPPA 출처 | — | **우리 구현** (주최측 `image_grappa` 금지) |

**보고할 축은 여섯이다.** `SSIM_full(acc4)`, `SSIM_full(acc8)`, `SSIM_bbox(acc4)`, `SSIM_bbox(acc8)`, `ms/slice`, `slow_count`. `recon_eval.py:141-151`이 다 찍어준다. 평균만 보면 acc8 퇴행이 acc4 개선에 가려진다.

**채택 판정**
1. `SSIM_bbox(acc4)` 퇴행 → 기각. (bbox가 점수의 절반)
2. `Δt > 50 ms/slice` → 구현 결함으로 보고 고친 뒤 재측정.
3. 위 둘을 통과하고 `0.5·Δfull + 0.5·Δbbox > 0` → 채택. §2 표에 따르면 시간 페널티는 어떤 α에서도 이 이득을 못 뒤집는다.
4. `|ΔSSIM| < 0.001` → 채택하지 마라. 재현성 위험(§8)과 코드 복잡도를 무료로 사는 셈이다.

**단일 레버 위반을 조심하라.** 이 repo의 A/B에서 반복 지적된 항목이다(`training_anal/2026-07-26-promptmr-4runs-analysis.md §2`). GRAPPA를 켜면서 lr이나 epoch을 같이 만지면 결과를 못 읽는다.

---

## 8. 함정 목록

**① 학습/추론 GRAPPA 불일치 — 세 갈래 조사가 모두 최상위 조용한 위험으로 꼽았다.**
주최측 `image_grappa`로 학습하고 우리 것으로 추론하면, val 곡선은 끝까지 예쁘게 나오고 리더보드에서만 무너진다. val도 주최측 GRAPPA를 쓰기 때문이다. 크래시가 안 나고 SSIM만 떨어지는 종류라 발견이 늦다. **학습에서도 우리 함수를 쓰는 것 외에 안전한 길이 없다.**

**② acc8 역설.**
도움이 절실한 곳이 정확히 GRAPPA가 굶는 곳이다. 전체 결손의 63%가 acc8인데(acc4 0.0517 vs acc8 0.0878), ACS 30줄에 R=8 커널은 §4.3의 표대로 방정식이 부족하다. 결과물은 앨리어싱이 아니라 **구조화된 g-factor 잡음**으로 나오고, SSIM의 contrast·structure 항이 여기 민감하다. 병변 bbox는 작고 고주파라 SSIM_bbox가 먼저 깎인다. 차단책은 두 가지 — MoE 전문가가 이미 갈려 있으니 acc4에만 켜고, acc8엔 ACS-only 앵커를 별도 arm으로 검토한다.
한 가지 반대 방향의 주의도 있다. **prior를 그 자체 SSIM으로 판정하지 마라.** prior의 가치는 "노이즈가 적은가"가 아니라 "앨리어싱이 풀렸는가"다. 관문 1(b)의 SSIM은 중단 조건으로만 쓰고, 최종 판정은 모델 출력의 A/B로 한다.

**③ 커널 조건수.**
`torch.linalg.lstsq`(CUDA=gels)는 full-rank를 전제하므로 오프셋이 모자란 조건에서 **에러 없이 쓰레기를 낸다.** 정규방정식 + Tikhonov + Cholesky로 가고, R=8은 n_b를 낮춰라. 그리고 W가 320~480으로 가변이라 ACS도 26~38로 흔들린다 — 커널 크기를 acc4/acc8 공통으로, 혹은 W와 무관하게 하드코딩하지 마라. W=320 볼륨에서 n_b=4는 오프셋이 2개다.

**④ 재앙 구현.**
`np.linalg.pinv/svd/lstsq`는 complex128로 승격하고, Pascal은 FP64가 FP32의 1/32다. 이 repo의 `coil_compress`가 이미 같은 함정 때문에 호출부 캐스팅 방어를 달아뒀다(`test_part.py:88`, `transforms.py:489`). pygrappa는 구멍 하나마다 파이썬 for를 도는데 우리 슬라이스의 구멍 수가 acc4 97,536 / acc8 114,432개다. **오프라인 정답 대조용으로만 써라.** 런타임은 conv2d 기반 자체 구현(~80줄, 새 의존성 0 — numpy 1.26.4·torch 2.3.1+cu121로 충분).

**⑤ 60초 경고.**
코드상 순수 출력이다 — `slow_count`는 `run_acc`에서 세어 print에만 쓰이고(`recon_eval.py:88-89, 147-151`) 점수 계산 경로에 안 들어간다. 그래도 심사자가 눈으로 보는 신호다. 지금 0건이고, 볼륨당 최대 슬라이스 수 S_max가 [40, 83]에만 묶여 있어 첫 경고 지점이 717~1500 ms/slice로 미확정이다. 관문 1(d)에서 h5 헤더만 읽어 10초에 확정하라.

**⑥ 재현성.**
seed 고정으로는 GRAPPA의 수치 경로를 못 묶는다. cuSOLVER/LAPACK 버전, 스레드 수, cuDNN 알고 선택이 바뀌면 출력이 비트 단위로 달라질 수 있다. Cholesky는 SVD보다 결정론적이라 이 축에서도 유리하다. `--grappa`를 켠 런은 재현 확인을 한 번 더 돌려라.

**⑦ 격자 정합.**
W=368/372는 384보다 **작아서** `utils/common/utils.py:85-105`의 `center_crop`이 zero-pad 경로를 탄다. V2는 `sens_reduce`로 모델 격자에 바로 얹으니 이 문제를 피해가지만, 혹시 영상 도메인 prior를 별도로 만들 일이 생기면 `center_crop`을 그대로 재사용하고 새로 짜지 마라.

**⑧ VRAM.**
추가 피크는 60~100 MB로 작다(추정: A 15.5 MB, conv 출력 12.7 MB, 채워진 k-space 복사 17 MB, cuFFT 작업공간 ~30 MB). 다만 현재 cascade 10 · 15코일 구성이 8 GB에 가깝게 돈다. GRAPPA 중간 텐서를 모델 forward 전에 명시적으로 `del`하고 prior만 남겨라. W=480 볼륨에서 한 번 확인이 필요하다.

**⑨ 아직 안 정한 것.**
readout_crop 전(H=640)과 후(H=384) 중 어느 격자에서 캘리브레이션할지, coil_compress를 켤 경우 GRAPPA를 압축 전에 둘지 후에 둘지, 병변 bbox의 공간 분포가 무릎 코일의 g-factor 취약 영역과 겹치는지. 마지막 것은 `scripts/lesion_stats.py`·`bbox_ssim_breakdown.py`가 이미 있어 CPU로 확인 가능하다.

---

## 9. 문서 교정

| 위치 | 지금 적힌 것 | 사실 | 파급 |
|---|---|---|---|
| `strategy/preprocessing.md:16` | "`image_grappa`는 **항상 제공**(2026-07-03 확인) — §1은 조건부가 아니라 무조건 채택 전제" | **거짓.** 리더보드엔 없다 | "무조건 채택 전제"가 무너진다. 이 줄을 근거로 "아래 C군은 그만큼 후순위"라고 쓴 판단도 다시 봐야 한다 |
| `strategy/model_strategy.md:66` | "GRAPPA는 미리 계산돼 제공된 것이라 **재생성 불가**(fully-sampled 없음)" | **거짓.** ACS에서 캘리브레이션한다 (Griswold 2002 Eq.[2]) | 아래 세 줄이 연쇄로 무효 |
| `strategy/model_strategy.md:63` (표) | "마스크 증강 시 GRAPPA는 ❌ 미적용 — 라인 더 지우면 prior를 스스로 훼손" | **무효.** 재생성한 마스크에서 GRAPPA를 다시 뽑으면 된다. 오히려 **동기시켜야** 입력과 prior의 가속률이 안 어긋난다 | 결론이 뒤집힌다 |
| `strategy/augmentation.md:109` | "`--remask`와 GRAPPA는 배타" | **조건부로만 참.** 주최측 `image_grappa`를 쓸 때만 배타 | 자체 계산하면 remask×GRAPPA 동시 사용이 열린다 |
| `utils/data/load_data.py:306-310` | 배타 가드 + "제거한 라인의 정보가 옆문으로 새어" | 코드는 지금 상태에선 맞다. 자체 계산 경로가 생기면 조건을 좁혀야 한다 | 가드를 지우지 말고 "주최측 GRAPPA를 쓸 때만"으로 문구를 좁혀라 |
| `strategy/rejected.md:30, :32` | GRAPPA→fft, GRAPPA로 x0 대체 기각 | **기각 사유가 무효.** 다만 :32의 결론(x0 직접 대체 안 함)은 **다른 이유로 유지** — `img_zf`가 DC 기준점을 겸한다(§3.3) | 기각 사유를 갱신하고, "x0 대체 대신 buffer 항 추가(V2)"를 새 행으로 넣어라 |
| `strategy/model_strategy.md:56` (표 헤더 아래) | "GRAPPA는 이미 복원된 **깨끗한** prior" | acc4에선 대체로 맞고 acc8에선 의심스럽다 | 관문 1(b)(c)의 실측 결과로 대체하라 |

한 가지 더. `CLAUDE.md §7`의 "추론시간은 ~0.00n 단위" 서술은 §2.1에 적은 대로 **인용 없는 자기 문서 한 줄**이 열여섯 곳으로 복제된 것이다. 이 병리는 위 두 오류와 정확히 같은 형태다. 시간 항의 정의를 사람이 대회 공지에서 직접 확인해 출처를 달아라. 그 확인이 붙으면 `run8`을 "추론 1.86배라 순손해"로 기각한 판정(`training_anal/2026-07-09-run8-capacity-analysis.md`)과, 4-view TTA를 "직접 감점"이라며 배제한 `summary.md:229`도 함께 재검토 대상이 된다 — 회수된 환산식이 맞다면 둘 다 뒤집힌다. 다만 그건 이번 임무 밖이다.