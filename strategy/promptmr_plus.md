# PromptMR+ 백본 — 현재 확정 모델 (2026-07-19~)

> 상태: **현재 주 백본**. 이 문서가 PromptMR+ 파이프라인의 단일 기준(source of truth).
> 이전 SDUM SPEC(`confirmed_model.md`)은 **superseded** — SDUM 코드는 보존하되 배선에서 제거됨.

## 왜 바꿨나
SDUM(NV cascaded)에서 작년 대회 우승 코드 `sMartify-fastmri/2025_final`의 **PromptMR+**로 백본 교체.
PromptMR+의 self-adaptive gradient(기여 #1)·momentum feature aggregation(기여 #2)이 정적 knee에도
유효(논문 Table 3, knee acc4/8 검증). 인접-슬라이스 SME(기여 #3)는 우리 단일 슬라이스 파이프라인과
안 맞아 제거.

## 확정 구조
| 항목 | 값 |
|---|---|
| cascade 수 (`--num-cascades`) | **10** |
| recon U-Net 폭 (`--n-feat0`) | **48** (feature_dim `[72,96,120]`, 3레벨=다운샘플 2단) |
| 감도맵 U-Net 폭 (`--sens-n-feat0`) | **24** (sens_feature_dim `[36,48,60]`) |
| momentum history (`--n-history`) | 3 (원본 11, 메모리 레버) |
| adaptive gradient | `adaptive_input=True`, `n_buffer=4` |
| 슬라이스 | **단일**(num_adj_slices=1, 배선 제거) |
| 마스크 | **Cartesian 전용**(poisson/kt_radial 배선 제거) |
| gradient checkpointing | **ON**(`--use-checkpoint`, 8GB VRAM) |

> ⚠️ **얕은 피라미드 유지**: SDUM 논문 ablation(Table 3b) — 다운샘플 2단 최적, 더 깊으면 작은 병변 대비
> 약화. 점수 50%가 SSIM_bbox(병변)이므로 `feature_dim`을 3레벨보다 깊게 만들지 말 것.

## 학습 설정 (사용자 지정)
- **손실 = masked SSIM × √(foreground 면적비)** — `--fg-loss --area-weight`. foreground 작은 슬라이스
  손실을 낮춰 가중(작년 팀 방식, 곱셈). masked SSIM은 채점 정합 `SSIMLoss(mask=fg_mask)` 재사용.
- **선택 add-on**: bbox 병변 손실 `--bbox-loss-lambda λ` (직교, 기본 off). `L += λ·bbox_ssim_loss`.
- **옵티마이저 = AdamW, `-l 1e-6`**(weight_decay=1e-3).
- **gradient clipping** `--grad-clip 0.1` (clip_grad_norm_).
- **gradient accumulation** `--accum-steps 2` (batch≈2 효과; loss/accum 스케일, 2스텝마다 step, 꼬리 처리).
- 스케줄러: 기본 상수 LR. `--warmup-iters>0`이면 linear warmup→cosine(promptmr 게이트, optimizer-step 단위·accum 보정).

## 코드 접합부 (`utils/model/promptmr/`)
- `promptmr_plus.py` — 백본(벤더·slim). 원본 커밋 `c7087f5`. `import fastmri`는 `utils/model/fastmri`(varnet과 공용, sens_expand/sens_reduce 추가)로 해석.
- `config.py` — `build_promptmr_net(args)`(CLI 레버 → `PromptMR(**kwargs)`).
- `adapter.py` — `build_promptmr(args)` → `PromptMRReconModel`. 계약 `forward(kspace (B,C,H,W,2), mask (B,1,1,W,1)) -> (B,384,384)`. mask를 `(B,1,H,W,1)`로 확장해 PromptMR의 SNU-정렬 블록 우회 + Cartesian ACS 자동 추출. **iFFT·외부 정규화 없음**(PromptMR 내부 처리).
- `reentrant_wrapper.py` — reentrant gradient checkpointing.
- 배선: `train.py`(CLI) · `train_part.py`(빌드·AdamW·스케줄러·area/clip/accum·resume 필드) · `test_part.py`(load_model·forward, 추론은 checkpoint off).

## 실행 예 (GPU 큐 슬롯)
```
python train.py --model promptmr --num-cascades 10 --n-feat0 48 --sens-n-feat0 24 \
  --fg-loss --area-weight -l 1e-6 --grad-clip 0.1 --accum-steps 2 --use-checkpoint \
  -n <net_name> -e <epochs> -t <train> -v <val>
# 채점: python recon_eval.py -n <net_name> -p /root/Data/leaderboard
```

## slim 등가성
인접-슬라이스·비-Cartesian 분기 제거는 원본(num_adj_slices=1)과 **bit-단위 등가**(동일 가중치·입력 →
max_abs_diff 0.0)로 검증. adaptive gradient·momentum·cascade·DC·Cartesian ACS는 원본 유지.

## 유지되는 것 (모델 무관)
데이터로더/transforms, 채점 하네스(`recon_eval.py`·`metrics.py`), 저장(`reconstruction` key), fg/bbox 손실
인프라, VarNet 경로(`--model varnet`). 단 `--normalize`는 off(PromptMR 내부 정규화).

## 후속(선택·2단계)
Progressive cascade expansion(SDUM 논문 §3.6): 얕게(예: 5) 수렴 후 warm-start로 10 확장 →
lr 1e-6 scratch 학습이 불안정할 때의 카드. `--init-from`(다른 num_cascades 허용) 별도 경로 필요,
`n_history` 버퍼 정렬 주의. 아직 미구현.

---

## 8GB VRAM 메모리 실측 · 실전 fitting config (2026-07-19 추가)

> 위 「확정 구조」(nc=10·nf=48·full-640)는 **설계 의도(design intent)** 로 유지한다.
> 아래는 **GTX 1080 8GB에서 실제로 돌릴 수 있는 config** — 위 SPEC 기본값은 이 GPU에서 OOM이라
> 별도로 명시한다. (더 큰 GPU면 위 SPEC 그대로 사용.)

**문제:** SPEC 기본값 `--num-cascades 10 --n-feat0 48`(readout-crop 없이 full H=640, 실코일 15)은
GTX 1080 8GB에서 **CUDA OOM**. checkpointing ON이어도 마찬가지.

**원인 (probe 실측):** 메모리 peak는 **coil/sens 경로가 지배**한다. 감도맵 추정·per-cascade
`sens_expand/sens_reduce`(15-coil FFT)·최종 RSS가 만드는 ~7GB 바닥이 cascade·nf와 거의 무관하게
8GB에 닿는다.
- coil 수가 지배적: C=15→OOM, C=8→~6.3GB, C=4→~3.3GB.
- **cascade 수·`n_feat0`는 peak에 거의 무영향** (C=15에서 nc 1→3 = +0.07GB, nf 16→48 = +0.01GB).
  → checkpointing이 cascade를 감싸도 무효였던 이유(병목이 cascade가 아니라 sens/coil).
- spatial(readout-crop 640→384)은 coil보다 효과 작지만(7.5→7.2), **무손실**이라 보조 지렛대.
- ⚠️ **k-space 폭(PE)이 가변**: 학습셋 W=320~480(대부분 368/372, 넓은 파일 386·388·480 존재). readout-crop은
  **높이만** 자르고 폭은 native → 넓은 슬라이스가 메모리를 키운다. **config는 최악폭(W=480)에서 fit해야 한다.**

**함정 — 최악폭 W=480에선 model 축소가 무의미:** 15 coil 그대로면 W=480에서 **nc=1(nf48)조차 OOM(7.57GB)**.
cascade/nf/sens 축소로는 **어떤 config도 8GB에 못 넣는다.** (구 shell12가 nc4·readout-crop 단독으로 iter 0
통과 후 넓은 슬라이스에서 OOM난 원인 — 좁은폭에서만 검증한 fitting config의 함정.)

**해법 = coil compression (필수):** `--coil-compress --n-virtual-coils 8`. 15→8 가상코일,
**에너지 99.999% 보존·RSS corr 1.0**(`preprocessing.md`, 사실상 무손실) + `recon_slice`(test_part.py)에 이미 배선 → 채점 자동 정합.

**실전 8GB config (최악폭 W=480, AdamW 포함 실측):**

| config | W=480 실peak | 여유 | 비고 |
|---|---|---|---|
| **nc=4·nf48 + coil-compress(8) + readout-crop (권장)** | 6.06GB | 1.94GB | 실검증: 200 iter 완주, OOM 없음 |
| nc=3·nf48 + 〃 | 5.15GB | 2.85GB | 더 여유 |
| nc≥6 + 〃 | — | OOM | coil압축해도 안 됨 |
| nc=4 · **coil압축 없이**(구 config) | OOM 7.65 | — | W=480에서 터짐(구 shell12) |

```
python train.py --model promptmr --readout-crop --coil-compress --n-virtual-coils 8 \
  --num-cascades 4 --n-feat0 48 --sens-n-feat0 24 --n-history 3 --use-checkpoint \
  --fg-loss --area-weight -l 1e-6 --grad-clip 0.1 --accum-steps 2 \
  -n <net_name> -e <epochs> -t <train> -v <val>
```
- **정합·무손실**: readout-crop은 어댑터 출력단 center_crop(384)과 수학적 등가(`confirmed_model.md:71`, 상대오차 1e-16),
  coil compression은 recon_slice에 배선 → 학습·검증·리더보드 채점 모두 자동 정합. 둘 다 사실상 무손실.
- 여유 1.94GB로 폭 가변(W≤480)에 robust. 그래도 다른 GPU 작업과 동시 실행은 피할 것.
- 슬롯: `shell11.sh`(스모크, --max-iters 120으로 넓은폭 관통) · `shell12.sh`(실학습 5ep+채점) · `shell12_resume.sh`(재개).
- 메모리 탐침: `scripts/probe_oom_promptmr.py`.

---

## 재검토: coil compression 을 빼고 구조적으로 절감 (2026-07-20)

> **위 「해법 = coil compression (필수)」의 근거가 무너졌다.** 결론을 폐기하는 게 아니라,
> **측정 전제가 틀렸으니 다시 재야 한다**는 뜻이다. 아래 탐침(`shell13.sh`) 결과가 나오기 전까지
> 위 표와 이 절 중 어느 쪽도 확정이 아니다.

**무엇이 틀렸나 — 위 실측은 checkpointing 이 실제로 안 걸린 상태에서 잰 값이다.**

1. **cascade 는 checkpointing 된 적이 없다.** 위 문서는 "checkpointing이 cascade를 감싸도 무효였다"고
   적었지만, `promptmr_plus.py` 의 cascade 루프는 `checkpoint` 없이 생으로 돌고 있었다.
   `ReentrantWrapper` 는 `sens_net` 과 `UpBlock.up` 에만 걸려 있었다. 감싼 적이 없으니 무효였는지
   알 수 없었다.
2. **코일별 sens 분할이 무력화돼 있었다.** `compute_sens_per_coil=True` 로 코일을 하나씩 돌면서도,
   그 바깥에서 `sens_net` 전체를 `use_reentrant=True` 로 한 번에 감쌌다. reentrant 는 역전파 때
   구간 전체를 다시 실행하므로, 재계산 도중 15코일분 activation 이 동시에 살아난다.
   순전파 peak 만 줄고 역전파 peak 은 그대로 — 코일별 분할의 목적이 사라진 상태였다.

**고친 것** (`promptmr_plus.py`):

- 코일 **하나씩** checkpoint (`compute_sens` 내부, `use_reentrant=False`). 바깥 통짜 wrapper 는
  per-coil 경로에서 사용하지 않는다(겹쳐 걸면 다시 통째로 되살아난다).
- cascade **하나씩** checkpoint (`PromptMR.forward` 루프).
- ⚠️ 위 두 개를 켜려면 선결 버그 하나를 먼저 고쳐야 했다: `NormPromptUnet.forward` 가 인자로 받은
  `history_feat` **리스트를 제자리 수정**하고 있었다. checkpoint 는 역전파 때 순전파를 다시 돌리는데,
  그때 이미 갱신된 리스트를 입력으로 읽어 **조용히 틀린 gradient** 가 나온다. checkpoint 없이는
  호출자가 반환값으로 즉시 재바인딩해서 증상이 안 보인다 — cascade checkpointing 을 켜는 순간
  발현되는 잠복 버그였다. 새 리스트를 반환하도록 바꿨다(결과는 동일).
  회귀 방지: `scripts/tests/test_checkpoint_equivalence.py` 가 checkpoint on/off 의 출력·전 파라미터
  gradient 일치를 float64 로 강제한다.

**측정된 것 (CPU, 역전파용으로 붙들고 있는 activation 총량, C=15 고정):**

| cascades | ckpt 끔 | ckpt 켬 | 절감 |
|---|---|---|---|
| 2 | 132.5 MB | 6.4 MB | 95.2% |
| 4 | 161.8 MB | 7.9 MB | 95.1% |
| 8 | 221.5 MB | 10.1 MB | 95.4% |

cascade 당 증가분이 **14.8MB → 0.6MB (약 25배)** 로 떨어진다. `nc` 를 10 으로 되돌릴 근거가 여기 있다.

> ⚠️ **이 95% 를 CUDA peak 절감으로 읽지 말 것.** activation 만 잰 값이다. 파라미터·AdamW 상태·
> cascade 루프 내내 붙들려 있는 `sens_maps (B,C,H,W,2)`·transient 워크스페이스는 그대로다.
> 위 표의 "C=15 에서 nc=1 조차 OOM(7.57GB)" 이 말해주는 **코일 의존 바닥**이 얼마나 깎이는지는
> **아직 실측 안 됨** — GPU 를 다른 작업이 점유 중이라 재지 못했다.

**다음:** `bash shell_scripting/shell13.sh` (GPU 빈 상태) → 코일 15 로 들어가는 최대 `nc` 확인
→ 그 값으로 `shell14.sh` 실학습. 탐침에서 여전히 전부 OOM 이면 위 「coil compression 필수」가
(이번엔 제대로 된 근거로) 재확인되는 것이고, 그때는 `--coil-compress` 를 되살린다.
