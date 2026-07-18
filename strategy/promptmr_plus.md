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
