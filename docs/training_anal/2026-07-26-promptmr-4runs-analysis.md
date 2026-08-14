# PromptMR+ 4런 묶음 분석 — run_base / run_kfold_old / run_moe / run_kfold_f2 (2026-07-26)

> ⚠️ **관례 예외:** 원래 1 run = 1 파일인데, 이 4개는 같은 시기에 연달아 돌았고 사용자가 간결본을 요청해 한 파일로 묶었다.
> ⚠️ **4런 전부 recon_eval 채점이 0회다.** 아래 숫자는 전부 **fg-val**(= 슬라이스 평균 `1 − masked SSIM_full`)이지 채점 점수가 아니다.

**한 줄 결론:** PromptMR+ 백본이 fg-val 기준 **SDUM run11(현 best, 0.9041)을 +0.0060 앞선다(run_base 0.9101)**. 다만 **4런 모두 채점을 못 해 3축(full/bbox/시간) 확인이 전무**하고, MoE 미세조정은 **3 epoch 내내 val이 평평해 효과 미판정**이다.

## §0 핵심 요약

- ✅ **PromptMR+ > SDUM (같은 지표·같은 val 폴더)**: run_base fg-val **0.9101** vs SDUM run11 **0.9041**. run12 분석이 "fg-val이 recon full을 잘 예측한다"고 세운 결론을 믿으면 실채점도 앞설 개연성.
- ✅ **재현성 실증**: `run_base`와 `run_moe`의 ①단계는 **완전히 같은 명령**인데 10 epoch val 곡선이 소수 4째 자리까지 일치(최저 0.08991 vs 0.08990). seed 고정이 실제로 작동한다.
- ⚠️ **MoE 미세조정 효과 미판정**: acc4 전문가 0.0769 / acc8 전문가 0.1022는 **검증셋이 서로 달라 비교 불가**하고, shared(0.0899, 두 셋 합산)와도 직접 비교가 안 된다. → §6 P0.
- ⚠️ **미세조정 3 epoch이 사실상 무변화**: acc4 `0.07694→0.07690→0.07694`, acc8 `0.10220→0.10220→0.10242`. lr 5e-5가 너무 낮았거나 이미 수렴. 어느 쪽인지 미확정.
- ⚠️ **fold 간 비교 불가**: `run_kfold_old`(f1)와 `run_kfold_f2`는 분할 방식이 달라(이름그룹 vs 볼륨) hold-out이 다르다. val 숫자를 나란히 놓으면 안 된다.
- 🐛 **버그 3건 발견·수정** (§4): volume 식별자 충돌 · MoE 라우터 임계값 · 채점 경로 부재.

## §1 4런 개요

| 이름 | 슬롯 | net_name | 데이터 | epoch | fg-val 최저 | peak |
|---|---|---|---|---|---|---|
| `run_base` | shell17 | `promptmr_run4` | 고정 분할 train 170 / val 30 (4651 slice) | 10 | **0.08991** (ep8) | 3.35GB |
| `run_kfold_old` | shell19 | `promptmr_kfold_f1` | K-Fold **이름그룹** 분할 f1 (4264 slice) | 10 | 0.09357 (ep9) | 3.35GB |
| `run_moe` | shell20 | `promptmr_moe_*` | 고정 분할 + `--acc` 필터 | 10/3/3 | shared **0.08990** (ep8) | 2.94GB |
| `run_kfold_f2` | shell21 | `promptmr_kfold_f2` | K-Fold **볼륨** 분할 f2 (4343 slice) | 10(진행) | 0.09315 (ep1) | 3.36GB |

**공통 config** (4런 전부 동일): `--model promptmr --readout-crop --num-cascades 10 --n-feat0 48 --sens-n-feat0 24 --n-history 3 -b 1 -l 2e-4 -e 10 --use-checkpoint --fg-loss --area-weight --grad-clip 0.1 --accum-steps 2`, seed 430.

**런별 diff (무엇을 검증하려 했나)**

| 런 | 직전 대비 바뀐 것 | 검증 의도 |
|---|---|---|
| `run_base` | (기준선) full coil + nc10 + lr 2e-4 | PromptMR+ 백본이 SDUM을 이기나 |
| `run_kfold_old` | `--k-fold 5 --val-fold 1` 추가 | 200 volume 전부 학습에 쓰면 이득인가 |
| `run_moe` | `--acc {all,4,8}` + `--init-from` 3단계 | acc4/acc8 전문가 분리가 이득인가 |
| `run_kfold_f2` | 분할 단위를 이름→**볼륨**으로 교정 후 f2 | (버그 수정본 재측정) |

## §2 직전 분석(run12, SDUM) 지적 추적

| 지적 | 상태 | 근거 |
|---|---|---|
| bbox 손실(C1) 3축 퇴행 → 롤백 | ✅ 유지 | 4런 전부 `--bbox-loss-lambda 0` |
| fg-loss는 이득 → 계승 | ✅ 계승 | 4런 전부 `--fg-loss --area-weight` |
| fg-val이 실채점 예측기 | ⬜ 재확인 불가 | 채점 0회 → 예측력 검증 자체가 막힘 |
| UC 순효과 분리(run13) | 🚫 무효 | 백본이 SDUM→PromptMR+로 교체되어 UC 계열 폐기 |
| 단일-레버 A/B 컨벤션 | ⬜ 여전 위반 | `run_moe`가 acc필터+init-from+3단계 동시 |

## §3 손실 분석

**곡선** (fg-val, 낮을수록 좋음)

```
run_base       0.09775 0.09362 0.09234 0.09163 0.09095 0.09077 0.09059 0.09033 0.08991 0.09010
run_moe shared 0.09685 0.09426 0.09227 0.09170 0.09089 0.09089 0.09066 0.09040 0.08990 0.09006
                                                                    ↑ 같은 명령 → 곡선 일치(재현성)
run_kfold_old  0.10300 0.09877 0.09759 0.09599 0.09501 0.09474 0.09442 0.09409 0.09489 0.09357
run_kfold_f2   0.09643 0.09315 …(진행 중)
run_moe acc4   0.07694 0.07690 0.07694        ← 3ep 무변화
run_moe acc8   0.10220 0.10220 0.10242        ← 3ep 무변화
```

- **ep8 부근 plateau**: `run_base`·`run_moe shared` 둘 다 ep8이 최저이고 ep9에서 소폭 반등. 10 epoch가 이 config의 대략적 수렴점 — 더 늘려도 이득이 작을 개연성(미확정, epoch A/B 필요).
- **fg-val → SSIM 환산**: 0.08991 ⇒ masked SSIM_full ≈ **0.9101**. SDUM run11의 0.9041 대비 **+0.0060**. *같은 지표·같은 val 폴더*라 이 비교는 성립한다.
- **⚠️ `run_kfold_*`는 위 둘과 비교 금지**: 검증셋이 val/ 30볼륨이 아니라 hold-out fold(43 / 40 볼륨)다. 절대값이 높다고 나쁜 게 아니다.
- **미세조정 무변화**: acc4·acc8 모두 3 epoch 동안 4째 자리만 흔들렸다. lr 5e-5 × 3ep이 shared에서 사실상 못 벗어났다는 뜻. **미세조정을 했다고 말할 수 있는 상태가 아니다.**

## §4 발견·수정한 버그 3건

| # | 버그 | 영향 | 상태 |
|---|---|---|---|
| 1 | train/·val/에 이름이 같은 다른 볼륨 30쌍 → 파일명으로 볼륨 식별 | 200 volume을 170으로 셈, `validate()`의 `reconstructions[fname]`에서 슬라이스 덮어씀 | ✅ `volume_id()` 도입 (`39dfa42`) |
| 2 | MoE 라우터 임계값 0.1875 (이론값 기반) | 실측 acc4 0.309~0.312 / acc8 0.196~0.200 → 임계가 **두 분포보다 아래** → acc8 100볼륨 전부 acc4 전문가로. MoE가 단일 모델로 전락 | ✅ 0.25로 교정 + 재병합 (`88c7e3b`) |
| 3 | `recon_eval -p /root/Data/leaderboard` 경로가 이 서버에 없음 | `run_base`·`run_kfold_old`·`run_moe` 전부 학습 완주 후 채점에서 AssertionError | ⚠️ 코드 아님 — 채점을 leaderboard 보유 환경으로 이관 결정 |

> 버그 2는 **추론 전용**이라 학습에는 안 닿았다(`--acc` 필터는 파일명 판별). 재학습 없이 재병합만으로 복구됨(200볼륨 전수 라우팅 오분류 0).

## §5 일반 ML 구현 점검

| 범주 | 항목 | 상태 | 근거/비고 |
|---|---|---|---|
| 재현성 | seed 고정 | ✅ | `run_base`≡`run_moe shared` 곡선 일치로 실증 |
| 재현성 | cudnn deterministic | ⬜ | `seed_fix`에 미포함 |
| 최적화 | LR 스케줄 | ⬜ | `--warmup-iters 0` = 상수 LR |
| 최적화 | optimizer·weight decay | ✅ | AdamW wd 1e-3 (`train_part.py:334`) |
| 최적화 | gradient clipping | ✅ | `--grad-clip 0.1` |
| 최적화 | 유효 batch | ✅ | `--accum-steps 2` |
| 체크포인트 | best/last + iter단위 | ✅ | `model.pt`·`best_model.pt`·`latest.pt`(500 iter) |
| 체크포인트 | resume | ✅ | k-fold config 가드 포함 (`39dfa42`) |
| 검증 | 매 epoch validation | ✅ | |
| 검증 | best 선택 기준 | ✅ | fg-val (채점 SSIM_full 정합) |
| 지표 | SSIM_full/bbox 로깅 | ⚠️ | val은 full만. bbox는 `recon_eval`뿐인데 **4런 다 미실행** |
| 지표 | baseline(zero-filled) 대비 | ⬜ | PromptMR+ 전환 후 한 번도 안 잼 |
| 데이터 | augmentation | ⬜ | `--aug-*` 전부 off |
| 안정성 | NaN/발산 감지 | ⬜ | 없음(현재까지 발산 사례 없음) |
| 효율 | grad checkpointing·workers | ✅ | `--use-checkpoint`, `nw 4` |
| 효율 | mixed precision | ⚠️ | CLAUDE.md는 Pascal 전제로 금지라 적었으나 **실제 서버는 RTX 4000 Ada(cap 8.9)** — 재검토 여지 |

## §6 다음 액션

- **P0 — 채점 1회.** 4런 전부 3축이 공백이다. leaderboard 보유 환경에서 `promptmr_moe` + `promptmr_run4`를 각각 돌려 **MoE가 이득인지, PromptMR+가 SDUM run11(0.9161/0.9095)을 이기는지** 확정. 옮길 것은 repo(`2966dbd` 이상) + `../result/<net>/checkpoints/best_model.pt`.
- **P0 — MoE 미세조정 효과 판정.** shared 모델을 acc4/acc8 **부분집합으로 각각 재검증**해야 전문가 0.0769/0.1022와 비교가 된다. 지금은 미세조정이 이득인지 손해인지조차 모른다.
- **P1 — 미세조정 하이퍼 재검토.** 3 epoch 무변화는 lr 5e-5가 낮았을 가능성. lr↑ 또는 epoch↑ A/B.
- **P2 — epoch 스윕.** `run_base`·`run_moe shared` 둘 다 ep8 plateau → 10 epoch 이상이 이득인지 확인.
- **P2 — AMP/BF16 재검토.** 서버가 Ada(cap 8.9)면 FP32 강제 제약이 유효하지 않다. 속도가 점수 요소인 만큼 실측 가치 있음.

**다음 학습 고려사항(탐색거리, 확정 아님)**
- MoE를 K-Fold로 검증하려면 **acc 층화 분할**이 먼저다 — 지금 `--acc`는 필터 후 분할이라 단계마다 fold 경계가 달라진다(`load_data.kfold_split` docstring 참조).
- `run_kfold_f2` 완주 후에도 **비교 대상이 없다**(같은 볼륨 분할의 다른 fold가 0개). f0/f3/f4를 더 돌려 CV 평균을 만들지, 아니면 k-fold를 접고 고정 분할로 통일할지 결정 필요.
