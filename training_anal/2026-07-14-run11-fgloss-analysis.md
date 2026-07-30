# run11 분석 — fg-loss(B5) 정합, 완료 (2026-07-14, 완주·채점 2026-07-15)

> ✅ **완료.** run11 10ep 완주 + recon_eval 채점 끝. **판정: fg-loss 승 = 새 best.**
> **최종 recon_eval:** SSIM_full **0.9161** / SSIM_bbox **0.9095** / **1514 ms/slice** (acc4 full 0.9390·bbox 0.9361 / acc8 full 0.8933·bbox 0.8828).
> → run10(0.9118 / 0.9021 / 1523ms) 대비 **full +0.0043 · bbox +0.0074 · 시간 −9ms = 3축 전부 개선.** fg-loss가 SSIM_full을 올리면서 **우려했던 bbox 하락도 없이 동반 상승**. **run11 = 현재 best.**
> 최종 fg-val(foreground-masked) = ep8 **0.9041**(1−0.09587). val이 실채점 SSIM_full(0.9161)을 방향·근사값까지 예측함(신뢰 예측기 확정).

**한 줄 결론:** run10과 **완전히 동일한 config([16,32]×4, 10ep)에 `--fg-loss`(B5)만 켰다** — 손실·검증을 대회 채점처럼 **foreground(몸통) 마스크 안에서만** 계산. fg-정합 val이 0.8908→**0.9011**(ep3)로 상승 중. **단 fg-loss는 val 지표 자체를 바꿔(whole→foreground) run10 val과 직접 비교가 안 됨** — 진짜 효과(SSIM_full↑·bbox 유지)는 **recon_eval A/B로만** 확정. (미측정)

> ℹ️ **이 run에 반영된 개선 = `--fg-loss` (B5, @esel34 구현, 커밋 347fb43).** 손실·검증을 채점 정합(`metrics.foreground_mask` 안에서만 SSIM). 마스크 손실이 채점식과 소수점 8자리까지 일치, 옵션 끄면 기존과 완전 동일. run10=off / run11=on의 유일한 차이.

## §0 핵심 요약

- 🔑 **개선 = fg-loss**: 기존 손실·검증은 배경 포함 전체 384²로 SSIM을 재서 **배경이 점수를 부풀리고 val이 실채점을 예측 못 함**. fg-loss는 채점과 같은 foreground 영역만 봐서 이 어긋남을 잡는다.
- ⚠️ **run10 val과 직접 비교 금지(함정)**: run10 val=whole-image, run11 val=**foreground-masked(≈SSIM_full)**. 지표가 달라 run11 val(0.9011)이 run10 val(0.8927)보다 높은 건 **일부가 지표 차이**(fg-masked가 원래 더 높게 나옴). fg-loss "효과"를 val 숫자 빼기로 주장하면 안 됨.
- ✅ **효과 확정(recon_eval)**: run11 완주 채점 **full 0.9161 / bbox 0.9095** > run10 **0.9118 / 0.9021** → fg-loss가 full·bbox **동반 상승**(bbox 하락 우려 기각). 시간도 1514ms로 동등. **fg-loss 승, 새 best.**
- ⚠️ **bbox 하락 감시**(@esel34 인계): fg-loss는 몸통 전체 정합이라 병변 박스(bbox) 점수가 떨어질 수 있음 → recon_eval에서 **bbox 안 떨어지는지 꼭 확인**.
- ✅ **fg-val이 실채점을 예측**: run11 fg-val(≈SSIM_full)이 ep3에 0.9011 — run10의 실채점 SSIM_full 0.9118(ep9)에 근접, 아직 6 epoch 남음. val이 이제 점수를 제대로 추종.

## §1 직전(run10) 대비 diff  *(필수)*

| 항목 | run10 | **run11** | 비고 |
|---|---|---|---|
| `--fg-loss` | OFF (전체 384²) | **ON (foreground 마스크)** | ← **유일한 변경** |
| 손실 | SSIMLoss 통짜 | SSIMLoss × fg_mask (채점 정합) | loss_function.py 확장 |
| val 지표 | whole-image SSIM | **foreground-masked SSIM(≈SSIM_full)** | train_part.validate 확장 |
| 그 외 | [16,32]×4, 10ep, readout-crop, LR 2.4e-4, warmup 200 | **동일** | |

**목적:** 손실·검증을 채점(foreground)에 맞추면 실제 SSIM_full이 run10(0.9118)보다 오르나. 배경 제외로 학습 신호가 몸통에 집중되는 효과 기대.

## §2 직전 분석 지적 추적  *(필수)*

| 지적 (출처) | 판정 | 근거 |
|---|---|---|
| whole-image val이 실채점 과소보고 (run8·run10) | ✅**정합됨** | fg-loss로 val=foreground-masked → 이제 SSIM_full 추종 |
| ⑩ 대회 지표 검증 배선(점수 절반 눈감음) | ⚠️→✅**부분해결** | val이 SSIM_full 정합(masked). 단 **bbox는 여전 val 미로깅**(full만) |
| ⑨ best-model이 채점축 아님 | ✅**개선** | 이제 foreground-masked val로 best 선택(SSIM_full 정합) |
| 작은+오래 우월 (run10) | (계승) | run11도 [16,32]×4·10ep 유지, 여기에 fg-loss만 추가 |

## §3 손실 · 지표 (진행 중, ep0~3)

**fg-정합 val 궤적 (foreground-masked ≈ SSIM_full):**
| epoch | Train(fg) | Val(fg-masked SSIM) | Δ |
|---|---|---|---|
| 0 | — | 0.8908 | — |
| 1 | — | 0.8972 | +0.0064 |
| 2 | 0.1036 | 0.8994 | +0.0022 |
| 3 | 0.1011 | **0.9011** | +0.0017 |

- 상승 중(아직 plateau 전, 6 epoch 남음). run10의 whole-val이 epoch 8에 멈춘 것처럼 run11도 후반 plateau 예상 → 최종 fg-val은 ~0.91+ 가능성.
- ~5.4h/epoch, -e 10이라 완주 시 총 ~54h(진행 중).

**⚠️ 비교의 정직성:**
- run11 fg-val(0.9011, 15-vol val)과 run10 SSIM_full(0.9118, ~80-vol 리더보드)은 **집합도 epoch도 다름** → 직접 대소 비교 부적절.
- fg-loss의 순수 효과 = **같은 지표·같은 데이터에서 on vs off**. 그게 통제 arm(run10_off2ep vs run11_on2ep) recon_eval, 또는 run11 완주 후 recon_eval vs run10 0.9118. **둘 다 대기.**

## §3.5 일반 ML 구현 점검 (델타)

- ⑩ **대회 지표 검증 정합: ⬜→⚠️부분** — val이 이제 채점식 SSIM_full과 정합(masked). **큰 개선.** 단 SSIM_bbox는 여전 val 미로깅.
- ⑨ **best-model 선택: 개선** — foreground-masked val 기준 = 채점축(full)에 근접.
- ⑦ resume ✅. 나머지(grad clip·NaN·augmentation·bbox 손실) 여전 ⬜.

## §7 다음 액션 (run11 완주 후)

- ⭐ **recon_eval A/B로 fg-loss 효과 확정**: `recon_eval -n sdum_run11` (완주 후) vs run10(0.9118/0.9021). 또는 통제 arm `sdum_run10_off2ep` vs `sdum_run11_on2ep` 2ep 채점. **SSIM_full 오르나 + bbox 안 떨어지나** 둘 다 확인.
- **bbox가 떨어지면**: fg-loss는 full엔 좋아도 bbox엔 손해일 수 있음 → bbox 가중 손실을 추가로 검토(팀원 우선순위).
- **이 문서 완성**: run11 완주 + recon_eval 나오면 §3·§0에 최종 3축·A/B 결과 채우고 판정 확정.
