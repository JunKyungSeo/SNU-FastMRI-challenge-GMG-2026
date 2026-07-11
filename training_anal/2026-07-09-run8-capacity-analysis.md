# run8 (capacity) 분석 — 완료 (착수 2026-07-09 · 종료 2026-07-10)

**한 줄 결론:** capacity(cascades 4→6, channels [16,32]→[24,48])는 **SSIM을 아주 조금(+0.006)** 올렸지만 **추론시간을 ~1.86× 폭증**(1515→2820 ms/slice, 조직 "느림" 경고 1→**79건**)시켰다. 새 채점(0.5·full+0.5·bbox+**추론시간**)에서 **capacity 베팅은 시간으로 손해** — 이득이 시간 페널티에 먹힐 공산이 크다. VRAM은 문제없었음(≈7.4GB). → **더 큰 모델이 아니라 "같은/작은 capacity로 더 오래·더 똑똑히"가 방향**(별도 run으로 검증할 것).

> ✅ 목적: readout-crop이 비운 VRAM 헤드룸을 capacity에 투자해 성능이 오르나. **결과: SSIM은 소폭↑ 맞지만 새 규칙상 추론시간 손해로 상쇄.**

---

## §1 직전(run7) 대비 diff  *(무엇을 바꿨나 / 왜)*

| 항목 | run7 | **run8** | 의도 |
|---|---|---|---|
| `--num-cascades` | 4 | **6** | iterative-refinement 깊이↑. grad-checkpointing 덕에 **VRAM 거의 안 씀**(activation-flat) → 사실상 공짜 용량, 시간만 +50% |
| `--channels` | [16,32] | **[24,48]** | Restormer 폭↑(~1.5×). readout-crop이 비운 **activation 헤드룸을 실제로 쓰는** 유일한 CLI 레버 |
| 그 외 | — | 동일 | -b 1, -l 2.4e-4, -e 2, --readout-crop, --warmup-iters 200 (run7과 같음) |

> ⚠️ **2개 레버 동시 변경** = run7과의 순수 1-레버 A/B는 아님(의도적). 시간 여유가 있어 depth+width를 함께 밀어보고, **효과가 있으면 다음 run에서 어느 쪽이 기여했는지 분리**한다.

## §2 근거 / 가설

- run7에서 **readout-crop+cosine LR(2.4e-4)** 이 optimization 병목을 풀어 train loss 0.037 달성(run4 plateau ~0.11 대비 확연). → 이제 **용량이 도움될 국면**이라는 판단(팀: 과적합 아님).
- **gradient checkpointing 확정 활성**(latent_recon.py:112-125, use_reentrant=False) → cascade 깊이는 activation-flat이라 T=6도 VRAM 안전, width만 헤드룸 소비.
- augmentation은 **다른 팀원이 진행 중이라 안 건드림**.

## §3 리스크 / 게이트 (훈련 전 확인)

- **OOM 위험 borderline~medium**: [24,48]×6 추정 VRAM **~7.1–8.1GB**(8192 근접). 워크플로우 적대적 검증이 "[24,48] 단독도 ~7.4–7.9GB로 낙관 편향" 경고 → cascade 2개 추가분(경계텐서 ×wider)까지 얹힘.
  - **선점검(선택)**: run7이 GPU 비운 뒤 `python scripts/probe_oom.py --cascades 6 --channels 24 48` (peak≤~7.0GB면 안전; nvidia-smi는 probe보다 0.5–0.9GB 높음).
  - **OOM 시**: 다음 run에서 `[20,40]` 또는 cascades 5로 후퇴. (시간 낭비 아님 — fail-fast)
- **시간**: ~7–8 s/iter 예상 → **~18–20h**(cascades 6 + 넓은 채널).
- → **실측 업데이트: VRAM ~6GiB로 OOM은 기우로 판명(§4 참조). CSME(C=15 고정)가 활성 지배 → 폭 확장 비용이 추정보다 작았음.**

## §4 결과 (진행 중 — 초기 실측 반영, 나머지 훈련 후)

**초기 실측 (iter 0~100, 2026-07-09):**
- ✅ **VRAM ~5997 MiB(≈5.85 GiB)로 안정 — OOM 걱정은 기우.** 여유 ~2GiB. 추정(7.1–8.1GB)이 과대평가였다.
  - 이유: 활성 메모리의 지배자는 **CSME(코일 C=15 고정)**이고 `--channels`와 무관 → Restormer 폭만 넓혀도 총 VRAM은 예상만큼 안 늘었다. 워크플로우 mem 에이전트도 "CSME/Restormer 비율은 가정"이라 실토했던 부분.
  - nvidia-smi=예약+cuDNN workspace라 config별 ±수백 MiB 노이즈(실제 run8 5997 < run7 6375로 더 큰 모델이 오히려 더 적게 뜨기도). 시작 직후 5.22GiB는 램프업 전 값.
- ✅ **~7.45 s/iter**(745s/100iter) → epoch ~9.6h, **총 ~19h** (예측 18–20h 부합).

**최종 학습 결과 (2 epoch 완주, exit 0):**
| epoch | Train | Val SSIM(whole) | train시간 |
|---|---|---|---|
| 0 | 0.1126 | 0.8845 | ~9.67h |
| 1 | **0.0972** | **0.8874** | ~9.68h |

- run7(whole val 0.8846) 대비 whole SSIM **+0.0028**. train도 계속 하강(0.113→0.097) = 여전히 under-training(과적합 아님, 팀 판단 맞음).
- ⚠️ epoch당 **~9.67h**(run7 ~5.2h의 **1.86×**). cascades 6 + 넓은 채널이 학습·추론 둘 다 느리게 만듦.

## §5 recon_eval 3축 실채점 (shell7, leaderboard) — **핵심**

새 규칙 = `0.5·SSIM_full + 0.5·SSIM_bbox + 추론시간`. 리더보드(`/root/Data/leaderboard`, ~80 볼륨)에서 `recon_eval.py`로 run7·run8 동시 채점:

| | SSIM_full | SSIM_bbox | SSIM 평균(0.5:0.5) | Recon Time | 느림경고(>60s) |
|---|---|---|---|---|---|
| **run7** ([16,32]×4) | 0.9004 | 0.8911 | **0.8958** | **1515 ms/slice** (3352s) | **1**건 |
| **run8** ([24,48]×6) | 0.9062 | 0.8979 | **0.9020** | **2820 ms/slice** (6241s) | **79**건 |
| Δ (run8−run7) | +0.0058 | +0.0068 | **+0.0063** | **+1305 ms/slice (~1.86×)** | +78 |
| (acc4 / acc8) | run8 0.9313/0.8811 | run8 0.9262/0.8696 | | acc8이 더 어려움(점수↓) | |

> baseline(zero-filled) full 0.833 / bbox 0.820 대비 두 run 모두 **+0.07~0.08**로 확실히 학습됨. acc8(고가속)이 acc4보다 낮음(예상).

**해석 (정직하게):**
- **whole-image val(0.887)은 recon_eval SSIM_full(0.906)을 과소보고**했다 — foreground mask + `attrs['max']` data_range라 채점식이 더 높게 나옴. 즉 우리 모델은 SSIM_full ~0.90으로 baseline보다 확실히 좋다.
- **run8의 SSIM 이득(+0.0063)은 실재하나 미미**하고, **추론시간이 ~1.86× 폭증**(조직 harness가 "너무 느림" 79회 경고). 시간항이 "~0.00n" 규모라도 **~2배 느려짐이 +0.006 SSIM을 상쇄/역전**시킬 공산이 크다 → **새 규칙에서 capacity 베팅은 순손해 가능성.**
- ⚠️ 정직성: 시간→점수 정확 매핑은 미공개("약 0.00n")라 net 부호 단정 불가. 하지만 **"느림 경고 79건"은 조직이 명시적으로 페널티 신호를 준 것** → 방향은 분명(추론 무겁게 만들지 말 것).

## §6 일반 ML 구현 점검 (run7 대비 델타만)

코드는 run7과 동일(run8은 config 변경뿐) → 체크리스트 대부분 그대로. **run7 분석 이후 바뀐 것:**
- ⑦ **resume(이어학습): ⬜없음 → ✅있음** — `--resume` 구현·머지됨(단 run8은 미사용). 이제 긴 run 중단복구 가능.
- ⑩ **대회 지표: ⚠️부분 → 사후 채점 ✅** — `recon_eval.py`로 3축 실채점 가능해짐(cv2 headless 해결). 단 학습 루프 내 로깅은 여전 미배선.
- 나머지(gradient clip·NaN감지·baseline 자동비교·augmentation)는 여전 ⬜(run7과 동일).

## §7 다음 액션 + 다음 학습 고려사항

- ⭐ **방향 = capacity가 아니라 "작은 모델 더 오래"**: run8의 교훈(capacity는 SSIM 소폭↑·추론 대폭↑ = 새 규칙상 손해)에 비추어, **추론 빠른 작은 모델([16,32]×4)을 epoch 늘려 더 오래 학습**하는 게 SSIM·시간 양축에 유리. → 별도 run으로 검증(recon_eval로 "시간 안 늘리고 SSIM 오르나").
- **추론시간이 이제 1급 제약**: run8의 79 느림경고는 red flag. 향후 capacity 키우기는 지양, 오히려 **추론 경량화**(작은 채널/cascade, 혹은 배포 최적화)가 점수 레버가 됨.
- **acc8 약점**: acc8 SSIM(0.87)이 acc4(0.93)보다 확연히 낮음 → 고가속 쪽 개선(예: acc8 오버샘플링·손실 가중)이 평균 점수에 효과적일 수 있음.
- bbox(0.898)와 full(0.906)이 근접 → 현재 whole-image 손실이 bbox도 그럭저럭 커버. bbox 전용 손실은 후순위 유지.
