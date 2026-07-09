# run8 (capacity) 분석 — **훈련 전(pre)** 목적·변경 기재 (2026-07-09)

> ⏳ **이 문서는 훈련 전에 미리 쓴 것.** 목적·가설·변경만 기록하고 **결과는 훈련 후 채운다.**
> run8은 run7 종료 후 큐(shell6)에서 자동 시작. base LR 2.4e-4(논문값), readout-crop 유지.

**한 줄 목적:** run7(readout-crop+cosine)이 loss 0.037까지 내려간 걸 보고, **readout-crop이 비운 VRAM 헤드룸(~1.8GB)을 용량(capacity)에 투자** — depth와 width를 동시에 올려 성능이 더 오르는지 본다.

---

## §1 직전(run7) 대비 diff  *(무엇을 바꿨나 / 왜)*

| 항목 | run7 | **run8** | 의도 |
|---|---|---|---|
| `--num-cascades` | 4 | **6** | iterative-refinement 깊이↑. grad-checkpointing 덕에 **VRAM 거의 안 씀**(activation-flat) → 사실상 공짜 용량, 시간만 +50% |
| `--channels` | [16,32] | **[24,48]** | Restormer 폭↑(~1.5×). readout-crop이 비운 **activation 헤드룸을 실제로 쓰는** 유일한 CLI 레버 |
| 그 외 | — | 동일 | -b 1, -l 2.4e-4, -e 2, --readout-crop, --warmup-iters 200 (run7과 같음) |

> ⚠️ **2개 레버 동시 변경** = run7과의 순수 1-레버 A/B는 아님(의도적). 시간 여유가 있어 depth+width를 함께 밀어보고, **효과가 있으면 run9에서 어느 쪽이 기여했는지 분리**한다.

## §2 근거 / 가설

- run7에서 **readout-crop+cosine LR(2.4e-4)** 이 optimization 병목을 풀어 train loss 0.037 달성(run4 plateau ~0.11 대비 확연). → 이제 **용량이 도움될 국면**이라는 판단(팀: 과적합 아님).
- **gradient checkpointing 확정 활성**(latent_recon.py:112-125, use_reentrant=False) → cascade 깊이는 activation-flat이라 T=6도 VRAM 안전, width만 헤드룸 소비.
- augmentation은 **다른 팀원이 진행 중이라 안 건드림**.

## §3 리스크 / 게이트 (훈련 전 확인)

- **OOM 위험 borderline~medium**: [24,48]×6 추정 VRAM **~7.1–8.1GB**(8192 근접). 워크플로우 적대적 검증이 "[24,48] 단독도 ~7.4–7.9GB로 낙관 편향" 경고 → cascade 2개 추가분(경계텐서 ×wider)까지 얹힘.
  - **선점검(선택)**: run7이 GPU 비운 뒤 `python scripts/probe_oom.py --cascades 6 --channels 24 48` (peak≤~7.0GB면 안전; nvidia-smi는 probe보다 0.5–0.9GB 높음).
  - **OOM 시**: run9에서 `[20,40]` 또는 cascades 5로 후퇴. (시간 낭비 아님 — fail-fast)
- **시간**: ~7–8 s/iter 예상 → **~18–20h**(cascades 6 + 넓은 채널).
- → **실측 업데이트: VRAM ~6GiB로 OOM은 기우로 판명(§4 참조). CSME(C=15 고정)가 활성 지배 → 폭 확장 비용이 추정보다 작았음.**

## §4 결과 (진행 중 — 초기 실측 반영, 나머지 훈련 후)

**초기 실측 (iter 0~100, 2026-07-09):**
- ✅ **VRAM ~5997 MiB(≈5.85 GiB)로 안정 — OOM 걱정은 기우.** 여유 ~2GiB. 추정(7.1–8.1GB)이 과대평가였다.
  - 이유: 활성 메모리의 지배자는 **CSME(코일 C=15 고정)**이고 `--channels`와 무관 → Restormer 폭만 넓혀도 총 VRAM은 예상만큼 안 늘었다. 워크플로우 mem 에이전트도 "CSME/Restormer 비율은 가정"이라 실토했던 부분.
  - nvidia-smi=예약+cuDNN workspace라 config별 ±수백 MiB 노이즈(실제 run8 5997 < run7 6375로 더 큰 모델이 오히려 더 적게 뜨기도). 시작 직후 5.22GiB는 램프업 전 값.
- ✅ **~7.45 s/iter**(745s/100iter) → epoch ~9.6h, **총 ~19h** (예측 18–20h 부합).

**훈련 후 채울 것:**
- [ ] TrainLoss·Val SSIM (run7의 val 0.881 대비 개선?)
- [ ] run7 과적합 판정 시 방향(→ run9). **헤드룸 ~2GiB 남아** run9에서 [32,64]·mlp_ratio 등 더 키울 여지.
