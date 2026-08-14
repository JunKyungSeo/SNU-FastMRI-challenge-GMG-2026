# run7 recon_eval 스코어카드 — 첫 리더보드 실채점 (2026-07-10)

> run7(`sdum_run5`, [16,32]×4 + readout-crop + cosine 2.4e-4, 2ep)의 **공식 채점**. shell7에서 `recon_eval.py -n sdum_run5 -p /root/Data/leaderboard`로 산출.
> 학습 분석은 [run7 training 분석](2026-07-09-run7-crop-cosine-analysis.md) 참고 — 이 문서는 **실채점 3축 숫자**만 간단히.

**한 줄:** run7은 리더보드에서 **SSIM_full 0.9004 / SSIM_bbox 0.8911 / 1515 ms·slice**. baseline(zero-filled) 대비 **+0.07** = 확실히 학습됨. 추론시간 경고 1건으로 양호(run8은 79건).

## 3축 점수 (새 규칙 = `0.5·SSIM_full + 0.5·SSIM_bbox + 추론시간`)

| 지표 | 전체 | acc4 | acc8 |
|---|---|---|---|
| **SSIM_full** | **0.9004** | 0.9275 | 0.8733 |
| **SSIM_bbox** | **0.8911** | 0.9205 | 0.8616 |
| **Recon Time** | **1515 ms/slice** (총 3352s) | 1528 ms/slice | 1500 ms/slice |
| SSIM 평균(0.5:0.5) | **0.8958** | | |

- 느림 경고(>60s/volume): **1건**(acc4 1). 대체로 harness 기준 통과.

## 해석 (간단)

- **whole-image val(0.8846)은 채점식 SSIM_full(0.9004)을 과소보고**했다 — 채점은 foreground mask + `attrs['max']` data_range라 더 높게 나옴. 즉 학습 중 보던 whole SSIM보다 **실제 점수가 좋다**.
- **baseline 대비**: zero-filled full 0.833 / bbox 0.820 → run7은 **full +0.067 / bbox +0.071**. 모델이 확실히 일함.
- **acc8이 acc4보다 확연히 낮다**(full 0.873 vs 0.928, bbox 0.862 vs 0.921) — 고가속(acc8)이 어려운 게 점수로 드러남 → 개선 여지는 acc8 쪽.
- full(0.900)과 bbox(0.891)이 근접 → whole-image 손실이 병변 박스도 그럭저럭 커버(bbox 전용 손실 후순위 근거).

## 다음
- run8 대비는 [run8 분석 §5](2026-07-09-run8-capacity-analysis.md) 참고(run8은 SSIM +0.006지만 추론시간 1.86×로 새 규칙상 손해).
- run9(진행 중) 채점 나오면 "run7 대비 시간 안 늘리고 SSIM 오르나" 비교.
