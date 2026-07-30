| 파일 | delta / 역할 |
|---|---|
| `shell1.sh` | run11(--fg-loss) 리더보드 채점 — shell3의 BASELINE(run10 vs run11) 결정용 |
| `shell2.sh` | run12 전 UC+bbox 조합 crash-check 스모크 (disposable) |
| `shell3.sh` | run12 = bbox 가중 손실 ON-팔 학습 (+UC 묶음이라 단일 레버 A/B 아님) |
| `shell4.sh` | run13 = run11 + --uc 하나만 얹은 clean UC A/B |
| `shell4_resume.sh` | run13을 --resume으로 epoch1부터 이어학습 (AUTO 오케스트레이터용) |
| `shell5.sh` | run7 = readout-crop + cosine+warmup, 2 epoch (태그충돌로 결과는 sdum_run5) |
| `shell6.sh` | run8 = capacity 실험 — cascades 4→6, channels [16,32]→[24,48] |
| `shell7.sh` | run7·run8 리더보드 채점 (SSIM_full/bbox/시간 3축) |
| `shell8.sh` | run9 = 장기학습 — run7 config 그대로 epoch 2→4 |
| `shell9.sh` | run10 = run9 동일 config, epoch 4→10 (plateau 확인) |
| `shell10.sh` | run11 = run10 + --fg-loss (fg 손실 A/B ON-팔) |
| `shell11.sh` | PromptMR+ 8GB 스모크 — coil압축(15→8)+readout-crop+nc4, 120 iter |
| `shell12.sh` | promptmr_run1 = PromptMR+ 첫 실학습 (coil 8 압축·nc4, 5 epoch + 채점) |
| `shell12_resume.sh` | promptmr_run1을 --resume으로 epoch 6부터 이어학습 |
| `shell13.sh` | 코일 무압축(15개) 메모리 탐침 — 학습 없음, cascade 상한 실측 |
| `shell14.sh` | promptmr_run2 = 코일 15개 무압축 + nc10 (run1의 대조군) |
| `shellB.sh` | SDUM --resume 기능 스모크 — scheduler fast-forward·val_loss_log 이어쓰기 검증 |
