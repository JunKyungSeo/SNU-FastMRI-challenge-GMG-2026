# SDUM run1 학습 로그 분석 (2026-07-07)

첫 GPU 학습(밤샘)에서 나온 로그 분석. **결론부터: 속도도 문제지만, 손실이 안 떨어지는 게 더 큰 문제.**

## 0. TL;DR

1. **⚠️ 손실이 학습 안 됨 (최우선).** Loss가 ~1.0에 고정 = SSIM ≈ 0 = 출력이 타깃과 거의 무상관. 스케일 불일치 의심. **속도 고치기 전에 이것부터.**
2. **속도: ~7초/iter → ~9시간/epoch → 30 epoch ≈ 11일.** 현재 config(제일 작은 4×[16,32])로도 비현실적.
3. **공짜 개선점: `num_workers=0`** (기본값). 데이터 로딩이 GPU와 안 겹침 → `--num-workers 4~8`만 줘도 빨라짐.

---

## 1. 실행 조건

```
python train.py --model sdum --num-cascades 4 --channels 16 32 \
  -b 1 -l 2e-4 -e 30 -r 50 -n sdum_run1 -t /root/Data/train/ -v /root/Data/val/
```
(뒤에 `; ... --channels 32 64` 도 이어 붙였지만, 첫 run이 11일짜리라 두 번째는 사실상 시작 못 함. 아래 로그 = **4×[16,32] run**.)

- 데이터: train 4651 슬라이스/epoch (batch=1), 30 epoch.
- 손실: `SSIMLoss` = `1 − SSIM(output, target, data_range=maximum)` (utils/common/loss_function.py).
- 옵티마이저: `AdamW(lr=2e-4, weight_decay=1e-3)`, **스케줄러·warmup 없음**.
- 이 config는 우리 조합 중 **가장 작음**(더 키우면 더 느리거나 OOM).

---

## 2. 속도 분석

로그의 `Time`은 **50 iter 블록 소요시간**(report_interval=50):

| 구간(iter) | Time(50 iter) | 초/iter |
|---|---|---|
| 1–50 | 347.2s | 6.94 |
| 51–100 | 352.7s | 7.05 |
| 101–150 | 347.4s | 6.95 |
| 151–200 | 349.1s | 6.98 |
| … | ~348s | ~6.97 |

- **≈ 7.0 초/iter** (forward + backward + AdamW step, 풀 해상도).
- **1 epoch** = 4651 × 7.0 ≈ **9.0 시간** (+ 매 epoch val).
- **30 epoch** ≈ **270시간 ≈ 11.3일.** → 제출 일정상 불가.

### 왜 이렇게 느린가
1. **모델이 근본적으로 무거움**: 풀 k-space 해상도(≈640×368)에서 Restormer(transformer, MDTA attention) + CSME(BasicUNet)를 **4 cascade** 반복. 맨 끝에만 384² crop.
2. **FP32 강제**: GTX 1080은 Pascal이라 BF16/AMP 불가 → FP16 대비 2배 느리고 2배 메모리.
3. **gradient checkpointing ON**: 메모리 아끼려고 backward에서 forward 재계산 → 시간 +30~50%. (메모리↔속도 맞바꾼 것.)
4. **`num_workers=0`** (load_data.py 기본): h5 읽기 + transform(FFT·마스킹)이 **메인 프로세스에서 직렬** 실행 → GPU가 데이터 기다리며 놀 수 있음. **가장 쉬운 개선점.**

### 속도 개선 (효과 큰 순)
- **`--num-workers 4~8`**: 데이터 로딩을 GPU 연산과 겹침. 코드 이미 지원(`getattr(args,"num_workers",0)`) — train.py에 인자만 있으면 됨. 공짜.
- **디버깅은 subset으로**: 전체 4651로 loss 곡선 볼 필요 없음. 몇백 슬라이스만으로 "배우는지" 먼저 확인 → iteration 시간 단축.
- **epoch 현실화**: 30은 불가. 학습이 되는 걸 확인한 뒤 시간 예산에 맞춰 재설정.
- **checkpointing OFF 시도**(메모리 되면): 작은 config는 끄면 30~50% 빨라질 수 있음.
- 냉정한 사실: 1080 FP32에서 이 모델은 태생적으로 느림. 최종 제출 학습엔 **GPU 시간 계획**이 별도로 필요.

---

## 3. ⚠️ 손실 분석 (최우선 — 속도보다 먼저)

로그의 Loss:
```
0.9894, 1, 1, 1, 0.9894, 1, 1, 1, 0.8677, 0.9894
```
- `Loss ≈ 1.0` = `1 − SSIM` 이므로 **SSIM ≈ 0** (출력이 타깃과 거의 무상관).
- 450 iter 동안 **감소 추세 없이 1.0 근처에서 진동**. 정상 학습이면 첫 epoch 안에 0.9→0.8→… 내려가야 함.
- 비교: E2E-VarNet은 DC 덕에 **초기부터** zero-filled 수준 SSIM(≈0.5~0.7)에서 시작해 내려감. 여기 SDUM은 **바닥(≈0)에서 시작해 안 움직임** → 구조적 문제 신호.

### 의심 원인 (검증 필요, 확정 아님)
1. **[유력] 출력 스케일 불일치.** `normalize=off`(load_data 기본)라 타깃은 원 스케일, `data_range=attrs['max']`. SDUM 어댑터 출력(RSS→384 crop)의 **절대 크기(magnitude)가 타깃과 다르면** SSIM이 붕괴. VarNet은 DC가 스케일을 잡아주지만, 우리 어댑터엔 그 보장이 없음.
2. warmup 없는 lr 2e-4 → transformer scratch 초기 불안정 가능. 단 그러면 loss가 **요동**치지 완전 flat 1.0은 아님 → 스케일 쪽이 더 의심.
3. CSME/DC 초기화로 출력이 near-constant → SSIM 0.

### 조사 방법 (systematic-debugging 권장)
- 첫 배치에서 찍어보기:
  `output.min/max/mean` vs `target.min/max/mean` vs `maximum`.
  → 스케일이 몇 배 어긋나는지 즉시 보임.
- 어긋나면: 어댑터 출력에 스케일 정합(타깃 정규화 or 출력 스케일링) 추가, 혹은 `--normalize` 경로 검토.
- **이게 안 잡히면 속도를 아무리 개선해도 무의미** (빠르게 안 배움).

---

## 4. 하이퍼파라미터 평가 (입력값 기준)

| 인자 | 값 | 평가 |
|---|---|---|
| `--num-cascades` | 4 | 최소급. 그래도 7s/iter. 손실 문제 해결 전엔 키울 이유 없음. |
| `--channels` | 16 32 | 가장 작은 폭. 병변 과평활 우려 있으나 지금은 "배우는지"가 먼저. |
| `-b` (batch) | 1 | 8GB 강제. gradient accumulation으로 유효 배치↑ 하면 안정성 도움 가능. |
| `-l` (lr) | 2e-4 | AdamW+wd 1e-3. transformer scratch엔 **warmup 권장**. 손실 문제와 별개로 개선 여지. |
| `-e` (epochs) | 30 | 시간상 **비현실(11일)**. 먼저 subset/1 epoch로 학습 여부 확인 후 결정. |
| `-r` (report) | 50 | ~6분 간격. OK. 초기 디버깅은 `-r 10`으로 촘촘히 보는 게 유리. |

**누락되어 아쉬운 것**: `--num-workers`(기본 0!), lr warmup/scheduler, gradient accumulation, 고정 seed.

---

## 5. 권고 — 다음 액션 (우선순위)

1. **[P0] 손실 원인부터.** 첫 배치 output/target 스케일 비교 → 스케일 정합 수정. 배우기 시작하는지 확인.
2. **[P1] `--num-workers 4~8` 추가** + **subset·짧은 epoch**로 빠르게 loss 곡선 확인. (속도 공짜 개선)
3. **[P2] 학습 확인되면** epoch/시간 예산 재설계, 필요시 checkpointing off·config 조정.
4. 32 64 등 큰 config 스윕은 위 1~2가 끝난 뒤에.

> 요약: 밤샘 run의 성과 = **파이프라인이 GPU에서 끝까지 돈다는 확인**. 하지만 **손실이 안 떨어져 아직 "학습"은 아님.** 스케일 정합이 먼저다.
