# Stage-A 배선 — 분기 판정 + C1(bbox 가중 손실) — 2026-07-14

**한 줄 결론**: 교란요인 D(acc8)·E(작은 박스)를 숫자로 배제한 뒤 **분기 C** 확정 →
**bbox 가중 손실 λ만** 배선(`--bbox-loss-lambda`, 기본 0.0=off, off는 git HEAD와 **비트 동일** 실측 확인).
오버샘플링(w)은 **p=0.392 라 기각**. 학습은 실행하지 않았고 슬롯 명령만 작성해뒀다.

> 근거 진단: [`2026-07-14-stage0-lesion-diagnosis.md`](2026-07-14-stage0-lesion-diagnosis.md)

---

## 0. 자기 결과 재검토 (이전 판정을 믿지 않고 숫자에서 재유도)

### 0-1. p (병변 슬라이스 비율)

| | raw | 값 |
|---|---|---|
| 전체 학습 슬라이스 | 170 볼륨(acc4 85 + acc8 85) | **4,651** |
| 병변 슬라이스(≥1 유효 박스) | — | **1,824** |
| **p** | 1824 / 4651 | **0.392174 (39.22%)** |

정합성: 4,651 = run10 의 1 epoch iteration 수와 **정확히 일치** → 중복 계수 없음.
유효 박스 = 2,593개(win_size 필터로 버려진 박스 **0개** — 데이터셋이 이미 w·h>16 만 담음).

**경계선 판단 아님**: 임계 0.30 대비 p=0.392 는 **0.09 위**, 0.15 대비는 2.6배.
분포가 경계에 걸쳐 있지 않다 → 사람 판단으로 넘기지 않고 진행.

### 0-2. label × acc SSIM_bbox (n 포함) + 셀 간 표준편차

n≥30 셀만(작은 n은 평균이 노이즈). 전체 박스 평균 **0.9039** 대비 편차:

| label | acc | n | mean | dev(전체평균 대비) |
|---|---|---:|---:|---:|
| Bone- Subchondral edema | 8 | 37 | 0.8117 | **−0.0922** |
| Bone-Fracture/Contusion/dislocation | 8 | 96 | 0.8118 | **−0.0921** |
| Ligament - ACL High Grade Sprain | 8 | 51 | 0.8422 | **−0.0617** |
| Ligament - ACL Low Grade sprain | 8 | 57 | 0.8549 | **−0.0490** |
| Bone- Subchondral edema | 4 | 31 | 0.8841 | −0.0198 |
| Joint Effusion | 8 | 121 | 0.8934 | −0.0105 |
| Ligament - ACL Low Grade sprain | 4 | 38 | 0.8947 | −0.0092 |
| Ligament - ACL High Grade Sprain | 4 | 54 | 0.9035 | −0.0004 |
| Bone-Fracture/Contusion/dislocation | 4 | 72 | 0.9041 | +0.0002 |
| Meniscus Tear | 8 | 124 | 0.9225 | +0.0186 |
| Periarticular cysts | 4 | 74 | 0.9327 | **+0.0288** |
| Joint Effusion | 4 | 154 | 0.9457 | **+0.0418** |
| Meniscus Tear | 4 | 134 | 0.9488 | **+0.0449** |
| Cartilage - Partial Thickness loss/defect | 4 | 34 | 0.9541 | **+0.0502** |

**셀 간 표준편차 = 0.0460**, 범위 = **0.1424** (−0.0922 … +0.0502). 14개 중 8개가 ±0.02 **밖**.
→ **전혀 평평하지 않다.**

### 0-3. 박스 면적 구간별 SSIM_bbox

| 면적(px²) | n | SSIM | delta(=박스−자기슬라이스) | 95% CI(볼륨 클러스터) |
|---|---:|---:|---:|---|
| <500 | 106 | **0.9152** | −0.0026 | [−0.0145, +0.0084] (유의 아님) |
| 500–1,000 | 382 | 0.9010 | −0.0121 | [−0.0226, −0.0028] 유의 |
| 1,000–2,000 | 250 | 0.8967 | −0.0096 | [−0.0173, −0.0017] 유의 |
| 2,000–4,000 | 113 | 0.8978 | −0.0109 | [−0.0218, −0.0004] 유의 |
| 4,000–8,000 | 217 | 0.9089 | −0.0033 | [−0.0156, +0.0074] (유의 아님) |
| 8,000+ | 161 | 0.9121 | −0.0016 | [−0.0090, +0.0050] (유의 아님) |

corr(area, SSIM) = **+0.035** (Spearman −0.014) · corr(area, delta) = **+0.055**.
가장 작은 구간이 오히려 **평균 최고**이고 delta도 유의하지 않다.

### 0-4. 전체 평균 sanity

| | 재집계(CPU) | run10 채점값 | 일치 |
|---|---:|---:|---|
| SSIM_full | 0.9118 | 0.9118 | ✅ |
| SSIM_bbox | 0.9021 | 0.9021 | ✅ |

집계 로직 정확. 이후 모든 판정은 이 재집계 위에서 한다.

---

## 1. 분기 판정 — D·E 배제 후 **C**

순서대로 교란요인부터 배제했다.

### ❌ D 배제 — "label 무관하게 acc8만 꺼져 있다"가 **아니다**

D가 참이면 **acc4 안에서는** 박스가 자기 슬라이스만큼 잘 나와야 한다(delta ≈ 0).
acc4 만으로 재계산:

| | mean delta | 95% CI | n |
|---|---:|---|---:|
| acc4 전체 | −0.0037 | [−0.0091, +0.0018] (유의 아님) | 653 |
| acc8 전체 | −0.0123 | [−0.0226, −0.0023] 유의 | 576 |

전체 평균만 보면 D처럼 보인다. **그러나 label 별로 쪼개면 acc4 안에서도 무너진다:**

| label | **acc4** delta | 95% CI | n₄ | acc8 delta | n₈ |
|---|---:|---|---:|---:|---:|
| Ligament - ACL High Grade Sprain | **−0.0405** | [−0.063, −0.023] 유의 | 54 | −0.0520 | 51 |
| Ligament - ACL Low Grade sprain | **−0.0391** | [−0.053, −0.026] 유의 | 38 | −0.0446 | 57 |
| Bone- Subchondral edema | **−0.0379** | [−0.053, −0.024] 유의 | 31 | −0.0600 | 37 |
| Bone-Fracture/Contusion/dislocation | **−0.0277** | [−0.046, −0.012] 유의 | 72 | −0.0360 | 96 |
| Meniscus Tear | +0.0151 | [+0.008, +0.024] 유의 | 134 | +0.0194 | 124 |
| Joint Effusion | +0.0116 | [+0.007, +0.017] 유의 | 154 | +0.0071 | 121 |
| Cartilage - Partial Thickness | +0.0250 | [+0.015, +0.040] 유의 | 34 | +0.0305 | 23 |

**acc4 전체 delta 가 0 처럼 보인 건 어려운 label(−0.04)과 쉬운 label(+0.02)이 상쇄됐기 때문**이다.
가속률만의 문제가 아니다 → **D 배제.**
(acc8 이 전반적으로 나쁜 건 사실이고 그 격차(0.0567)가 더 크다 — 그건 §5 에서 별도 후보로 남긴다.)

### ❌ E 배제 — "작은 박스만 꺼져 있다"가 **아니다**

§0-3 그대로: 가장 작은 구간(<500px²)의 delta = **−0.0026, 유의하지 않음**. 오히려 SSIM 평균은 최고.
음의 delta 는 **중간 크기(500–4,000)** 에 있고, corr(area, delta) = +0.055 ≈ 0.
→ 해상도/고주파 문제로 환원되지 않는다 → **E 배제.**

### ❌ A 배제 — 평평하지 않다

n≥30 셀 14개 중 **8개가 평균 ±0.02 밖**, 셀 간 sd 0.0460, 범위 0.1424. → **A 아님.**

### ✅ **C 확정**

- D·E 배제됨.
- 꺼진 셀 **있음**: ACL sprain(High/Low), Subchondral edema, Bone-Fracture 4개 label 이
  자기 슬라이스 대비 **−0.028 ~ −0.060**, acc4/acc8 **양쪽에서** 유의(볼륨 클러스터 부트스트랩).
- **p = 0.392 ≳ 0.30** → 노출 부족이 아니라 **주의(attention) 부족**.

→ **오버샘플링(w) 배선 안 함. bbox 가중 손실(λ)만 배선.**

**w 를 기각하는 추가 근거**: q=2p=0.784 → w=5.637 이지만 이는 배치의 78%를 병변으로 채운다는 뜻이고,
q=3p=1.176 은 **q>1 이라 w 가 음수 = 정의 자체가 붕괴**. p 가 이미 0.392 인 데이터에서 풀 문제가 아니다.

---

## 2. 배선 내용 (C1 = `--bbox-loss-lambda`)

손실: **L = (1 − SSIM_full) + λ · (1 − SSIM_bbox)** · 박스 없는 슬라이스는 λ항 = 0.

### 변경 파일 (4개)

| 파일 | 변경 |
|---|---|
| `utils/common/loss_function.py` | **`bbox_ssim_loss()` 추가.** 유효 박스 판정·crop·평균을 채점(`metrics.ssim_bbox`)과 동일하게. 유효 박스 0개면 `0` 반환. |
| `train.py` | `--bbox-loss-lambda` (기본 **0.0 = off**) 추가. `>0` 이면 `args.return_bbox=True` 로 파생. |
| `utils/data/load_data.py` | `return_bbox` 를 **`and is_train`** 으로 게이트 → **val/leaderboard 로더엔 절대 안 붙음**(자연 분포·기존 언패킹 유지). |
| `utils/learning/train_part.py` | `train_epoch` 에서 bbox 소비. bbox 는 튜플 **맨 끝** 규약이라 먼저 떼어냄 → `--fg-loss` 와 **조합 가능**(run11이 이기면 fg+bbox 동시 필요). |

**고정 파일 미수정 확인**: `recon_eval.py`, `utils/common/metrics.py` — **손대지 않음**(읽기만).
기존 bbox 배관(`_parse_bbox`·`_bbox_collate`·flip 시 bbox 동기)은 이미 있어서 **손실 소비부만** 붙였다.

### 채점식과의 일치 (가장 중요)

학습 신호가 채점식과 어긋나면 bbox 를 미는 의미가 없다. `bbox_ssim_loss` 는
`metrics.ssim_bbox`(고정 파일)와 동일하게:
- 박스를 이미지에 클리핑 → **클리핑 후** w·h 가 `win_size`(=7) 미만이면 **제외**(채점도 `None` 처리)
- crop 한 뒤 그 위에서 SSIM 맵 계산 후 평균 (전체 맵을 잘라내는 방식이 아님)
- 배치의 모든 유효 박스에 대한 **박스 단위 평균** = 채점의 `bbox_total/bbox_idx` 와 같은 단위

**실측 검증: `1 - bbox_ssim_loss` == `metrics.ssim_bbox` 를 소수점 8자리(abs 1e-8)까지 재현** ✅

---

## 3. off 일 때 비트 동일 검증 (실측)

`--fg-loss` 선례와 같은 방식. **git HEAD 원본 `train_epoch`** 과 **C1 이 들어간 현재 `train_epoch`** 을
같은 seed(430)·같은 데이터·같은 모델로 CPU 에서 돌려 iter별 loss 를 비교.

```
scripts/verify_bbox_off_identical.py    (CPU, GPU·실모델 불필요)
```

| | epoch total loss |
|---|---|
| git HEAD 원본 | `0.004068452799419519` |
| 현재 코드, `--bbox-loss-lambda 0.0` (off) | `0.004068452799419519` |
| 현재 코드, `--bbox-loss-lambda 0.1` (on) | `0.004131599492752986` |

- **off: 20 iter 전부 + epoch total 까지 완전 일치 (`==`, float 전체 자릿수)** ✅
- **on: iter14 부터 갈라짐** — 테스트 볼륨(`knee_acc4_1.h5`)의 첫 병변 슬라이스가 **13번**이라,
  거기서 처음 λ 그래디언트가 들어간 **직후** 발산. λ항이 **병변 슬라이스에서만** 작동한다는 증거. ✅

### pytest (`scripts/test_bbox_loss.py`, 6 passed)

| 테스트 | 확인 |
|---|---|
| `test_bbox_loss_matches_scoring_function` | 학습 손실 == `metrics.ssim_bbox` (1e-8) |
| `test_win_size_filter_matches_scoring` | w=6 / 클리핑 후 w=4 박스를 채점도 학습도 똑같이 버림. 경계값 7은 유효 |
| `test_no_box_slice_is_exactly_zero` | 박스 없는 슬라이스 → λ항 정확히 0 → 손실 **한 비트도** 안 바뀜 |
| `test_bbox_loss_has_gradient` | λ항에 실제로 grad 가 흐름(신호가 됨) |
| `test_off_is_bit_identical` | off → 6-튜플, 앞 6개 원소 on/off 비트 동일 |
| `test_val_loader_never_gets_bbox` | λ>0 여도 **val 로더는 6-튜플** (자연 분포 유지) |

**전체 스위트 25 passed** (기존 19 + 신규 6) — 회귀 없음.
재현성: 샘플러를 안 쓰므로(§1에서 w 기각) 기존 seed 경로 그대로. `--bbox-loss-lambda` 는 데이터
순서에 영향을 주지 않는다.

---

## 4. λ 유도

run10 실측 손실 항 크기:

| 항 | 값 |
|---|---|
| 1 − SSIM_full | **0.0882** |
| 1 − SSIM_bbox | **0.0979** (full 항의 1.11배) |

두 항의 크기가 비슷해서 λ 가 곧 "bbox 에 주는 상대 비중"이 된다:

| λ | 병변 슬라이스에서 λ항/총손실 | epoch 평균 기여(p=0.392 반영) |
|---:|---:|---:|
| 0.05 | 5.3% | 2.1% |
| **0.10** | **10.0%** | **3.9%** |
| 0.20 | 18.2% | 7.1% |
| 0.30 | 25.0% | 9.8% |
| 0.50 | 35.7% | 14.0% |
| 1.00 | 52.6% | 20.6% |

**λ = 0.1 채택.** 근거:
1. **A/B 게이트가 "SSIM_full 비퇴행"** 이다. bbox 를 미는 개입이라 full 을 깎으면 순손실.
   λ 가 클수록 full 항의 상대 비중이 줄어 full 이 위험해진다.
2. **먹을 게 애초에 적다.** 진단상 bbox 를 **완벽히** 고쳐도 총점 상한 **+0.0099**.
   0.5·full 이 0.002 만 깎여도 이득이 절반 날아간다. 큰 λ 로 도박할 이유가 없다.
3. 승급 경로가 있다: run12 에서 **bbox↑ & full 평평**이면 다음 run 에서 λ=0.3 검토.
   반대로 full 이 깎이면 λ 가 과한 것 → λ↓ 또는 롤백.

---

## 5. 정직하게 남기는 한계 (읽고 결정해야 할 것)

1. **균일 λ 는 절반이 헛일이다.** 진단상 Meniscus Tear·Joint Effusion·Cartilage 박스는
   자기 슬라이스보다 **이미 쉽다**(delta > 0). 이 3개가 leaderboard 박스의 **48%**(590/1229).
   균일 λ 는 여기에도 똑같이 힘을 준다. 그럼에도 균일로 간 이유: **label 조건부 λ 는 검증이 불가능**하다
   (대부분 label 의 n < 30, `Ligament - PCL High Grade` 는 leaderboard 박스 **0개** = 데이터 없음).
   내 진단 문서 스스로가 "n 작은 label 로 가중치 튜닝하지 말 것"이라고 했으니 그걸 지킨다.
2. **더 큰 물고기는 acc8 이다.** bbox 완전 해결 상한 +0.0099 vs acc4↔acc8 격차 **0.0567**(bbox)/**0.0481**(full).
   C1 은 "싸니까 해보는" 개입이지 최우선 레버가 아니다. acc8(예: acc별 GRAPPA 융합 가중,
   `strategy/candidates.md`)은 **다음 A/B 후보로 남긴다** — 지금 같이 켜면 A/B 가 오염된다.
3. **λ항은 SSIM_full 항과 같은 영역을 일부 중복해서 민다**(박스는 foreground 안에 있다).
   그래서 full 이 깎일 수도, 같이 오를 수도 있다. 그래서 게이트를 full 비퇴행으로 잡았다.

---

## 6. A/B 게이트 (run12 판정 기준)

| | baseline (run10 또는 run11 중 승자) | run12 (= baseline + λ=0.1) | 판정 |
|---|---|---|---|
| **SSIM_full** | 기준값 | **≥ baseline (비퇴행)** | ⛔ 깎이면 실패 → λ↓ 또는 롤백 |
| **SSIM_bbox** | 기준값 | **> baseline** | 이게 오르는 게 목적 |
| 추론시간 | 1523 ms/slice | 동일해야 정상 | 손실만 바뀜 → 변할 이유 없음 |
| label별 재분해 | ACL/edema/fracture delta −0.03~−0.05 | **이 4개 delta 가 0 쪽으로** | 진짜 원인을 건드렸는지 확인 |

> ⚠️ 평소 A/B(=full 을 올리는 개입)와 **방향이 반대**다. C1 은 bbox 를 미는 대신 full 을 깎을 수 있는
> 개입이라, "full 비퇴행"이 통과 조건이지 부수 조건이 아니다.

---

## 7. 슬롯 명령 (작성만 함 — **실행 안 됨**)

`shell_scripting/shell11.sh` (README 규칙: 11번째부터 `_template.sh` 복사).

```bash
python train.py --model sdum --num-cascades 4 --channels 16 32 \
    -b 1 -l 2.4e-4 -e 10 -r 100 --num-workers 4 \
    --readout-crop --warmup-iters 200 $FG_FLAG \
    --bbox-loss-lambda 0.1 \
    -n sdum_run12 --run-tag run12 -t /root/Data/train/ -v /root/Data/val/
```

**의존성 가드 내장**: 스크립트 상단 `BASELINE=""` 이 비어 있으면 **실행을 거부한다(exit 2)**.
run11(`--fg-loss`)이 아직 돌고 있어서 A/B baseline 이 아직 안 정해졌기 때문:

- `BASELINE=run11` → run11 승 → `FG_FLAG="--fg-loss"` (fg 유지한 채 C1 추가)
- `BASELINE=run10` → run11 패 → `FG_FLAG=""` (fg 빼고 C1만 추가)

**실행 확인**: `bash shell_scripting/shell11.sh` → `[STOP] BASELINE 이 비어 있다` + exit 2.
GPU 미사용, 학습 미시작. ✅

---

## 8. 다음에 사람이 결정할 것

1. **run11 채점** → `python recon_eval.py -n sdum_run11 -p /root/Data/leaderboard`
   (run10 = full 0.9118 / bbox 0.9021 과 비교)
2. 그 승자를 `shell11.sh` 의 `BASELINE=` 에 적기.
3. **54h 를 C1(상한 +0.0099)에 쓸지, acc8(격차 0.0567)에 쓸지** — 나는 C1 을 배선했지만,
   기대 이득만 보면 acc8 이 더 크다. 순서는 사람이 정할 일이다.
4. 커밋 전 **"지금 사용자가 누구인지"** 확인 필요(팀 2명, identity 오염 방지). 아직 커밋 안 함.
