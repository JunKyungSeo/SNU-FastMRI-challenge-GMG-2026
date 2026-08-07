# 2026 SNU FastMRI Challenge - Team GMG
전기정보공학부 서준경 - esel343434@snu.ac.kr
전기정보공학부 최현수 - bedso@snu.ac.kr

## 결과 (Public Leaderboard Data Evaluation)

| 모델 | SSIM_full | SSIM_bbox | ms/slice |
|---|---|---|---|
| zero-filled | 0.833 | 0.820 | — |
| PromptMR+ 단일 | 0.9309 | 0.9284 | 733.6 |
| PromptMR+ MoE | 0.9321 | 0.9300 | 734.3 |

점수 사본은 [shell_scripting/results/](shell_scripting/results/), 실행 로그는 [shell_scripting/logs/](shell_scripting/logs/)

## 베이스라인에서 바꾼 것

### 1. PromptMR+

[utils/model/promptmr/](utils/model/promptmr/)입니다. undersampled k-space와 mask를 넣으면 감도맵 추정부터 data consistency, 정규화, RSS, 384 crop까지 모델 안에서 다 끝납니다. 바깥에서 손댈 게 없습니다. 접합부는 [adapter.py](utils/model/promptmr/adapter.py) 하나이고, 계약은 `forward(kspace (B,C,H,W,2), mask (B,1,1,W,1)) -> (B,384,384)`입니다.

원본에서 이번 대회에 안 쓰는 갈래는 쳐냈습니다. 인접 슬라이스 SME와 비-Cartesian 분기를 빼고 knee·정적 2D·단일 슬라이스 전용으로 남겼는데, 원본과 bit 단위로 같은 값이 나오는 걸 확인하고 잘랐습니다.

설정은 cascade 10 / `n_feat0` 48 / `sens_n_feat0` 24입니다. momentum history는 원본 11에서 3으로 줄이고 gradient checkpointing을 켰습니다. 

### 2. 손실을 채점식에 맞춤

채점이 foreground mask 안에서만 SSIM을 재니 손실도 같은 영역에서 잽니다(`--fg-loss`). 여기에 √(foreground 면적비)를 곱해(`--area-weight`) 배경만 넓은 슬라이스가 점수를 부풀리지 못하게 했습니다. 구현은 [loss_function.py](utils/common/loss_function.py)의 `SSIMLoss`입니다.

### 3. Readout crop 640 → 384

readout 방향은 완전샘플이라 잘라도 정보가 안 없어집니다. 그래서 네트워크에 넣기 전에 높이만 384로 줄입니다([`readout_crop()`](utils/data/transforms.py), `--readout-crop`). 폭은 native 그대로 둡니다(acc4 368, acc8 372). VRAM이 40%쯤 빠지고 그만큼 빨라지는데, crop 전후 재구성 결과의 상관계수는 1.0입니다.

### 4. acc4 / acc8 MoE

가속률별로 전문가를 따로 학습시키고 추론 때 라우팅합니다([moe.py](utils/model/moe.py)). 라우터는 mask의 샘플링 비율을 보고 acc를 판정합니다(임계 0.25).

핵심은 슬라이스마다 전문가 하나만 돈다는 점입니다. 추론시간이 단일 모델과 같습니다. 

만드는 순서는 공용 pretrain → acc4 미세조정 → acc8 미세조정 → [merge_moe.py](scripts/merge_moe.py) 병합입니다. 

### 5. 마스크를 다시 씌워 학습 볼륨 2배

학습 h5에는 full k-space가 그대로 들어 있습니다. 대회 마스크 생성기를 재현해 같은 볼륨에 반대 가속률 마스크를 새로 씌웠습니다([`equispaced_mask()`](utils/data/transforms.py), `--remask`). acc4 볼륨을 acc8 학습에 쓸 수 있으니 전문가당 볼륨이 80개에서 160개가 됩니다. center_fraction 0.08로 만든 마스크가 원본과 bit 단위로 같은 걸 확인하고 넣었습니다. 로더를 만들 때 [`verify_remaskable()`](utils/data/load_data.py)이 학습 볼륨 전수로 두 전제를 다시 재고, 어긋나면 학습을 시작하지 않습니다.

### 6. 층화 K-fold

train/val 고정 분할 대신 볼륨 단위 층화 K-fold를 씁니다([`_stratified_folds()`](utils/data/load_data.py), `--k-fold 5 --val-fold N`). 볼륨 단위로 잘라야 같은 볼륨의 슬라이스가 train과 val에 섞이지 않습니다. `--init-from`으로 다른 run의 가중치를 가져올 때는 k-fold·fold·seed를 대조합니다. 다르면 hold-out이 새기 때문에 스크립트가 막습니다.

### 7. 증강은 좌우 flip 하나

phase-encode 방향 좌우 flip만 씁니다([`MRIAugmentor`](utils/data/transforms.py), `--aug-on`). k-space와 target, 병변 박스를 같이 뒤집고 샘플별 RNG를 결정적으로 뽑아 다시 돌려도 같게 나오도록 했습니다.

## 실행

```bash
pip3 install -r requirements.txt
```

Python 3.10.12 / torch 2.3.1+cu121에서 검증 완료.

학습:

```bash
bash final_run_train.sh
```

가중치는 `../result/<net_name>/checkpoints/`(repo 바깥의 `result` 폴더)에, epoch별 validation loss는 `../result/<net_name>/val_loss_log*.npy`에 쌓입니다. seed는 430 고정이라 같은 인자로 다시 돌리면 같은 결과가 나옵니다.

채점:

```bash
LEADERBOARD_DATA_DIR="" bash recon_eval.sh
```

GPU를 쓰는 작업은 [shell_scripting/](shell_scripting/)의 큐 슬롯으로 돌립니다. 슬롯 하나가 작업 하나이고 로그는 자동으로 남습니다. 규칙은 [shell_scripting/README.md](shell_scripting/README.md)에 있습니다.

## 제출물

- 코드 repo — 이 README
- loss 기록 — `../result/<net_name>/val_loss_log*.npy`, [training_anal/](training_anal/)
- 모델 weight — `../result/<net_name>/checkpoints/best_model.pt`
- 모델 설명 ppt
