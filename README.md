# FastMRI_SNU_2026_GMG

2026 SNU FastMRI Challenge(knee annotation track) 참가 코드입니다. 베이스라인의 E2E-VarNet 대신 PromptMR+를 백본으로 세우고, 채점식에 맞춰 손실과 전처리, 추론 경로를 다시 짰습니다.

## 대회 조건

knee 데이터만 쓰고, 가속률은 acc4와 acc8, 출력은 384×384입니다.

점수 = `0.5·SSIM_full + 0.5·SSIM_bbox + 추론시간`

`SSIM_full`은 foreground mask 안에서 잰 평균 SSIM이고(data_range는 `attrs['max']`), `SSIM_bbox`는 fastMRI+ 병변 bounding box 안에서만 잰 SSIM입니다. 둘은 5:5라 한쪽으로 치우치면 손해입니다. 추론시간은 0.00n 단위로 작지만 SSIM이 비슷하게 붙으면 여기서 순위가 갈립니다. 모델을 키우는 게 곧 이득은 아니라는 뜻입니다.

병변 박스는 각 image h5의 `attrs['annotations']`에 JSON으로 들어 있습니다. `{"<slice>": [{x, y, width, height, label}, ...]}` 형식이고, 384² 좌표계 기준으로 width·height가 16을 넘는 박스만 담겨 있습니다.

사전학습 가중치와 외부 데이터는 금지입니다. 여기 있는 가중치는 전부 처음부터 학습했습니다. 채점은 `recon_eval.py`와 `utils/common/metrics.py`가 하는데, 대회가 고정한 파일이라 건드리지 않습니다.

## 결과

리더보드 데이터(`/root/Data/leaderboard`) 자가 채점입니다.

| 모델 | SSIM_full | SSIM_bbox | ms/slice |
|---|---|---|---|
| zero-filled (재구성 없음) | 0.833 | 0.820 | — |
| PromptMR+ 단일 | 0.9309 | 0.9284 | 733.6 |
| PromptMR+ MoE | 0.9321 | 0.9300 | 734.3 |

아래 두 행은 같은 pretrain에서 갈라져 나온 쌍이라 그대로 비교됩니다. MoE는 0.7ms를 더 쓰고 full +0.0012, bbox +0.0016을 가져갔습니다.

MoE를 가속률로 쪼개면 acc4가 0.9497 / 0.9505, acc8이 0.9145 / 0.9094입니다. acc8이 0.035쯤 낮고, 여기가 지금 가장 큰 약점입니다.

따로 돌린 K-fold f2 + flip 증강 run은 0.9303 / 0.9279 / 716.9ms입니다. 점수는 조금 낮은데 17ms 빠릅니다.

점수 사본은 [shell_scripting/results/](shell_scripting/results/), 실행 로그는 [shell_scripting/logs/](shell_scripting/logs/)에 있습니다.

## 베이스라인에서 바꾼 것

### 1. 백본을 PromptMR+로

[utils/model/promptmr/](utils/model/promptmr/)입니다. undersampled k-space와 mask를 넣으면 감도맵 추정부터 data consistency, 정규화, RSS, 384 crop까지 모델 안에서 다 끝납니다. 바깥에서 손댈 게 없습니다. 접합부는 [adapter.py](utils/model/promptmr/adapter.py) 하나이고, 계약은 `forward(kspace (B,C,H,W,2), mask (B,1,1,W,1)) -> (B,384,384)`입니다.

원본에서 이번 대회에 안 쓰는 갈래는 쳐냈습니다. 인접 슬라이스 SME와 비-Cartesian 분기를 빼고 knee·정적 2D·단일 슬라이스 전용으로 남겼는데, 원본과 bit 단위로 같은 값이 나오는 걸 확인하고 잘랐습니다.

설정은 cascade 10 / `n_feat0` 48 / `sens_n_feat0` 24입니다. momentum history는 원본 11에서 3으로 줄이고 gradient checkpointing을 켰습니다. GTX 1080 8GB에 밀어넣으려면 이 둘이 필요합니다. Pascal이라 FP32, batch 1로 돕니다. VarNet(`--model varnet`)과 SDUM 코드는 레거시로 남겨만 뒀습니다.

### 2. 손실을 채점식에 맞춤

채점이 foreground mask 안에서만 SSIM을 재니 손실도 같은 영역에서 잽니다(`--fg-loss`). 여기에 √(foreground 면적비)를 곱해(`--area-weight`) 배경만 넓은 슬라이스가 점수를 부풀리지 못하게 했습니다. 구현은 [loss_function.py](utils/common/loss_function.py)의 `SSIMLoss`입니다.

SDUM 백본 시절 이 플래그 하나로 full 0.9118 → 0.9161, bbox 0.9021 → 0.9095가 됐습니다. 시간은 그대로였고요. 반대로 bbox에 가중치를 더 주는 손실(`--bbox-loss-lambda`)은 구현해 뒀지만 기본 0입니다. 켜봤더니 노렸던 bbox까지 같이 떨어졌습니다.

### 3. Readout crop 640 → 384

readout 방향은 완전샘플이라 잘라도 정보가 안 없어집니다. 그래서 네트워크에 넣기 전에 높이만 384로 줄입니다([`readout_crop()`](utils/data/transforms.py), `--readout-crop`). 폭은 native 그대로 둡니다(acc4 368, acc8 372). VRAM이 40%쯤 빠지고 그만큼 빨라지는데, crop 전후 재구성 결과의 상관계수는 1.0입니다.

### 4. acc4 / acc8 MoE

가속률별로 전문가를 따로 학습시키고 추론 때 라우팅합니다([moe.py](utils/model/moe.py)). 라우터는 mask의 샘플링 비율을 보고 acc를 판정합니다(임계 0.25).

핵심은 슬라이스마다 전문가 하나만 돈다는 점입니다. 추론시간이 단일 모델과 같습니다. 점수에 시간이 들어가서 앙상블을 못 쓰는 대회라 이건 거의 공짜입니다.

만드는 순서는 공용 pretrain → acc4 미세조정 → acc8 미세조정 → [merge_moe.py](scripts/merge_moe.py) 병합입니다. [shell26.sh](shell_scripting/shell26.sh)가 이 5단계를 한 번에 돕니다.

### 5. 마스크를 다시 씌워 학습 볼륨 2배

학습 h5에는 full k-space가 그대로 들어 있습니다. 대회 마스크 생성기를 재현해 같은 볼륨에 반대 가속률 마스크를 새로 씌웠습니다([`equispaced_mask()`](utils/data/transforms.py), `--remask`). acc4 볼륨을 acc8 학습에 쓸 수 있으니 전문가당 볼륨이 80개에서 160개가 됩니다. center_fraction 0.08로 만든 마스크가 원본과 bit 단위로 같은 걸 확인하고 넣었습니다.

### 6. 층화 K-fold

train/val 고정 분할 대신 볼륨 단위 층화 K-fold를 씁니다([`_stratified_folds()`](utils/data/load_data.py), `--k-fold 5 --val-fold N`). 볼륨 단위로 잘라야 같은 볼륨의 슬라이스가 train과 val에 섞이지 않습니다. `--init-from`으로 다른 run의 가중치를 가져올 때는 k-fold·fold·seed를 대조합니다. 다르면 hold-out이 새기 때문에 스크립트가 막습니다.

### 7. 증강은 좌우 flip 하나

phase-encode 방향 좌우 flip만 씁니다([`MRIAugmentor`](utils/data/transforms.py), `--aug-on`). k-space와 target, 병변 박스를 같이 뒤집고 샘플별 RNG를 결정적으로 뽑아 다시 돌려도 같게 나오도록 했습니다.

한때 마스크 thinning과 k-space 위상·노이즈도 넣었다가 뺐습니다. 대회 입력이 acc4/acc8로 고정이라 마스크를 흔들면 MoE 라우팅과 부딪힙니다. 위아래 flip이나 회전, elastic, MixUp 계열은 아예 안 씁니다. 병변 모양을 바꾸는 변형은 SSIM_bbox를 직접 깎으니까요. 자세한 이유는 [strategy/rejected.md](strategy/rejected.md)에 적어 뒀습니다.

## 실행

```bash
pip3 install -r requirements.txt
```

Python 3.10.12 / torch 2.3.1+cu121에서 검증했습니다.

학습:

```bash
python train.py --model promptmr --readout-crop \
  --num-cascades 10 --n-feat0 48 --sens-n-feat0 24 --n-history 3 \
  -b 1 -l 2e-4 -e 10 --use-checkpoint \
  --fg-loss --area-weight --grad-clip 0.1 --accum-steps 2 \
  --k-fold 5 --val-fold 2 --acc all \
  --aug-on --aug-schedule constant --aug-max-prob 0.5 \
  -n <net_name> -t <train_dir> -v <val_dir>
```

가중치는 `../result/<net_name>/checkpoints/`(repo 바깥의 `result` 폴더)에, epoch별 validation loss는 `../result/<net_name>/val_loss_log*.npy`에 쌓입니다. seed는 430 고정이라 같은 인자로 다시 돌리면 같은 결과가 나옵니다.

채점:

```bash
python recon_eval.py -n <net_name> -p /root/Data/leaderboard
```

SSIM_full, SSIM_bbox, 추론시간을 한 번에 뽑고 acc4/acc8로도 쪼개서 보여줍니다.

GPU를 쓰는 작업은 [shell_scripting/](shell_scripting/)의 큐 슬롯으로 돌립니다. 슬롯 하나가 작업 하나이고 로그는 자동으로 남습니다. 규칙은 [shell_scripting/README.md](shell_scripting/README.md)에 있습니다.

## 제출물

- 코드 repo — 이 README
- loss 기록 — `../result/<net_name>/val_loss_log*.npy`, [training_anal/](training_anal/)
- 모델 weight — `../result/<net_name>/checkpoints/best_model.pt`
- 모델 설명 ppt

## 더 읽을 것

- [strategy/README.md](strategy/README.md) — 전략 문서 인덱스. 백본 SPEC은 [promptmr_plus.md](strategy/promptmr_plus.md)
- [training_anal/](training_anal/) — run별 학습 분석과 누적 이력표
