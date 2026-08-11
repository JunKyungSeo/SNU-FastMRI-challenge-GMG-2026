# 2026 SNU FastMRI Challenge | Team GMG
#### JunKyung Seo
* Undergrad. @ Department of Electrical & Computer  Engineering | $\tiny\textsf{Mar 2022 - Current}$
   * Double major in Department of Industrial Engineering
* Email | esel343434@snu.ac.kr
#### HyunSoo Choi
* Undergrad. @ Department of Electrical & Computer  Engineering | $\tiny\textsf{Mar 2022 - Current}$
   * Double major in Department of Sociology
* Email | bedso@snu.ac.kr

## [1. 대회 개요](https://github.com/LISTatSNU/FastMRI_challenge)  

본 대회는 [서울대학교 전기정보공학부](https://ece.snu.ac.kr/)의 [LIST](https://list.snu.ac.kr/) 연구실이 주관하고 [AIRS Medical](https://airsmed.com/)이 후원하는 대회로, NYU의 [FastMRI dataset](https://fastmri.med.nyu.edu/)과 Facebook AI 및 NeurIPS의 [FASTMRI Challenge](https://github.com/facebookresearch/fastMRI?tab=readme-ov-file)를 기반으로 합니다.

MRI(Magnetic Resonance Imaging)는 자기장을 이용해 인체 내부 구조를 비침습적으로 촬영하는 기술입니다. 자기장을 통해 얻을 수 있는 k-space data를 신속한 촬영을 위해 부분 샘플링하면 정보 손실로 인해 이미지 화질이 저하되는데(aliasing, Nyquist-Shannon sampling Theorem), 이를 Alias-Free Image로 복원하는 모델을 설계하는 것이 본 대회의 과제입니다. 

지난 대회와 달리, 본 6회 대회는 Annotation-Guided라는 조건이 추가되었습니다. SSIM은 구조 기반 점수라 MSE/PSNR 등 다른 지표보다 사람의 눈과 유사하게 점수를 매깁니다. 점수 계산 과정에서 noise는 SSIM을 낮추는데, LPF를 적용하면 noise가 없어지고 전체 이미지가 뭉개지는 대신 SSIM 점수는 상승합니다. 하지만 이는 세부 구조 및 병변 등 의학적 진단에 정말로 중요한 영역에 대한 정보를 없애는 것이므로 바람직하지 않습니다. 따라서 full image에 대한 SSIM_full과 함께, 오직 lesion이 존재하는 영역에 대해 계산한 SSIM_bbox도 점수에 포함하게 되었습니다. 

`
Score : SSIM_full + SSIM_bbox + InferenceTime
`

#### [2026 제 6회 SNU FastMRI Challenge 상세](https://fastmri.snu.ac.kr/)
* 학부생 1-2인 1팀
* 대회 기간: 2026년 7월 1일 - 8월 20일
* 참여 인원: 271명 (186팀, 등록 마감 전날 기준)
* 사용 환경: NVIDIA GTX 1080, 8GB VRAM 제한 ([VESSL AI](https://vessl.ai/ko) 제공)
* Dataset
    * [FastMRI+ dataset](https://www.nature.com/articles/s41597-022-01255-z)의 재가공 데이터 (from NYU FastMRI dataset)
    * 무릎 MRI 영상, multi-coiled k-space & image data with annotation
    * 가속: X4, X8
    * 학습용 데이터 크기
        * 부위-가속 쌍 각각에 대해 85개의 train data | 15개의 validation data
        * 각 파일 당 18 - 40 slices | annotation 18종
* 사전 학습된 모델 가중치 또는 외부 데이터 사용 불가

---

## 2. Team GMG - 성과

#### (1) 중간평가 12위 @ 12:00, July 29
SSIM_full = 0.9321 | SSIM_bbox = 0.9300 | Inference Time = 732.8 ms
#### (2) 공개 리더보드 #위 @ 00:00, Aug 14
SSIM_full = 0.9336 | SSIM_bbox = 0.9322 | Inference Time = 737.5 ms  
#### (3) 최종 제출 @ 23:59, Aug 19
SSIM_full = 0.9336 | SSIM_bbox = 0.9322 | Inference Time = 737.5 ms  
#### (4) 최종 순위 #위  

점수 사본 : [shell_scripting/results/](shell_scripting/results/)  
실행 로그 : [shell_scripting/logs/](shell_scripting/logs/)

## 3. 모델 구조 및 학습 전략

### 1. PromptMR+ & SDUM
[Github] [PromptMR-plus](https://github.com/hellopipu/PromptMR-plus)  
[Paper] PromptMR-plus: [*Rethinking Deep Unrolled Model for Accelerated
MRI Reconstruction*](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09565.pdf)  
[Github] [SDUM](https://github.com/NVIDIA-Medtech/NV-Raw2insights-MRI)    
[Paper] SDUM: [*SDUM: A Scalable Deep
Unrolled Model for Universal MRI Reconstruction*](https://arxiv.org/abs/2512.17137)  

#### 1.1. Model Structure
PromptMR-plus 원본에서 이번 대회에 필요 없거나, 성능을 낮추는 요소들은 제외했습니다. 인접 슬라이스 SME와 비-Cartesian 분기를 빼고 knee·정적 2D·단일 슬라이스 전용으로 남겼고, SDUM의 SWDC(Sampling-aware Weighted Data Consistency)를 추가했습니다. 

#### 1.2. Model Parameter  
 다른 값들은 원 논문에 기재된 값과 동일하게 사용했고, cascade 12 / `n_feat0` 48 / `sens_n_feat0` 24 / `n_history` 11로 설정했습니다. 8GB VRAM에 모델을 올리기 위해 gradient checkpointing을 켰고, batch = 1로 설정했으며, gradient accumulation = 2로 설정하여 effective batch size = 2로 설정했습니다. 

 #### 1.3. GPU VRAM 절약
 #### 1.3.1. Read-out Crop
 kspace data의 readout(높이)방향은 crop해도 정보 손실이 없고, 제출물에서 이미지 사이즈를 384*384로 crop하므로, 이를 미리 진행하여 입력 데이터 크기를 줄였습니다. kspace data를 Inverse Fourier Transform을 적용하여 Image domain으로 바꾸고 crop한 뒤 다시 Fourier Transform을 진행하여 Frequency Domain으로 돌렸습니다. 
 #### 1.3.2. Gradient Checkpointing
 cascade를 지날 때 값을 VRAM에 저장하지 않고, 필요할 때 다시 계산하는 방식입니다. 학습 시간은 증가하지만, VRAM 사용량을 획기적으로 낮출 수 있습니다. 
 #### 1.3.3. Coil별 sensitivity map estimation & checkpointing
 Coil Sensitivity map estimation을 위해 15개의 coil을 한 번에 Unet에 넣지 않고, 하나씩 나누어 넣었습니다. 또한 이때에도 checkpointing을 적용하였습니다. 

### 2. Loss Function
`Loss = 0.9·(1−SSIM) + 0.1·L1`  
#### 2.1. SSIMLoss
배경을 제외한 foreground mask 안에서만 SSIM을 측정하도록 하였습니다. 여기에 √(foreground 면적비)를 곱하여 (area-weight) 배경만 넓은 슬라이스가 점수를 부풀리지 않도록, 그리고 foreground mask가 큰 슬라이스가 점수에 더 큰 영향을 주도록 했습니다.
#### 2.2. L1 Loss
기존의 SSIMLoss에 L1을 10% 섞었습니다. SSIM은 픽셀 단위 밝기 오차를 잘 반영하지 못하는데, 그 부분을 L1 loss를 이용하여 보완합니다. L1은 `data_range`로 나눠 SSIM과 자릿수를 맞췄습니다. 안 맞추면 픽셀 절대값 스케일이라 섞는 즉시 L1이 SSIM을 압도합니다. 검증 손실은 순수 SSIM 그대로 둬서 채점식과 어긋나지 않게 했습니다.

### 3. acc4 / acc8 MoE

가속률별로 전문가를 따로 학습시키고 추론 때 라우팅합니다. 라우터는 mask의 sampling rate를 보고 acc를 판정합니다.  
학습 순서는 아래와 같습니다.  
`shared model -> acc8 expert 미세조정 -> acc4 expert 미세조정 -> 병합 및 채점`

### 4. Augmentation
#### 4.1. Remasking
#### 4.1.1. Equispace Remasking
학습 데이터는 full k-space에서 mask를 적용시켜, acc4와 acc8 under-sampled k-space data를 생성하는 방식입니다. 즉 제공된 mask와 다른 mask를 적용시키면 acc4 train data에서 acc8 train data를, acc8 train data에서 acc4 train data를 생성할 수 있습니다. shared에선 이 augmentation을 적용하지 않았고, expert 학습을 시작할 때 적용했습니다.
#### 4.1.2. Random offset Remasking
격자 위상(offset)을 샘플마다 새로 뽑습니다. train data와 validation data는 마스크의 offset이 고정되어있지만, public leaderboard data를 관찰해보니 offset이 random인 것을 발견했습니다.  공용 단계는 여기에 더해 가속률까지 4|8로 재추첨하고(`--mask-aug full`), 전문가 단계는 가속률을 자기 것으로 고정합니다(`--remask 4|8`).

전문가 단계는 볼륨이 2배가 되지만 epoch 길이는 그대로 둡니다(`--max-iters 2179` = 4358의 절반). 로더가 매 epoch 섞이니 epoch마다 다른 절반을 보게 되고, 같은 시간 예산 안에서 데이터 다양성만 2배가 됩니다.

#### 4.2. 좌우 flip + affine
phase-encode 방향 좌우 flip과 affine(회전·확대) 두 가지를 씁니다([`MRIAugmentor`](utils/data/transforms.py), `--aug-on`). k-space·target·병변 박스를 같이 변환하고 샘플별 RNG를 `(seed, epoch, 파일명, slice)`에서 결정적으로 뽑아, 다시 돌리면 같은 증강이 나옵니다.

**flip**은 전 단계 확률 0.5 고정입니다. 이미지 도메인 왕복 대신 DFT 성질로 닫아 `K' = roll(flip(K),1)·ramp` 한 줄로 처리합니다. 픽셀 정확하고 옛 구현보다 15배 빠릅니다(470ms → 31ms).

**affine**은 회전 U(−5°,+5°)와 확대 U(1.0,1.1)입니다. label은 384² 고정 크롭이라 돌리면 프레임 밖 정보가 없으므로, 재사용하지 않고 코일별 IFFT → 워프 → FFT로 "변형된 환자의 획득"을 만들고 정답도 같은 워프 영상의 RSS에서 새로 뽑습니다. 보간은 LANCZOS4입니다.

affine은 공용 단계가 끝난 뒤에 들어온 레버라 전문가 단계에서 처음 켜집니다. 0.5로 바로 올리면 계단이 생기므로 코사인으로 태워 올립니다(`--affine-delay`/`--affine-peak-epoch`, 양 끝 기울기 0). acc8은 5→10 epoch, acc4는 3→5 epoch에 걸쳐 0에서 0.5까지 갑니다. flip은 이 곡선과 무관하게 계속 0.5입니다. 같이 0부터 올리면 없애려던 계단보다 큰 계단(0.5→0→0.5)이 새로 생깁니다.

### 5. 학습 스케줄 — 공용 30 + 전문가 20/10

최종 run은 `promptmrp_GMG_final_0807`입니다. 공통 인자는 [_final_base.sh](shell_scripting/_final_base.sh) 한 곳에 모아 두고 단계별 슬롯이 델타만 얹습니다. 세 단계가 같은 값을 써야 `--init-from`이 성립하기 때문입니다.

| 단계 | epoch | LR |
|---|---|---|
| 공용 (acc 무관) | 30 | 2e-4 유지 5 → cosine 25 |
| acc8 전문가 | 18 + tail 2 | warm-up 3 → 유지 5 → cosine 12 |
| acc4 전문가 | 8 + tail 2 | warm-up 2 → 유지 2 → cosine 6 |

optimizer는 AdamW(weight_decay 1e-3), base LR 2e-4, 하한 1.8e-5(base×0.09)입니다. `--grad-clip 0.1`과 `--accum-steps 2`(batch 1 × 2)를 씁니다. 스케줄러는 linear warm-up → base 유지 → cosine 감쇠 세 구간입니다([`build_scheduler()`](utils/learning/train_part.py)).

acc8에 예산을 2배 준 건 남은 헤드룸이 그쪽에 있어서입니다. 직전 run의 리더보드 점수를 가속률로 쪼개 보면 acc4는 SSIM_full 0.9505 / SSIM_bbox 0.9516인데 acc8은 0.9167 / 0.9126입니다. 올릴 여지가 있는 쪽이 명확합니다.

**EMA.** 학습은 원본 가중치로 하되 지수이동평균(decay 0.999)을 따로 굴리고, 검증과 `best_model.pt`에는 평균 쪽을 남깁니다. 제출 가중치는 전부 이 EMA 가중치입니다.

**tail.** 각 전문가의 마지막 2 epoch은 hold-out fold까지 학습에 합쳐 돕니다(`--holdout-tail`). epoch당 슬라이스 수는 본 런과 같게 두고 그 안에 hold-out을 전량 넣습니다. LR이 이미 하한 근처인 구간이라 여기서 검증을 포기하고 데이터를 마저 쓰는 쪽이 낫다고 봤습니다. 대신 val이 학습에 들어간 뒤라 `best_model.pt`·early-stop 판정은 전부 의미를 잃습니다. 그래서 tail 슬롯은 마지막 epoch의 EMA 가중치를 `best_model.pt` 자리로 직접 옮깁니다. 채점부가 `best_model.pt`만 읽기 때문입니다.

### 6. 층화 K-fold

train/val 고정 분할 대신 볼륨 단위 층화 K-fold를 씁니다(`--k-fold 5 --val-fold 2`). 

---

## 4. 실행 가이드

```bash
pip3 install -r requirements.txt
```
학습:
```bash
bash final_run_train.sh
```
가중치는 `../result/<net_name>/checkpoints/`(repo 바깥의 `result` 폴더)에, epoch별 validation loss는 `../result/<net_name>/val_loss_log*.npy`에 쌓입니다. seed는 430 고정이라 같은 인자로 다시 돌리면 같은 결과가 나옵니다.  
채점:
```bash
LEADERBOARD_DATA_DIR="" bash recon_eval.sh
```

---
### 코드 작성 과정에서 생성형 AI의 도움을 받았음을 밝힙니다. 