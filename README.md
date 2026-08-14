# 2026 SNU FastMRI Challenge | Team GMG
#### JunKyung Seo
* Undergrad. @ Department of Electrical & Computer  Engineering | $\tiny\textsf{Mar 2022 - Current}$
   * Double major in Department of Industrial Engineering
* Email | esel343434@snu.ac.kr
#### HyunSoo Choi
* Undergrad. @ Department of Electrical & Computer  Engineering | $\tiny\textsf{Mar 2022 - Current}$
   * Double major in Department of Sociology
* Email | bedso@snu.ac.kr

## 4. 실행 가이드 및 재현성 확보

```bash
pip3 install -r requirements.txt
```
학습:
```bash
bash final_run_train.sh
```
`shared 25 → acc8 18+2 → acc4 6+2 → merge` 를 순서대로 실행합니다 (GTX 1080 8GB 기준 약 278시간).

- 채점 대상 가중치 : `../result/promptmrp_GMG_final_0807_moe/checkpoints/best_model.pt` (repo 바깥)
- 단계별 가중치 : `../result/promptmrp_GMG_final_0807_{shared,acc8,acc8_tail,acc4,acc4_tail}/checkpoints/`
- epoch별 validation loss : `../result/promptmrp_GMG_final_0807_*/val_loss_log*.npy`

채점:

```bash
LEADERBOARD_DATA_DIR="" bash recon_eval.sh
```

#### 경로를 바꿔야 할 때

산출물을 다른 디스크에 쌓으려면 인자로 경로를 줍니다. `train.py` 와 `recon_eval.py` 가 `../result` 를 고정으로 참조하므로, 스크립트가 `../result` 를 지정한 경로로 심볼릭 링크합니다.

```bash
bash final_run_train.sh /mnt/data/result
```

채점은 인자 없이 그대로입니다 — `../result` 가 이미 그 경로를 가리킵니다.

leaderboard 데이터가 `/root/Data/leaderboard`(또는 `/workspace/Data/leaderboard`) 가 아니라면 경로를 넘깁니다.

```bash
LEADERBOARD_DATA_DIR=/mnt/leaderboard bash recon_eval.sh
```
---
## [1. 대회 개요](https://github.com/LISTatSNU/FastMRI_challenge)  

본 대회는 [서울대학교 전기정보공학부](https://ece.snu.ac.kr/)의 [LIST](https://list.snu.ac.kr/) 연구실이 주관하고 [AIRS Medical](https://airsmed.com/)이 후원하는 대회로, NYU의 [FastMRI dataset](https://fastmri.med.nyu.edu/)과 Facebook AI 및 NeurIPS의 [FASTMRI Challenge](https://github.com/facebookresearch/fastMRI?tab=readme-ov-file)를 기반으로 합니다.

MRI(Magnetic Resonance Imaging)는 자기장을 이용해 인체 내부 구조를 비침습적으로 촬영하는 기술입니다. 자기장을 통해 얻을 수 있는 k-space data를 신속한 촬영을 위해 부분 샘플링하면 정보 손실로 인해 이미지 화질이 저하되는데(aliasing, Nyquist-Shannon sampling Theorem), 이를 Alias-Free Image로 복원하는 모델을 설계하는 것이 본 대회의 과제입니다. 

지난 대회와 달리 이번 대회에는 Annotation-Guided라는 내용이 추가되었습니다. 대회에서 사용하는 지표 SSIM은 구조 기반 지표이므로 MSE/PSNR 등 다른 지표보다 사람의 눈과 유사하게 점수를 매깁니다. 이때, 점수 계산 과정에서 noise는 SSIM을 낮추는데, LPF를 적용하면 noise가 없어지고 전체 이미지가 뭉개지는 대신 SSIM 점수는 상승합니다. 하지만 이를 모델이 학습하는 것은 세부 구조 및 병변 등 의학적 진단에 정말로 중요한 영역에 대한 정보를 없애는 것이므로 바람직하지 않습니다. 따라서 full image에 대한 SSIM_full과 함께, 오직 lesion이 존재하는 영역에 대해 계산한 SSIM_bbox도 점수에 포함하게 되었습니다. 

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
#### (2) 최종 제출 @ 23:59, Aug 19
SSIM_full = 0.9336 | SSIM_bbox = 0.9322 | Inference Time = 737.5 ms  
#### (3) 최종 순위 #위  

점수 사본 : [shell_scripting/results/](shell_scripting/results/)  
실행 로그 : [shell_scripting/logs/](shell_scripting/logs/)

## 3. 모델 구조 및 학습 전략

### 0. 디렉터리 구조

```
fastMRI_SNU_2026_GMG/
├── train.py                        학습 진입점 (인자 정의)
├── final_run_train.sh              최종 제출 모델의 학습 실행
├── recon_eval.py                   재구성 + 채점 (SSIM_full / SSIM_bbox / 시간)
├── recon_eval.sh                   채점 실행
├── requirements.txt
├── README.md
│
├── utils/
│   ├── common/
│   │   ├── loss_function.py        SSIM + L1 혼합 손실, foreground/area-weight
│   │   ├── metrics.py              채점 지표
│   │   └── utils.py                seed 고정, 재구성 h5 저장
│   ├── data/
│   │   ├── load_data.py            데이터셋·로더, K-Fold 분할, 가속률 필터
│   │   └── transforms.py           마스크 재생성·offset 재추첨, flip/affine 증강
│   ├── learning/
│   │   ├── train_part.py           학습 루프, EMA, 스케줄러, 체크포인트 저장
│   │   └── test_part.py            추론 — prep_volume / recon_slice, MoE 라우팅
│   └── model/
│       ├── promptmr/               백본 PromptMR+
│       │   ├── promptmr_plus.py    cascade 본체
│       │   ├── adapter.py          k-space ↔ 모델 입출력, readout crop
│       │   ├── config.py           모델 하이퍼파라미터
│       │   └── reentrant_wrapper.py  gradient checkpointing 래퍼
│       ├── fastmri/                원본 fastMRI 유틸 (FFT, coil combine, subsample …)
│       ├── moe.py                  가속률 판정 router
│       ├── unet.py
│       └── varnet.py               baseline (미사용, 참고용)
│
├── scripts/
│   └── merge_moe.py                acc4·acc8 expert → MoE 결합 체크포인트
│
└── shell_scripting/                학습·채점 실행 스크립트 모음
```

학습·채점 산출물은 `<net_name>` 폴더 아래에 저장됩니다. 최종 제출 모델의 `<net_name>` 은 `promptmrp_GMG_final_0807_moe` 이며, acc4·acc8 expert 를 결합한 단일 체크포인트입니다. 추론 시 router 가 마스크의 sampling rate 로 가속률을 판정해 해당 expert 로 보냅니다.

### 1. PromptMR+
- [Directory] `utils/model/promptmr`  

#### 1.1. Model Structure
PromptMR-plus 원본에서 이번 대회에 필요 없거나, 성능을 낮추는 요소들은 제외했습니다. 인접 슬라이스 SME을 제외했고, SDUM의 SWDC(Sampling-aware Weighted Data Consistency)를 추가했습니다. 

#### 1.2. Model Parameter  
 다른 값들은 원 논문에 기재된 값과 동일하게 사용했고, cascade 12 / features 48 / sensitivity map estimation features 24 / history features 11로  설정했습니다. 

 #### 1.3. GPU VRAM 절약
 #### 1.3.1. Read-out Crop
 kspace data의 readout(높이)방향은 crop해도 정보 손실이 없고, 제출물에서 이미지 사이즈를 384*384로 crop하므로, 이를 미리 진행하여 입력 데이터 크기를 줄였습니다. kspace data를 Inverse Fourier Transform을 통해 Image domain으로 바꾸고 crop한 뒤 다시 Fourier Transform을 진행하여 Frequency Domain으로 돌렸습니다. 
 #### 1.3.2. Gradient Checkpointing
 cascade를 지날 때 값을 VRAM에 저장하지 않고, 필요할 때 다시 계산하는 방식입니다. 학습 시간은 증가하지만, VRAM 사용량을 획기적으로 낮출 수 있습니다. 
 #### 1.3.3. Coil별 sensitivity map estimation & checkpointing
 Coil Sensitivity map estimation을 위해 15개의 coil을 한 번에 Unet에 넣지 않고, 하나씩 나누어 넣었습니다. 또한 이때에도 checkpointing을 적용하였습니다. 
 #### 1.3.4. Gradient accumulation
 VRAM 제한 때문에 batch를 높일 수 없으므로, gradient accumualation = 2를 설정하여 effective batch size = 2가 되도록 하였습니다. 

### 2. Loss Function
`Loss = 0.9·(1−SSIM) + 0.1·L1`  
#### 2.1. SSIMLoss
배경을 제외한 foreground mask 안에서만 SSIM을 측정했습니다. 여기에 √(foreground 면적비)를 곱하여(area-weight) 배경보다 foreground mask가 큰 슬라이스가 점수에 더 큰 영향을 주도록 했습니다.
#### 2.2. L1 Loss
기존의 SSIMLoss에 L1 Loss를 추가했습니다. SSIM은 픽셀 단위 밝기 오차를 잘 반영하지 못하는데, 그 부분을 L1 loss를 이용하여 보완했습니다. 두 Loss의 크기를 맞추고 정규화하기 위해 L1을 data_range로 나눴습니다. 이때, Validation Loss는 수정하지 않고 그대로 둬서 채점식과 동일하게 유지했습니다.

### 3. acc4 / acc8 Mixture of Expert

가속률별로 expert를 따로 학습시키고 Inference 때 routing합니다. Router는 mask의 sampling rate를 보고 acc를 판정합니다.  
따라서 전체 학습 순서는 아래와 같습니다.  
`shared model -> acc8 expert -> acc4 expert -> 병합 및 채점`

### 4. Augmentation
#### 4.1. Remasking
#### 4.1.1. Equispace Remasking
학습 데이터는 full sampled k-space에서 undersampling mask를 적용시켜 acc4와 acc8 under-sampled k-space data를 생성하는 방식으로 제공됩니다. 즉 제공된 mask와 다른 mask를 적용시키면 acc4 train data에서 acc8 train data를, acc8 train data에서 acc4 train data를 생성할 수 있습니다. shared와 각 expert에서 적용했습니다. 
#### 4.1.2. Random offset Remasking
격자 위상(offset)을 샘플마다 새로 뽑습니다. train data와 validation data는 마스크의 offset이 고정되어있지만, public leaderboard data는  offset이 random인 것을 발견했습니다. 따라서 이 부분을 augmentation을 통해 반영하고자 했습니다.  
augmentation을 적용하면 expert 학습 시 데이터 수가 2배가 되고, 시간도 2배가 됩니다. 이 경우 학습 시간이 지나치게 길어지므로, 이를 막기 위해 epoch의 길이는 그대로 뒀습니다. 로더가 매 epoch 섞이니 epoch마다 다른 데이터를 보게 되고, 같은 시간 예산 안에서 데이터 다양성을 2배가 되도록 했습니다. 

#### 4.2. 좌우 flip + affine
**flip**은 phase-encode 방향 좌우 flip을 적용했으며, 각 슬라이스에서 0.5 확률로 일어나도록 설정했습니다. 

**affine**은 5°이내 회전과 1배~1.1배 확대입니다. 
affine은 다른 augmentation과 달리 세부 구조 손실이 존재할 수 있으므로, expert 학습에서 처음 켰습니다. acc8은 5→10 epoch cosine warm-up, acc4는 3→5 epoch cosine warm-up을 걸쳐 적용 확률을 0부터 0.5까지 늘렸습니다.

### 5. 학습 스케줄 — Shared 30 + Expert 20/10

최종 run은 `promptmrp_GMG_final_0807`입니다. 

| 단계 | epoch | LR |
|---|---|---|
| shared | 30 | 2e-4 유지 5 → cosine 25 |
| acc8 expert | 18 + tail 2 | warm-up 3 → 유지 5 → cosine 12 |
| acc4 expert | 8 + tail 2 | warm-up 2 → 유지 2 → cosine 6 |

**최종 제출 모델에선 시간이 부족하여, shared 학습을 25epoch까지만 진행했습니다. 즉, cosine dacay를 중간에 끊었습니다.**  

**Optimizer.** AdamW(weight_decay 1e-3)  
**Learning Rate.** base 2e-4 / 하한 1.8e-5  
**Gradient clipping.** 0.1  
**tail.** 각 expert의 마지막 두 epoch은 validation data까지 포함하여 학습을 진행합니다. 최종 제출본인 해당 run에서만 이를 이용했습니다. 

## 5. Reference
### 5.1. Past Winning works
#### 5.1.1. 2025 SNU FastMRI challenge | sMaRtlfy
- https://youtu.be/-Tau6N7PNHs?si=9MQpTjnVBKG9HVAj
- https://github.com/sMartify-fastmri/2025_final
#### 5.1.2. 2025 SNU FastMRI challenge | 2rm1000
- https://www.youtube.com/watch?v=m5hwaRKvDUw&t=8s
- https://github.com/gates02/fastmri-2-1000
#### 5.1.3. 2025 SNU FastMRI challenge | SNUnet
- https://www.youtube.com/watch?v=Sufe7FQ0F6E
### 5.2. PromptMR-plus
- https://github.com/hellopipu/PromptMR-plus  
- [[Paper]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09565.pdf) *Rethinking Deep Unrolled Model for Accelerated
MRI Reconstruction*  
### 5.3. SDUM
- SDUM 논문의 SWDC 내용을 기존 PromptMR-plus 모델에 추가했습니다. 
- https://github.com/NVIDIA-Medtech/NV-Raw2insights-MRI    
- [[Paper]](https://arxiv.org/abs/2512.17137) *SDUM: A Scalable Deep
Unrolled Model for Universal MRI Reconstruction*


### 코드 작성 과정에서 생성형 AI의 도움을 받았음을 밝힙니다. 