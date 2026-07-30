import torch
import argparse
import shutil
import os, sys
from pathlib import Path

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')
from utils.learning.train_part import train

if os.getcwd() + '/utils/common/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/common/')
from utils.common.utils import seed_fix


def parse():
    parser = argparse.ArgumentParser(description='Train Varnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-r', '--report-interval', type=int, default=500, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='test_varnet', help='Name of network')
    parser.add_argument('-t', '--data-path-train', type=Path, default='/Data/train/', help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='/Data/val/', help='Directory of validation data')
    
    parser.add_argument('--cascade', type=int, default=1, help='Number of cascades | Should be less than 12') ## important hyperparameter
    parser.add_argument('--chans', type=int, default=9, help='Number of channels for cascade U-Net | 18 in original varnet') ## important hyperparameter
    parser.add_argument('--sens_chans', type=int, default=4, help='Number of channels for sensitivity map U-Net | 8 in original varnet') ## important hyperparameter
    parser.add_argument('--model', type=str, default='varnet', choices=['varnet', 'promptmr'], help='Backbone: varnet or promptmr (SDUM은 코드만 보존·배선 제거)')
    parser.add_argument('--num-cascades', type=int, default=10, help='PromptMR+ cascade 수 T (기본 10)')
    parser.add_argument('--n-feat0', type=int, default=48, help='PromptMR+ recon U-Net base 폭 (feature)')
    parser.add_argument('--sens-n-feat0', type=int, default=24, help='PromptMR+ 감도맵 U-Net base 폭 (sens feature)')
    parser.add_argument('--n-history', type=int, default=3, help='PromptMR+ momentum history 길이 (원본 11, 메모리 레버)')
    parser.add_argument('--use-checkpoint', action=argparse.BooleanOptionalAction, default=True, help='PromptMR+ gradient checkpointing (8GB VRAM용, 기본 on). 끄려면 --no-use-checkpoint')
    parser.add_argument('--input-key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader worker processes (0=main proc)')
    parser.add_argument('--max-iters', type=int, default=0, help='Cap train iters/epoch for debugging (0=full)')
    parser.add_argument('--readout-crop', action='store_true', help='SDUM: k-space readout(높이) 640→384 crop을 입력단에 적용(무손실, ~40%% 공간연산·메모리 절감). --normalize와 함께 켜지 말 것(어댑터가 self-normalize)')
    parser.add_argument('--crop-h', type=int, default=384, help='--readout-crop 시 crop 높이 (기본 384 = 대회 출력 크기)')
    parser.add_argument('--coil-compress', action='store_true', help='SVD coil compression: 물리코일(15)→가상코일(--n-virtual-coils)로 축소(속도/VRAM). ACS 기반이라 추론 안전. 학습·추론 동일 적용(config가 ckpt에 저장돼 recon_slice가 재적용).')
    parser.add_argument('--n-virtual-coils', type=int, default=8, help='--coil-compress 시 남길 가상코일 수 (기본 8). ACS 에너지 보존율은 scripts/coil_energy_report.py로 확인.')
    parser.add_argument('--fg-loss', action='store_true', help='B5: SSIM 손실·검증을 채점(metrics.foreground_mask) 정합 foreground 영역 안에서만 계산. 기본 off=전체 384². 소비부 train_part(train_epoch/validate) 확장됨.')
    parser.add_argument('--bbox-loss-lambda', type=float, default=0.0, help='C1: 병변 bbox 가중 손실. L = (1-SSIM_full) + λ·(1-SSIM_bbox). 기본 0.0=off(기존과 비트 동일). >0이면 train 로더가 bbox를 함께 반환(val/leaderboard는 자연 분포 유지 위해 미적용). 유효 박스 판정은 채점(metrics.ssim_bbox, win_size=7)과 동일.')
    parser.add_argument('--area-weight', action='store_true',
        help='PromptMR+ 손실: masked SSIM × sqrt(foreground 면적비). foreground 작은 슬라이스 손실을 낮춰 가중. '
             '--fg-loss와 함께 켜야 함(fg_mask 필요). 작년 팀 방식.')
    parser.add_argument('--grad-clip', type=float, default=0.0, help='gradient clipping max_norm (0=off). PromptMR+ 권장 0.1')
    parser.add_argument('--accum-steps', type=int, default=1, help='gradient accumulation 스텝 수(1=off, batch 유사 효과). PromptMR+ 권장 2')
    parser.add_argument('--warmup-iters', type=int, default=0, help='linear LR warmup iters 후 cosine 감쇠 (0=off → 상수 LR). promptmr에서만 활성')
    parser.add_argument('--run-tag', type=str, default='', help='val_loss_log 파일명 suffix (run별 분리 저장, 덮어쓰기 방지). 예: run4')
    parser.add_argument('--resume', action='store_true', help='<net_name>/checkpoints/의 model.pt(epoch 경계) 또는 latest.pt(epoch 중간, 500 iter마다 저장) 중 더 진행된 쪽에서 이어학습. 같은 config로만.')
    parser.add_argument('--k-fold', type=int, default=0, help='K-Fold Cross Validation 분할 수 (0=off, 기존 train/val 구분 모드)')
    parser.add_argument('--val-fold', type=int, default=0, help='Validation으로 사용할 fold 인덱스 (0 <= val-fold < k-fold)')
    parser.add_argument('--kfold-seed', type=int, default=430, help='K-Fold Volume Split용 Random Seed (재현성)')
    # MoE(acc4/acc8 분리) — 학습 땐 파일명 accN 토큰으로 acc를 골라 학습, 추론 땐 마스크로 라우팅(test_part).
    parser.add_argument('--acc', type=str, default='all', choices=['all', '4', '8'], help='학습 데이터를 가속률로 필터(파일명 accN 기준). all=전체. MoE expert 학습 시 4 또는 8.')
    parser.add_argument('--init-from', type=str, default='', help='Option C 미세조정: 지정한 체크포인트의 model 가중치만 로드해 초기화(optimizer/epoch은 새로 시작). --resume과 배타적(resume 우선).')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    
    # fix seed
    if args.seed is not None:
        seed_fix(args.seed)

    # C1: bbox 가중 손실을 켜면 train 로더가 bbox를 실어와야 한다(λ=0이면 off → 기존 6-튜플 그대로).
    args.return_bbox = args.bbox_loss_lambda > 0

    # --area-weight는 fg_mask가 필요 → --fg-loss 없이 주면 조용히 무시되므로 명시 경고.
    if getattr(args, "area_weight", False) and not getattr(args, "fg_loss", False):
        print("[warn] --area-weight는 --fg-loss와 함께 켜야 동작합니다(fg_mask 필요). 지금은 무시됩니다.")

    args.exp_dir = '../result' / args.net_name / 'checkpoints'
    args.val_dir = '../result' / args.net_name / 'reconstructions_val'
    args.main_dir = '../result' / args.net_name / __file__
    args.val_loss_dir = '../result' / args.net_name

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)

    train(args)
