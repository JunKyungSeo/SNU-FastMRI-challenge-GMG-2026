"""k-space <-> 영상 어댑터.

우리 파이프라인 계약  forward(masked_kspace (B,C,H,W,2), mask (B,1,1,W,1)) -> image (B,384,384)
를 유지하면서, NV SDUM cascaded 모델(영상 도메인, (B,T,C,H,W,2) 입력)을 호출한다.
"""
import torch
import torch.nn as nn
from monai.apps.reconstruction.complex_utils import complex_abs_t
from monai.apps.reconstruction.mri_utils import root_sum_of_squares_t
from monai.data.fft_utils import ifftn_centered

from utils.common.utils import center_crop
from utils.model.sdum.config import build_cascaded


def _extract_acc(mask_1d: torch.Tensor) -> int:
    # mask_1d: (W,) of {0,1}. frac<=0.25 -> acc8 else acc4 (strategy/candidates.md)
    frac = mask_1d.float().mean().item()
    return 8 if frac <= 0.25 else 4


class SDUMReconModel(nn.Module):
    """어댑터: (masked_kspace (B,C,H,W,2), mask (B,1,1,W,1)) -> image (B,384,384)."""

    def __init__(self, args, out_size=(384, 384)):
        super().__init__()
        self.cascaded = build_cascaded(args)
        self.out_size = out_size
        self.mask_type = str(args.train_mask_types[0])  # "uniform"
        self.acq_type = str(args.acq_types[0])          # "knee"
        # UC(Universal Conditioning) on 여부. on이면 forward에서 timestep을 넘겨 cascade-index 조건화.
        # ⚠️ acc는 _extract_acc(마스크 frac) 휴리스틱. 마스크 증강을 켜면 frac이 낮아져 acc4→acc8 오판 →
        #    UC 실험은 마스크 증강 off에서만.
        self.uc = bool(getattr(args, "time_cond", False) or getattr(args, "label_cond", False))

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, C, H, W, two = masked_kspace.shape
        # 언더샘플 영상 = ifft(masked_kspace) -> (B,C,H,W,2)
        img = ifftn_centered(masked_kspace, spatial_dims=2, is_complex=True)

        # per-sample 스케일 정규화: 신경망에 ~단위 스케일 입력을 주고, 출력은 다시 곱해
        # 타깃 물리 스케일로 복원. 이게 없으면 SSIM≈0(스케일 불일치 + bias로 학습 불가).
        # zero-filled RSS magnitude의 표준편차를 스케일로 사용.
        zf_mag = root_sum_of_squares_t(complex_abs_t(img), spatial_dim=1)  # (B,H,W)
        scale = zf_mag.reshape(B, -1).std(dim=1).clamp_min(1e-12)          # (B,)
        img = img / scale.view(B, 1, 1, 1, 1)                              # broadcast over C,H,W,2

        x = img.unsqueeze(1)                      # (B,1,C,H,W,2)
        # (B,1,1,W,1) 마스크에서 (B,1,C,H,W,2) boolean 마스크로 broadcast
        mask_b = mask.bool().view(B, 1, 1, 1, W, 1).expand(B, 1, C, H, W, two)
        acc = _extract_acc(mask.reshape(B, W)[0])

        if self.uc:
            # timestep=0 → latent_recon가 actual_timestep = 0*T + i = i (cascade 번호)로 사용.
            # timestep!=None이면 (x, cas_skips, sensitivity_maps) 3-튜플 반환 → [0]으로 텐서만.
            out = self.cascaded(x, mask_b, self.mask_type, acc, self.acq_type, timestep=0)
            out = out[0]
        else:
            out = self.cascaded(x, mask_b, self.mask_type, acc, self.acq_type)

        # non-flow 경로는 영상 (B,T,C,H,W,2) 반환
        out = out[:, 0]                           # T 제거 -> (B,C,H,W,2)
        mag = root_sum_of_squares_t(complex_abs_t(out), spatial_dim=1)  # (B,H,W)
        mag = mag * scale.view(B, 1, 1)           # 출력 스케일 복원 -> 타깃 물리 스케일
        # center_crop은 (C,H,W)를 받음 -> B를 C자리로 취급, (B,384,384) 반환(폭 372->384 zero-pad 포함)
        # center_crop이 cuda 가용 시 출력을 cuda로 강제 이동하므로, 입력과 동일 디바이스로 재정렬(정석 계약).
        return center_crop(mag, *self.out_size).to(masked_kspace.device)


def build_sdum(args, out_size=(384, 384)) -> SDUMReconModel:
    return SDUMReconModel(args, out_size=out_size)
