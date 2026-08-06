"""k-space <-> 영상 어댑터 (SDUM의 adapter.py 대체).

우리 파이프라인 계약  forward(masked_kspace (B,C,H,W,2), mask (B,1,1,W,1)) -> image (B,384,384)
를 유지하면서 PromptMR+ 백본을 호출한다.

SDUM 어댑터와 달리 **iFFT·외부 정규화가 없다** — PromptMR+가 감도맵 추정·data consistency·정규화를
모두 내부에서 처리하고, forward가 이미 RSS + center_crop(384)까지 수행한다. 어댑터가 하는 일은
mask 형태 변환 하나뿐.
"""
import torch
import torch.nn as nn

from utils.common.utils import center_crop
from utils.model.promptmr.config import build_promptmr_net


class PromptMRReconModel(nn.Module):
    """어댑터: (masked_kspace (B,C,H,W,2), mask (B,1,1,W,1)) -> image (B,384,384)."""

    def __init__(self, args, out_size=(384, 384)):
        super().__init__()
        self.net = build_promptmr_net(args)
        self.out_size = out_size
        self.use_checkpoint = bool(getattr(args, "use_checkpoint", True))
        # 감도맵 U-Net 에 한 번에 넣을 코일 수. 1 = 기존(코일 하나씩) 과 비트 동일.
        self.sens_chunk = max(1, int(getattr(args, "sens_chunk", 1)))

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, C, H, W, two = masked_kspace.shape
        # 폭(PE)축 마스크 (B,1,1,W,1) → (B,1,H,W,1)로 H축 확장.
        # PromptMR.forward의 "SNU 정렬" 블록(mask.shape[2]==1일 때 발동해 레이아웃을 바꿈)을 우회하고,
        # KspaceACSExtractor는 mask[:,0,0,:,0](=폭축)에서 ACS를 자동 추출(num_low_frequencies=None).
        mask5d = mask.expand(B, 1, H, W, 1).contiguous()
        out = self.net(
            masked_kspace, mask5d,
            num_low_frequencies=None,
            mask_type=("cartesian",),
            use_checkpoint=self.use_checkpoint,
            compute_sens_per_coil=True,     # 8GB VRAM 절감(코일별 감도 추정)
            sens_chunk=self.sens_chunk,
        )
        # PromptMR.forward가 이미 RSS + center_crop(384,384) 수행 → (B,384,384).
        # 방어적으로 크기 재확인(원본이 384가 아니면 crop; 384면 idempotent).
        if tuple(out.shape[-2:]) != tuple(self.out_size):
            out = center_crop(out, *self.out_size)
        return out.to(masked_kspace.device)


def build_promptmr(args, out_size=(384, 384)) -> PromptMRReconModel:
    return PromptMRReconModel(args, out_size=out_size)
