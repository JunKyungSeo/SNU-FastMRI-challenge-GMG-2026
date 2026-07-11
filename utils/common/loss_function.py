"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X, Y, data_range, mask=None):
        """mask=None이면 기존 그대로(전체 384² 평균). mask((B,H,W) 0/1)를 주면
        채점(utils/common/metrics.py::ssim_full)과 동일하게 foreground 안에서만 평균:
        입력을 마스크로 곱한 뒤 SSIM 맵을 win//2 valid 영역으로 crop한 마스크로 가중평균."""
        X = X.unsqueeze(1)
        Y = Y.unsqueeze(1)
        data_range = data_range[:, None, None, None]
        if mask is not None:
            m = mask.unsqueeze(1)                 # (B,1,H,W)
            X = X * m
            Y = Y * m
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)
        uy = F.conv2d(Y, self.w)
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        if mask is None:
            return 1 - S.mean()
        # ssim_full 정합: 마스크를 conv-valid 영역(win//2 pad)만큼 crop 후 샘플별 가중평균.
        pad = self.win_size // 2
        m_valid = m[:, :, pad:-pad, pad:-pad]
        num = (S * m_valid).sum(dim=(1, 2, 3))
        den = m_valid.sum(dim=(1, 2, 3)).clamp_min(1.0)  # 빈 마스크(드묾) 방어
        return 1 - (num / den).mean()
