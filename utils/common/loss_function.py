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


def bbox_ssim_loss(ssim, output, target, data_range, bboxes):
    """C1: 병변 박스 안에서만 계산한 SSIM 손실 = 1 - mean_over_boxes SSIM(box).

    유효 박스 판정·crop·평균 방식을 채점(utils/common/metrics.py::ssim_bbox, 고정 파일)과
    **동일**하게 맞춘다 — 학습 신호와 채점식이 어긋나면 안 되므로:
      · 박스를 이미지 경계로 클리핑 → 클리핑 '후' w·h 가 win_size(7) 미만이면 제외(채점도 None 처리)
      · crop 한 뒤 그 위에서 SSIM 맵을 구하고 평균 (전체 이미지 맵을 잘라내는 게 아님)
    ssim(=SSIMLoss)이 `1 - S.mean()`을 돌려주므로 `1 - 반환값` = metrics.ssim_bbox 값.

    배치의 모든 유효 박스에 대해 **박스 단위 평균** (채점의 bbox_total/bbox_idx 와 같은 단위 —
    박스가 많은 슬라이스가 그만큼 더 기여). 유효 박스가 하나도 없으면 0 을 돌려주므로
    박스 없는 슬라이스에서는 손실이 기존과 정확히 같다.

    Args:
        ssim: SSIMLoss 인스턴스 (win_size 등 채점과 동일 설정)
        output, target: (B, H, W)
        data_range: (B,)  — 각 샘플의 max (정규화 쓰면 정규화 공간의 max)
        bboxes: 길이 B 의 리스트. 각 원소는 dict(x, y, width, height) 의 리스트.
    """
    win = ssim.win_size
    H, W = output.shape[-2], output.shape[-1]
    terms = []
    for b, boxes in enumerate(bboxes):
        for box in boxes:
            x0, y0 = max(0, int(box["x"])), max(0, int(box["y"]))
            x1 = min(W, int(box["x"]) + int(box["width"]))
            y1 = min(H, int(box["y"]) + int(box["height"]))
            if (x1 - x0) < win or (y1 - y0) < win:
                continue                                  # 채점이 None 으로 버리는 박스 → 학습에서도 제외
            terms.append(ssim(output[b:b + 1, y0:y1, x0:x1],
                              target[b:b + 1, y0:y1, x0:x1],
                              data_range[b:b + 1]))
    if not terms:
        return output.new_zeros(())                       # 유효 박스 없음 → λ항 0 (= 기존 손실과 동일)
    return torch.stack(terms).mean()
