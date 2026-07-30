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


BBOX_CAP = 2   # C1: 슬라이스당 박스 개수 상한. batch=1에서 박스 많은 슬라이스의 과대 gradient 스파이크 억제.


def bbox_ssim_loss(ssim, output, target, data_range, bboxes):
    """C1: 병변 박스 안에서 계산한 SSIM 손실. 집계 = 슬라이스별 cap=2 합(스케일 X).

    per-box 값은 채점(utils/common/metrics.py::ssim_bbox, 고정 파일)과 **동일**하게 맞춘다
    — 학습 신호와 채점식이 어긋나면 안 되므로:
      · 박스를 이미지 경계로 클리핑 → 클리핑 '후' w·h 가 win_size(7) 미만이면 제외(채점도 None 처리)
      · crop 한 뒤 그 위에서 SSIM 맵을 구하고 평균 (전체 이미지 맵을 잘라내는 게 아님)
    ssim(=SSIMLoss)이 `1 - S.mean()`을 돌려주므로 박스 하나의 반환값 = 1 - metrics.ssim_bbox 값.

    집계는 **박스 평균이 아니라 슬라이스별 cap=2 합**이다(설계: docs/archive/superpowers/specs/2026-07-12-bbox-interslice-saliency-design.md):
      한 슬라이스의 유효 박스 손실이 ℓ₁..ℓ_m 일 때
        slice_term = min(m, 2) · mean(ℓ₁..ℓ_m)     # m≤2면 합, m≥3면 2·평균(초과분 cap). 순서 무관.
      L_bbox = 박스가 있는 슬라이스들의 slice_term 평균 (batch=1이라 보통 그 슬라이스 하나).
    이유: 채점은 전역 박스 누산기라 박스를 다 세지만(박스 평균은 multi-box 슬라이스를 1/m로 저평가),
    순수 합은 batch=1에서 박스 많은 슬라이스가 학습을 흔든다. cap=2가 둘의 절충 — 박스의 82%(≤2개)는
    합과 동일하게 세고, 나머지(≥3개, drop 6.9%)만 2개어치로 자른다. 스케일 X = 채점 정합 배율 k를
    넣지 않음 → 단일 박스 gradient가 λ 그대로(비병변 대비 1.55×), cap으로 최대 2.11×.

    유효 박스가 하나도 없으면 0 을 돌려주므로 박스 없는 슬라이스에서는 손실이 기존과 정확히 같다.

    Args:
        ssim: SSIMLoss 인스턴스 (win_size 등 채점과 동일 설정)
        output, target: (B, H, W)
        data_range: (B,)  — 각 샘플의 max (정규화 쓰면 정규화 공간의 max)
        bboxes: 길이 B 의 리스트. 각 원소는 dict(x, y, width, height) 의 리스트.
    """
    win = ssim.win_size
    H, W = output.shape[-2], output.shape[-1]
    slice_terms = []
    for b, boxes in enumerate(bboxes):
        box_losses = []
        for box in boxes:
            x0, y0 = max(0, int(box["x"])), max(0, int(box["y"]))
            x1 = min(W, int(box["x"]) + int(box["width"]))
            y1 = min(H, int(box["y"]) + int(box["height"]))
            if (x1 - x0) < win or (y1 - y0) < win:
                continue                                  # 채점이 None 으로 버리는 박스 → 학습에서도 제외
            box_losses.append(ssim(output[b:b + 1, y0:y1, x0:x1],
                                   target[b:b + 1, y0:y1, x0:x1],
                                   data_range[b:b + 1]))
        if not box_losses:
            continue
        m = len(box_losses)
        # cap=2 합 = min(m,2)·mean. mean이라 박스 순서에 무관(합-of-first-2 방식은 조용히 틀림).
        slice_terms.append(min(m, BBOX_CAP) * torch.stack(box_losses).mean())
    if not slice_terms:
        return output.new_zeros(())                       # 유효 박스 없음 → λ항 0 (= 기존 손실과 동일)
    return torch.stack(slice_terms).mean()
