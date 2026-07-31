# Vendored from sMartify-fastmri/2025_final (FastMRI_challenge_sMaRtIfy_final/utils/model/promptmr_plus.py)
# 원본 커밋: c7087f58addfc2e6f0cec0b33443213d69e2ce52 (main). PromptMR+ (Xin et al., arXiv:2411/ECCV'24).
# 이 repo에선 knee·정적 2D·Cartesian 전용. num_adj_slices=1(기본) 또는 3(--num-adj-slices 3).
# `import fastmri`는 varnet과 동일하게 utils/model/fastmri로 해석됨(sens_expand/sens_reduce는 그쪽에 추가함).
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
from torch.utils.checkpoint import checkpoint

import fastmri
from utils.common.utils import center_crop

from utils.model.promptmr.reentrant_wrapper import ReentrantWrapper


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias, stride=stride)

def mask_center(x: torch.Tensor, mask_from: int, mask_to: int) -> torch.Tensor:
    """
    Initializes a mask with the center filled in.

    Args:
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.

    Returns:
        A mask with the center filled.
    """
    mask = torch.zeros_like(x)
    mask[:, :, :, mask_from:mask_to] = x[:, :, :, mask_from:mask_to]

    return mask

def batched_mask_center(
    x: torch.Tensor, mask_from: torch.Tensor, mask_to: torch.Tensor
) -> torch.Tensor:
    """
    Initializes a mask with the center filled in.

    Can operate with different masks for each batch element.

    Args:
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.

    Returns:
        A mask with the center filled.
    """
    if not mask_from.shape == mask_to.shape:
        raise ValueError("mask_from and mask_to must match shapes.")
    if not mask_from.ndim == 1:
        raise ValueError("mask_from and mask_to must have 1 dimension.")
    if not mask_from.shape[0] == 1:
        if (not x.shape[0] == mask_from.shape[0]) or (
            not x.shape[0] == mask_to.shape[0]
        ):
            raise ValueError("mask_from and mask_to must have batch_size length.")

    if mask_from.shape[0] == 1:
        mask = mask_center(x, int(mask_from), int(mask_to))
    else:
        mask = torch.zeros_like(x)
        for i, (start, end) in enumerate(zip(mask_from, mask_to)):
            mask[i, :, :, start:end] = x[i, :, :, start:end]

    return mask

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super().__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=False),   # GPT suggestion
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act, no_use_ca=False):
        super().__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        if not no_use_ca:
            self.CA = CALayer(n_feat, reduction, bias=bias)
        else:
            self.CA = nn.Identity()
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        return res + x


##########################################################################
# ---------- Prompt Block -----------------------

class PromptBlock(nn.Module):
    def __init__(self, prompt_dim=128, prompt_len=5, prompt_size=96, lin_dim=192, learnable_prompt=False):
        super().__init__()
        self.prompt_param = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size), 
                                         requires_grad=learnable_prompt)
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.dec_conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):

        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1))
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)
        prompt_param = self.prompt_param.unsqueeze(0).repeat(B, 1, 1, 1, 1, 1).squeeze(1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * prompt_param
        prompt = torch.sum(prompt, dim=1)

        prompt = F.interpolate(prompt, (H, W), mode="bilinear")
        prompt = self.dec_conv3x3(prompt)

        return prompt


class DownBlock(nn.Module):
    def __init__(self, input_channel, output_channel, n_cab, kernel_size, reduction, bias, act,
                 no_use_ca=False, first_act=False):
        super().__init__()
        if first_act:
            self.encoder = [CAB(input_channel, kernel_size, reduction,bias=bias, act=nn.PReLU(), no_use_ca=no_use_ca)]
            self.encoder = nn.Sequential(
                    *(self.encoder+[CAB(input_channel, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca) 
                                    for _ in range(n_cab-1)]))
        else:
            self.encoder = nn.Sequential(
                *[CAB(input_channel, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca) 
                  for _ in range(n_cab)])
        self.down = nn.Conv2d(input_channel, output_channel,kernel_size=3, stride=2, padding=1, bias=True)

    def forward(self, x):
        enc = self.encoder(x)
        x = self.down(enc)
        return x, enc


class UpBlock(nn.Module):
    def __init__(self, in_dim, out_dim, prompt_dim, n_cab, kernel_size, reduction, bias, act,
                 no_use_ca=False, n_history=0):
        super().__init__()
        # momentum layer
        self.n_history = n_history
        if n_history > 0:
            self.momentum = nn.Sequential(
                nn.Conv2d(in_dim*(n_history+1), in_dim, kernel_size=1, bias=bias),
                CAB(in_dim, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca)
            )

        self.fuse = nn.Sequential(*[CAB(in_dim+prompt_dim, kernel_size, reduction,
                                        bias=bias, act=act, no_use_ca=no_use_ca) for _ in range(n_cab)])
        self.reduce = nn.Conv2d(in_dim+prompt_dim, in_dim, kernel_size=1, bias=bias)

        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_dim, out_dim, 1, stride=1, padding=0, bias=False))
        self.up = ReentrantWrapper(self.up)

        self.ca = CAB(out_dim, kernel_size, reduction, bias=bias, act=act, no_use_ca=no_use_ca)

    def forward(self, x, prompt_dec, skip, history_feat: Optional[torch.Tensor] = None):
        # momentum layer
        if self.n_history > 0:
            if history_feat is None:
                x = torch.cat([torch.tile(x, (1, self.n_history+1, 1, 1))], dim=1)
            else:
                x = torch.cat([x, history_feat], dim=1)

            x = self.momentum(x)

        x = torch.cat([x, prompt_dec], dim=1)

        x = self.fuse(x)
            
        x = self.reduce(x)

        x = self.up(x) + skip
        x = self.ca(x)

        return x


class SkipBlock(nn.Module):
    def __init__(self, enc_dim, n_cab, kernel_size, reduction, bias, act, no_use_ca=False):
        super().__init__()
        if n_cab == 0:
            self.skip_attn = nn.Identity()
        else:
            self.skip_attn = nn.Sequential(*[CAB(enc_dim, kernel_size, reduction, bias=bias, act=act,
                                                 no_use_ca=no_use_ca) for _ in range(n_cab)])

    def forward(self, x):
        x = self.skip_attn(x)
        return x
    
    
class KspaceACSExtractor:
    '''
    Extract ACS lines from k-space data
    '''

    def __init__(self, mask_center):
        self.mask_center = mask_center

    def get_pad_and_num_low_freqs(
        self, mask: torch.Tensor, num_low_frequencies: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        get the padding size and number of low frequencies for the center mask. For fastmri and cmrxrecon dataset
        '''
        if num_low_frequencies is None or (num_low_frequencies == -1).all():
            # get low frequency line locations and mask them out
            squeezed_mask = mask[:, 0, 0, :, 0].to(torch.int8)
            cent = squeezed_mask.shape[1] // 2
            # running argmin returns the first non-zero
            left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
            right = torch.argmin(squeezed_mask[:, cent:], dim=1)
            num_low_frequencies_tensor = torch.max(
                2 * torch.min(left, right), torch.ones_like(left)
            )  # force a symmetric center unless 1
        else:
            num_low_frequencies_tensor = num_low_frequencies * torch.ones(
                mask.shape[0], dtype=mask.dtype, device=mask.device
            )

        pad = (mask.shape[-2] - num_low_frequencies_tensor + 1) // 2
        return pad.type(torch.long), num_low_frequencies_tensor.type(torch.long)

    def __call__(self, masked_kspace: torch.Tensor,
                 mask: torch.Tensor,
                 num_low_frequencies: Optional[int] = None,
                 mask_type: Tuple[str] = ("cartesian",),
                 ) -> torch.Tensor:
        # slim: Cartesian 전용. poisson_disc(cc-brain)·kt_radial(cmrxrecon) 분기 제거.
        if not self.mask_center:
            return masked_kspace
        pad, num_low_freqs = self.get_pad_and_num_low_freqs(mask, num_low_frequencies)
        return batched_mask_center(masked_kspace, pad, pad + num_low_freqs)



class PromptUnet(nn.Module):
    def __init__(self,
                in_chans: int,
                out_chans: int,
                n_feat0: int,
                feature_dim: List[int],
                prompt_dim: List[int],
                len_prompt: List[int],
                prompt_size: List[int],
                n_enc_cab: List[int],
                n_dec_cab: List[int],
                n_skip_cab: List[int],
                n_bottleneck_cab: int,
                kernel_size=3,
                reduction=2,  # modified by sMaRtIfy
                act=nn.PReLU(),
                bias=False,
                no_use_ca=False,
                learnable_prompt=False,
                adaptive_input=False,
                n_buffer=0,
                n_history=0,
                 ):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_history = n_history
        self.n_buffer = n_buffer if adaptive_input else 0

        in_chans = in_chans * (1+self.n_buffer) if adaptive_input else in_chans
        out_chans = out_chans * (1+self.n_buffer) if adaptive_input else in_chans

        # Feature extraction
        self.feat_extract = conv(in_chans, n_feat0, kernel_size, bias=bias)

        # Encoder - 3 DownBlocks
        self.enc_level1 = DownBlock(n_feat0, feature_dim[0], n_enc_cab[0], kernel_size, reduction, bias, act, no_use_ca, first_act=True)
        self.enc_level2 = DownBlock(feature_dim[0], feature_dim[1], n_enc_cab[1], kernel_size, reduction, bias, act, no_use_ca)
        self.enc_level3 = DownBlock(feature_dim[1], feature_dim[2], n_enc_cab[2], kernel_size, reduction, bias, act, no_use_ca)

        # Skip Connections - 3 SkipBlocks
        self.skip_attn1 = SkipBlock(n_feat0, n_skip_cab[0], kernel_size, reduction, bias, act, no_use_ca)
        self.skip_attn2 = SkipBlock(feature_dim[0], n_skip_cab[1], kernel_size, reduction, bias, act, no_use_ca)
        self.skip_attn3 = SkipBlock(feature_dim[1], n_skip_cab[2], kernel_size, reduction, bias, act, no_use_ca)

        # Bottleneck
        self.bottleneck = nn.Sequential(*[CAB(feature_dim[2], kernel_size, reduction, bias, act, no_use_ca)
                                          for _ in range(n_bottleneck_cab)])
        # Decoder - 3 UpBlocks
        self.prompt_level3 = PromptBlock(prompt_dim[2], len_prompt[2], prompt_size[2], feature_dim[2], learnable_prompt)
        self.dec_level3 = UpBlock(feature_dim[2], feature_dim[1], prompt_dim[2], n_dec_cab[2], kernel_size, reduction, bias, act, no_use_ca, n_history)

        self.prompt_level2 = PromptBlock(prompt_dim[1], len_prompt[1], prompt_size[1], feature_dim[1], learnable_prompt)
        self.dec_level2 = UpBlock(feature_dim[1], feature_dim[0], prompt_dim[1], n_dec_cab[1], kernel_size, reduction, bias, act, no_use_ca, n_history)

        self.prompt_level1 = PromptBlock(prompt_dim[0], len_prompt[0], prompt_size[0], feature_dim[0], learnable_prompt)
        self.dec_level1 = UpBlock(feature_dim[0], n_feat0, prompt_dim[0], n_dec_cab[0], kernel_size, reduction, bias, act, no_use_ca, n_history)

        # OutConv
        self.conv_last = conv(n_feat0, out_chans, 5, bias=bias)

    def forward(self, x, history_feat: Optional[List[torch.Tensor]] = None):
        if history_feat is None:
            history_feat = [None, None, None]

        history_feat3, history_feat2, history_feat1 = history_feat
        current_feat = []

        # 0. featue extraction
        x = self.feat_extract(x)

        # 1. encoder
        x, enc1 = self.enc_level1(x)
        x, enc2 = self.enc_level2(x)
        x, enc3 = self.enc_level3(x)

        # 2. bottleneck
        x = self.bottleneck(x)

        # 3. decoder
        current_feat.append(x.clone())
        dec_prompt3 = self.prompt_level3(x)
        x = self.dec_level3(x, dec_prompt3, self.skip_attn3(enc3), history_feat3)

        current_feat.append(x.clone())
        dec_prompt2 = self.prompt_level2(x)
        x = self.dec_level2(x, dec_prompt2, self.skip_attn2(enc2), history_feat2)

        current_feat.append(x.clone())
        dec_prompt1 = self.prompt_level1(x)
        x = self.dec_level1(x, dec_prompt1, self.skip_attn1(enc1), history_feat1)

        # 4. last conv
        if self.n_history > 0:
            # ⚠️ 원본과 다른 점: 원본은 인자로 받은 history_feat 리스트를 제자리(in-place)로 갱신한다.
            #    그러면 cascade 를 gradient checkpointing 으로 감쌀 때 조용히 틀린 gradient 가 나온다 —
            #    역전파 재계산이 "이미 갱신된" 리스트를 입력으로 다시 읽어서 순전파와 다른 값을 만든다.
            #    (checkpoint 없이는 호출자가 반환값으로 바로 재바인딩해서 차이가 안 보인다.)
            #    새 리스트를 만들어 반환하면 결과는 그대로면서 재계산이 항상 같은 입력을 본다.
            new_history_feat = []
            for i, history_feat_i in enumerate(history_feat):
                if history_feat_i is None:  # for the first cascade, repeat the current feature
                    new_history_feat.append(torch.cat([torch.tile(current_feat[i], (1, self.n_history, 1, 1))], dim=1))
                else:  # for the rest cascades: pop the oldest feature and append the current feature
                    new_history_feat.append(torch.cat([current_feat[i], history_feat_i[:, :-self.feature_dim[2-i]]], dim=1))
            history_feat = new_history_feat
        return self.conv_last(x), history_feat

class NormPromptUnet(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        n_feat0: int,
        feature_dim: List[int],
        prompt_dim: List[int],
        len_prompt: List[int],
        prompt_size: List[int],
        n_enc_cab: List[int],
        n_dec_cab: List[int],
        n_skip_cab: List[int],
        n_bottleneck_cab: int,
        no_use_ca: bool = False,
        learnable_prompt=False,
        adaptive_input=False,
        n_buffer=0,
        n_history=0,
    ):

        super().__init__()
        self.n_history = n_history
        self.n_buffer = n_buffer
        self.unet = PromptUnet(in_chans=in_chans,
                               out_chans=out_chans,
                               n_feat0=n_feat0,
                               feature_dim=feature_dim,
                               prompt_dim=prompt_dim,
                               len_prompt=len_prompt,
                               prompt_size=prompt_size,
                               n_enc_cab=n_enc_cab,
                               n_dec_cab=n_dec_cab,
                               n_skip_cab=n_skip_cab,
                               n_bottleneck_cab=n_bottleneck_cab,
                               no_use_ca=no_use_ca,
                               learnable_prompt=learnable_prompt,
                               adaptive_input=adaptive_input,
                               n_buffer=n_buffer,
                               n_history=n_history,
                               )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return einops.rearrange(x, 'b c h w two -> b (two c) h w')

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        return einops.rearrange(x, 'b (two c) h w -> b c h w two', two=2).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        x = x.reshape(b, c * h * w)

        mean = x.mean(dim=1).view(b, 1, 1, 1)
        std = x.std(dim=1).view(b, 1, 1, 1)

        x = x.view(b, c, h, w)
        return (x - mean) / std, mean, std

    def unnorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * std + mean

    def pad(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 7) + 1
        h_mult = ((h - 1) | 7) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(self, x: torch.Tensor,
              h_pad: List[int], w_pad: List[int], h_mult: int, w_mult: int) -> torch.Tensor:
        return x[..., h_pad[0]: h_mult - h_pad[1], w_pad[0]: w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor,
                history_feat: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
                buffer: torch.Tensor = None):
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")
        cc = x.shape[1]
        if buffer is not None:
            x = torch.cat([x, buffer], dim=1)

        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        x, history_feat = self.unet(x, history_feat)

        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)

        if buffer is not None:
            x, _, latent, _ = torch.split(x, [cc, cc, cc, x.shape[1] - 3*cc], dim=1)
        else:
            latent = None
        return x, latent, history_feat

class PromptMRBlock(nn.Module):
    def __init__(self, model: nn.Module, num_adj_slices: int = 1):

        super().__init__()
        self.model = model
        self.num_adj_slices = num_adj_slices   # DC(sens_expand/reduce)가 슬라이스 그룹을 알아야 한다
        self.dc_weight = nn.Parameter(torch.ones(1))

    def forward(
        self,
        current_img: torch.Tensor,
        img_zf: torch.Tensor,
        latent: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
        history_feat: Optional[Tuple[torch.Tensor, ...]] = None
    ):
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_img)
        current_kspace = fastmri.sens_expand(current_img, sens_maps, self.num_adj_slices)
        ffx = fastmri.sens_reduce(torch.where(mask != 0, current_kspace, zero), sens_maps,
                                  self.num_adj_slices) # mask != 0 for silence warning
        if self.model.n_buffer > 0:
            # adaptive input. buffer: A^H*A*x_i, s_i, x0, A^H*A*x_i-x0
            buffer = torch.cat([ffx, latent, img_zf] + [ffx-img_zf]*(self.model.n_buffer-3), dim=1)
        else:
            buffer = None
            
        soft_dc = (ffx - img_zf) * self.dc_weight
        model_term, latent, history_feat = self.model(current_img, history_feat, buffer)
        img_pred = current_img - soft_dc - model_term
        return img_pred, latent, history_feat

class PromptMR(nn.Module):

    def __init__(
        self,
        num_cascades: int = 12,
        n_feat0: int = 48,
        feature_dim: List[int] = [72, 96, 120],  
        prompt_dim: List[int] = [24, 48, 72],
        sens_n_feat0: int = 24,
        sens_feature_dim: List[int] = [36, 48, 60],
        sens_prompt_dim: List[int] = [12, 24, 36],
        len_prompt: List[int] = [3, 3, 3],
        prompt_size: List[int] = [64, 32, 16],
        n_enc_cab: List[int] = [2, 3, 3],
        n_dec_cab: List[int] = [2, 2, 3],
        n_skip_cab: List[int] = [1, 1, 1],
        n_bottleneck_cab: int = 3,
        no_use_ca: bool = False,
        sens_len_prompt: Optional[List[int]] = None,
        sens_prompt_size: Optional[List[int]] = None,
        sens_n_enc_cab: Optional[List[int]] = None,
        sens_n_dec_cab: Optional[List[int]] = None,
        sens_n_skip_cab: Optional[List[int]] = None,
        sens_n_bottleneck_cab: Optional[List[int]] = None,
        sens_no_use_ca: Optional[bool] = None,
        mask_center: bool = True,
        learnable_prompt: bool = False,
        adaptive_input: bool = True,   # Plus
        n_buffer: int = 4,             # Plus
        n_history: int = 11,           # Plus
        num_adj_slices: int = 1,       # 2.5D 인접 슬라이스 (1=단일, 3=±1). 상류 원본과 동일 규약
    ):
        # 인접 슬라이스 규약(상류 promptmr_v2.py와 동일):
        #   masked_kspace 코일축은 **슬라이스-major** 배열 — (B, adj*C, H, W, 2) 에서
        #   [adj0의 C코일, adj1의 C코일, ...] 순서. einops 'b (adj coil) ...' 와 같은 뜻이다.
        #   sens_reduce/sens_expand(coil_combine.py)가 이 배열을 전제로 이미 num_adj_slices를 받는다.
        #   use_sens_adj는 미사용(감도는 코일별 독립 추정) — 8GB에서 compute_sens_per_coil을 쓰므로
        #   sens U-Net in_chans는 2로 둔다. 상류의 use_sens_adj=False 분기와 동일.
        super().__init__()
        self.num_cascades = num_cascades
        self.n_history = n_history
        self.n_buffer = n_buffer
        self.num_adj_slices = num_adj_slices
        self.center_slice = num_adj_slices // 2
        self.sens_net = SensitivityModel(
            num_adj_slices=num_adj_slices,
            n_feat0=sens_n_feat0,
            feature_dim=sens_feature_dim,
            prompt_dim=sens_prompt_dim,
            len_prompt=sens_len_prompt if sens_len_prompt is not None else len_prompt,
            prompt_size=sens_prompt_size if sens_prompt_size is not None else prompt_size,
            n_enc_cab=sens_n_enc_cab if sens_n_enc_cab is not None else n_enc_cab,
            n_dec_cab=sens_n_dec_cab if sens_n_dec_cab is not None else n_dec_cab,
            n_skip_cab=sens_n_skip_cab if sens_n_skip_cab is not None else n_skip_cab,
            n_bottleneck_cab=sens_n_bottleneck_cab if sens_n_bottleneck_cab is not None else n_bottleneck_cab,
            no_use_ca=sens_no_use_ca if sens_no_use_ca is not None else no_use_ca,
            mask_center=mask_center,
            learnable_prompt=learnable_prompt,
        )
        self.sens_net_ckp = ReentrantWrapper(self.sens_net)

        # DC + denoiser in each cascade
        self.cascades = nn.ModuleList([
            PromptMRBlock(
                NormPromptUnet(
                    in_chans=2 * num_adj_slices,    # 상류: 실/허 2채널 × 인접 슬라이스 수
                    out_chans=2 * num_adj_slices,
                    n_feat0=n_feat0,
                    feature_dim=feature_dim,
                    prompt_dim=prompt_dim,
                    len_prompt=len_prompt,
                    prompt_size=prompt_size,
                    n_enc_cab=n_enc_cab,
                    n_dec_cab=n_dec_cab,
                    n_skip_cab=n_skip_cab,
                    n_bottleneck_cab=n_bottleneck_cab,
                    no_use_ca=no_use_ca,
                    learnable_prompt=learnable_prompt,
                    adaptive_input=adaptive_input,
                    n_buffer=n_buffer,
                    n_history=n_history
                ),
                num_adj_slices=num_adj_slices,
            ) for _ in range(num_cascades)
        ])


    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[torch.Tensor] = None,
        mask_type: Tuple[str] = ("cartesian",),
        use_checkpoint: bool = True,
        compute_sens_per_coil: bool = True, # can further reduce the memory usage
    ) -> torch.Tensor:
        '''
        Args:
            masked_kspace: (bs, nc, h, w, 2) input k-space data
            mask: (bs, 1, h, w) or (bs, nc, h, w) mask
            num_low_frequencies: (bs) number of low frequencies
            mask_type: (str) mask type
            use_checkpoint: (bool) whether to use checkpoint for memory saving
            compute_sens_per_coil: (bool) whether to compute sensitivity maps per coil for memory saving
        '''

        # Shape aligning to SNU FastMRI dataloader
        if mask.shape[2] == 1:
            if num_low_frequencies is None:
                num_low_frequencies = torch.tensor([round(0.08 * mask.shape[3]) for _ in range(mask.shape[0])]).cuda(non_blocking=True) # TODO Batch implementation
                
            mask = torch.stack([mask.squeeze(2).clone() for _ in range(masked_kspace.shape[2])], dim=2)

        if compute_sens_per_coil:
            # 코일 단위 checkpoint 를 compute_sens 안에서 직접 걸므로 바깥 통짜 wrapper 는 쓰지 않는다.
            # (겹쳐 걸면 바깥 재계산이 코일 루프를 통째로 되살려 코일 단위 절감이 무의미해진다.)
            sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies, mask_type,
                                      compute_sens_per_coil, use_checkpoint)
        else:
            self.sens_net_ckp.use_checkpoint = use_checkpoint
            sens_maps = self.sens_net_ckp(masked_kspace, mask, num_low_frequencies, mask_type, compute_sens_per_coil)

        img_zf = fastmri.sens_reduce(masked_kspace, sens_maps, self.num_adj_slices)
        img_pred = img_zf.clone()
        latent = img_zf.clone()
        history_feat = None

        for cascade in self.cascades:
            if use_checkpoint and self.training:
                # cascade 당 (B,C,H,W,2) 중간 텐서가 ~5개 살아남는다(sens_expand·fft2c·where·ifft2c
                # ·complex_mul). 코일 압축을 없애면 C=15 라 이게 cascade 수만큼 그대로 쌓인다.
                # cascade 단위로 끊으면 역전파 때 한 cascade 분만 재현되므로 nc 를 늘려도 평평해진다.
                img_pred, latent, history_feat = checkpoint(
                    cascade, img_pred, img_zf, latent, mask, sens_maps, history_feat,
                    use_reentrant=False,
                )
            else:
                img_pred, latent, history_feat = cascade(img_pred, img_zf, latent, mask, sens_maps, history_feat)

        # 인접 슬라이스면 **중앙 슬라이스만** 최종 출력이다(상류와 동일). img_pred는 (B,adj,H,W,2),
        # sens_maps는 (B,adj*C,H,W,2) — 각각 adj로 chunk해 center를 꺼낸 뒤 coil-combine한다.
        # num_adj_slices=1이면 chunk(...,1)[0]이 항등이라 기존 동작과 비트 동일하다.
        if self.num_adj_slices > 1:
            img_pred = torch.chunk(img_pred, self.num_adj_slices, dim=1)[self.center_slice]
            sens_maps = torch.chunk(sens_maps, self.num_adj_slices, dim=1)[self.center_slice]
        # complex_mul 은 torch.stack 으로 새 텐서를 만들어 입력을 건드리지 않는다 → clone 불필요.
        img_pred = fastmri.rss(fastmri.complex_abs(fastmri.complex_mul(img_pred, sens_maps)), dim=1)
        img_pred = center_crop(img_pred, 384, 384)

        return img_pred

class SensitivityModel(nn.Module):
    
    def __init__(
        self,
        n_feat0: int,
        feature_dim: List[int],
        prompt_dim: List[int],
        len_prompt: List[int],
        prompt_size: List[int],
        n_enc_cab: List[int],
        n_dec_cab: List[int],
        n_skip_cab: List[int],
        n_bottleneck_cab: int,
        no_use_ca: bool = False,
        mask_center: bool = True,
        learnable_prompt=False,
        num_adj_slices: int = 1,
    ):
        # use_sens_adj는 미사용(상류의 use_sens_adj=False 분기) → 감도는 **코일별 독립 추정**이라
        # unet in_chans는 인접 슬라이스 수와 무관하게 2다. num_adj_slices는 RSS 정규화를
        # 슬라이스 그룹별로 하기 위해서만 필요하다(divide_root_sum_of_squares 참고).
        super().__init__()
        self.num_adj_slices = num_adj_slices
        self.mask_center = mask_center
        self.norm_unet = NormPromptUnet(in_chans=2,
                                        out_chans=2,
                                        n_feat0=n_feat0,
                                        feature_dim=feature_dim,
                                        prompt_dim=prompt_dim,
                                        len_prompt=len_prompt,
                                        prompt_size=prompt_size,
                                        n_enc_cab=n_enc_cab,
                                        n_dec_cab=n_dec_cab,
                                        n_skip_cab=n_skip_cab,
                                        n_bottleneck_cab=n_bottleneck_cab,
                                        no_use_ca=no_use_ca,
                                        learnable_prompt=learnable_prompt,
                                        )
        self.kspace_acs_extractor = KspaceACSExtractor(mask_center)

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b = x.shape[0]
        x = einops.rearrange(x, 'b coil h w two -> (b coil) 1 h w two')   # 코일별 독립 추정
        return x, b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        return einops.rearrange(x, '(b coil) 1 h w two -> b coil h w two', b=batch_size)

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        # per-pixel coil normalization: sum_c |S_c|^2 = 1
        # ⚠️ 인접 슬라이스에서는 **슬라이스 그룹별로** 정규화해야 한다(상류와 동일). 코일축 전체로
        #    한 번에 나누면 각 그룹 합이 1/adj 로 줄어 스케일이 조용히 어긋난다.
        if self.num_adj_slices == 1:
            return x / fastmri.rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)
        b, adj_coil, h, w, two = x.shape
        coil = adj_coil // self.num_adj_slices
        x = x.view(b, self.num_adj_slices, coil, h, w, two)
        x = x / fastmri.rss_complex(x, dim=2).unsqueeze(-1).unsqueeze(2)
        return x.view(b, adj_coil, h, w, two)

    def compute_sens(self, model: nn.Module, images: torch.Tensor, compute_per_coil: bool,
                     use_checkpoint: bool = False) -> torch.Tensor:
        bc = images.shape[0]  # batch_size * n_coils
        if compute_per_coil:
            output = []
            for i in range(bc):
                image_i = images[i].unsqueeze(0)
                if use_checkpoint and self.training:
                    # ⚠️ 코일 "하나씩" checkpoint 하는 게 핵심이다. sens_net 전체를 한 덩어리로
                    # checkpoint 하면 순전파 peak 만 줄고, 역전파 재계산 때 이 루프가 통째로 다시
                    # 돌면서 B*C 코일분 activation 이 동시에 살아난다(= 절감 효과가 사라짐).
                    # 코일 단위로 끊으면 재계산 중에도 한 코일분만 살아 있다.
                    out_i = checkpoint(lambda t: model(t)[0], image_i, use_reentrant=False)
                else:
                    out_i = model(image_i)[0]
                output.append(out_i)
            output = torch.cat(output, dim=0)
        else:
            output = model(images)[0]
        return output

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[Union[int, torch.Tensor]] = None,
        mask_type: Tuple[str] = ("cartesian",),
        compute_per_coil: bool = False,
        use_checkpoint: bool = False,
    ) -> torch.Tensor:

        masked_kspace_acs = self.kspace_acs_extractor(masked_kspace, mask, num_low_frequencies, mask_type)
        # convert to image space
        images, batches = self.chans_to_batch_dim(fastmri.ifft2c(masked_kspace_acs))

        return self.divide_root_sum_of_squares(
            self.batch_chans_to_chan_dim(
                self.compute_sens(self.norm_unet, images, compute_per_coil, use_checkpoint), batches
            )
        )