# Vendored subset of NVIDIA NV-Raw2insights-MRI scripts/utils.py (Apache-2.0).
# https://github.com/NVIDIA-Medtech/NV-Raw2insights-MRI
# Only the helper functions the SDUM model imports are included; HuggingFace
# auto-download code is intentionally NOT vendored (competition rule).
import random  # noqa: F401

import torch  # noqa: F401
from monai.apps.reconstruction.complex_utils import complex_conj_t, complex_mul_t
from monai.data.fft_utils import fftn_centered, ifftn_centered

def get_acs_region(mask: torch.Tensor) -> tuple[int, int, int, int]:
    """
    Given a 2D binary mask,
    finds the maximum acs region containing only positive (nonzero)
    elements that includes the center of the mask.

    Returns:
        A tuple (left, right, top, bottom) where:
          - left and right are the minimum and maximum x indices,
          - top and bottom are the minimum and maximum y indices.
    """
    h, w = mask.shape[-3], mask.shape[-2]
    # Determine the center coordinates.
    cy, cx = h // 2, w // 2

    # Ensure the center is positive.
    if not torch.all(mask[..., cy, cx, :]):
        raise ValueError("The center of the mask is not positive.")

    # Initialize the bounding box to the center pixel.
    left, right = cx, cx
    top, bottom = cy, cy

    # Try to expand the box outward while the newly included row/column is all positive.
    expanded = True
    while expanded:
        expanded = False

        # Attempt to expand to the left.
        if left > 0 and torch.all(mask[..., top : bottom + 1, left - 1, :] > 0):
            left -= 1
            expanded = True

        # Attempt to expand to the right.
        if right < w - 1 and torch.all(mask[..., top : bottom + 1, right + 1, :] > 0):
            right += 1
            expanded = True

        # Attempt to expand upward.
        if top > 0 and torch.all(mask[..., top - 1, left : right + 1, :] > 0):
            top -= 1
            expanded = True

        # Attempt to expand downward.
        if bottom < h - 1 and torch.all(mask[..., bottom + 1, left : right + 1, :] > 0):
            bottom += 1
            expanded = True

    return top, bottom, left, right


def get_acs_image(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    top, bottom, left, right = get_acs_region(mask)
    x_k = fftn_centered(x, spatial_dims=2, is_complex=True)
    num_low_freqs_x = right - left  # size of the fully-sampled center
    num_low_freqs_y = bottom - top

    # take out the fully-sampled region and set the rest of the data to zero
    x_k_acs = torch.zeros_like(x_k)
    start_x = (mask.shape[-2] - num_low_freqs_x + 1) // 2  # this marks the start of center extraction
    start_y = (mask.shape[-3] - num_low_freqs_y + 1) // 2

    x_k_acs[..., start_y : start_y + num_low_freqs_y, start_x : start_x + num_low_freqs_x, :] = x_k[
        ..., start_y : start_y + num_low_freqs_y, start_x : start_x + num_low_freqs_x, :
    ]
    x_acs = ifftn_centered(x_k_acs, spatial_dims=2, is_complex=True)
    return x_acs


def sensitivity_map_reduce(img: torch.Tensor, sens_maps: torch.Tensor, k: int = 1, mode: str = "rand") -> torch.Tensor:
    if k != 1:
        res = complex_mul_t(img, complex_conj_t(sens_maps))
        coils = list(range(sens_maps.shape[1]))
        if mode == "rand":
            random.shuffle(coils)
        return torch.stack([res[:, i::k].sum(dim=1) for i in range(k)], dim=1)
    else:
        return complex_mul_t(img, complex_conj_t(sens_maps)).sum(dim=-4, keepdim=True)


def sensitivity_map_expand(img: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    return complex_mul_t(img.sum(dim=-4, keepdim=True), sens_maps)


def reshape_channel_to_batch_dim(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    """
    Combines batch and channel dimensions.

    Args:
        x: input of shape (B,C,H,W,2) for 2D data or (B,C,H,W,D,2) for 3D data

    Returns:
        A tuple containing:
            (1) output of shape (B*C,1,...)
            (2) batch size
    """

    if len(x.shape) == 5:  # this is 2D
        b, c, h, w, two = x.shape
        return x.contiguous().view(b * c, 1, h, w, two), b

    elif len(x.shape) == 6:  # this is 3D
        b, c, h, w, d, two = x.shape
        return x.contiguous().view(b * c, 1, h, w, d, two), b

    else:
        raise ValueError(f"only 2D (B,C,H,W,2) and 3D (B,C,H,W,D,2) data are supported but x has shape {x.shape}")


def reshape_batch_channel_to_channel_dim(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Detaches batch and channel dimensions.

    Args:
        x: input of shape (B*C,1,H,W,2) for 2D data or (B*C,1,H,W,D,2) for 3D data
        batch_size: batch size

    Returns:
        output of shape (B,C,...)
    """
    if len(x.shape) == 4:  # this is 2D real
        bc, chan, h, w = x.shape  # bc represents B*C
        c = bc // batch_size
        return x.view(batch_size, c, chan, h, w)
    if len(x.shape) == 5:  # this is 2D
        bc, one, h, w, two = x.shape  # bc represents B*C
        c = bc // batch_size
        return x.view(batch_size, c, h, w, two)

    elif len(x.shape) == 6:  # this is 3D
        bc, one, h, w, d, two = x.shape  # bc represents B*C
        c = bc // batch_size
        return x.view(batch_size, c, h, w, d, two)

    else:
        raise ValueError(f"only 2D (B*C,1,H,W,2) and 3D (B*C,1,H,W,D,2) data are supported but x has shape {x.shape}")
