'''
Adapted from https://github.com/meta-llama/llama/blob/main/llama/model.py

Original Copyright Notice:
  Copyright (c) Meta Platforms, Inc. and affiliates.
  This software may be used and distributed according to the terms of the Llama Community License Agreement.
'''

import math
from typing import Optional, Tuple

import torch

def apply_scaling(freqs: torch.Tensor, old_context_length: int, scale_factor:float = 32.0, low_freq_factor: float = 1.0, high_freq_factor:float = 4.0,) -> torch.Tensor:
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < (old_context_length / high_freq_factor):
            new_freqs.append(freq)
        elif wavelen > (old_context_length / low_freq_factor):
            new_freqs.append(freq / scale_factor)
        else:
            new_freqs.append(freq / (scale_factor * 0.5))

    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 500000.0,
    use_scaled: bool = False,
    old_context_length: int = 32768,
    accelerator=None,
):
    """
    Produces a tensor of shape (end, dim) that encodes the cos/sin phases used
    for rotary embeddings. We do NOT slice to (dim//2), so if dim=64,
    the output shape is (end, 64).

    Inside this function:
      - We generate 'dim/2' distinct frequency values (because we step by +2).
      - The final shape is (end, dim/2) in frequency space. Then we convert
        to complex representation, effectively doubling to 'dim'.

    But to keep it simpler for assertion checks, we directly arrange it as
    shape (end, dim). That matches x.shape[-1] if x has dimension 64.

    Steps:
      1) We create a range [0, 2, 4, ..., dim) => length = dim//2
      2) Then do outer product => (end, dim//2)
      3) Convert to a complex representation => shape still (end, dim//2),
         but each entry is complex => effectively 2 * (dim//2) = dim real floats.
      4) We'll store it as a 'real dimension' of 'dim', so we flatten real+imag
         in apply_rotary_emb. This means the final .shape is (end, dim).

    Alternatively, you can store it as shape (end, dim//2) in a complex dtype.
    For clarity, we'll keep it as is and do an assertion in reshape_for_broadcast.
    """
    if accelerator is not None:
        accelerator.print(
            f"precompute_freqs_cis(dim={dim}, end={end}, theta={theta},"
            f" use_scaled={use_scaled}, old_context_length={old_context_length})"
        )

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    if use_scaled:
        freqs = apply_scaling(freqs, old_context_length=old_context_length)

    t = torch.arange(end, device=freqs.device)  

    freqs = torch.outer(t, freqs)

    freqs_cis = torch.polar(
        torch.ones_like(freqs),
        freqs
    )  

    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """

    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f"freqs_cis.shape: {freqs_cis.shape} != x.shape: {x.shape}"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

@torch._dynamo.disable
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

    """

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)