# Safe Triton RoPE: dtype-safe, stride-safe, with auto-fallbacks.
# Replaces the buggy Unsloth kernel usage.

from __future__ import annotations
import os
from typing import Tuple, Optional

import torch
from packaging.version import Version

# --- Optional Triton import / feature flags ---
_HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False

# Allow users to force disable Triton even if available
if os.getenv("QLLAMA_DISABLE_TRITON_ROPE", "0") == "1":
    _HAS_TRITON = False

# -----------------------
# Slow, always-correct RoPE (PyTorch)
# -----------------------
class _SlowRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, cos, sin, position_ids: Optional[torch.Tensor]):
        # X: [B, S, H, D]
        # cos/sin: either [S, D/2] or [1, S, 1, D/2] or [S, 1, D/2]
        if position_ids is not None:
            # Expect cos/sin shape [S, D/2] (squeeze safe), then index by positions
            cos_ = cos.squeeze()
            sin_ = sin.squeeze()
            # position_ids: [B, S]
            cos_ = cos_[position_ids]  # [B, S, D/2]
            sin_ = sin_[position_ids]
            # expand to [B, S, 1, D/2]
            cos = cos_.unsqueeze(2)
            sin = sin_.unsqueeze(2)
        else:
            # Make shapes broadcast to [B,S,H,D/2]
            cos = cos.squeeze()
            sin = sin.squeeze()
            while cos.dim() < 3:
                cos = cos.unsqueeze(0)
                sin = sin.unsqueeze(0)
            # cos/sin should be [S, D/2] -> [1,S,1,D/2]
            if cos.dim() == 2:
                cos = cos.unsqueeze(0).unsqueeze(2)
                sin = sin.unsqueeze(0).unsqueeze(2)

        half = X.size(-1) // 2
        x1 = X[..., :half]
        x2 = X[..., half:]
        # rotate_half(X) = [-x2, x1]
        rot1 = -x2
        rot2 = x1

        # Ensure same dtype for math
        math_dtype = torch.float32
        x1m = x1.to(math_dtype)
        x2m = x2.to(math_dtype)
        cos_m = cos.to(math_dtype)
        sin_m = sin.to(math_dtype)

        y1 = (x1m * cos_m) + (rot1.to(math_dtype) * sin_m)
        y2 = (x2m * cos_m) + (rot2.to(math_dtype) * sin_m)

        Y = torch.cat([y1, y2], dim=-1).to(X.dtype)
        ctx.save_for_backward(cos_m, sin_m)
        return Y

    @staticmethod
    def backward(ctx, dY):
        cos_m, sin_m = ctx.saved_tensors
        half = dY.size(-1) // 2
        dy1 = dY[..., :half]
        dy2 = dY[..., half:]
        math_dtype = torch.float32
        dy1m = dy1.to(math_dtype)
        dy2m = dy2.to(math_dtype)

        # For rotate_half backward:
        # d/dx1 ([-x2, x1]) = [0, I]; d/dx2 ([-x2, x1]) = [-I, 0]
        # So inverse rotation for grad is [dy2, -dy1]
        rot1 = -dy2m
        rot2 = dy1m

        dx1 = (dy1m * cos_m) + (rot1 * sin_m)
        dx2 = (dy2m * cos_m) + (rot2 * sin_m)
        dX = torch.cat([dx1, dx2], dim=-1).to(dY.dtype)
        return dX, None, None, None


def _slow_rope_apply(Q, K, cos, sin, position_ids=None):
    Q = _SlowRoPE.apply(Q, cos, sin, position_ids)
    K = _SlowRoPE.apply(K, cos, sin, position_ids)
    return Q, K


def _slow_rope_apply_key_only(K, cos, sin, position_ids=None):
    K = _SlowRoPE.apply(K, cos, sin, position_ids)
    return K


# -----------------------
# Fast Triton RoPE (safe)
# -----------------------
MAX_FUSED_SIZE = 65536

def _calc_block(head_half: int) -> Tuple[int, int]:
    # round up to power of 2, cap to MAX_FUSED_SIZE
    if not _HAS_TRITON:
        return 0, 0
    bs = triton.next_power_of_2(head_half)
    if bs > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"RoPE BLOCK_SIZE {bs} exceeds MAX_FUSED_SIZE {MAX_FUSED_SIZE}. "
            "Use Python RoPE."
        )
    num_warps = 4
    if bs >= 32768: num_warps = 32
    elif bs >=  8192: num_warps = 16
    elif bs >=  2048: num_warps = 8
    return int(bs), int(num_warps)


if _HAS_TRITON:
    @triton.jit
    def _rope_kernel(
        Q_ptr,                  # *mut T
        cos_ptr,                # *const f32
        sin_ptr,                # *const f32
        n_rows: tl.constexpr,   # rows = B*S
        n_heads: tl.constexpr,
        head_dim: tl.constexpr,
        seqlen: tl.constexpr,
        row_stride: tl.constexpr,  # contiguous: row_stride = n_heads*head_dim
        cos_stride: tl.constexpr,  # cos[seqlen, head_half]
        sin_stride: tl.constexpr,
        backward: tl.constexpr,    # bool
        BLOCK_SIZE: tl.constexpr
    ):
        row = tl.program_id(0)
        group = tl.program_id(1)   # group of heads processed by this program
        head_half = head_dim // 2

        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < head_half

        # Load cos / sin for (row % seqlen) at offsets
        cos = tl.load(cos_ptr + (row % seqlen) * cos_stride + offs, mask=mask, other=0.0)
        sin = tl.load(sin_ptr + (row % seqlen) * sin_stride + offs, mask=mask, other=0.0)

        if backward:
            sin = -sin

        # compute heads in groups of 4 (like Unsloth), but small groups are fine too.
        ROPE_GROUP = 4
        h_start = group * ROPE_GROUP
        h_end = tl.minimum(h_start + ROPE_GROUP, n_heads)

        for h in range(h_start, h_end):
            base = row * row_stride + h * head_dim
            q1_ptr = Q_ptr + base + offs
            q2_ptr = Q_ptr + base + head_half + offs

            # Do math in f32 for stability, then cast back to Q dtype on store
            q1 = tl.load(q1_ptr, mask=mask, other=0.0).to(tl.float32)
            q2 = tl.load(q2_ptr, mask=mask, other=0.0).to(tl.float32)

            o1 = q1 * cos - q2 * sin
            o2 = q2 * cos + q1 * sin

            # Cast back to original dtype
            o1 = o1.to(q1_ptr.dtype.element_ty)
            o2 = o2.to(q2_ptr.dtype.element_ty)

            tl.store(q1_ptr, o1, mask=mask)
            tl.store(q2_ptr, o2, mask=mask)


class _FastRoPEFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, cos, sin):
        # Q: [B,S,H,D], cos/sin: [S, D/2] (preferred) or broadcastable versions
        B, S, H, D = Q.shape
        assert D % 2 == 0, "head_dim must be even for RoPE"
        half = D // 2

        # Prepare cos/sin to [S, half], float32, contiguous, on same device
        cos = cos.squeeze()
        sin = sin.squeeze()
        if cos.dim() == 4:  # [1,S,1,half]
            cos = cos[0, :, 0, :]
            sin = sin[0, :, 0, :]
        elif cos.dim() == 3:  # [S,1,half] or [1,S,half]
            if cos.size(0) == 1 and cos.size(1) == S:
                cos = cos[0]
                sin = sin[0]
            elif cos.size(1) == 1 and cos.size(0) == S:
                cos = cos[:, 0]
                sin = sin[:, 0]
        # now [S, half]
        cos = cos.to(torch.float32, copy=False).contiguous()
        sin = sin.to(torch.float32, copy=False).contiguous()

        # Make Q as [B*S, H, D] contiguous along last dim
        Qc = Q.contiguous()
        Q2 = Qc.view(B * S, H, D)

        # Kernel launch params
        n_rows = B * S
        row_stride = H * D
        BLOCK, WARPS = _calc_block(half)
        # number of head groups
        group = (H + 3) // 4  # ROPE_GROUP=4

        # Launch
        _rope_kernel[(n_rows, group)](
            Q2, cos, sin,
            n_rows=n_rows,
            n_heads=H,
            head_dim=D,
            seqlen=S,
            row_stride=row_stride,
            cos_stride=cos.stride(0),
            sin_stride=sin.stride(0),
            backward=False,
            BLOCK_SIZE=BLOCK,
            num_warps=WARPS,
        )

        ctx.saved_shape = (B, S, H, D)
        ctx.row_stride = row_stride
        ctx.n_rows = n_rows
        ctx.H = H
        ctx.D = D
        ctx.S = S
        ctx.cos = cos
        ctx.sin = sin
        ctx.BLOCK = BLOCK
        ctx.WARPS = WARPS
        return Q2.view(B, S, H, D)

    @staticmethod
    def backward(ctx, dY):
        B, S, H, D = ctx.saved_shape
        dYc = dY.contiguous().view(ctx.n_rows, H, D)

        group = (H + 3) // 4
        _rope_kernel[(ctx.n_rows, group)](
            dYc, ctx.cos, ctx.sin,
            n_rows=ctx.n_rows,
            n_heads=H,
            head_dim=D,
            seqlen=S,
            row_stride=ctx.row_stride,
            cos_stride=ctx.cos.stride(0),
            sin_stride=ctx.sin.stride(0),
            backward=True,
            BLOCK_SIZE=ctx.BLOCK,
            num_warps=ctx.WARPS,
        )
        return dYc.view(B, S, H, D), None, None


def _can_use_triton(Q, cos, sin) -> bool:
    if not _HAS_TRITON:
        return False
    if not torch.cuda.is_available():
        return False
    # Triton likes CUDA tensors
    if any(t.device.type != "cuda" for t in (Q, cos, sin)):
        return False
    # Require even head dim
    if Q.size(-1) % 2 != 0:
        return False
    # Size limits
    if (Q.size(-1) // 2) > MAX_FUSED_SIZE:
        return False
    return True


# -----------------------
# Public API (drop-in)
# -----------------------
@torch.compiler.disable  # keep behavior stable across PyTorch versions
def fast_rope_embedding(Q: torch.Tensor,
                        K: torch.Tensor,
                        cos: torch.Tensor,
                        sin: torch.Tensor,
                        position_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Safe RoPE application:
      - Uses Triton kernel when it is safe & beneficial
      - Otherwise falls back to the slow PyTorch version
    Shapes:
      Q,K: [B, S, H, D]
      cos,sin: preferably [S, D/2] or broadcastable to it
    """
    # If position_ids is used, we need per-token indexing -> Python path
    if position_ids is not None:
        return _slow_rope_apply(Q, K, cos, sin, position_ids)

    # Triton path only when safe
    if _can_use_triton(Q, cos, sin):
        try:
            Qout = _FastRoPEFn.apply(Q.to(torch.float32), cos, sin).to(Q.dtype)
            Kout = _FastRoPEFn.apply(K.to(torch.float32), cos, sin).to(K.dtype)
            return Qout, Kout
        except Exception as e:
            # Safety first: fallback if anything goes wrong
            # You can optionally log this.
            pass

    # Slow but correct
    return _slow_rope_apply(Q, K, cos, sin, position_ids=None)


@torch.compiler.disable
def fast_rope_embedding_key_only(K: torch.Tensor,
                                 cos: torch.Tensor,
                                 sin: torch.Tensor,
                                 position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
    if position_ids is not None or not _can_use_triton(K, cos, sin):
        return _slow_rope_apply_key_only(K, cos, sin, position_ids)
    try:
        return _FastRoPEFn.apply(K.to(torch.float32), cos, sin).to(K.dtype)
    except Exception:
        return _slow_rope_apply_key_only(K, cos, sin, position_ids=None)
