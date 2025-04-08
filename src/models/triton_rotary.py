## COPIED FROM UNSLOTH
## BUG HUNT: UNUSED. CURRENTLY `rotary.py` IS USED IN ITS PLACE BY model_arch.py
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import triton
MAX_FUSED_SIZE : int = 65536
next_power_of_2 = triton.next_power_of_2

# torch.cuda.amp.custom_fwd is deprecated >= 2.4
import torch
from packaging.version import Version
if Version(torch.__version__) < Version("2.4.0"):
    torch_amp_custom_fwd = torch.cuda.amp.custom_fwd
    torch_amp_custom_bwd = torch.cuda.amp.custom_bwd
else:
    torch_amp_custom_fwd = torch.amp.custom_fwd(device_type = "cuda")
    torch_amp_custom_bwd = torch.amp.custom_bwd(device_type = "cuda")
pass


# tl.math.tanh now is libdevice.tanh
from packaging.version import Version
import triton
import triton.language as tl
if Version(triton.__version__) >= Version("3.0.0"):
    from triton.language.extra import libdevice
    triton_tanh = libdevice.tanh
    triton_cast = tl.cast
else:
    triton_tanh = tl.math.tanh
    # No casting in old Triton versions
    @triton.jit
    def triton_cast(x, dtype):
        return x.to(dtype)
    pass
pass


def calculate_settings(n : int) -> (int, int,):
    BLOCK_SIZE : int = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch Triton kernel since n = {n} exceeds "\
                           f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}.")
    num_warps : int = 4
    if   BLOCK_SIZE >= 32768: num_warps = 32
    elif BLOCK_SIZE >=  8192: num_warps = 16
    elif BLOCK_SIZE >=  2048: num_warps = 8
    return BLOCK_SIZE, num_warps
pass


import bitsandbytes as bnb
# https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1330/files
HAS_CUDA_STREAM = Version(bnb.__version__) > Version("0.43.3")
global CUDA_STREAM
CUDA_STREAM = None
get_ptr = bnb.functional.get_ptr
import ctypes
cdequantize_blockwise_fp32      = bnb.functional.lib.cdequantize_blockwise_fp32
cdequantize_blockwise_fp16_nf4  = bnb.functional.lib.cdequantize_blockwise_fp16_nf4
cdequantize_blockwise_bf16_nf4  = bnb.functional.lib.cdequantize_blockwise_bf16_nf4
cgemm_4bit_inference_naive_fp16 = bnb.functional.lib.cgemm_4bit_inference_naive_fp16
cgemm_4bit_inference_naive_bf16 = bnb.functional.lib.cgemm_4bit_inference_naive_bf16


def QUANT_STATE(W):
    return getattr(W, "quant_state", None)
pass


def get_lora_parameters(proj):
    # For DPO or disabled adapters
    base_layer = (proj.base_layer if hasattr(proj, "base_layer") else proj)
    W = base_layer.weight

    if not hasattr(proj, "disable_adapters") or proj.disable_adapters or proj.merged:
        return W, QUANT_STATE(W), None, None, None
    pass

    active_adapter = proj.active_adapters[0] if \
        hasattr(proj, "active_adapters") else proj.active_adapter
    A = proj.lora_A [active_adapter].weight
    B = proj.lora_B [active_adapter].weight
    s = proj.scaling[active_adapter]
    return W, QUANT_STATE(W), A, B, s
pass


def get_lora_parameters_bias(proj):
    # For DPO or disabled adapters
    base_layer = (proj.base_layer if hasattr(proj, "base_layer") else proj)
    W = base_layer.weight
    bias = base_layer.bias

    if not hasattr(proj, "disable_adapters") or proj.disable_adapters or proj.merged:
        return W, QUANT_STATE(W), None, None, None, bias
    pass

    active_adapter = proj.active_adapters[0] if \
        hasattr(proj, "active_adapters") else proj.active_adapter
    A = proj.lora_A [active_adapter].weight
    B = proj.lora_B [active_adapter].weight
    s = proj.scaling[active_adapter]
    return W, QUANT_STATE(W), A, B, s, bias
pass


if HAS_CUDA_STREAM:
    def fast_dequantize(W, quant_state = None, out = None):
        if quant_state is None: return W
        if type(quant_state) is not list:
            # New quant_state as a class
            # https://github.com/TimDettmers/bitsandbytes/pull/763/files
            absmax     = quant_state.absmax
            shape      = quant_state.shape
            dtype      = quant_state.dtype
            blocksize  = quant_state.blocksize
            offset     = quant_state.offset
            state2     = quant_state.state2
            absmax2    = state2.absmax
            code2      = state2.code
            blocksize2 = state2.blocksize
        else:
            # Old quant_state as a list of lists
            absmax, shape, dtype, blocksize, compressed_stats, _, _ = quant_state
            offset, state2 = compressed_stats
            absmax2, code2, blocksize2, _, _, _, _ = state2
        pass
        global CUDA_STREAM
        if CUDA_STREAM is None: CUDA_STREAM = torch.cuda.current_stream("cuda:0")

        # Create weight matrix
        if out is None:
            out = torch.empty(shape, dtype = dtype, device = "cuda:0")
        else:
            assert(out.shape == shape)
            assert(out.dtype == dtype)

        # NF4 dequantization of statistics
        n_elements_absmax = absmax.numel()
        out_absmax = torch.empty(n_elements_absmax, dtype = torch.float32, device = "cuda:0")

        # Do dequantization
        ptr_out_absmax = get_ptr(out_absmax)
        cdequantize_blockwise_fp32(
            get_ptr(code2), get_ptr(absmax), get_ptr(absmax2), ptr_out_absmax,
            ctypes.c_int(blocksize2), ctypes.c_int(n_elements_absmax), CUDA_STREAM,
        )
        out_absmax += offset

        fx = cdequantize_blockwise_fp16_nf4 if dtype == torch.float16 else \
             cdequantize_blockwise_bf16_nf4
        fx(get_ptr(None), get_ptr(W), ptr_out_absmax, get_ptr(out),
           ctypes.c_int(blocksize), ctypes.c_int(out.numel()), CUDA_STREAM,)

        # Careful returning transposed data
        is_transposed = (True if W.shape[0] == 1 else False)
        return out.t() if is_transposed else out
    pass
else:
    def fast_dequantize(W, quant_state = None, out = None):
        if quant_state is None: return W
        if type(quant_state) is not list:
            # New quant_state as a class
            # https://github.com/TimDettmers/bitsandbytes/pull/763/files
            absmax     = quant_state.absmax
            shape      = quant_state.shape
            dtype      = quant_state.dtype
            blocksize  = quant_state.blocksize
            offset     = quant_state.offset
            state2     = quant_state.state2
            absmax2    = state2.absmax
            code2      = state2.code
            blocksize2 = state2.blocksize
        else:
            # Old quant_state as a list of lists
            absmax, shape, dtype, blocksize, compressed_stats, _, _ = quant_state
            offset, state2 = compressed_stats
            absmax2, code2, blocksize2, _, _, _, _ = state2
        pass

        # Create weight matrix
        if out is None:
            out = torch.empty(shape, dtype = dtype, device = "cuda:0")
        else:
            assert(out.shape == shape)
            assert(out.dtype == dtype)

        # NF4 dequantization of statistics
        n_elements_absmax = absmax.numel()
        out_absmax = torch.empty(n_elements_absmax, dtype = torch.float32, device = "cuda:0")

        # Do dequantization
        ptr_out_absmax = get_ptr(out_absmax)
        cdequantize_blockwise_fp32(
            get_ptr(code2), get_ptr(absmax), get_ptr(absmax2), ptr_out_absmax,
            ctypes.c_int(blocksize2), ctypes.c_int(n_elements_absmax),
        )
        out_absmax += offset

        fx = cdequantize_blockwise_fp16_nf4 if dtype == torch.float16 else \
             cdequantize_blockwise_bf16_nf4
        fx(get_ptr(None), get_ptr(W), ptr_out_absmax, get_ptr(out),
           ctypes.c_int(blocksize), ctypes.c_int(out.numel()),)

        # Careful returning transposed data
        is_transposed = (True if W.shape[0] == 1 else False)
        return out.t() if is_transposed else out
    pass
pass


if HAS_CUDA_STREAM:
    def fast_gemv(X, W, quant_state, out = None):
        if quant_state is None: return torch.matmul(X, W, out = out)
        # For fast X @ W where seq_len == 1
        # From https://github.com/TimDettmers/bitsandbytes/blob/main/bitsandbytes/functional.py#L1469
        _, q_len, hd = X.shape
        # assert(q_len == 1)

        if type(quant_state) is not list:
            # https://github.com/TimDettmers/bitsandbytes/pull/763/files
            absmax     = quant_state.absmax
            shape      = quant_state.shape
            dtype      = quant_state.dtype
            blocksize  = quant_state.blocksize
            stats      = quant_state.code
            offset     = quant_state.offset
            state2     = quant_state.state2
            absmax2    = state2.absmax
            code2      = state2.code
            blocksize2 = state2.blocksize
        else:
            absmax, shape, dtype, blocksize, compressed_stats, quant_type, stats = quant_state
            offset, state2 = compressed_stats
            absmax2, code2, blocksize2, _, _, _, _ = state2
        pass
        global CUDA_STREAM
        if CUDA_STREAM is None: CUDA_STREAM = torch.cuda.current_stream("cuda:0")
        
        # assert(dtype == X.dtype)
        bout = shape[0]

        if out is None:
            out = torch.empty((1, 1, bout,), dtype = dtype, device = "cuda:0")
        # else:
        #     assert(out.shape == (1, 1, bout,))
        # pass

        n = 1
        m = shape[0]
        k = shape[1]
        lda = shape[0]
        ldc = shape[0]
        ldb = (hd+1)//2
        m = ctypes.c_int32(m)
        n = ctypes.c_int32(n)
        k = ctypes.c_int32(k)
        lda = ctypes.c_int32(lda)
        ldb = ctypes.c_int32(ldb)
        ldc = ctypes.c_int32(ldc)

        df = torch.empty(absmax.shape, dtype = torch.float32, device = "cuda:0")
        cdequantize_blockwise_fp32(
            get_ptr(code2), get_ptr(absmax), get_ptr(absmax2), get_ptr(df),
            ctypes.c_int(blocksize2), ctypes.c_int(df.numel()), CUDA_STREAM,
        )
        df += offset
        absmax = df

        fx = cgemm_4bit_inference_naive_fp16 if dtype == torch.float16 else \
            cgemm_4bit_inference_naive_bf16

        blocksize = ctypes.c_int32(blocksize)
        fx(m, n, k, get_ptr(X), get_ptr(W), get_ptr(absmax), get_ptr(stats), get_ptr(out),
           lda, ldb, ldc, blocksize, CUDA_STREAM,)

        return out
    pass
else:
    def fast_gemv(X, W, quant_state, out = None):
        if quant_state is None: return torch.matmul(X, W, out = out)
        # For fast X @ W where seq_len == 1
        # From https://github.com/TimDettmers/bitsandbytes/blob/main/bitsandbytes/functional.py#L1469
        _, q_len, hd = X.shape
        # assert(q_len == 1)

        if type(quant_state) is not list:
            # https://github.com/TimDettmers/bitsandbytes/pull/763/files
            absmax     = quant_state.absmax
            shape      = quant_state.shape
            dtype      = quant_state.dtype
            blocksize  = quant_state.blocksize
            stats      = quant_state.code
            offset     = quant_state.offset
            state2     = quant_state.state2
            absmax2    = state2.absmax
            code2      = state2.code
            blocksize2 = state2.blocksize
        else:
            absmax, shape, dtype, blocksize, compressed_stats, quant_type, stats = quant_state
            offset, state2 = compressed_stats
            absmax2, code2, blocksize2, _, _, _, _ = state2
        pass
        # assert(dtype == X.dtype)
        bout = shape[0]

        if out is None:
            out = torch.empty((1, 1, bout,), dtype = dtype, device = "cuda:0")
        # else:
        #     assert(out.shape == (1, 1, bout,))
        # pass

        n = 1
        m = shape[0]
        k = shape[1]
        lda = shape[0]
        ldc = shape[0]
        ldb = (hd+1)//2
        m = ctypes.c_int32(m)
        n = ctypes.c_int32(n)
        k = ctypes.c_int32(k)
        lda = ctypes.c_int32(lda)
        ldb = ctypes.c_int32(ldb)
        ldc = ctypes.c_int32(ldc)

        df = torch.empty(absmax.shape, dtype = torch.float32, device = "cuda:0")
        cdequantize_blockwise_fp32(
            get_ptr(code2), get_ptr(absmax), get_ptr(absmax2), get_ptr(df),
            ctypes.c_int(blocksize2), ctypes.c_int(df.numel()),
        )
        df += offset
        absmax = df

        fx = cgemm_4bit_inference_naive_fp16 if dtype == torch.float16 else \
            cgemm_4bit_inference_naive_bf16

        blocksize = ctypes.c_int32(blocksize)
        fx(m, n, k, get_ptr(X), get_ptr(W), get_ptr(absmax), get_ptr(stats), get_ptr(out),
           lda, ldb, ldc, blocksize,)

        return out
    pass
pass


def fast_linear_forward(proj, X, temp_lora = None, out = None):

    W, W_quant, lora_A, lora_B, lora_S, bias = get_lora_parameters_bias(proj)
    bsz, q_len, in_dim = X.shape
    if q_len != 1: return matmul_lora(X, W, W_quant, lora_A, lora_B, lora_S)

    if W_quant is None:
        out = torch.matmul(X, W.t(), out = out)
    elif bsz == 1 and q_len == 1:
        out = fast_gemv(X, W, W_quant, out = out)
    else:
        W = fast_dequantize(W.t(), W_quant)
        out = torch.matmul(X, W, out = out)
    pass

    # Add in LoRA weights
    if lora_A is not None:
        out_dim = out.shape[2]
        dtype = X.dtype

        if not hasattr(lora_A, "_fast_lora"):
            lora_A._fast_lora = lora_A.to(dtype)
            lora_B._fast_lora = lora_B.to(dtype)
        pass
        
        if bsz == 1:
            out = out.view(out_dim)
            temp_lora = torch.mv(lora_A._fast_lora, X.ravel(), out = temp_lora)
            out.addmv_(lora_B._fast_lora, temp_lora, alpha = lora_S)
        else:
            out = out.view(bsz, out_dim)
            temp_lora = torch.mm(X.view(bsz, in_dim), lora_A._fast_lora.t(), out = temp_lora)
            out.addmm_(temp_lora, lora_B._fast_lora.t(), alpha = lora_S)
        pass
        out = out.view(bsz, 1, out_dim)
    pass

    if bias is not None: out += bias

    return out
pass


def matmul_lora(X, W, W_quant, A, B, s, out = None):
    dtype = X.dtype
    W = fast_dequantize(W.t(), W_quant)

    if X.dim() == 3:
        batch, seq_len, d = X.shape
        X = X.view(-1, X.shape[-1])
        reshape = True
    else:
        reshape = False
    pass

    out = torch.matmul(X, W, out = out)
    if W_quant is not None: del W

    if A is not None:
        # LoRA is enabled
        A, B = A.t(), B.t()
        out += (X @ A.to(dtype)) @ (s * B.to(dtype))
    pass
    
    return out.view(batch, seq_len, -1) if reshape else out
pass




import triton
import triton.language as tl
import torch
ROPE_GROUP_SIZE : int = 4

def _rope_embedding(
    Q,     Q_row_stride,
    cos, cos_row_stride,
    sin, sin_row_stride,
    seqlen,
    head_dim      : tl.constexpr,
    n_heads       : tl.constexpr,
    BACKWARD_PASS : tl.constexpr,
    BLOCK_SIZE    : tl.constexpr,
):
    """
        Calculates the RoPE Embedding quickly
        RoPE is Q * cos + rotate_half(Q) * sin
        See our blog post for more info
    """
    ROPE_GROUP_SIZE = 4
    row_position  = tl.program_id(0)
    group_head_position = tl.program_id(1)
    col_offsets  = tl.arange(0, BLOCK_SIZE)
    half_head_dim = head_dim // 2
    mask = col_offsets < half_head_dim

    sin1 = tl.load(sin + (row_position % seqlen)*sin_row_stride + \
                   half_head_dim*0 + col_offsets, mask = mask, other = 0)
    cos1 = tl.load(cos + (row_position % seqlen)*cos_row_stride + \
                   half_head_dim*0 + col_offsets, mask = mask, other = 0)

    if BACKWARD_PASS:
        # See our blog post for more info.
        sin1 = -sin1
    pass

    # [TODO] Autotune ROPE_GROUP_SIZE to be 1, 2, 4, 8
    head_start = group_head_position * ROPE_GROUP_SIZE
    head_end = min((head_start + ROPE_GROUP_SIZE), n_heads)

    # 10% Faster kernel from [HuyNguyen-hust](https://github.com/unslothai/unsloth/pull/238)
    for k in range(head_start, head_end):
        offs_q1 = row_position * Q_row_stride + k * head_dim + col_offsets
        offs_q2 = row_position * Q_row_stride + k * head_dim + col_offsets + half_head_dim

        # For Gemma - sometimes RoPE must be done in float32 and not bfloat16
        Q1 = tl.load(Q + offs_q1, mask = mask, other = 0).to(sin1.dtype)
        Q2 = tl.load(Q + offs_q2, mask = mask, other = 0).to(sin1.dtype)

        tl.store(Q + offs_q1, Q1*cos1 - Q2*sin1, mask = mask)
        tl.store(Q + offs_q2, Q2*cos1 + Q1*sin1, mask = mask)
    pass
pass
_rope_embedding = triton.jit(_rope_embedding)
_rope_embedding = triton.heuristics(
    {
        "BACKWARD_PASS": lambda args: bool(args["BACKWARD_PASS"]),
    }
)(_rope_embedding)


class Fast_RoPE_Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, cos, sin):
        cos, sin = cos.squeeze(), sin.squeeze()
        batch    : int
        seq_len  : int
        n_heads  : int
        head_dim : int
        batch, seq_len, n_heads, head_dim = Q.shape
        Q = Q.view(batch*seq_len, n_heads*head_dim)
        n_rows : int
        n_cols : int
        n_rows, n_cols = Q.shape
        assert(seq_len <= cos.shape[0])

        # [TODO] Changing blocksize to head_dim//2 seems to have
        # some concurrency / un-deterministic issues.
        BLOCK_SIZE, num_warps = calculate_settings(head_dim//2) # (head_dim//2)
        
        # group_size = 4 # 4 or 8, too large group_size can hurt performance.
        div : int
        mod : int
        div, mod = divmod(n_heads, ROPE_GROUP_SIZE)
        n_groups : int = div + (mod != 0)

        _rope_embedding[(n_rows, n_groups, )](
              Q,   Q.stride(0),
            cos, cos.stride(0),
            sin, sin.stride(0),
            seq_len,
            head_dim, n_heads,
            BACKWARD_PASS = False,
            BLOCK_SIZE = BLOCK_SIZE,
            num_warps  = num_warps,
        )
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps  = num_warps
        ctx.n_groups = n_groups
        ctx.cos = cos
        ctx.sin = sin
        return Q.view(batch, seq_len, n_heads, head_dim)
    pass

    @staticmethod
    def backward(ctx, dY):
        batch    : int
        seq_len  : int
        n_heads  : int
        head_dim : int
        batch, seq_len, n_heads, head_dim = dY.shape
        dY = dY.reshape(batch*seq_len, n_heads*head_dim)
        # Must be reshape not view
        n_rows : int
        n_cols : int
        n_rows, n_cols = dY.shape

        cos = ctx.cos
        sin = ctx.sin

        _rope_embedding[(n_rows, ctx.n_groups, )](
            dY,  dY .stride(0),
            cos, cos.stride(0),
            sin, sin.stride(0),
            seq_len, head_dim, n_heads,
            BACKWARD_PASS = True,
            BLOCK_SIZE = ctx.BLOCK_SIZE,
            num_warps  = ctx.num_warps,
        )
        dY = dY.view(batch, seq_len, n_heads, head_dim)
        return dY, None, None,
    pass
pass

# [TODO] Unsure why RoPE Embedding is not torch.compiling properly
@torch.compiler.disable
def fast_rope_embedding(Q, K, cos, sin):
    Q = Fast_RoPE_Embedding.apply(Q.to(torch.float32), cos, sin)
    K = Fast_RoPE_Embedding.apply(K.to(torch.float32), cos, sin)
    # Q = Fast_RoPE_Embedding.apply(Q.transpose(1, 2), cos, sin).transpose(1, 2)
    # K = Fast_RoPE_Embedding.apply(K.transpose(1, 2), cos, sin).transpose(1, 2)
    return Q, K
pass


# [TODO] Unsure why RoPE Embedding is not torch.compiling properly
@torch.compiler.disable
def fast_rope_embedding_key_only(K, cos, sin):
    K = Fast_RoPE_Embedding.apply(K, cos, sin)
    # Q = Fast_RoPE_Embedding.apply(Q.transpose(1, 2), cos, sin).transpose(1, 2)
    # K = Fast_RoPE_Embedding.apply(K.transpose(1, 2), cos, sin).transpose(1, 2)
    return K
pass


class Slow_RoPE_Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, cos, sin, position_ids):
        if position_ids is not None:
            # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
            cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
            sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
            cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
            sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]

        # Q * cos + rotate_half(Q) * sin
        half = Q.shape[-1]//2
        RH_Q = torch.cat((-Q[..., half:], Q[..., :half]), dim = -1)
        Q *= cos
        Q.addcmul_(RH_Q, sin)
        # RH_Q *= sin
        # Q += RH_Q
        ctx.save_for_backward(cos, sin)
        return Q
    pass

    @staticmethod
    def backward(ctx, dY):
        cos, sin = ctx.saved_tensors
        # Q * cos + rotate_half.T(Q) * sin
        half = dY.shape[-1]//2
        RH_dY = torch.cat((dY[..., half:], -dY[..., :half]), dim = -1)
        dY *= cos
        dY.addcmul_(RH_dY, sin)
        # RH_dY *= sin
        # dY += RH_dY
        return dY, None, None, None
    pass
pass


def inplace_rope_embedding(Q, K, cos, sin, position_ids):
    Q = Slow_RoPE_Embedding.apply(Q, cos, sin, position_ids)
    K = Slow_RoPE_Embedding.apply(K, cos, sin, position_ids)
    return Q, K
pass