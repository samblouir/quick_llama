"""
===============================================================================
BASE MODEL
-------------------------------------
Supports:
  - Rotary embeddings
  - GQA (Group Query Attention) (set (gqa_num_heads < num_heads) and (gqa_num_heads % num_heads == 0))
  - RMSNorm
  - (Optional) fused cross-entropy that does not materialize logits
  - Segment-aware block mask
  - FAN-in (as seen in JAX, similar to OLMO 2) param inits
===============================================================================
"""

from transformers import AutoModelForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
import math
from typing import Optional, Any
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import birdie_dna.utils

import einops
import torch.nn.attention.flex_attention as flex_attention
from torch.nn.attention.flex_attention import create_block_mask
from birdie_dna.modeling import rotary, softcap
from torch.utils.checkpoint import checkpoint
from birdie_dna.modeling.rope_embedding import fast_rope_embedding
import logging
from safetensors.torch import load_file as load_safetensors_file
import torch

prenorm = 1

torch_compile_options = {}
from torch.nn.attention.flex_attention import (
    flex_attention as _flex_attention,
)

_flex_attention = torch.compile(
    _flex_attention, dynamic=True, options=torch_compile_options
)


def load_model_from_safetensors(
    config: dict, safetensors_path: str, device: str = "cpu"
):
    """
    Instantiates BaseModel and loads weights from a safetensors file.

    Args:
        config (dict): Configuration dictionary compatible with BaseModel.
        safetensors_path (str): Path to the .safetensors file.
        device (str): Device to load the model onto initially ("cpu" recommended).

    Returns:
        BaseModel: The loaded model instance.
    """
    required_keys = [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "vocab_size",
    ]
    if not all(key in config for key in required_keys):
        missing = [key for key in required_keys if key not in config]
        logging.warning(
            f"Config dict missing required keys for BaseModel: {missing}. Using defaults or expecting errors."
        )
        # Add defaults here if appropriate, e.g., config.setdefault('vocab_size', 128256)

    logging.info(f"Instantiating BaseModel...")
    # BaseModel expects config keys directly as layer_kwargs
    model = BaseModel(layer_kwargs=config)

    logging.info(f"Loading weights from safetensors file: {safetensors_path}")
    # Load the state dictionary from the safetensors file
    state_dict = load_safetensors_file(safetensors_path, device=device)

    # Load the state dict into the model
    try:
        # Set strict=False initially, as buffers like freqs_cis might not be in the checkpoint
        load_result = model.load_state_dict(state_dict, strict=False)
        logging.info(f"Safetensors load result (strict=False): {load_result}")
        if load_result.missing_keys:
            # Re-check if missing keys are critical parameters or just buffers/non-persistent items
            critical_missing = [
                k for k in load_result.missing_keys if not k.endswith("freqs_cis")
            ]  # Example filter
            if critical_missing:
                logging.error(f"CRITICAL Missing keys during safetensors load: {critical_missing}")
                raise RuntimeError(f"Critical keys missing: {critical_missing}") # Optional: Fail hard
            else:
                logging.warning(f"Non-critical missing keys (potentially buffers): {load_result.missing_keys}")
                raise RuntimeError(f"Missing keys found: {load_result.missing_keys}")

        if load_result.unexpected_keys:
            logging.warning(
                f"Unexpected keys found in checkpoint: {load_result.unexpected_keys}"
            )
            raise RuntimeError(f"Unexpected keys found: {load_result.unexpected_keys}")

    except Exception as e:
        logging.error(f"Error loading state_dict from {safetensors_path}: {e}")
        raise

    # Return the model (likely on CPU), convert dtype after loading
    # The original load_model converted to bfloat16 at the end
    model = model.to(torch.bfloat16)
    logging.info("Model loaded from safetensors and converted to bfloat16.")
    return model


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        dtype: torch.dtype = torch.float32,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        torch_dtype = birdie_dna.utils.str_to_dtype(dtype)
        self.norm = nn.RMSNorm(
            hidden_size,
            eps=eps,
            elementwise_affine=True,
            dtype=torch_dtype,
            device=device,
        )

    def forward(
        self,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return self.norm(x)


class LinearProjection(nn.Module):
    """
    A single linear layer with forced dimension alignment (dims are made to be divisible by 128)
    Uses truncated normal initialization for weights.
    """

    def __init__(self, in_dim: int = None, out_dim: int = None, **kwargs: Any):
        super().__init__()

        if in_dim is None:
            in_dim = kwargs["hidden_size"]
        if out_dim is None:
            out_dim = kwargs.get("out_dim", in_dim)

        is_vocab_head = kwargs.get("is_vocab_head", False)
        vocab_size = kwargs.get("vocab_size", 32000)
        if is_vocab_head:
            out_dim = vocab_size

        param_dtype = kwargs.get("dtype", torch.float32)

        in_dim = birdie_dna.utils.make_divisible_by(in_dim, 128)
        out_dim = birdie_dna.utils.make_divisible_by(out_dim, 128)

        param_dtype = birdie_dna.utils.str_to_dtype(param_dtype)

        self.layer = nn.Linear(
            in_dim,
            out_dim,
            bias=kwargs.get("projection_layers_use_bias", False),
            dtype=param_dtype,
        )

        fan_in = in_dim
        std = 1.0 / math.sqrt(fan_in)
        nn.init.trunc_normal_(
            self.layer.weight, mean=0.0, std=std, a=-2 * std, b=2 * std
        )

    def forward(
        self,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return self.layer(x)


class self_attn(nn.Module):
    """
    Custom MHA that:
      - Splits Q,K,V
      - Applies flex_attention
      - Optionally uses post-attention RMSNorm
      - Applies rotary embeddings if provided
      - Supports GQA via gqa_num_heads
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.hidden_size = kwargs["hidden_size"]
        self.num_heads = kwargs["num_attention_heads"]
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_heads)
        self.gqa_num_heads = int(kwargs.get("num_key_value_heads", 8))

        qk_dims = self.num_heads * self.head_dim

        v_dims = self.gqa_num_heads * self.head_dim

        self.q_proj = LinearProjection(self.hidden_size, qk_dims, **kwargs)
        self.k_proj = LinearProjection(self.hidden_size, v_dims, **kwargs)
        self.v_proj = LinearProjection(self.hidden_size, v_dims, **kwargs)

        self.o_proj = LinearProjection(qk_dims, self.hidden_size, **kwargs)

        self.enable_gqa = self.gqa_num_heads != self.num_heads
        print(
            f"  self.num_heads: {self.num_heads}, self.gqa_num_heads: {self.gqa_num_heads}, enable_gqa: {self.enable_gqa}"
        )
        if self.enable_gqa:
            assert (
                self.num_heads % self.gqa_num_heads == 0
            ), "gqa_num_heads must be a multiple of num_heads"
            assert (
                self.gqa_num_heads < self.num_heads
            ), "gqa_num_heads must be less than num_heads"

    # @torch.compile
    def forward(
        self,
        x: torch.Tensor,
        block_mask=None,
        freqs_cis=None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for the MHA block.
        """
        residual = x

        # x = self.pre_rms_norm(x)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = einops.rearrange(q, "B S (H D) -> B S H D", H=self.num_heads)
        k = einops.rearrange(k, "B S (H D) -> B S H D", H=self.gqa_num_heads)

        # q, k = fast_rope_embedding(q, k, freqs_cis.real, freqs_cis.imag)
        (q, k) = rotary.apply_rotary_emb(q, k, freqs_cis)

        q = einops.rearrange(q, "B S H D -> B H S D", H=self.num_heads)
        k = einops.rearrange(k, "B S H D -> B H S D", H=self.gqa_num_heads)
        v = einops.rearrange(v, "B S (H D) -> B H S D", H=self.gqa_num_heads)

        attn_out = _flex_attention(
            query=q,
            key=k,
            value=v,
            block_mask=block_mask,
            enable_gqa=self.enable_gqa,
        )

        attn_out = einops.rearrange(attn_out, "B H S D -> B S (H D)", H=self.num_heads)

        out = self.o_proj(attn_out)
        # out = self.post_rms_norm(out)

        return (residual + out).to(x.dtype)


class mlp(nn.Module):
    """
    A feed-forward block using the 'SwiGLU' pattern:
      - gate = sigmoid(Linear(x))
      - ungated = Linear(x)
      - multiply gate * ungated
      - project down
      - RMSNorm
      - add residual
    """

    def __init__(self, **kwargs):
        super().__init__()
        hidden_size = kwargs["hidden_size"]
        mlp_mult = kwargs.get("mlp_dim_mult", 4.0)

        in_dim = hidden_size
        ffn_dim = int(in_dim * mlp_mult)

        self.gate_proj = LinearProjection(in_dim, ffn_dim, **kwargs)
        self.up_proj = LinearProjection(in_dim, ffn_dim, **kwargs)
        self.down_proj = LinearProjection(ffn_dim, in_dim, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass of SwiGLU feed-forward.
        """
        residual = x
        gated = torch.sigmoid(self.gate_proj(x))
        ungated = self.up_proj(x)
        ff_out = self.down_proj(gated * ungated)
        return ff_out + residual


class DecoderLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.input_layernorm = RMSNorm(
            **kwargs,
            # hidden_size=self.hidden_size,
            # eps=kwargs.get("eps", 1e-5),
            # dtype=kwargs.get("dtype", torch.float32),
            # device=kwargs.get("device", None),
        )

        self.post_attention_layernorm = RMSNorm(
            **kwargs,
            # hidden_size=self.hidden_size,
            # eps=kwargs.get("eps", 1e-5),
            # dtype=kwargs.get("dtype", torch.float32),
            # device=kwargs.get("device", None),
        )
        self.self_attn = self_attn(**kwargs)
        self.mlp = mlp(**kwargs)

    def forward(
        self,
        x: torch.Tensor,
        block_mask=None,
        freqs_cis=None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for the decoder layer.
        """
        residual = x

        x = self.input_layernorm(x)

        x = self.self_attn(x, block_mask=block_mask, freqs_cis=freqs_cis)

        x = self.post_attention_layernorm(x)

        x = self.mlp(x)

        return (residual + x).to(x.dtype)


class Embedding(nn.Embedding):
    """
    Wrapper to allow for *args and **kwargs.
    """

    def forward(
        self,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return super().forward(x)


class Dropout(nn.Dropout):
    """
    Wrapper to allow for *args and **kwargs.
    """

    def forward(
        self,
        x: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return super().forward(x)


class ScaledForwardNoGradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        # Store the scale factor for use in the backward pass
        ctx.scale = scale
        # Scale the input in the forward pass
        return input * scale

    @staticmethod
    def backward(ctx, grad_output):
        # Unscale the gradient in the backward pass
        return grad_output / ctx.scale, None


class BaseModel(nn.Module):
    """
    A flexible Transformer-like model that:
      1) Has an embedding layer (vocab_size x hidden_size).
      2) Stacks MHA + MLP layers (with optional RMSNorm, GQA, rotary, etc.).
      3) Ends with a final RMSNorm, and has a projection to vocab_size (assuming we're doing LM).

    If label_ids is provided, returns cross-entropy loss. Otherwise returns logits.
    """

    def __init__(self, layer_kwargs):
        super().__init__()

        self.num_layers = layer_kwargs.get("num_hidden_layers", 16)
        self.hidden_size = layer_kwargs.get("hidden_size", 2048)
        self.vocab_size = layer_kwargs.get("vocab_size", 32000)
        self.sequence_length = layer_kwargs.get("sequence_length", 512)
        self.batch_size = layer_kwargs.get("batch_size", 1)

        self.num_heads = layer_kwargs.get("num_attention_heads", 32)

        self.head_dim = layer_kwargs.get("head_dim", self.hidden_size // self.num_heads)

        self.use_precomputed_block_mask = int(
            layer_kwargs.get("use_precomputed_block_mask", 0)
        )
        self.use_fusedlce = int(layer_kwargs.get("use_fusedlce", 1))
        self.bidirectional = int(layer_kwargs.get("bidirectional", 0))
        self.sliding_window = int(layer_kwargs.get("sliding_window", 0))
        self.window_size = int(layer_kwargs.get("window_size", 4096))

        layer_kwargs["hidden_size"] = self.hidden_size
        layer_kwargs["num_attention_heads"] = self.num_heads
        layer_kwargs["num_hidden_layers"] = self.num_layers
        layer_kwargs["num_key_value_heads"] = layer_kwargs.get("num_key_value_heads", 8)
        layer_kwargs["sequence_length"] = self.sequence_length
        layer_kwargs["batch_size"] = self.batch_size

        if int(layer_kwargs.get("use_embeddings", 1)):
            self.embeddings = Embedding(self.vocab_size, self.hidden_size)
            fan_in = self.hidden_size
            std = 1.0 / math.sqrt(fan_in)
            nn.init.trunc_normal_(
                self.embeddings.weight, mean=0.0, std=std, a=-2 * std, b=2 * std
            )
        else:
            self.embeddings = LinearProjection(
                int(layer_kwargs.get("input_dims", 8)),
                self.hidden_size,
                **layer_kwargs,
            )

        freqs_cis = rotary.precompute_freqs_cis(
            dim=(self.head_dim),
            # end=8192,
            end=self.sequence_length,
            theta=layer_kwargs.get("base_decay_rate", 500_000),
            use_scaled=True,
            # old_context_length=layer_kwargs.get("pretraining_sequence_length", self.sequence_length)
            old_context_length=8192,
        )

        self.register_buffer(
            "freqs_cis",
            freqs_cis,
            persistent=False,
        )

        embed_dropout = layer_kwargs.get("embed_dropout", 0.0)
        residual_dropout = layer_kwargs.get("residual_dropout", 0.0)

        layers = []
        seen_layers = 0
        while seen_layers < self.num_layers:
            if layer_kwargs.get("use_attention", True):
                layers.append(
                    DecoderLayer(
                        **layer_kwargs, freqs_cis=freqs_cis, layer_idx=seen_layers
                    )
                )
                # mha = self_attn(
                # 	**layer_kwargs,
                # 	freqs_cis=self.freqs_cis,
                # )
                # layers.append(mha)
                # ffn = mlp(**layer_kwargs)
                # layers.append(ffn)
                seen_layers += 1

        head_in_dim = birdie_dna.utils.make_divisible_by(self.hidden_size, 128)
        head_out_dim = self.vocab_size
        self.lm_head = nn.Parameter(
            torch.randn(head_out_dim, head_in_dim), requires_grad=True
        )
        fan_in_head = head_out_dim
        std_head = 1.0 / math.sqrt(fan_in_head)
        nn.init.trunc_normal_(
            self.lm_head, mean=0.0, std=std_head, a=-2 * std_head, b=2 * std_head
        )

        self.norm = RMSNorm(
            hidden_size=self.hidden_size,
            eps=layer_kwargs.get("eps", 1e-5),
            dtype=layer_kwargs.get("dtype", torch.float32),
            device=layer_kwargs.get("device", None),
        )

        if self.use_fusedlce:
            from cut_cross_entropy import LinearCrossEntropy

            self.LCE = LinearCrossEntropy()

            none_loss = LinearCrossEntropy(reduction='none')
            def lce_noner(*args, **kwargs):
                return none_loss(*args, **kwargs).mean(dim=-1)
            self.LCE_none = lce_noner

        self.layers = nn.ModuleList()
        # self.layers.append(self.embeddings)
        if 0.0 < embed_dropout:
            self.layers.append(Dropout(p=embed_dropout, inplace=True))
        self.layers.extend(layers)

        if self.use_precomputed_block_mask:

            def mask_mod(b, h, q_idx, kv_idx):
                return q_idx >= kv_idx

            self.block_mask = create_block_mask(
                mask_mod,
                B=self.batch_size,
                H=1,
                Q_LEN=self.sequence_length,
                KV_LEN=self.sequence_length,
                device=layer_kwargs.get("device", "cuda"),
                _compile=True,
            )
        else:
            self.block_mask = None

        def cross_entropy_per_sample(logits, label_ids):
            """
            logits: (B, L, vocab_size)
            label_ids: (B, L) with -100 to ignore
            Returns per-sample average, shape (B,)
            """
            logits_t = logits.permute(0, 2, 1)
            label_ids_ = label_ids.to(torch.long)
            loss_per_pos = F.cross_entropy(logits_t, label_ids_, reduction="none")
            mask = label_ids_ != -100
            sum_loss = (loss_per_pos * mask).sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1)
            return sum_loss / count

        self.cross_entropy_per_sample = cross_entropy_per_sample

    def update_freqs_cis(self, freqs_cis):
        try:
            self.register_buffer(
                "freqs_cis",
                freqs_cis,
                persistent=False,
            )
        except RuntimeError:
            self.freqs_cis = freqs_cis
        print(f"  Updated model.freqs_cis.shape to {self.freqs_cis.shape}")

        for layer in self.layers:
            if isinstance(layer, self_attn):
                try:
                    layer.register_buffer(
                        "freqs_cis",
                        freqs_cis,
                        persistent=False,
                    )
                except RuntimeError:
                    layer.freqs_cis = freqs_cis

    def reset_freq_cis(
        self,
        seq_len: int,
        base_decay_rate: float = 500_000,
        old_context_length: int = None,
        accelerator=None,
    ):
        """
        Recompute rotary embeddings if sequence length changes.
        """
        if old_context_length is None:
            old_context_length = seq_len

        self.sequence_length = seq_len
        freqs_cis = rotary.precompute_freqs_cis(
            dim=(self.head_dim),
            end=seq_len,
            theta=base_decay_rate,
            use_scaled=(old_context_length < seq_len),
            old_context_length=old_context_length,
            accelerator=accelerator,
        )
        try:
            self.register_buffer(
                "freqs_cis",
                freqs_cis,
                persistent=False,
            )
        except RuntimeError:
            self.freqs_cis = freqs_cis
        print(f"  Updated model.freqs_cis.shape to {self.freqs_cis.shape}")

        for layer in self.layers:
            if isinstance(layer, self_attn):
                try:
                    layer.register_buffer(
                        "freqs_cis",
                        freqs_cis,
                        persistent=False,
                    )
                except RuntimeError:
                    layer.freqs_cis = freqs_cis

        print("After reset_freq_cis, model.freqs_cis.shape =", self.freqs_cis.shape)
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, self_attn):
                assert layer.freqs_cis.shape == self.freqs_cis.shape
                assert layer.freqs_cis.shape[0] == seq_len

        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        label_ids: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_per_sample_loss: bool = False,
        softmax_temperature: float = 1.0,
        reduction=None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass. If label_ids provided, return scalar cross-entropy. Otherwise logits.
        """


        x = input_ids
        
        # B, L = input_ids.shape[:2]
        # if x.shape[1] < self.sequence_length:


        #     zeros = torch.zeros(
        #                 (x.shape[0], self.sequence_length - x.shape[1]),
        #                 dtype=torch.long,
        #                 device=x.device,
        #             )
        #     # print(f"  input_ids.shape: {input_ids.shape},  zeros.shape: {zeros.shape}")
            
        #     # if segment_ids is None:
        #     segment_ids = torch.cat(
        #         [
        #             torch.ones_like(input_ids),
        #             zeros,
        #         ],
        #         dim=1,
        #     )

        #     x = torch.cat(
        #         [
        #             x,
        #             zeros,
        #         ],
        #         dim=1,
        #     )

        #     label_ids = torch.cat(
        #         [
        #             label_ids,
        #             zeros - 100,
        #         ],
        #         dim=1,
        #     )

            

        B, L = x.shape[:2]
        # if segment_ids is None:
        #     segment_ids = torch.ones_like(
        #         x, dtype=torch.long, device=x.device
        #     )
            

        def mask_mod(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            segment_mask = segment_ids[b, q_idx] == segment_ids[b, kv_idx]
            return causal_mask & segment_mask

        block_mask = create_block_mask(
            mask_mod,
            B=B,
            H=1,
            Q_LEN=L,
            KV_LEN=L,
            device=x.device,
            _compile=True,
        )

        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x, block_mask=block_mask, freqs_cis=self.freqs_cis)
        x = self.norm(x)

        if label_ids is None:
            B, L, D = x.shape
            logits = torch.matmul(x.view(-1, D), self.lm_head.to(x.dtype))
            logits = logits.view(B, L, self.vocab_size)
            return logits

        # if self.use_fusedlce:
        if True:
            logits_16 = x.to(torch.float16)
            w_16 = self.lm_head.to(torch.float16)

            if reduction is None:
                loss_fn = self.LCE
            else: # mean: 21.4 negative, 27.6 positive, 28.2 norm
                    

                loss_fn = self.LCE_none

            # if softmax_temperature == 1.0:
            #     return loss_fn(logits_16, self.lm_head, label_ids).to(torch.float32)
            scaled_weights = ScaledForwardNoGradScale.apply(self.lm_head, softmax_temperature)
            loss = loss_fn(logits_16.to(torch.float16), scaled_weights.to(torch.float16), label_ids)
            # print(f"  logits_16.shape: {logits_16.shape},  w_16.shape: {w_16.shape},  label_ids.shape: {label_ids.shape},  loss.shape: {loss.shape}")
            # exit()
            return loss.to(torch.float32)

            loss = self.LCE(logits_16, w_16, label_ids)
            if return_per_sample_loss:
                return loss
            return loss.mean()

        x = x.to(torch.bfloat16)
        logits = torch.matmul(x.view(-1, x.shape[-1]), self.lm_head.to(x.dtype))
        logits = logits.view(B, L, self.vocab_size)
        per_sample_loss = self.cross_entropy_per_sample(logits, label_ids)
        if return_per_sample_loss:
            return per_sample_loss
        return per_sample_loss.mean()


def _load_model(
    target_model,
    config,
):
    instruct = config.get("instruct", False)

    if instruct:
        source_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct"
        )
    else:
        source_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

    _config = LlamaConfig(
        vocab_size=128256,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=16,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=128000,
        eos_token_id=128001,
        pretraining_tp=1,
        tie_word_embeddings=True,
        rope_theta=500000.0,
        rope_scaling={
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        },
        attention_bias=False,
    )

    # target_model = CausalLlama(config=_config, use_fusedlce=True, **config)

    # Copy embeddings
    target_model.embeddings.weight.data = (
        source_model.model.embed_tokens.weight.data.clone()
    )

    # For each layer
    for i in range(len(source_model.model.layers)):
        # Copy attention weights
        target_i = i
        target_model.layers[
            target_i
        ].self_attn.q_proj.layer.weight.data = source_model.model.layers[
            i
        ].self_attn.q_proj.weight.data.clone()
        target_model.layers[
            target_i
        ].self_attn.k_proj.layer.weight.data = source_model.model.layers[
            i
        ].self_attn.k_proj.weight.data.clone()
        target_model.layers[
            target_i
        ].self_attn.v_proj.layer.weight.data = source_model.model.layers[
            i
        ].self_attn.v_proj.weight.data.clone()
        target_model.layers[
            target_i
        ].self_attn.o_proj.layer.weight.data = source_model.model.layers[
            i
        ].self_attn.o_proj.weight.data.clone()
        target_model.layers[
            target_i
        ].mlp.gate_proj.layer.weight.data = source_model.model.layers[
            i
        ].mlp.gate_proj.weight.data.clone()
        target_model.layers[
            target_i
        ].mlp.up_proj.layer.weight.data = source_model.model.layers[
            i
        ].mlp.up_proj.weight.data.clone()
        target_model.layers[
            target_i
        ].mlp.down_proj.layer.weight.data = source_model.model.layers[
            i
        ].mlp.down_proj.weight.data.clone()
        target_model.layers[
            target_i
        ].input_layernorm.norm.weight.data = source_model.model.layers[
            i
        ].input_layernorm.weight.data.clone()
        target_model.layers[
            target_i
        ].post_attention_layernorm.norm.weight.data = source_model.model.layers[
            i
        ].post_attention_layernorm.weight.data.clone()

    # Copy final norm and lm_head
    target_model.norm.norm.weight.data = source_model.model.norm.weight.data.clone()
    target_model.lm_head.data = source_model.lm_head.weight.data.clone()

    del source_model
    return target_model.to(torch.bfloat16)


def load_model(config):
    model = BaseModel(config)
    # source_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct").model
    model = _load_model(model, config)
    return model


if __name__ == "__main__":
    config = dict(
        vocab_size=128256,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=16,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=128000,
        eos_token_id=128001,
        pretraining_tp=1,
        tie_word_embeddings=True,
        rope_theta=500000.0,
        rope_scaling={
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        },
        attention_bias=False,
    )

    expected = """Layer 14:
  LlamaDecoderLayer
  self_attn.q_proj.weight: torch.Size([2048, 2048])
  self_attn.k_proj.weight: torch.Size([512, 2048])
  self_attn.v_proj.weight: torch.Size([512, 2048])
  self_attn.o_proj.weight: torch.Size([2048, 2048])
  mlp.gate_proj.weight: torch.Size([8192, 2048])
  mlp.up_proj.weight: torch.Size([8192, 2048])
  mlp.down_proj.weight: torch.Size([2048, 8192])
  input_layernorm.weight: torch.Size([2048])
  post_attention_layernorm.weight: torch.Size([2048])
""".strip()

    actual = """Layer 16:
  DecoderLayer
  input_layernorm.norm.weight: torch.Size([2048])
  post_attention_layernorm.norm.weight: torch.Size([2048])
  self_attn.q_proj.layer.weight: torch.Size([2048, 2048])
  self_attn.k_proj.layer.weight: torch.Size([512, 2048])
  self_attn.v_proj.layer.weight: torch.Size([512, 2048])
  self_attn.o_proj.layer.weight: torch.Size([2048, 2048])
  mlp.gate_proj.layer.weight: torch.Size([8192, 2048])
  mlp.up_proj.layer.weight: torch.Size([8192, 2048])
  mlp.down_proj.layer.weight: torch.Size([2048, 8192])
  mlp.input_layernorm.norm.weight: torch.Size([2048])
""".strip()

    model = BaseModel(config)
    source_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct"
    ).model

    model = load_model(model, config)

    for i in range(len(model.layers)):
        print(f"Layer {i}:")
        print(f"  {model.layers[i].__class__.__name__}")
        param_shapes = []
        for name, param in model.layers[i].named_parameters():
            if param.requires_grad:
                param_shapes.append(param.shape)
                # print std and mean
                print(
                    f"  {name}: {param.shape}, std: {param.std()}, mean: {param.mean()}"
                )

    # for name, param in model.layers[i].named_parameters():
    # 	# if param.requires_grad:
    # 	print(f"  {name}: {param.shape}")

llama_layers = """
Layer 0:
  LlamaDecoderLayer
  self_attn.q_proj.weight: torch.Size([2048, 2048])
  self_attn.k_proj.weight: torch.Size([512, 2048])
  self_attn.v_proj.weight: torch.Size([512, 2048])
  self_attn.o_proj.weight: torch.Size([2048, 2048])
  mlp.gate_proj.weight: torch.Size([8192, 2048])
  mlp.up_proj.weight: torch.Size([8192, 2048])
  mlp.down_proj.weight: torch.Size([2048, 8192])
  input_layernorm.weight: torch.Size([2048])
  post_attention_layernorm.weight: torch.Size([2048])
Layer 1:
  LlamaDecoderLayer
  self_attn.q_proj.weight: torch.Size([2048, 2048])
  self_attn.k_proj.weight: torch.Size([512, 2048])
  self_attn.v_proj.weight: torch.Size([512, 2048])
  self_attn.o_proj.weight: torch.Size([2048, 2048])
  mlp.gate_proj.weight: torch.Size([8192, 2048])
  mlp.up_proj.weight: torch.Size([8192, 2048])
  mlp.down_proj.weight: torch.Size([2048, 8192])
  input_layernorm.weight: torch.Size([2048])
  post_attention_layernorm.weight: torch.Size([2048])
Layer 2:
  LlamaDecoderLayer
  self_attn.q_proj.weight: torch.Size([2048, 2048])
  self_attn.k_proj.weight: torch.Size([512, 2048])
  self_attn.v_proj.weight: torch.Size([512, 2048])
  self_attn.o_proj.weight: torch.Size([2048, 2048])
  mlp.gate_proj.weight: torch.Size([8192, 2048])
  mlp.up_proj.weight: torch.Size([8192, 2048])
  mlp.down_proj.weight: torch.Size([2048, 8192])
  input_layernorm.weight: torch.Size([2048])
  post_attention_layernorm.weight: torch.Size([2048])
Layer 3:
  LlamaDecoderLayer
  self_attn.q_proj.weight: torch.Size([2048, 2048])
  self_attn.k_proj.weight: torch.Size([512, 2048])
  self_attn.v_proj.weight: torch.Size([512, 2048])
  self_attn.o_proj.weight: torch.Size([2048, 2048])
  mlp.gate_proj.weight: torch.Size([8192, 2048])
  mlp.up_proj.weight: torch.Size([8192, 2048])
  mlp.down_proj.weight: torch.Size([2048, 8192])
  input_layernorm.weight: torch.Size([2048])
  post_attention_layernorm.weight: torch.Size([2048])
Layer 4:
  LlamaDecoderLayer
  self_attn.q_proj.weight: torch.Size([2048, 2048])
  self_attn.k_proj.weight: torch.Size([512, 2048])
  self_attn.v_proj.weight: torch.Size([512, 2048])
  self_attn.o_proj.weight: torch.Size([2048, 2048])
  mlp.gate_proj.weight: torch.Size([8192, 2048])
  mlp.up_proj.weight: torch.Size([8192, 2048])
  mlp.down_proj.weight: torch.Size([2048, 8192])
  input_layernorm.weight: torch.Size([2048])
  post_attention_layernorm.weight: torch.Size([2048])
Layer 5:
  LlamaDecoderLayer
  self_attn.q_proj.weight: torch.Size([2048, 2048])
  self_attn.k_proj.weight: torch.Size([512, 2048])
  self_attn.v_proj.weight: torch.Size([512, 2048])
  self_attn.o_proj.weight: torch.Size([2048, 2048])
  mlp.gate_proj.weight: torch.Size([8192, 2048])
  mlp.up_proj.weight: torch.Size([8192, 2048])
  mlp.down_proj.weight: torch.Size([2048, 8192])
  input_layernorm.weight: torch.Size([2048])
  post_attention_layernorm.weight: torch.Size([2048])
Layer 6:
  LlamaDecoderLayer
  self_attn.q_proj.weight: torch.Size([2048, 2048])
  self_attn.k_proj.weight: torch.Size([512, 2048])
  self_attn.v_proj.weight: torch.Size([512, 2048])
  self_attn.o_proj.weight: torch.Size([2048, 2048])
  mlp.gate_proj.weight: torch.Size([8192, 2048])
  mlp.up_proj.weight: torch.Size([8192, 2048])
  mlp.down_proj.weight: torch.Size([2048, 8192])
  input_layernorm.weight: torch.Size([2048])
  post_attention_layernorm.weight: torch.Size([2048])
Layer 7:
  LlamaDecoderLayer
  self_attn.q_proj.weight: torch.Size([2048, 2048])
  self_attn.k_proj.weight: torch.Size([512, 2048])
  self_attn.v_proj.weight: torch.Size([512, 2048])
  self_attn.o_proj.weight: torch.Size([2048, 2048])
  mlp.gate_proj.weight: torch.Size([8192, 2048])
  mlp.up_proj.weight: torch.Size([8192, 2048])
  mlp.down_proj.weight: torch.Size([2048, 8192])
  input_layernorm.weight: torch.Size([2048])
  post_attention_layernorm.weight: torch.Size([2048])
Layer 8:
  LlamaDecoderLayer
  self_attn.q_proj.weight: torch.Size([2048, 2048])
  self_attn.k_proj.weight: torch.Size([512, 2048])
  self_attn.v_proj.weight: torch.Size([512, 2048])
  self_attn.o_proj.weight: torch.Size([2048, 2048])
  mlp.gate_proj.weight: torch.Size([8192, 2048])
  mlp.up_proj.weight: torch.Size([8192, 2048])
  mlp.down_proj.weight: torch.Size([2048, 8192])
  input_layernorm.weight: torch.Size([2048])
  post_attention_layernorm.weight: torch.Size([2048])
Layer 9:
  LlamaDecoderLayer
  self_attn.q_proj.weight: torch.Size([2048, 2048])
  self_attn.k_proj.weight: torch.Size([512, 2048])
  self_attn.v_proj.weight: torch.Size([512, 2048])
  self_attn.o_proj.weight: torch.Size([2048, 2048])
  mlp.gate_proj.weight: torch.Size([8192, 2048])
  mlp.up_proj.weight: torch.Size([8192, 2048])
  mlp.down_proj.weight: torch.Size([2048, 8192])
  input_layernorm.weight: torch.Size([2048])
  post_attention_layernorm.weight: torch.Size([2048])
Layer 10:
  LlamaDecoderLayer
  self_attn.q_proj.weight: torch.Size([2048, 2048])
  self_attn.k_proj.weight: torch.Size([512, 2048])
  self_attn.v_proj.weight: torch.Size([512, 2048])
  self_attn.o_proj.weight: torch.Size([2048, 2048])
  mlp.gate_proj.weight: torch.Size([8192, 2048])
  mlp.up_proj.weight: torch.Size([8192, 2048])
  mlp.down_proj.weight: torch.Size([2048, 8192])
  input_layernorm.weight: torch.Size([2048])
  post_attention_layernorm.weight: torch.Size([2048])
Layer 11:
  LlamaDecoderLayer
  self_attn.q_proj.weight: torch.Size([2048, 2048])
  self_attn.k_proj.weight: torch.Size([512, 2048])
  self_attn.v_proj.weight: torch.Size([512, 2048])
  self_attn.o_proj.weight: torch.Size([2048, 2048])
  mlp.gate_proj.weight: torch.Size([8192, 2048])
  mlp.up_proj.weight: torch.Size([8192, 2048])
  mlp.down_proj.weight: torch.Size([2048, 8192])
  input_layernorm.weight: torch.Size([2048])
  post_attention_layernorm.weight: torch.Size([2048])
Layer 12:
  LlamaDecoderLayer
  self_attn.q_proj.weight: torch.Size([2048, 2048])
  self_attn.k_proj.weight: torch.Size([512, 2048])
  self_attn.v_proj.weight: torch.Size([512, 2048])
  self_attn.o_proj.weight: torch.Size([2048, 2048])
  mlp.gate_proj.weight: torch.Size([8192, 2048])
  mlp.up_proj.weight: torch.Size([8192, 2048])
  mlp.down_proj.weight: torch.Size([2048, 8192])
  input_layernorm.weight: torch.Size([2048])
  post_attention_layernorm.weight: torch.Size([2048])
Layer 13:
  LlamaDecoderLayer
  self_attn.q_proj.weight: torch.Size([2048, 2048])
  self_attn.k_proj.weight: torch.Size([512, 2048])
  self_attn.v_proj.weight: torch.Size([512, 2048])
  self_attn.o_proj.weight: torch.Size([2048, 2048])
  mlp.gate_proj.weight: torch.Size([8192, 2048])
  mlp.up_proj.weight: torch.Size([8192, 2048])
  mlp.down_proj.weight: torch.Size([2048, 8192])
  input_layernorm.weight: torch.Size([2048])
  post_attention_layernorm.weight: torch.Size([2048])
Layer 14:
  LlamaDecoderLayer
  self_attn.q_proj.weight: torch.Size([2048, 2048])
  self_attn.k_proj.weight: torch.Size([512, 2048])
  self_attn.v_proj.weight: torch.Size([512, 2048])
  self_attn.o_proj.weight: torch.Size([2048, 2048])
  mlp.gate_proj.weight: torch.Size([8192, 2048])
  mlp.up_proj.weight: torch.Size([8192, 2048])
  mlp.down_proj.weight: torch.Size([2048, 8192])
  input_layernorm.weight: torch.Size([2048])
  post_attention_layernorm.weight: torch.Size([2048])
Layer 15:
  LlamaDecoderLayer
  self_attn.q_proj.weight: torch.Size([2048, 2048])
  self_attn.k_proj.weight: torch.Size([512, 2048])
  self_attn.v_proj.weight: torch.Size([512, 2048])
  self_attn.o_proj.weight: torch.Size([2048, 2048])
  mlp.gate_proj.weight: torch.Size([8192, 2048])
  mlp.up_proj.weight: torch.Size([8192, 2048])
  mlp.down_proj.weight: torch.Size([2048, 8192])
  input_layernorm.weight: torch.Size([2048])
  post_attention_layernorm.weight: torch.Size([2048])
"""

basemodel_layers = """
Layer 0:
  Embedding
  weight: torch.Size([128256, 2048])
Layer 1:
  DecoderLayer
  input_layernorm.norm.weight: torch.Size([2048])
  post_attention_layernorm.norm.weight: torch.Size([2048])
  self_attn.q_proj.layer.weight: torch.Size([2048, 2048])
  self_attn.k_proj.layer.weight: torch.Size([512, 2048])
  self_attn.v_proj.layer.weight: torch.Size([512, 2048])
  self_attn.o_proj.layer.weight: torch.Size([2048, 2048])
  mlp.gate_proj.layer.weight: torch.Size([8192, 2048])
  mlp.up_proj.layer.weight: torch.Size([8192, 2048])
  mlp.down_proj.layer.weight: torch.Size([2048, 8192])
  mlp.input_layernorm.norm.weight: torch.Size([2048])
Layer 2:
  DecoderLayer
  input_layernorm.norm.weight: torch.Size([2048])
  post_attention_layernorm.norm.weight: torch.Size([2048])
  self_attn.q_proj.layer.weight: torch.Size([2048, 2048])
  self_attn.k_proj.layer.weight: torch.Size([512, 2048])
  self_attn.v_proj.layer.weight: torch.Size([512, 2048])
  self_attn.o_proj.layer.weight: torch.Size([2048, 2048])
  mlp.gate_proj.layer.weight: torch.Size([8192, 2048])
  mlp.up_proj.layer.weight: torch.Size([8192, 2048])
  mlp.down_proj.layer.weight: torch.Size([2048, 8192])
  mlp.input_layernorm.norm.weight: torch.Size([2048])
Layer 3:
  DecoderLayer
  input_layernorm.norm.weight: torch.Size([2048])
  post_attention_layernorm.norm.weight: torch.Size([2048])
  self_attn.q_proj.layer.weight: torch.Size([2048, 2048])
  self_attn.k_proj.layer.weight: torch.Size([512, 2048])
  self_attn.v_proj.layer.weight: torch.Size([512, 2048])
  self_attn.o_proj.layer.weight: torch.Size([2048, 2048])
  mlp.gate_proj.layer.weight: torch.Size([8192, 2048])
  mlp.up_proj.layer.weight: torch.Size([8192, 2048])
  mlp.down_proj.layer.weight: torch.Size([2048, 8192])
  mlp.input_layernorm.norm.weight: torch.Size([2048])
Layer 4:
  DecoderLayer
  input_layernorm.norm.weight: torch.Size([2048])
  post_attention_layernorm.norm.weight: torch.Size([2048])
  self_attn.q_proj.layer.weight: torch.Size([2048, 2048])
  self_attn.k_proj.layer.weight: torch.Size([512, 2048])
  self_attn.v_proj.layer.weight: torch.Size([512, 2048])
  self_attn.o_proj.layer.weight: torch.Size([2048, 2048])
  mlp.gate_proj.layer.weight: torch.Size([8192, 2048])
  mlp.up_proj.layer.weight: torch.Size([8192, 2048])
  mlp.down_proj.layer.weight: torch.Size([2048, 8192])
  mlp.input_layernorm.norm.weight: torch.Size([2048])
Layer 5:
  DecoderLayer
  input_layernorm.norm.weight: torch.Size([2048])
  post_attention_layernorm.norm.weight: torch.Size([2048])
  self_attn.q_proj.layer.weight: torch.Size([2048, 2048])
  self_attn.k_proj.layer.weight: torch.Size([512, 2048])
  self_attn.v_proj.layer.weight: torch.Size([512, 2048])
  self_attn.o_proj.layer.weight: torch.Size([2048, 2048])
  mlp.gate_proj.layer.weight: torch.Size([8192, 2048])
  mlp.up_proj.layer.weight: torch.Size([8192, 2048])
  mlp.down_proj.layer.weight: torch.Size([2048, 8192])
  mlp.input_layernorm.norm.weight: torch.Size([2048])
Layer 6:
  DecoderLayer
  input_layernorm.norm.weight: torch.Size([2048])
  post_attention_layernorm.norm.weight: torch.Size([2048])
  self_attn.q_proj.layer.weight: torch.Size([2048, 2048])
  self_attn.k_proj.layer.weight: torch.Size([512, 2048])
  self_attn.v_proj.layer.weight: torch.Size([512, 2048])
  self_attn.o_proj.layer.weight: torch.Size([2048, 2048])
  mlp.gate_proj.layer.weight: torch.Size([8192, 2048])
  mlp.up_proj.layer.weight: torch.Size([8192, 2048])
  mlp.down_proj.layer.weight: torch.Size([2048, 8192])
  mlp.input_layernorm.norm.weight: torch.Size([2048])
Layer 7:
  DecoderLayer
  input_layernorm.norm.weight: torch.Size([2048])
  post_attention_layernorm.norm.weight: torch.Size([2048])
  self_attn.q_proj.layer.weight: torch.Size([2048, 2048])
  self_attn.k_proj.layer.weight: torch.Size([512, 2048])
  self_attn.v_proj.layer.weight: torch.Size([512, 2048])
  self_attn.o_proj.layer.weight: torch.Size([2048, 2048])
  mlp.gate_proj.layer.weight: torch.Size([8192, 2048])
  mlp.up_proj.layer.weight: torch.Size([8192, 2048])
  mlp.down_proj.layer.weight: torch.Size([2048, 8192])
  mlp.input_layernorm.norm.weight: torch.Size([2048])
Layer 8:
  DecoderLayer
  input_layernorm.norm.weight: torch.Size([2048])
  post_attention_layernorm.norm.weight: torch.Size([2048])
  self_attn.q_proj.layer.weight: torch.Size([2048, 2048])
  self_attn.k_proj.layer.weight: torch.Size([512, 2048])
  self_attn.v_proj.layer.weight: torch.Size([512, 2048])
  self_attn.o_proj.layer.weight: torch.Size([2048, 2048])
  mlp.gate_proj.layer.weight: torch.Size([8192, 2048])
  mlp.up_proj.layer.weight: torch.Size([8192, 2048])
  mlp.down_proj.layer.weight: torch.Size([2048, 8192])
  mlp.input_layernorm.norm.weight: torch.Size([2048])
Layer 9:
  DecoderLayer
  input_layernorm.norm.weight: torch.Size([2048])
  post_attention_layernorm.norm.weight: torch.Size([2048])
  self_attn.q_proj.layer.weight: torch.Size([2048, 2048])
  self_attn.k_proj.layer.weight: torch.Size([512, 2048])
  self_attn.v_proj.layer.weight: torch.Size([512, 2048])
  self_attn.o_proj.layer.weight: torch.Size([2048, 2048])
  mlp.gate_proj.layer.weight: torch.Size([8192, 2048])
  mlp.up_proj.layer.weight: torch.Size([8192, 2048])
  mlp.down_proj.layer.weight: torch.Size([2048, 8192])
  mlp.input_layernorm.norm.weight: torch.Size([2048])
Layer 10:
  DecoderLayer
  input_layernorm.norm.weight: torch.Size([2048])
  post_attention_layernorm.norm.weight: torch.Size([2048])
  self_attn.q_proj.layer.weight: torch.Size([2048, 2048])
  self_attn.k_proj.layer.weight: torch.Size([512, 2048])
  self_attn.v_proj.layer.weight: torch.Size([512, 2048])
  self_attn.o_proj.layer.weight: torch.Size([2048, 2048])
  mlp.gate_proj.layer.weight: torch.Size([8192, 2048])
  mlp.up_proj.layer.weight: torch.Size([8192, 2048])
  mlp.down_proj.layer.weight: torch.Size([2048, 8192])
  mlp.input_layernorm.norm.weight: torch.Size([2048])
Layer 11:
  DecoderLayer
  input_layernorm.norm.weight: torch.Size([2048])
  post_attention_layernorm.norm.weight: torch.Size([2048])
  self_attn.q_proj.layer.weight: torch.Size([2048, 2048])
  self_attn.k_proj.layer.weight: torch.Size([512, 2048])
  self_attn.v_proj.layer.weight: torch.Size([512, 2048])
  self_attn.o_proj.layer.weight: torch.Size([2048, 2048])
  mlp.gate_proj.layer.weight: torch.Size([8192, 2048])
  mlp.up_proj.layer.weight: torch.Size([8192, 2048])
  mlp.down_proj.layer.weight: torch.Size([2048, 8192])
  mlp.input_layernorm.norm.weight: torch.Size([2048])
Layer 12:
  DecoderLayer
  input_layernorm.norm.weight: torch.Size([2048])
  post_attention_layernorm.norm.weight: torch.Size([2048])
  self_attn.q_proj.layer.weight: torch.Size([2048, 2048])
  self_attn.k_proj.layer.weight: torch.Size([512, 2048])
  self_attn.v_proj.layer.weight: torch.Size([512, 2048])
  self_attn.o_proj.layer.weight: torch.Size([2048, 2048])
  mlp.gate_proj.layer.weight: torch.Size([8192, 2048])
  mlp.up_proj.layer.weight: torch.Size([8192, 2048])
  mlp.down_proj.layer.weight: torch.Size([2048, 8192])
  mlp.input_layernorm.norm.weight: torch.Size([2048])
Layer 13:
  DecoderLayer
  input_layernorm.norm.weight: torch.Size([2048])
  post_attention_layernorm.norm.weight: torch.Size([2048])
  self_attn.q_proj.layer.weight: torch.Size([2048, 2048])
  self_attn.k_proj.layer.weight: torch.Size([512, 2048])
  self_attn.v_proj.layer.weight: torch.Size([512, 2048])
  self_attn.o_proj.layer.weight: torch.Size([2048, 2048])
  mlp.gate_proj.layer.weight: torch.Size([8192, 2048])
  mlp.up_proj.layer.weight: torch.Size([8192, 2048])
  mlp.down_proj.layer.weight: torch.Size([2048, 8192])
  mlp.input_layernorm.norm.weight: torch.Size([2048])
Layer 14:
  DecoderLayer
  input_layernorm.norm.weight: torch.Size([2048])
  post_attention_layernorm.norm.weight: torch.Size([2048])
  self_attn.q_proj.layer.weight: torch.Size([2048, 2048])
  self_attn.k_proj.layer.weight: torch.Size([512, 2048])
  self_attn.v_proj.layer.weight: torch.Size([512, 2048])
  self_attn.o_proj.layer.weight: torch.Size([2048, 2048])
  mlp.gate_proj.layer.weight: torch.Size([8192, 2048])
  mlp.up_proj.layer.weight: torch.Size([8192, 2048])
  mlp.down_proj.layer.weight: torch.Size([2048, 8192])
  mlp.input_layernorm.norm.weight: torch.Size([2048])
Layer 15:
  DecoderLayer
  input_layernorm.norm.weight: torch.Size([2048])
  post_attention_layernorm.norm.weight: torch.Size([2048])
  self_attn.q_proj.layer.weight: torch.Size([2048, 2048])
  self_attn.k_proj.layer.weight: torch.Size([512, 2048])
  self_attn.v_proj.layer.weight: torch.Size([512, 2048])
  self_attn.o_proj.layer.weight: torch.Size([2048, 2048])
  mlp.gate_proj.layer.weight: torch.Size([8192, 2048])
  mlp.up_proj.layer.weight: torch.Size([8192, 2048])
  mlp.down_proj.layer.weight: torch.Size([2048, 8192])
  mlp.input_layernorm.norm.weight: torch.Size([2048])
Layer 16:
  DecoderLayer
  input_layernorm.norm.weight: torch.Size([2048])
  post_attention_layernorm.norm.weight: torch.Size([2048])
  self_attn.q_proj.layer.weight: torch.Size([2048, 2048])
  self_attn.k_proj.layer.weight: torch.Size([512, 2048])
  self_attn.v_proj.layer.weight: torch.Size([512, 2048])
  self_attn.o_proj.layer.weight: torch.Size([2048, 2048])
  mlp.gate_proj.layer.weight: torch.Size([8192, 2048])
  mlp.up_proj.layer.weight: torch.Size([8192, 2048])
  mlp.down_proj.layer.weight: torch.Size([2048, 8192])
  mlp.input_layernorm.norm.weight: torch.Size([2048])
Layer 17:
  RMSNorm
  norm.weight: torch.Size([2048])
"""
