"""
===============================================================================
BASE MODEL
-------------------------------------
Supports:
  - Rotary embeddings
  - GQA (Group Query Attention)
  - RMSNorm
  - (Optional) fused cross-entropy
  - Segment-aware block mask
  - FAN-in param inits
  - (Optional) Triton-based rotary embeddings (enable with use_triton_rope=True)
===============================================================================
"""

import math
import os
import logging
from typing import Optional, Any, Union, Dict

import torch
import torch.nn as nn
import einops

try:
    from torch.nn.attention.flex_attention import create_block_mask
    from torch.nn.attention.flex_attention import flex_attention as _raw_flex_attention

    _flex_attention = torch.compile(_raw_flex_attention, dynamic=True, options={})
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    logging.warning("flex_attention not found; fallback or error may occur if used.")

from safetensors.torch import load_file as load_safetensors_file
from transformers import AutoModelForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig

from cut_cross_entropy import LinearCrossEntropy
from . import rotary

TRITON_ROPE_AVAILABLE = False
try:
    raise Exception("TODO: Double-check correctness. Fallback to Python-based RoPE.")
    from .triton_rotary import fast_rope_embedding

    TRITON_ROPE_AVAILABLE = True
except Exception as e:
    logging.warning(f"Triton-based RoPE not available; falling back to Python-based RoPE. e: {e}")
    TRITON_ROPE_AVAILABLE = False


def str_to_dtype(dtype_str: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(dtype_str, torch.dtype):
        return dtype_str
    if isinstance(dtype_str, str) and "bf16" in dtype_str.lower():
        return torch.bfloat16
    return torch.float32


def make_divisible_by(val: int, divisor: int) -> int:
    if divisor == 0:
        raise ValueError("Divisor cannot be zero")
    return (val + divisor - 1) // divisor * divisor


def get_default_config() -> Dict[str, Any]:
    # Default config for Llama 3.2 1B
    return {
        "vocab_size": 128256,
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_hidden_layers": 16,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_act": "silu",
        "max_position_embeddings": 131072,
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-5,
        "use_cache": True,
        "pad_token_id": None,
        "bos_token_id": 128000,
        "eos_token_id": 128001,
        "pretraining_tp": 1,
        "tie_word_embeddings": True,
        "rope_theta": 500000.0,
        "rope_scaling": {
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        },
        "attention_bias": False,
        "sequence_length": 512,
        "batch_size": 1,
        "use_precomputed_block_mask": 0,
        "use_fusedlce": 1,
        "bidirectional": 0,
        "sliding_window": 0,
        "window_size": 4096,
        "eps": 1e-5,
        "dtype": "bfloat16",
        "device": "cpu",
        "embed_dropout": 0.0,
        "residual_dropout": 0.0,
        "mlp_dim_mult": 4.0,
        "use_attention": True,
        "use_triton_rope": False,
    }


def merge_config_overrides(user_config: Optional[dict] = None) -> Dict[str, Any]:
    defaults = get_default_config()
    if user_config is not None:
        defaults.update(user_config)
    return defaults


class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        torch_dtype = str_to_dtype(dtype)
        self.norm = nn.RMSNorm(
            hidden_size,
            eps=eps,
            elementwise_affine=True,
            dtype=torch_dtype,
            device=device,
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.norm(x)


class LinearProjection(nn.Module):
    def __init__(self, in_dim: int = None, out_dim: int = None, **kwargs):
        super().__init__()
        if in_dim is None:
            in_dim = kwargs["hidden_size"]
        if out_dim is None:
            out_dim = kwargs.get("out_dim", in_dim)

        is_vocab_head = kwargs.get("is_vocab_head", False)
        vocab_size = kwargs.get("vocab_size", 32000)
        if is_vocab_head:
            out_dim = vocab_size

        param_dtype = str_to_dtype(kwargs.get("dtype", torch.float32))
        in_dim = make_divisible_by(in_dim, 128)
        out_dim = make_divisible_by(out_dim, 128)

        use_bias = kwargs.get("projection_layers_use_bias", False)
        self.layer = nn.Linear(in_dim, out_dim, bias=use_bias, dtype=param_dtype)

        fan_in = in_dim
        std = 1.0 / math.sqrt(fan_in)
        nn.init.trunc_normal_(
            self.layer.weight, mean=0.0, std=std, a=-2 * std, b=2 * std
        )
        if self.layer.bias is not None:
            nn.init.zeros_(self.layer.bias)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.layer(x)


class self_attn(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.hidden_size = kwargs["hidden_size"]
        self.num_heads = kwargs["num_attention_heads"]
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_heads)
        self.gqa_num_heads = int(kwargs.get("num_key_value_heads", self.num_heads))

        qk_dims = self.num_heads * self.head_dim
        v_dims = self.gqa_num_heads * self.head_dim

        self.q_proj = LinearProjection(self.hidden_size, qk_dims, **kwargs)
        self.k_proj = LinearProjection(self.hidden_size, v_dims, **kwargs)
        self.v_proj = LinearProjection(self.hidden_size, v_dims, **kwargs)
        self.o_proj = LinearProjection(qk_dims, self.hidden_size, **kwargs)
        self.enable_gqa = self.gqa_num_heads != self.num_heads
        if self.enable_gqa:
            if not (
                self.gqa_num_heads > 0 and self.num_heads % self.gqa_num_heads == 0
            ):
                raise ValueError("num_key_value_heads must divide num_attention_heads")
            if not (self.gqa_num_heads < self.num_heads):
                raise ValueError("num_key_value_heads must be < num_attention_heads")

        self.use_triton_rope = bool(kwargs.get("use_triton_rope", False))

    def forward(
        self, x: torch.Tensor, block_mask=None, freqs_cis=None, *args, **kwargs
    ):
        q = einops.rearrange(self.q_proj(x), "b s (h d) -> b s h d", h=self.num_heads)
        k = einops.rearrange(
            self.k_proj(x), "b s (h d) -> b s h d", h=self.gqa_num_heads
        )
        v = einops.rearrange(
            self.v_proj(x), "b s (h d) -> b s h d", h=self.gqa_num_heads
        )

        if freqs_cis is not None:
            if (
                self.use_triton_rope
                and TRITON_ROPE_AVAILABLE
                and (self.gqa_num_heads == self.num_heads)
            ):
                cos_part = freqs_cis.real[: q.shape[1], : q.shape[-1]]
                sin_part = freqs_cis.imag[: q.shape[1], : q.shape[-1]]
                q_f32 = q.to(torch.float32)
                k_f32 = k.to(torch.float32)
                q_out, k_out = fast_rope_embedding(q_f32, k_f32, cos_part, sin_part)
                q = q_out.to(q.dtype)
                k = k_out.to(k.dtype)
            else:
                q, k = rotary.apply_rotary_emb(q, k, freqs_cis.to(q.device))

        q = einops.rearrange(q, "b s h d -> b h s d")
        k = einops.rearrange(k, "b s h d -> b h s d")
        v = einops.rearrange(v, "b s h d -> b h s d", h=self.gqa_num_heads)

        attn_out = _flex_attention(
            query=q, key=k, value=v, block_mask=block_mask, enable_gqa=self.enable_gqa
        )
        attn_out = einops.rearrange(attn_out, "b h s d -> b s (h d)")
        return self.o_proj(attn_out)


class mlp(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        hidden_size = kwargs["hidden_size"]
        mlp_mult = kwargs.get("mlp_dim_mult", 4.0)
        ffn_dim = int(hidden_size * mlp_mult)

        self.gate_proj = LinearProjection(hidden_size, ffn_dim, **kwargs)
        self.up_proj = LinearProjection(hidden_size, ffn_dim, **kwargs)
        self.down_proj = LinearProjection(ffn_dim, hidden_size, **kwargs)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class DecoderLayer(nn.Module):
    def __init__(self, layer_idx: int = -1, **kwargs):
        super().__init__()
        self.hidden_size = kwargs["hidden_size"]
        self.layer_idx = layer_idx

        self.input_layernorm = RMSNorm(
            hidden_size=self.hidden_size,
            eps=kwargs.get("rms_norm_eps", 1e-5),
            dtype=kwargs.get("dtype", torch.float32),
        )

        self.self_attn = self_attn(**kwargs)
        self.post_attention_layernorm = RMSNorm(
            hidden_size=self.hidden_size,
            eps=kwargs.get("rms_norm_eps", 1e-5),
            dtype=kwargs.get("dtype", torch.float32),
        )
        self.mlp = mlp(**kwargs)

    def forward(
        self, x: torch.Tensor, block_mask=None, freqs_cis=None, *args, **kwargs
    ):
        residual = x
        x = self.input_layernorm(x)
        x = residual + self.self_attn(x, block_mask=block_mask, freqs_cis=freqs_cis)
        residual = x
        x = self.post_attention_layernorm(x)
        x = residual + self.mlp(x)
        return x.to(residual.dtype)


class Embedding(nn.Embedding):
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return super().forward(x)


class Dropout(nn.Dropout):
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return super().forward(x)


class BaseModel(nn.Module):
    def __init__(self, layer_kwargs: dict):
        super().__init__()
        self.config = layer_kwargs
        self.num_layers = self.config.get("num_hidden_layers", 16)
        self.hidden_size = self.config.get("hidden_size", 2048)
        self.vocab_size = self.config.get("vocab_size", 32000)
        self.num_heads = self.config.get("num_attention_heads", 32)
        self.head_dim = self.hidden_size // self.num_heads

        self.embeddings = Embedding(self.vocab_size, self.hidden_size)
        fan_in_embed = self.hidden_size
        std_embed = 1.0 / math.sqrt(fan_in_embed)
        nn.init.trunc_normal_(
            self.embeddings.weight,
            mean=0.0,
            std=std_embed,
            a=-2 * std_embed,
            b=2 * std_embed,
        )

        base_decay_rate = self.config.get("rope_theta", 500000.0)
        pretraining_seq_len = self.config.get("rope_scaling", {}).get(
            "original_max_position_embeddings", 8192
        )
        max_position_embeddings = self.config.get("max_position_embeddings", 131072)
        freqs_cis = rotary.precompute_freqs_cis(
            dim=self.head_dim,
            end=max_position_embeddings,
            theta=base_decay_rate,
            use_scaled=(max_position_embeddings > pretraining_seq_len),
            old_context_length=pretraining_seq_len,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        embed_dropout = self.config.get("embed_dropout", 0.0)
        self.embed_dropout = (
            nn.Dropout(p=embed_dropout) if embed_dropout > 0 else nn.Identity()
        )

        self.layers = nn.ModuleList(
            [DecoderLayer(layer_idx=i, **self.config) for i in range(self.num_layers)]
        )
        self.norm = RMSNorm(
            hidden_size=self.hidden_size,
            eps=self.config.get("rms_norm_eps", 1e-5),
            dtype=self.config.get("dtype", torch.float32),
        )

        self.tie_word_embeddings = self.config.get("tie_word_embeddings", True)
        if not self.tie_word_embeddings:
            head_in_dim = make_divisible_by(self.hidden_size, 128)
            head_out_dim = self.vocab_size
            self.lm_head_weight = nn.Parameter(
                torch.randn(head_out_dim, head_in_dim), requires_grad=True
            )
            fan_in_head = head_in_dim
            std_head = 1.0 / math.sqrt(fan_in_head)
            nn.init.trunc_normal_(
                self.lm_head_weight,
                mean=0.0,
                std=std_head,
                a=-2 * std_head,
                b=2 * std_head,
            )
        else:
            self.lm_head_weight = None

        self.use_fusedlce = int(self.config.get("use_fusedlce", 0))
        if self.use_fusedlce:
            try:
                self.LCE = LinearCrossEntropy(reduction="mean")
                self.LCE_none = LinearCrossEntropy(reduction="none")
                logging.info("Using cut_cross_entropy for loss calculation.")
            except ImportError:
                logging.error("cut_cross_entropy not found. Disabling fused LCE.")
                self.use_fusedlce = 0
                self.LCE = None
                self.LCE_none = None

        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.cross_entropy_none = nn.CrossEntropyLoss(reduction="none")

    def get_input_embeddings(self):
        return self.embeddings

    @property
    def lm_head(self):
        if self.tie_word_embeddings:
            return self.embeddings.weight
        else:
            return self.lm_head_weight

    def get_output_embeddings_weight(self) -> torch.Tensor:
        return self.lm_head

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
        def mask_mod(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            segment_mask = segment_ids[b, q_idx] == segment_ids[b, kv_idx]
            return causal_mask & segment_mask

        B, L = input_ids.shape
        block_mask = create_block_mask(
            mask_mod,
            B=B,
            H=1,
            Q_LEN=L,
            KV_LEN=L,
            device=input_ids.device,
            _compile=True,
        )

        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = layer(x, block_mask=block_mask, freqs_cis=self.freqs_cis)
        x = self.norm(x)

        if label_ids is None:
            B, L, D = x.shape
            logits = torch.matmul(x.view(-1, D), self.lm_head.to(x.dtype))
            return logits.view(B, L, self.vocab_size)

        logits_16 = x.to(torch.float16)
        w_16 = self.lm_head.to(torch.float16)
        if reduction is None:
            loss_fn = self.LCE
        else:
            assert reduction == "none", "Only 'none' reduction is supported."
            loss_fn = self.LCE_none

        loss = loss_fn(logits_16, w_16, label_ids)
        return loss.to(torch.float32)

    def reset_freq_cis(self, seq_len: int, accelerator=None):
        base_decay_rate = self.config.get("rope_theta", 500000.0)
        old_context_length = self.config.get("rope_scaling", {}).get(
            "original_max_position_embeddings", 8192
        )
        new_freqs_cis = rotary.precompute_freqs_cis(
            dim=self.head_dim,
            end=seq_len,
            theta=base_decay_rate,
            use_scaled=(seq_len > old_context_length),
            old_context_length=old_context_length,
            accelerator=accelerator,
        )
        try:
            self.register_buffer("freqs_cis", new_freqs_cis, persistent=False)
        except Exception as e:
            logging.warning(f"Could not register buffer for freqs_cis: {e}")
            self.freqs_cis = new_freqs_cis.to(self.freqs_cis.device)

        logging.info(f"Updated model.freqs_cis to shape {self.freqs_cis.shape}")
        self.config["max_position_embeddings"] = seq_len


def load_model_from_safetensors(
    config: dict, safetensors_path: str, device: str = "cpu"
) -> BaseModel:
    merged_config = merge_config_overrides(config)
    logging.info("Instantiating BaseModel with merged config...")
    model = BaseModel(layer_kwargs=merged_config)

    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"safetensors file not found: {safetensors_path}")
    logging.info(f"Loading weights from: {safetensors_path}")
    state_dict = load_safetensors_file(safetensors_path, device=device)
    load_result = model.load_state_dict(state_dict, strict=False)
    logging.info(f"load_state_dict result: {load_result}")

    missing_crit = [k for k in load_result.missing_keys if not k.endswith("freqs_cis")]
    if missing_crit:
        logging.error(f"Missing critical keys: {missing_crit}")

    final_dtype = str_to_dtype(merged_config.get("dtype", "bfloat16"))
    model = model.to(final_dtype)
    logging.info(f"Model loaded and converted to {final_dtype}.")
    return model


def _load_model(target_model, config):
    instruct = config.get("instruct", True)
    if instruct:
        source_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct"
        )
    else:
        source_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

    # aaa = LlamaConfig(
    #     vocab_size=128256,
    #     hidden_size=2048,
    #     intermediate_size=8192,
    #     num_hidden_layers=16,
    #     num_attention_heads=32,
    #     num_key_value_heads=8,
    #     hidden_act="silu",
    #     max_position_embeddings=131072,
    #     initializer_range=0.02,
    #     rms_norm_eps=1e-5,
    #     use_cache=True,
    #     pad_token_id=None,
    #     bos_token_id=128000,
    #     eos_token_id=128001,
    #     pretraining_tp=1,
    #     tie_word_embeddings=True,
    #     rope_theta=500000.0,
    #     rope_scaling={
    #         "factor": 32.0,
    #         "high_freq_factor": 4.0,
    #         "low_freq_factor": 1.0,
    #         "original_max_position_embeddings": 8192,
    #         "rope_type": "llama3",
    #     },
    #     attention_bias=False,
    # )

    target_model.embeddings.weight.data = (source_model.model.embed_tokens.weight.data.clone())
    
    for i in range(len(source_model.model.layers)):
        target_model.layers[i].self_attn.q_proj.layer.weight.data = source_model.model.layers[i].self_attn.q_proj.weight.data.clone()
        target_model.layers[i].self_attn.k_proj.layer.weight.data = source_model.model.layers[i].self_attn.k_proj.weight.data.clone()
        target_model.layers[i].self_attn.v_proj.layer.weight.data = source_model.model.layers[i].self_attn.v_proj.weight.data.clone()
        target_model.layers[i].self_attn.o_proj.layer.weight.data = source_model.model.layers[i].self_attn.o_proj.weight.data.clone()
        target_model.layers[i].mlp.gate_proj.layer.weight.data = source_model.model.layers[i].mlp.gate_proj.weight.data.clone()
        target_model.layers[i].mlp.up_proj.layer.weight.data = source_model.model.layers[i].mlp.up_proj.weight.data.clone()
        target_model.layers[i].mlp.down_proj.layer.weight.data = source_model.model.layers[i].mlp.down_proj.weight.data.clone()
        target_model.layers[i].input_layernorm.norm.weight.data = source_model.model.layers[i].input_layernorm.weight.data.clone()
        target_model.layers[i].post_attention_layernorm.norm.weight.data = source_model.model.layers[i].post_attention_layernorm.weight.data.clone()

    target_model.norm.norm.weight.data = source_model.model.norm.weight.data.clone()
    target_model.lm_head.data = source_model.lm_head.weight.data.clone()

    del source_model
    return target_model.to(torch.bfloat16)


def load_model(
    config: Optional[dict] = None,
    hf_model_name: Optional[str] = None,
    instruct: bool = None,
) -> BaseModel:
    merged_config = merge_config_overrides(config)

    logging.info("Creating BaseModel using merged config...")
    model = BaseModel(merged_config)
    if instruct is not None:
        merged_config['instruct'] = instruct
    model = _load_model(model, merged_config)
    final_dtype = str_to_dtype(merged_config.get("dtype", "bfloat16"))
    model = model.to(final_dtype)
    logging.info(f"BaseModel ready, dtype={final_dtype}")
    return model


if __name__ == "__main__":
    m1 = load_model()
    print(
        "Model1 with default config param count:",
        sum(p.numel() for p in m1.parameters()),
    )
    overrides = {"dtype": "float32"}
    m2 = load_model(config=overrides)
    print(
        "Model2 partial overrides param count:", sum(p.numel() for p in m2.parameters())
    )
