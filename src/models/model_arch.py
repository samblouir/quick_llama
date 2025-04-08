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
  - FAN-in param inits
===============================================================================
"""

import math
import os
import logging
from typing import Optional, Any, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

try:
    from torch.nn.attention.flex_attention import create_block_mask
    from torch.nn.attention.flex_attention import flex_attention as _raw_flex_attention

    # Example compile options if needed:
    _flex_attention = torch.compile(_raw_flex_attention, dynamic=True, options={})
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    logging.warning("flex_attention not found; fallback or error will occur if used.")

from safetensors.torch import load_file as load_safetensors_file
from transformers import AutoModelForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig

# External utilities - placeholders for your environment
from cut_cross_entropy import LinearCrossEntropy 
from . import rotary
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
    """
    Returns a dictionary containing the default configuration
    for the LLaMA-like BaseModel in this file.
    """
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
    }

def merge_config_overrides(user_config: Optional[dict] = None) -> Dict[str, Any]:
    """
    Merges user_config with the default config,
    allowing user config to override any defaults.

    Returns:
        A merged dictionary containing the final config.
    """
    defaults = get_default_config()
    if user_config is not None:
        defaults.update(user_config)
    return defaults


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    """
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
    """
    A single linear layer with forced dimension alignment (dims are made to be
    divisible by 128). Uses truncated normal initialization for weights.
    """
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

        param_dtype = kwargs.get("dtype", torch.float32)
        param_dtype = str_to_dtype(param_dtype)

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
    """
    Custom MHA that:
      - Splits Q,K,V
      - Applies flex_attention if available (fallback to PyTorch if not)
      - Applies rotary embeddings if provided
      - Supports GQA via gqa_num_heads
    """
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

        self.enable_gqa = (self.gqa_num_heads != self.num_heads)
        if self.enable_gqa:
            if not (self.gqa_num_heads > 0 and self.num_heads % self.gqa_num_heads == 0):
                raise ValueError("num_key_value_heads must divide num_attention_heads")
            if not (self.gqa_num_heads < self.num_heads):
                raise ValueError("num_key_value_heads must be less than num_attention_heads")

    def forward(
        self,
        x: torch.Tensor,
        block_mask=None,
        freqs_cis=None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = einops.rearrange(q, "b s (h d) -> b s h d", h=self.num_heads)
        k = einops.rearrange(k, "b s (h d) -> b s h d", h=self.gqa_num_heads)

        if freqs_cis is not None:
            q, k = rotary.apply_rotary_emb(q, k, freqs_cis.to(q.device))

        q = einops.rearrange(q, "b s h d -> b h s d")
        k = einops.rearrange(k, "b s h d -> b h s d")
        v = einops.rearrange(v, "b s (h d) -> b h s d", h=self.gqa_num_heads)

        attn_out = _flex_attention(
            query=q,
            key=k,
            value=v,
            block_mask=block_mask,
            enable_gqa=self.enable_gqa,
        )

        attn_out = einops.rearrange(attn_out, "b h s d -> b s (h d)")
        out = self.o_proj(attn_out)
        return out


class mlp(nn.Module):
    """
    A feed-forward block (SwiGLU or similar). This is a minimal example.
    """
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
        self,
        x: torch.Tensor,
        block_mask=None,
        freqs_cis=None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)
        attn_out = self.self_attn(x, block_mask=block_mask, freqs_cis=freqs_cis)
        x = residual + attn_out

        residual = x
        x = self.post_attention_layernorm(x)
        mlp_out = self.mlp(x)
        x = residual + mlp_out

        return x.to(residual.dtype)


class Embedding(nn.Embedding):
    """
    Simple embedding wrapper to allow *args, **kwargs.
    """
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return super().forward(x)


class Dropout(nn.Dropout):
    """
    Simple dropout wrapper.
    """
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return super().forward(x)


class BaseModel(nn.Module):
    """
    A flexible Transformer-like model.

    Usage:
        model = BaseModel(layer_kwargs=some_config_dict)
        outputs = model(input_ids, label_ids=labels, ...)
    """
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
            self.embeddings.weight, mean=0.0, std=std_embed, a=-2 * std_embed, b=2 * std_embed
        )

        base_decay_rate = self.config.get("rope_theta", 500000.0)
        pretraining_seq_len = self.config.get("rope_scaling", {}).get("original_max_position_embeddings", 8192)
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
        if embed_dropout > 0.0:
            self.embed_dropout = Dropout(p=embed_dropout)
        else:
            self.embed_dropout = nn.Identity()

        self.layers = nn.ModuleList([
            DecoderLayer(layer_idx=i, **self.config) for i in range(self.num_layers)
        ])

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
                self.lm_head_weight, mean=0.0, std=std_head, a=-2 * std_head, b=2 * std_head
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
                logging.error("cut_cross_entropy not found, but use_fusedlce=True. Disabling fused LCE.")
                self.use_fusedlce = 0
                self.LCE = None
                self.LCE_none = None

        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.cross_entropy_none = nn.CrossEntropyLoss(reduction="none")

    def get_input_embeddings(self):
        return self.embeddings

    def get_output_embeddings_weight(self) -> torch.Tensor:
        if self.tie_word_embeddings:
            return self.embeddings.weight
        else:
            return self.lm_head_weight

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
            H=1, # Broadcast to all heads
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
            # if no labels are provided, return the logits
            B, L, D = x.shape
            logits = torch.matmul(x.view(-1, D), self.lm_head.to(x.dtype))
            logits = logits.view(B, L, self.vocab_size)
            return logits

        logits_16 = x.to(torch.float16)
        w_16 = self.lm_head.to(torch.float16)

        if reduction is None:
            loss_fn = self.LCE
        else:
            assert reduction == "none", "Only 'none' reduction is supported."
            loss_fn = self.LCE_none

        loss = loss_fn(logits_16.to(torch.float16), w_16, label_ids)
        return loss.to(torch.float32)

    def reset_freq_cis(self, seq_len: int, accelerator=None):
        """
        Reset (and recompute) the RoPE frequencies for a new sequence length.
        """
        base_decay_rate = self.config.get("rope_theta", 500000.0)
        old_context_length = self.config.get("rope_scaling", {}).get("original_max_position_embeddings", 8192)
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
            logging.warning(f"Could not register buffer for freqs_cis: {e}. Assigning directly.")
            self.freqs_cis = new_freqs_cis.to(self.freqs_cis.device)

        logging.info(f"Updated model.freqs_cis to shape {self.freqs_cis.shape}")
        self.config["max_position_embeddings"] = seq_len


def load_model_from_safetensors(
    config: dict,
    safetensors_path: str,
    device: str = "cpu"
) -> BaseModel:
    """
    Instantiates BaseModel with merged config and loads weights from a safetensors file.

    Args:
        config (dict): Configuration overrides (merged with defaults).
        safetensors_path (str): Path to the .safetensors file.
        device (str): Device to load the model on initially.

    Returns:
        BaseModel: The loaded model instance.
    """
    merged_config = merge_config_overrides(config)
    logging.info("Instantiating BaseModel with merged config...")
    model = BaseModel(layer_kwargs=merged_config)

    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"safetensors file not found: {safetensors_path}")
    logging.info(f"Loading weights from: {safetensors_path}")
    state_dict = load_safetensors_file(safetensors_path, device=device)

    load_result = model.load_state_dict(state_dict, strict=False)
    logging.info(f"load_state_dict result: {load_result}")

    critical_missing = [
        k for k in load_result.missing_keys if not k.endswith("freqs_cis")
    ]
    if critical_missing:
        logging.error(f"Missing critical keys: {critical_missing}")

    final_dtype = str_to_dtype(merged_config.get("dtype", "bfloat16"))
    model = model.to(final_dtype)
    logging.info(f"Model loaded and converted to {final_dtype}.")
    return model




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
        target_model.layers[i].self_attn.q_proj.layer.weight.data = source_model.model.layers[i].self_attn.q_proj.weight.data.clone()
        target_model.layers[i].self_attn.k_proj.layer.weight.data = source_model.model.layers[i].self_attn.k_proj.weight.data.clone()
        target_model.layers[i].self_attn.v_proj.layer.weight.data = source_model.model.layers[i].self_attn.v_proj.weight.data.clone()
        target_model.layers[i].self_attn.o_proj.layer.weight.data = source_model.model.layers[i].self_attn.o_proj.weight.data.clone()
        target_model.layers[i].mlp.gate_proj.layer.weight.data = source_model.model.layers[i].mlp.gate_proj.weight.data.clone()
        target_model.layers[i].mlp.up_proj.layer.weight.data = source_model.model.layers[i].mlp.up_proj.weight.data.clone()
        target_model.layers[i].mlp.down_proj.layer.weight.data = source_model.model.layers[i].mlp.down_proj.weight.data.clone()
        target_model.layers[i].input_layernorm.norm.weight.data = source_model.model.layers[i].input_layernorm.weight.data.clone()
        target_model.layers[i].post_attention_layernorm.norm.weight.data = source_model.model.layers[i].post_attention_layernorm.weight.data.clone()

    # Copy final norm and lm_head
    target_model.norm.norm.weight.data = source_model.model.norm.weight.data.clone()
    target_model.lm_head.data = source_model.lm_head.weight.data.clone()

    del source_model
    return target_model.to(torch.bfloat16)




def load_model(
    config: Optional[dict] = None,
    hf_model_name: Optional[str] = None,
    instruct: bool = True
) -> BaseModel:
    """
    Create a new BaseModel instance from (optionally) a Hugging Face model
    or local logic. If hf_model_name is provided, attempts to load from HF.

    Args:
        config (dict, optional): Config overrides.
        hf_model_name (str, optional): If provided, loads weights from HF.
        instruct (bool): If True, modifies logic to load an instruct variant.

    Returns:
        BaseModel
    """
    merged_config = merge_config_overrides(config)
    logging.info("Creating BaseModel using merged config...")

    model = BaseModel(layer_kwargs=merged_config)

    if hf_model_name:
        llama_model_id = hf_model_name
        logging.info(f"Loading from HF base checkpoint: {llama_model_id}")

        try:
            source_model = AutoModelForCausalLM.from_pretrained(
                llama_model_id,
                torch_dtype=str_to_dtype(merged_config["dtype"]),
                low_cpu_mem_usage=True
            )
        except Exception as e:
            logging.error(f"Failed to load from HF: {e}")
            raise

        source_sd = source_model.state_dict()
        target_sd = model.state_dict()

        # Key mapping logic would go here

        model.load_state_dict(target_sd, strict=False)
        del source_model, source_sd

    final_dtype = str_to_dtype(merged_config.get("dtype", "bfloat16"))
    model = model.to(final_dtype)
    logging.info(f"BaseModel ready, dtype={final_dtype}")
    return model



def load_model(
        config: Optional[dict] = None,
        hf_model_name: Optional[str] = None,
        instruct: bool = True
) -> BaseModel:
    """
    Load a model with the given configuration.
    hf_model_name: Optional[str] is currently not used.
    instruct: this bool is currently not used.
    """

    merged_config = merge_config_overrides(config)
    logging.info("Creating BaseModel using merged config...")

    model = BaseModel(merged_config)
    model = _load_model(model, merged_config)

    final_dtype = str_to_dtype(merged_config.get("dtype", "bfloat16"))
    model = model.to(final_dtype)
    logging.info(f"BaseModel ready, dtype={final_dtype}")
    return model


if __name__ == "__main__":
    # 1) Minimal usage with default config
    model1 = load_model()
    print("Model1 created with defaults:", sum(p.numel() for p in model1.parameters()))

    # 2) Provide partial overrides
    user_overrides = {
        "dtype": "float32",
    }
    model2 = load_model(config=user_overrides)
    print("Model2 partial overrides:", sum(p.numel() for p in model2.parameters()))
