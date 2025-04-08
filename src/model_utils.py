import os
import torch
import torch.nn as nn
import math
from safetensors.torch import load_file as load_safetensors
from accelerate import Accelerator
from accelerate.utils import save_accelerator_state, load_accelerator_state
from accelerate.logging import get_logger
import glob
import shutil
import json

import trainable_llama # Assuming this exists and has TrainableLlama class

log = get_logger(__name__)

def create_model(config):
    """Creates the model instance using trainable_llama."""
    log.info(f"Creating model based on: {config.get('model_name', 'config setting missing')}")
    # Pass only necessary model-related config
    model_config = {k: v for k, v in config.items() if k in [
         'model_name', 'sequence_length', 'softmax_temperature' # Add others your model needs
    ]}
    try:
        model = trainable_llama.TrainableLlama(model_config)
        log.info("Model created successfully via trainable_llama.")
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"Trainable model parameters: {params:,}")
        return model
    except Exception as e:
        log.error(f"Failed to create model using trainable_llama: {e}", exc_info=True)
        raise

def create_optimizer_and_scheduler(model, config, num_training_steps):
    """Creates optimizer and learning rate scheduler."""
    no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"] # Common no_decay parameters
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n.lower() for nd in no_decay) and p.requires_grad],
            "weight_decay": config['weight_decay'],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n.lower() for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=config['lr'],
        eps=config['adam_epsilon']
    )
    log.info(f"Optimizer created: AdamW (lr={config['lr']}, eps={config['adam_epsilon']}, wd={config['weight_decay']})")

    num_warmup_steps = int(num_training_steps * config['num_warmup_steps_ratio'])
    log.info(f"Total training steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")

    try:
        from transformers import get_linear_schedule_with_warmup
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        log.info("LR Scheduler created: Linear warmup and decay (via transformers)")
    except ImportError:
         log.warning("transformers.get_linear_schedule_with_warmup not found. Using constant LR scheduler.")
         lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    return optimizer, lr_scheduler


def save_checkpoint(accelerator: Accelerator, step, checkpoint_dir, config, keep_limit=3):
    """Saves the model, optimizer, and scheduler state using Accelerate."""
    checkpoint_save_dir = os.path.join(checkpoint_dir, f"step_{step}")
    log.info(f"Saving checkpoint for step {step} to: {checkpoint_save_dir}")

    try:
        accelerator.save_state(output_dir=checkpoint_save_dir)

        if accelerator.is_main_process:
             config_path = os.path.join(checkpoint_save_dir, "training_config.json")
             try:
                 config_to_save = {k: v for k, v in config.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
                 with open(config_path, "w") as f:
                      json.dump(config_to_save, f, indent=4)
             except Exception as e:
                  log.error(f"Failed to save config with checkpoint: {e}")

             if keep_limit is not None and keep_limit > 0:
                 checkpoints = sorted(
                     glob.glob(os.path.join(checkpoint_dir, "step_*")),
                     key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else -1 # Handle potential non-step folders
                 )
                 checkpoints = [c for c in checkpoints if c.split("_")[-1].isdigit()] # Filter again just in case

                 if len(checkpoints) > keep_limit:
                     checkpoints_to_delete = checkpoints[:-keep_limit]
                     log.info(f"Pruning old checkpoints: Found {len(checkpoints)}, keeping {keep_limit}.")
                     for ckpt_path in checkpoints_to_delete:
                         log.info(f"  Deleting old checkpoint: {ckpt_path}")
                         try:
                              shutil.rmtree(ckpt_path)
                         except OSError as e:
                              log.error(f"    Error deleting checkpoint {ckpt_path}: {e}")

        accelerator.wait_for_everyone()
        log.info(f"Checkpoint for step {step} saved successfully.")

    except Exception as e:
        log.error(f"Failed to save checkpoint at step {step}: {e}", exc_info=True)


def load_checkpoint(accelerator: Accelerator, checkpoint_dir):
    """Loads state from the latest checkpoint directory using Accelerate."""
    checkpoints = sorted(
        glob.glob(os.path.join(checkpoint_dir, "step_*")),
        key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else -1
    )
    checkpoints = [c for c in checkpoints if c.split("_")[-1].isdigit()]

    if not checkpoints:
        log.warning(f"No step_* checkpoints found in {checkpoint_dir}. Cannot load.")
        return 0 # Start from step 0 if no checkpoint

    checkpoint_to_load = checkpoints[-1]
    log.info(f"Attempting to load checkpoint from latest: {checkpoint_to_load}")

    try:
        accelerator.load_state(input_dir=checkpoint_to_load)
        log.info(f"Successfully loaded state from {checkpoint_to_load}")
        try:
            resume_step = int(checkpoint_to_load.split("_")[-1])
            log.info(f"Resuming training from step {resume_step + 1}")
        except (ValueError, IndexError):
             log.warning("Could not determine resume step from checkpoint directory name.")
             resume_step = 0
        accelerator.wait_for_everyone()
        return resume_step + 1

    except FileNotFoundError:
        log.error(f"Checkpoint directory not found or incomplete: {checkpoint_to_load}. Starting from scratch.")
        return 0
    except Exception as e:
        log.error(f"Failed to load checkpoint from {checkpoint_to_load}: {e}", exc_info=True)
        return 0


def load_pretrained_weights(model, accelerator, ckpt_path, reset_head=False):
    """Loads weights from a .safetensors file into the model (not a full training state)."""
    log.info(f"Loading pretrained weights from: {ckpt_path}")
    if not os.path.isdir(ckpt_path):
         log.error(f"Provided pretrained weights path is not a directory: {ckpt_path}")
         # Maybe allow loading a direct .safetensors file path too?
         if os.path.isfile(ckpt_path) and ckpt_path.endswith(".safetensors"):
              full_ckpt_path = ckpt_path
         else:
              return model # Return original model if path is invalid

    else: # It's a directory
         full_ckpt_path = os.path.join(ckpt_path, "model.safetensors")
         if not os.path.exists(full_ckpt_path):
             found_files = glob.glob(os.path.join(ckpt_path, "*.safetensors"))
             if not found_files:
                  log.error(f"No .safetensors file found in directory {ckpt_path}")
                  return model
             full_ckpt_path = found_files[0]
             log.warning(f"model.safetensors not found, loading {os.path.basename(full_ckpt_path)} instead.")


    try:
        loaded_dict = load_safetensors(full_ckpt_path)
        processed_dict = {}
        prefix_to_strip = "_orig_mod."
        for k, v in loaded_dict.items():
             if k.startswith(prefix_to_strip):
                  processed_dict[k[len(prefix_to_strip):]] = v
             else:
                  processed_dict[k] = v

        unwrapped_model = accelerator.unwrap_model(model)
        missing_keys, unexpected_keys = unwrapped_model.load_state_dict(processed_dict, strict=False)

        if missing_keys:
             log.warning(f"Weights not found in checkpoint for keys: {missing_keys}")
        if unexpected_keys:
             log.warning(f"Checkpoint contained weights not found in model: {unexpected_keys}")

        log.info(f"Successfully loaded weights from {full_ckpt_path}.")

        if reset_head:
            log.warning("Resetting LM head weights.")
            try:
                lm_head_key = None
                for name, param in unwrapped_model.named_parameters():
                    if "lm_head" in name or "vocab_head" in name:
                        lm_head_key = name
                        break

                if lm_head_key and hasattr(unwrapped_model, lm_head_key.split('.')[0]):
                    lm_head_param = unwrapped_model.get_parameter(lm_head_key)
                    try:
                         # Try to get hidden size from input embeddings
                         hidden_size = unwrapped_model.get_input_embeddings().weight.shape[-1]
                    except AttributeError:
                         # Fallback: try to get hidden size from the head itself (input dim)
                         hidden_size = lm_head_param.shape[-1]
                         log.warning("Could not get hidden size from embeddings, using head input dim.")

                    std_head = 1.0 / math.sqrt(hidden_size)
                    nn.init.trunc_normal_(lm_head_param, mean=0.0, std=std_head, a=-2 * std_head, b=2 * std_head)
                    log.info(f"Reset LM head '{lm_head_key}' with std {std_head:.4f}")
                else:
                    log.error("Could not find LM head parameter to reset.")

            except Exception as e:
                log.error(f"Failed to reset LM head: {e}", exc_info=True)

        return model

    except Exception as e:
        log.error(f"Failed to load pretrained weights from {full_ckpt_path}: {e}", exc_info=True)
        return model