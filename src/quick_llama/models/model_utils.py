"""
model_utils.py

Utility functions for saving/loading checkpoints and pretrained weights.
    - save_checkpoint: Saves model/optimizer/lr_scheduler state.
    - load_checkpoint: Loads the most recent or specified checkpoint.
    - load_pretrained_weights: Loads pretrained weights from safetensors into a model.
"""

import os
import torch
import torch.nn as nn
import math
from safetensors.torch import load_file as load_safetensors, save_file
from accelerate import Accelerator
from accelerate.utils import save_accelerator_state, load_accelerator_state
from accelerate.logging import get_logger
import glob
import shutil
import json
import logging
from typing import Union

try:
    from .model_arch import BaseModel
except ImportError:
    try:
        from model_arch import BaseModel
    except ImportError:
        logging.warning("Could not import BaseModel for type hinting in model_utils.")
        BaseModel = nn.Module

accel_log = get_logger(__name__)
log = logging.getLogger(__name__)

def save_checkpoint(
    accelerator: Accelerator,
    model,
    optimizer,
    lr_scheduler,
    step,
    checkpoint_dir,
    config,
    keep_limit=3,
):
    """
    Save the entire training state using Accelerate's 'save_state'.
    Also saves a JSON version of the config.

    Args:
        accelerator (Accelerator): Accelerate instance to handle the saving.
        model (nn.Module): The model being trained (or wrapped model).
        optimizer (torch.optim.Optimizer): The optimizer in use.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): The LR scheduler in use.
        step (int): The current training step or iteration.
        checkpoint_dir (str): The directory where checkpoints are stored.
        config (dict): Training configuration, saved as JSON for reproducibility.
        keep_limit (int, optional): How many checkpoints to keep. Older ones are removed. Defaults to 3.

    Returns:
        str: The path to the saved checkpoint directory, or None if saving failed.
    """
    checkpoint_save_dir = os.path.join(checkpoint_dir, f"step_{step}")
    accel_log.info(
        f"Saving checkpoint for step {step} to: {checkpoint_save_dir}",
        main_process_only=True,
    )

    try:
        # Use Accelerator to save the training state (model, optimizer, scheduler, etc.)
        accelerator.save_state(output_dir=checkpoint_save_dir)

        if accelerator.is_main_process:
            # Also save a JSON config file for future reference
            config_path = os.path.join(checkpoint_save_dir, "training_config.json")
            try:
                config_to_save = {
                    k: v
                    for k, v in config.items()
                    if isinstance(v, (str, int, float, bool, list, dict, type(None)))
                }
                with open(config_path, "w") as f:
                    json.dump(config_to_save, f, indent=4)
            except Exception as e:
                log.error(f"Failed to save config with checkpoint: {e}", exc_info=True)

            # Prune older checkpoints if keep_limit is set
            if keep_limit is not None and keep_limit > 0:
                try:
                    checkpoints = sorted(
                        [
                            d
                            for d in glob.glob(os.path.join(checkpoint_dir, "step_*"))
                            if os.path.isdir(d)
                        ],
                        key=lambda x: int(x.split("_")[-1]),
                    )
                    if len(checkpoints) > keep_limit:
                        checkpoints_to_delete = checkpoints[:-keep_limit]
                        log.info(
                            f"Pruning old checkpoints: Found {len(checkpoints)}, keeping {keep_limit}."
                        )
                        for ckpt_path in checkpoints_to_delete:
                            log.info(f"  Deleting old checkpoint: {ckpt_path}")
                            try:
                                shutil.rmtree(ckpt_path)
                            except OSError as e_rm:
                                log.error(
                                    f"    Error deleting checkpoint {ckpt_path}: {e_rm}"
                                )
                except Exception as e_prune:
                    log.error(
                        f"Error during checkpoint pruning: {e_prune}", exc_info=True
                    )

        accelerator.wait_for_everyone()
        accel_log.info(
            f"Checkpoint for step {step} saved successfully.", main_process_only=True
        )
        return checkpoint_save_dir

    except Exception as e:
        log.error(f"Failed to save checkpoint at step {step}: {e}", exc_info=True)
        accelerator.wait_for_everyone()
        return None

def load_checkpoint(
    accelerator: Accelerator, checkpoint_dir_or_path, model, optimizer, lr_scheduler
):
    """
    Load a checkpoint from the specified directory or from "latest" inside that directory.

    Args:
        accelerator (Accelerator): The Accelerate instance to handle loading state.
        checkpoint_dir_or_path (str): The directory containing the checkpoints, or "latest".
        model (nn.Module): The model to load into.
        optimizer (torch.optim.Optimizer): The optimizer to load state into.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): The LR scheduler to load state into.

    Returns:
        tuple (str, int):
            - Path to the loaded checkpoint directory, or None if load failed.
            - The next training step (resume_step).
    """
    checkpoint_to_load = None
    if not checkpoint_dir_or_path or not isinstance(checkpoint_dir_or_path, str):
        log.error(f"Invalid checkpoint path provided: {checkpoint_dir_or_path}")
        return None, 0

    if checkpoint_dir_or_path.lower() == "latest":
        parent_dir = os.path.dirname(model.config.get("_name_or_path", "."))
        log.warning(
            f"Attempting to find latest checkpoint in parent directory: {parent_dir}. Specify explicit path if this fails."
        )
        if not os.path.isdir(parent_dir):
            log.error(
                f"'latest' specified, but parent directory '{parent_dir}' not found."
            )
            return None, 0
        checkpoint_dir_or_path = parent_dir

    if os.path.isdir(checkpoint_dir_or_path):
        step_folders = sorted(
            [
                d
                for d in glob.glob(os.path.join(checkpoint_dir_or_path, "step_*"))
                if os.path.isdir(d)
            ],
            key=lambda x: int(x.split("_")[-1]),
        )
        if step_folders:
            checkpoint_to_load = step_folders[-1]
            accel_log.info(
                f"Found latest checkpoint: {checkpoint_to_load}", main_process_only=True
            )
        elif (
            os.path.exists(os.path.join(checkpoint_dir_or_path, "pytorch_model.bin"))
            or os.path.exists(os.path.join(checkpoint_dir_or_path, "model.safetensors"))
            or os.path.exists(os.path.join(checkpoint_dir_or_path, "optimizer.bin"))
        ):
            checkpoint_to_load = checkpoint_dir_or_path
            accel_log.info(
                f"Using provided path as specific checkpoint directory: {checkpoint_to_load}",
                main_process_only=True,
            )
        else:
            log.error(
                f"No valid checkpoint steps or state files found in directory: {checkpoint_dir_or_path}"
            )
            return None, 0
    else:
        log.error(
            f"Checkpoint path not found or is not a directory: {checkpoint_dir_or_path}"
        )
        return None, 0

    accel_log.info(
        f"Attempting to load checkpoint state from: {checkpoint_to_load}",
        main_process_only=True,
    )

    try:
        accelerator.load_state(input_dir=checkpoint_to_load)
        accel_log.info(
            f"Successfully loaded state from {checkpoint_to_load}",
            main_process_only=True,
        )

        resume_step = 0
        try:
            step_str = os.path.basename(checkpoint_to_load).split("_")[-1]
            if step_str.isdigit():
                resume_step = int(step_str) + 1
                accel_log.info(
                    f"Resuming training from step {resume_step}", main_process_only=True
                )
            else:
                log.warning(
                    f"Could not parse step number from checkpoint directory name: {checkpoint_to_load}"
                )
        except (ValueError, IndexError, TypeError):
            log.warning(
                f"Could not determine resume step from checkpoint directory name: {checkpoint_to_load}"
            )

        accelerator.wait_for_everyone()
        return checkpoint_to_load, resume_step

    except FileNotFoundError as e:
        log.error(f"Checkpoint file/directory not found during load: {e}", exc_info=True)
    except Exception as e:
        log.error(f"Failed to load checkpoint from {checkpoint_to_load}: {e}", exc_info=True)

    accelerator.wait_for_everyone()
    return None, 0

def load_pretrained_weights(
    model: Union[BaseModel, nn.Module],
    accelerator: Accelerator,
    ckpt_path: str,
    strict: bool = False,
):
    """
    Load pretrained weights from a safetensors file into the model, optionally with strict key matching.

    Args:
        model (BaseModel or nn.Module): The model into which we load weights.
        accelerator (Accelerator): For unwrapping the model in distributed training.
        ckpt_path (str): Directory containing 'model.safetensors' or the direct .safetensors file path.
        strict (bool, optional): If True, will raise an error for unexpected or missing keys. Defaults to False.

    Returns:
        nn.Module: The model with updated state_dict (if load is successful).
    """
    accel_log.info(
        f"Loading pretrained weights from: {ckpt_path}", main_process_only=True
    )

    if os.path.isdir(ckpt_path):
        full_ckpt_path = os.path.join(ckpt_path, "model.safetensors")
    elif os.path.isfile(ckpt_path) and ckpt_path.endswith(".safetensors"):
        full_ckpt_path = ckpt_path
    else:
        log.error(
            f"Invalid checkpoint path for safetensors: {ckpt_path}. Must be dir or .safetensors file."
        )
        return model

    if not os.path.exists(full_ckpt_path):
        log.error(f"Pretrained weights file not found: {full_ckpt_path}")
        return model

    try:
        loaded_dict = load_safetensors(full_ckpt_path, device="cpu")
        processed_dict = {}
        prefix_to_strip = "_orig_mod."

        # Sometimes there's a prefix in keys (e.g., from FSDP). Remove it if present.
        for k, v in loaded_dict.items():
            if k.startswith(prefix_to_strip):
                processed_dict[k[len(prefix_to_strip) :]] = v
            else:
                processed_dict[k] = v

        unwrapped_model = accelerator.unwrap_model(model)
        missing_keys, unexpected_keys = unwrapped_model.load_state_dict(
            processed_dict, strict=strict
        )

        if missing_keys:
            log.warning(f"Weights not found in checkpoint for keys: {missing_keys}")
        if unexpected_keys:
            log.warning(f"Checkpoint contained weights not found in model: {unexpected_keys}")

        if strict and (missing_keys or unexpected_keys):
            raise RuntimeError("Strict weight loading failed due to missing or unexpected keys.")

        accel_log.info(
            f"Successfully loaded weights from {full_ckpt_path} into unwrapped model.",
            main_process_only=True,
        )
        return model

    except Exception as e:
        log.error(
            f"Failed to load pretrained weights from {full_ckpt_path}: {e}",
            exc_info=True,
        )
        return model
