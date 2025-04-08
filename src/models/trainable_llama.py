"""
Provides the TrainableLlama class, which encapsulates:
    - Creation/loading of a BaseModel using a config dictionary
    - Preparation of optimizer & scheduler
    - A convenient train_step method for usage in training loops
"""

import torch
from typing import Optional, Any, Dict, Union
import transformers
from accelerate import Accelerator
import torch.nn as nn
import logging

try:
    from .model_arch import BaseModel, load_model as wrap_load_model
except ImportError:
    from model_arch import BaseModel, load_model as wrap_load_model

log = logging.getLogger(__name__)

def prepare_optimizer_and_scheduler(model: nn.Module, config: Dict[str, Any]):
    """
    Prepare an AdamW optimizer and a learning rate scheduler using huggingface/transformers helpers.

    Args:
        model (nn.Module): The model with parameters to optimize.
        config (Dict[str, Any]): A config dictionary expected to contain:
            - 'num_training_steps'
            - 'lr_schedule'
            - 'num_warmup_steps' or 'num_warmup_steps_ratio'
            - 'lr'
            - 'adam_epsilon'
            - 'weight_decay'

    Returns:
        (torch.optim.Optimizer, transformers.optimization): The configured optimizer and LR scheduler.
    """
    if "num_training_steps" not in config:
        log.warning(
            "'num_training_steps' not found in config. Estimating based on placeholder or defaults."
        )
        num_steps = config.get("estimated_num_training_steps", 10000)
    else:
        num_steps = config["num_training_steps"]

    warmup_ratio = config.get("num_warmup_steps_ratio", 0.05)
    num_warmup_steps = config.get("num_warmup_steps", int(num_steps * warmup_ratio))
    lr_schedule = config.get("lr_schedule", "linear").lower()

    optimizer_grouped_parameters = [p for p in model.parameters() if p.requires_grad]
    if not optimizer_grouped_parameters:
        raise ValueError("No trainable parameters found in the model.")

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=config.get("lr", 1e-4),
        eps=config.get("adam_epsilon", 1e-8),
        weight_decay=config.get("weight_decay", 0.01),
    )

    log.info(
        f"Optimizer: AdamW (LR={config.get('lr', 1e-4)}, WD={config.get('weight_decay', 0.01)})"
    )
    log.info(
        f"Scheduler: {lr_schedule}, Total Steps: {num_steps}, Warmup Steps: {num_warmup_steps}"
    )

    if lr_schedule == "linear":
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_steps
        )
    elif lr_schedule == "cosine":
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_steps
        )
    elif lr_schedule == "constant":
        scheduler = transformers.get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps
        )
    else:
        log.warning(
            f"LR schedule '{lr_schedule}' not recognized or implemented. Using constant schedule as fallback."
        )
        scheduler = transformers.get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps
        )

    return optimizer, scheduler

class TrainableLlama:
    """
    Wrapper class for a trainable LLaMA-like model.

    Handles:
        - Model construction/loading
        - Optimizer & scheduler setup
        - Single-step training logic (train_step)
    """
    def __init__(self, config: Dict[str, Any], accelerator: Accelerator):
        """
        Initialize the TrainableLlama.

        Args:
            config (Dict[str, Any]): Configuration dictionary passed to the base model and training utilities.
            accelerator (Accelerator): The Hugging Face Accelerate Accelerator instance.

        Raises:
            ValueError: If accelerator is None.
        """
        self.config = config
        if accelerator is None:
            raise ValueError("Accelerator instance must be provided to TrainableLlama.")
        self.accelerator = accelerator

        log.info("Loading/Creating model via wrap_load_model (from model_arch)...")
        self.model = wrap_load_model(config)
        log.info("Model loaded/created.")

        log.info("Preparing optimizer and scheduler...")
        self.optimizer, self.scheduler = prepare_optimizer_and_scheduler(
            self.model, config
        )
        log.info("Optimizer and scheduler created.")

        log.info("Preparing model, optimizer, scheduler with Accelerator...")
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler
        )
        log.info("Accelerator preparation complete.")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Perform a single training step given a batch of data.

        Args:
            batch (Dict[str, torch.Tensor]): Should contain keys for 'input_ids' (and 'label_ids' if computing loss).

        Returns:
            torch.Tensor: Scalar loss for this step.
        """
        self.model.train()
        loss = self.model(**batch, return_loss=True, reduction="mean")
        return loss

    def unwrapped_optimizer(self) -> torch.optim.Optimizer:
        """
        Retrieve the raw (unwrapped) optimizer.

        Useful when needing to manipulate or inspect the optimizer outside
        the Accelerator environment.

        Returns:
            torch.optim.Optimizer: The underlying optimizer object.
        """
        return self.optimizer
