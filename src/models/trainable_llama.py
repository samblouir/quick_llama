"""
trainable_llama.py

Provides a TrainableLlama class that integrates a custom LLaMA-like model
with optimizer & scheduler, using Accelerate for distributed training.
"""

import torch
from typing import Optional, Any, Dict, Union
import transformers
from accelerate import Accelerator
import torch.nn as nn

from model_arch import BaseModel
from wrap_model import load_model as wrap_load_model

def prepare_optimizer_and_scheduler(model: nn.Module, config: Dict[str, Any]):
    """
    Create and return an optimizer and a scheduler based on config.
    """
    num_steps = config.get("num_training_steps", 200)
    num_warmup_steps = config.get("num_warmup_steps", int(num_steps * 0.05))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        eps=config.get("adam_epsilon", 1e-8),
    )
    if config["lr_schedule"] == "linear":
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_steps
        )
    elif config["lr_schedule"] == "cosine":
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_steps
        )
    else:
        # fallback: constant
        scheduler = transformers.get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps
        )
    return optimizer, scheduler

def _train_step(
    batch: Dict[str, torch.Tensor],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    accelerator: Accelerator,
    config: Dict[str, Any],
    return_loss: bool = True
):
    """
    One training step with optional gradient clipping and stepping the scheduler.
    """
    with accelerator.autocast():
        optimizer.zero_grad()
        loss = model(**batch)
        accelerator.backward(loss)
        if config.get("clip_grad_norm", 1.0) > 0:
            accelerator.clip_grad_norm_(model.parameters(), config["clip_grad_norm"])
        optimizer.step()
        scheduler.step()

    if return_loss:
        return loss.detach().cpu()
    return None

class TrainableLlama:
    """
    A wrapper class that loads a custom LLaMA-like model, sets up optimizer & scheduler,
    and provides a training step method for convenience.
    """

    def __init__(self, config: Dict[str, Any], accelerator: Optional[Accelerator] = None):
        self.config = config
        self.accelerator = accelerator or config["accelerator"]
        self.model = self.load_model(config)
        self.optimizer, self.scheduler = self.load_optimizer_scheduler(config)

    def load_model(self, config: Dict[str, Any]) -> nn.Module:
        """
        Load or instantiate the model. This could be replaced with your custom architecture.
        For demonstration, we call wrap_load_model, but in practice you'd use BaseModel or similar.
        """
        # If you want to skip "wrap_load_model", you can do:
        #   model = BaseModel(config)
        model = wrap_load_model(config)
        return self.accelerator.prepare(model)

    def load_optimizer_scheduler(self, config: Dict[str, Any]):
        optimizer, scheduler = prepare_optimizer_and_scheduler(self.model, config)
        optimizer, scheduler = self.accelerator.prepare(optimizer, scheduler)
        return optimizer, scheduler

    def train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Perform a single training step.
        """
        loss = _train_step(
            batch,
            self.model,
            self.optimizer,
            self.scheduler,
            self.accelerator,
            self.config,
            return_loss=True
        )
        return loss

    def unwrapped_optimizer(self) -> torch.optim.Optimizer:
        """Return the unwrapped optimizer for advanced inspection."""
        return self.accelerator.unwrap_model(self.optimizer)

    def __call__(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Shortcut to call train_step directly."""
        return self.train_step(batch)