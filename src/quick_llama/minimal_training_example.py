"""
This example shows:
1) Creating a minimal config
2) Loading a Llama model with our faster implementation
3) Tokenizing an input
4) Using packer_batcher to pack the input into a batch
5) Converting the batch to torch tensors
6) Preparing the model and optimizer
7) Running a training loop
8) Saving the model
9) Loading the model


Usage:
```
pip install git+https://www.github.com/SamBlouir/quick_llama
NUM_LOCAL_GPUS=1
accelerate launch --num_processes ${NUM_LOCAL_GPUS} minimal_training_example.py
```

"""

import os
from models import model_utils
import numpy as np
import torch
import torch.nn as nn
from models.model_arch import load_model
from packer_batcher import Batcher
import config as cfg
import data_utils
import packer_batcher

def main():
    # 1) Minimal config with overrides
    
    # --- 1. Configuration & Initialization ---
    config = cfg.get_config()
    accelerator = cfg.setup_accelerator(config)

    # Load the tokenizer and dataset
    model_config = {
        "dtype": "bfloat16", # parameter and compute dtype
        "sequence_length": 128,
        "batch_size": 1,
        "minimum_sequence_length": 0,
        "num_training_steps": 5,
    }

    # 2) Load in Llama 1B with our faster implementation
    model = load_model(config=model_config)
    print(f"Model parameter count: {sum(p.numel() for p in model.parameters())}")

    # 3) Tokenize two strings
    input_hello = data_utils.tokenize_text("This is getting out of hand, ")
    label_world = data_utils.tokenize_text("now there are two of them!")
    
    # 4) Use packer_batcher to pack them into a single batch
    #    We'll define a small sequence_length for this demonstration.
    batcher = Batcher(config=model_config)
    batcher.add(input_ids=input_hello, label_ids=label_world)

    # Retrieve the packed batch dictionary
    # Normally we'd check to see if batcher is ready to pop since it can't hold any more samples.
    status = batcher.is_ready()
    print(f"  status: {status}")
        
    # but for this example we'll just pop the batch.
    batch = batcher.pop()

    print(f"*" * 60,)
    # Debugging function to check alignment
    packer_batcher.debug_alignments(batch)
    print(f"*" * 60,)

    # 5) Convert from numpy arrays to torch Tensors and send them to the accelerator
    def send_to_gpu(batch):
        new_batch = {}
        for k, v in batch.items():
            new_batch[k] = torch.tensor(v, dtype=torch.int32).to(accelerator.device)
        return new_batch
    gpu_batch = send_to_gpu(batch)

    # 6) Prepare the model and optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    # 7) Example training loop
    num_steps = 3
    for step in range(num_steps):
        optimizer.zero_grad()
        loss = model(**gpu_batch)
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()

        if accelerator.is_main_process:
            print(f"Step {step+1}/{num_steps} - Loss: {loss.item():.4f}")

    print("Training complete.")

    # 8) Save the model
    last_step = step
    model_utils.save_checkpoint(
        accelerator,
        model,
        optimizer,
        lr_scheduler,
        last_step,
        config["checkpoint_dir"],
        config,
        config.get("save_limit", 3),
    )

    # 9) Load the model
    # model, optimizer, lr_scheduler, are updated in-place

    checkpoint_to_load = os.path.join(config['checkpoint_dir'], f"step_{int(last_step)}")
    _, loaded_step = model_utils.load_checkpoint(
        accelerator, checkpoint_to_load, model, optimizer, lr_scheduler
    )

    assert(last_step + 1 == loaded_step), f"Loaded step {loaded_step} does not match expected step {last_step}."

    print(f"Checkpoint from step {last_step} loaded successfully.")

if __name__ == "__main__":
    main()
