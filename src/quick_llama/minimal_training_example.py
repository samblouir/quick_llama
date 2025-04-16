"""
This example shows:
1) Loading in LLaMA 1B model via model_arch.load_model().
2) Tokenizing "Hello" and "World" and using them as input_ids and label_ids for the data preprocessor.
4) Showing the outputs of the data preprocessor
5) Showing a short training loop

Usage:
NUM_LOCAL_GPUS=1
accelerate launch --num_processes ${NUM_LOCAL_GPUS} minimal_training_example.py

"""
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
    config = {
        "dtype": "bfloat16", # parameter and compute dtype
        "sequence_length": 1024,
        "batch_size": 1,
        "minimum_sequence_length": 0,
    }

    # 2) Load in Llama 1B with a faster implementation
    model = load_model(config=config)
    print(f"Model parameter count: {sum(p.numel() for p in model.parameters())}")

    # 3) "Tokenize" two strings. Here we hardcode numeric IDs for brevity.
    #    label_ids are the same as input_ids for a standard LM.
    input_hello = data_utils.tokenize_text("Hello")
    label_world = data_utils.tokenize_text("World")
    
    # 4) Use packer_batcher to pack them into a single batch
    #    We'll define a small sequence_length for this demonstration.
    batcher = Batcher(config=config)
    batcher.add(input_ids=input_hello, label_ids=label_world)   # Add first example

    # Retrieve the packed batch dictionary
    # Normally we'd check to see if batcher is ready to pop since it can't hold any more samples.
    ## status = batcher.is_ready()
    # but for this example we'll just pop the batch.
    batch = batcher.pop()

    print(f"*" * 60,)
    # Debugging function to check alignment
    packer_batcher.debug_alignments(packer_batcher)
    print(f"*" * 60,)

    # 5) Convert from numpy arrays to torch Tensors and reshape to [batch_size, seq_len].
    #    We'll keep batch_size=1 from the Batcher in this example.
    def send_to_gpu(batch):
        new_batch = {}
        for k, v in batch.items():
            if isinstance(v, np.ndarray):
                new_batch[k] = torch.tensor(v).to(accelerator.device)
            else:
                new_batch[k] = v
        return new_batch
    gpu_batch = send_to_gpu(batch)

    # 6) Prepare the model and optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model, optimizer = accelerator.prepare(model, optimizer)

    # 7) Example training loop
    num_steps = 3
    for step in range(num_steps):
        optimizer.zero_grad()
        loss = model(**gpu_batch)
        accelerator.backward(loss)
        optimizer.step()

        if accelerator.is_main_process:
            print(f"Step {step+1}/{num_steps} - Loss: {loss.item():.4f}")

    print("Training complete.")

if __name__ == "__main__":
    main()
