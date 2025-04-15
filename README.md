# Quick LLaMa: Efficient and Flexible LLaMa

# This codebase is currently being refactored for Github and is/has been used for submissions to ACL, EMNLP, and NeurIPS 2025.

Quick Llama provides an optimized implementation targeting **Llama 3.2 1B**. It is designed for efficient **gradient-checkpoint-free training** on 80GB GPUs.


---
<div align="center">
  <img src="https://github.com/samblouir/quick_llama/blob/main/quickllama.png?raw=true" alt="quick llama logo - a llama wearing a racing helmet and running so fast flames" width="200" />

  <h3>You are here (Quick LLaMa)!</h3>
</div>

---
<div align="center">
<a href="https://github.com/samblouir/quick_ssm/">
  <img src="https://github.com/samblouir/quick_ssm/blob/main/quickssm.png?raw=true" alt="quick ssm logo of a rocket" width="200" />
  <h3>Check out the Quick SSM repo!</h3>
  </a>
</div>

---

## Key Features

* **Optimized Llama 3.2 1B:** A Torch-based implementation with some **Triton kernels** and **FlexAttention**.
* **Efficient Training:** Enables gradient-checkpoint-free training to reduce compute: instruction tune on 1 million training examples within two hours using 4x A100 80GB with a sequence length of 4096 and a global batch size of 32.
* **Distributed Training:** Built-in support for distributed data parallelism using [HuggingFace Accelerate](https://huggingface.co/docs/accelerate). Defaults to FP16 communication and overlapped backwards pass computation and communication.
* **Efficient Attention Implementation:** Leverages [Flex Attention](https://pytorch.org/blog/flexattention/) with efficient block-sparsity masks and proper sequence packing. This allows us to simply skip computing on tokens we would otherwise mask out, like padding tokens, or tokens from sequences other than a specific sample.
* **Optimized Vocab. Head and RoPE:** Includes cached Triton-based Rotary Positional Encodings (RoPE) and efficient cross-entropy implementations (from Apple and Unsloth.AI, with some tweaks to fix negative loss issues).
* **Flexible Data Handling:** Included data pre-processor automatically adds necessary auxiliary data for efficient FlexAttention, enabling block-sparsity masks and sequence packing. The processor also allows for proper sequence packing for SSMs and prefix-LMs.
* **Utilities Included:**
    * Automatic weight download and conversion of Llama 3.2 1B's weights from HuggingFace to our Torch format.
    * Example training script with configuration management, logging (TensorBoard support via Accelerate), and checkpointing.
    * Basic command-line interface for conversing with the model.
    * Evaluation using [EleutherAI's LM Harness](https://github.com/EleutherAI/lm-evaluation-harness).
* **Performance:** Compatible with `torch.compile` for optimal performance.


## Quick Llama includes techniques and code inspired by or adapted from:

* [Splash Attention](https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py)
* [FlexAttention](https://pytorch.org/blog/flexattention/)
* [Apple's Cross-Entropy Loss](https://github.com/apple/ml-cross-entropy)
* [Unsloth.AI](https://github.com/unslothai/) (RoPE Triton Kernels and Cross-Entropy Loss)
* [Meta's Llama Reference Implementation](https://github.com/meta-llama/llama/blob/main/llama/model.py) (RoPE)
* [HuggingFace's Llama 3.2 Implementation](https://huggingface.co/meta-llama/Llama-3.2-1B)



## Tested Setup

* Hardware: 4x A100 80GB GPUs
* Parameters: Sequence lengths >= 4096, batch sizes 32-64.

## Requirements

* Python >= 3.8
* PyTorch >= 2.5
* Triton >= 3.1
* Hugging Face Libraries: `transformers`, `accelerate`, `datasets`, `huggingface-hub`
* Other libraries: `safetensors`, `numpy`, `tqdm`

**Hugging Face Token:**
You must set the environment variable `HF_TOKEN` with your HuggingFace token that has been granted access to [Meta's Llama 3.2 repository](https://huggingface.co/meta-llama/Llama-3.2-1B) to download the weights.

```bash
export HF_TOKEN="your_hf_token_here"
python quick_llama/minimal_training_example.py

## Alternatively:

HF_TOKEN="your_hf_token_here"python quick_llama/minimal_training_example.py

```

## Installation

It is highly recommended to install in a virtual environment (e.g., using `venv` or `conda`).

```bash
# Create and activate a virtual environment (example using venv)
python -m venv .venv
source .venv/bin/activate # On Linux/macOS
# .venv\Scripts\activate # On Windows

# Clone the repository
git clone https://github.com/samblouir/quick_llama
cd quick_llama

# Install the base package in editable mode
pip install -e .

```

## Usage

This project uses HuggingFace Accelerate for handling distributed training and device placement.

```bash
# Minimal Example
# This will download the model weights and run a minimal training example.
# It also shows prints out a processed batch by the the data processor, which adds necessary auxiliary information for the Flex Attention block sparsity mask used.
python quick_llama/minimal_training_example.py
```

```bash

# 1 GPU on a local machine
python train_wrapper.py \
	--dataset_name "teknium/OpenHermes-2.5" \
	--batch_size_per_device 32 \
	--gradient_accumulation_steps 4 \
	--num_epochs 1 \
	--lr 5e-5 \
	--output_dir "./training_logs" \
	--steps_between_evals 200

```

```bash
# 4 GPUs on a local machine
accelerate launch --num_gpus 4 train_wrapper.py \
	--dataset_name "teknium/OpenHermes-2.5" \
	--batch_size_per_device 8 \
	--gradient_accumulation_steps 4 \
	--num_epochs 1 \
	--lr 5e-5 \
	--output_dir "./training_logs" \
	--steps_between_evals 200

```

```bash
# Multi-node setup
accelerate launch --num_processes 8 --num_machines 2 --machine_rank 0 train_wrapper.py \
	--dataset_name "teknium/OpenHermes-2.5" \
	--batch_size_per_device 8 \
	--gradient_accumulation_steps 4 \
	--num_epochs 1 \
	--lr 5e-5 \
	--output_dir "./training_logs" \
	--steps_between_evals 200
```


### Configuration

Training runs are configured via command-line arguments. Key arguments (defined in `config.py`) include:

* `--model_name`: HF model identifier (default: `meta-llama/Llama-3.2-1B`).
* `--dataset_name`: HF dataset identifier (default: `teknium/OpenHermes-2.5`).
* `--output_dir`: Directory to save logs and checkpoints (default: `logs/`).
* `--num_epochs`: Number of training epochs.
* `--batch_size_per_device`: Batch size for each GPU.
* `--lr`: Learning rate.
* `--steps_between_evals`: How often to run validation.
* `--mixed_precision`: `bf16` (default), `fp16`, or `no`.
* ... and others (see `config.py` or run `python train.py --help`).

Configuration files (`config.json`), logs, TensorBoard data, and checkpoints will be saved under `{output_dir}/{config_hash}/`.

### Training

Use `accelerate launch` to start a training run. Accelerate will automatically handle multi-GPU/multi-node setups based on its configuration.

1.  **Configure Accelerate (if needed):**
    Run `accelerate config` and follow the prompts to set up your distributed environment (number of GPUs, etc.).

2.  **Launch Training:**
    Adjust arguments as needed.

    ```bash
    accelerate launch train.py \
        --dataset_name "teknium/OpenHermes-2.5" \
        --batch_size_per_device 8 \
        --gradient_accumulation_steps 4 \
        --num_epochs 1 \
        --lr 5e-5 \
        --output_dir "./training_logs" \
        --steps_between_evals 200 \
        # Add other arguments from config.py parse_arguments() function as needed
    ```

	config.py get_config() automatically sets log_dir and checkpoint_dir.
    * Logs will be printed to the console and the project directory.
    * Checkpoints are saved periodically in the `output_dir`.
    * TensorBoard logs can be viewed by running `tensorboard --logdir ./training_logs`.

### Trouble-shooting

```bash
# Nuclear option for flash-attn issues:
## Re-installs flash-attn and dependencies
## Warning: This will changes the versions of some packages
pip uninstall flash-attn -y
pip install flash-attn --no-build-isolation --force-reinstall
```


### Evaluation (EleutherAI LM Harness Adapter)

Documentation for this module is coming soon.
This allows you to run the EleutherAI LM Harness without needing to re-export the model to the HuggingFace format and running lm_eval.

## TODO:

* Merge in support for [Birdie's reward-driven mixture-of-denoisers](https://www.github.com/samblouir/birdie) for second-stage pretraining.
* Merge in support for fine-tuning Llama Vision 11B.
* Merge in support for finetuning LoRA adapters.
* Merge in support for exporting back to VLLM and HuggingFace-compatible formats.
* Finish sequence parallelism implementation.
* Add optional Gradient Checkpointing for larger models/settings.
* Add detailed instructions for LM Harness evaluation and CLI inference.


## Contributing

Contributions are welcome! Please feel free to open an issue to report bugs or suggest features. Pull requests are also appreciated.

## License

Apache 2.0