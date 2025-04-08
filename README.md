# Quick LLaMa: Efficient and Flexible LLaMa

Quick Llama provides an optimized implementation of Llama 3.2 1B.
It is currently optimized to allow for **gradient-checkpoint-free fine-tuning and second-stage pre-training** of Llama 3.2 1B using modest settings.



<div align="center">
  <img src="https://github.com/samblouir/quick_llama/blob/main/quickllama.png?raw=true" alt="quick llama logo - a llama wearing a racing helmet and running so fast flames" width="200" />

  <h3>You are here (Quick LLaMa)!</h3>
</div>


<div align="center">
<a href="https://github.com/samblouir/quick_ssm/">
  <img src="https://github.com/samblouir/quick_ssm/blob/main/quickssm.png?raw=true" alt="quick ssm logo of a rocket" width="200" />
  <h3>Check out the Quick SSM repo!</h3>
  </a>
</div>


Code is included to:
- Automatically download Llama's weights from HuggingFace.
    
	> *(Note: You must set the environment variable `HF_TOKEN` with your HuggingFace token which has been granted access to [Meta's Llama 3.2 repository](https://huggingface.co/meta-llama/Llama-3.2-1B).)*
- Transfer the weights to the new, more efficient implementation.
- Instruction-tune the model with regular checkpointing.
- Evaluate the model with EleutherAI's LM Harness.
It currently supports distributed data parallelism, using Torch's DistributedDataParallel wrapped by HuggingFace Accelerate.




This implementation includes techniques and code found in:
* [Splash Attention](https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_kernel.py)
* [Flex Attention](https://pytorch.org/blog/flexattention/)
* [Cut Your Losses in Large-Vocabulary Language Models](https://github.com/apple/ml-cross-entropy)
* [Unsloth.AI Triton-based Rotary Position Encodings](https://github.com/unslothai/)
* [Meta's Llama Rotary Position Encodings](https://github.com/meta-llama/llama/blob/main/llama/model.py)
* [HuggingFace's Llama 3.2 Implementation](https://huggingface.co/meta-llama/Llama-3.2-1B)

This codebase is in use/has been used for several papers for ACL, EMNLP, and NeurIPS 2025.






## Installation

You can install the package directly from GitHub:

```bash
pip install git+https://github.com/samblouir/quick_llama
```

Alternatively, install it in editable mode for development:

```bash
git clone https://github.com/samblouir/quick_llama
cd quick_llama
pip install -e .
```


## Key Features
- **Prefix-LM and Sequence Packing**: Supports bidirectional prefix language modeling (Prefix-LM) and sequence packing, allowing for efficient batch-free training on long sequences without samples interfering with each other.
- **Efficient Attention with optional Sparsity:** Leverages Flex Attention for sparse Attention.
- **Integrated Flex Attention**: Flex Attention is the core Attention implementation in this codebase, allowing for a highly flexible attention computation. Block sparsity masks are dynamnically created per batch and shared across all Attention layers.
- **Grouped Query Attention**: Queries can share keys.
- **FP16 Communication**: Supports FP16 communication for distributed training, greatly reducing overhead.
- **Overlapping Backward Computation and Communication**: The backward pass is overlapped with communication, reducing distributed training overhead.
- **Torch Compile**: Support for `torch.compile` for optimized performance.
- **EleutherAI LM Harness Support**: Support for EleutherAI's LM Harness is built-in, allowing for evaluation on max-likelihood tasks.
- **Cached and Triton-based Rotary Positional Encodings**: Utilizes cached rotary positional embeddings applied using Triton.


## Repository Structure

## Tested Setup
 4x A100 80GB GPUs, modest sequence lengths (4096+) and smaller batch sizes (32-64).

## TODO:
- Merge in support for [Birdie's reward-driven mixture-of-denoisers for second-stage pretraining](https://www.github.com/samblouir/birdie) from BirdieDNA.
- Merge in support for fine-tuning Llama Vision 11B
- Merge in support for finetuning LoRA adapters
- Merge in support for exporting back to a VLLM and HuggingFace-compatible save
- Finish sequence parallelism
- Add Gradient Checkpointing for larger Llama models and/or batch sizes and sequence lengths.


## Contributing

Contributions are welcome!
Please feel free to open an issue to report bugs or suggest features.

## License

Apache 2.0
