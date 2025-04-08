import argparse
import os
import datetime
from accelerate import Accelerator, DDPCommunicationHookType, InitProcessGroupKwargs, DistributedDataParallelKwargs
import cache
import json

DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_DATASET_NAME = "teknium/OpenHermes-2.5"
DEFAULT_OUTPUT_DIR = "logs"


def get_default_config():
    """Returns a dictionary with default configuration values."""
    return {
        "model_name": DEFAULT_MODEL_NAME,
        "dataset_name": DEFAULT_DATASET_NAME,
        "sequence_length": 4096,
        "minimum_sequence_length": 64,
        "num_epochs": 1,
        "batch_size_per_device": 32,
        "gradient_accumulation_steps": 1,
        "lr": 1e-4,
        "weight_decay": 0.1,
        "adam_epsilon": 1e-8,
        "num_warmup_steps_ratio": 0.05,
        "clip_grad_norm": 1.0,
        "softmax_temperature": 1.0,
        "steps_between_evals": 1024,
        "save_limit": 3,
        "output_dir": DEFAULT_OUTPUT_DIR,
        "seed": 42,
        "mixed_precision": "bf16",
        "communication_timeout_seconds": 3000,
        "use_eval_cache": True,
        "test_set_path": None, # Allow overriding the default test set path
    }

def parse_arguments():
    """Parses command-line arguments to override defaults."""
    parser = argparse.ArgumentParser(description="Train a Llama model using Accelerate.")

    parser.add_argument("--model_name", type=str, help=f"Model name from Hugging Face Hub (default: {DEFAULT_MODEL_NAME})")
    parser.add_argument("--dataset_name", type=str, help=f"Dataset name from Hugging Face Hub (default: {DEFAULT_DATASET_NAME})")
    parser.add_argument("--sequence_length", type=int, help="Maximum sequence length.")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--batch_size_per_device", type=int, help="Batch size per GPU.")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="Number of steps to accumulate gradients over.")
    parser.add_argument("--lr", type=float, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, help="Weight decay.")
    parser.add_argument("--steps_between_evals", type=int, help="Frequency of evaluation.")
    parser.add_argument("--output_dir", type=str, help="Directory to save logs and checkpoints.")
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument("--mixed_precision", type=str, choices=["no", "fp16", "bf16"], help="Mixed precision type.")
    parser.add_argument("--test_set_path", type=str, help="Path to the multiple-choice test set JSON file.")
    parser.add_argument("--no_eval_cache", action="store_false", dest="use_eval_cache", help="Disable caching of evaluation generations.")

    args = parser.parse_args()
    overrides = {k: v for k, v in vars(args).items() if v is not None}
    return overrides

def setup_accelerator(config):
    """Initializes and returns the Accelerator based on config."""
    kwargs_handlers = [
        DistributedDataParallelKwargs(comm_hook=DDPCommunicationHookType.FP16),
        InitProcessGroupKwargs(
             timeout=datetime.timedelta(seconds=config['communication_timeout_seconds'])
        ),
    ]

    accelerator = Accelerator(
        mixed_precision=config['mixed_precision'],
        step_scheduler_with_optimizer=False,
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        log_with="tensorboard", # Or ["tensorboard", "wandb"] or None
        project_dir=config['output_dir'],
        kwargs_handlers=kwargs_handlers,
        device_placement=True,
    )
    return accelerator

def get_config():
    """Gets the final configuration dictionary."""
    config = get_default_config()
    cmd_line_overrides = parse_arguments()
    config.update(cmd_line_overrides)

    config['script_dir'] = os.path.dirname(os.path.abspath(__file__))
    config['run_name'] = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    config_to_hash = {k: v for k, v in config.items() if k not in ['script_dir', 'run_name', 'output_dir']}
    try:
        config_hash = cache.quick_key(config_to_hash)[:16]
    except Exception as e:
        print(f"Warning: Could not generate config hash using cache.quick_key: {e}. Using timestamp.")
        config_hash = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    config['config_hash'] = config_hash

    config['log_dir'] = os.path.join(config['output_dir'], config['config_hash'])
    config['checkpoint_dir'] = os.path.join(config['log_dir'], "checkpoints")

    return config

def _clean_dict_serializable_for_hash(d):
    """Prepares dict for hashing, converting non-primitives."""
    cleaned = {}
    for key, value in d.items():
        if isinstance(value, (str, int, float, bool, tuple)):
             cleaned[key] = value
        elif isinstance(value, list):
              cleaned[key] = tuple(value) # Convert lists to tuples for hashing
        # Ignore other types like dicts, objects for hashing purposes
    return cleaned