"""
parser.py

CLI argument parsing for training configurations.
"""

import argparse

def config_parser():
    """
    Parse command-line arguments for configuration settings.

    Returns:
        An argparse Namespace with training configuration.
    """
    parser = argparse.ArgumentParser(
        description="Argument parser for configuration settings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--sequence_length", type=int, default=4096,
                        help="The maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="The (global) batch size")
    parser.add_argument("--minimum_sequence_length", type=int, default=64,
                        help="The minimum sequence length for packing")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="The learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="The weight decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                        help="The Adam epsilon")
    parser.add_argument("--num_warmup_steps", type=int, default=512,
                        help="The number of warmup steps")
    parser.add_argument("--softmax_temperature", type=float, default=1.0,
                        help="The softmax temperature")
    parser.add_argument("--steps_between_evals", type=int, default=512,
                        help="The number of steps between evaluations")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="The number of epochs")
    parser.add_argument("--lr_schedule", type=str, default="linear",
                        help="The learning rate schedule")
    parser.add_argument("--num_steps", type=int, default=16384,
                        help="The number of training steps")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0,
                        help="The maximum gradient norm")

    args = parser.parse_args()
    return args