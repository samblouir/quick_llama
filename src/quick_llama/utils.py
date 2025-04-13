import json
import psutil
import os
import signal
import sys
import random
import time
import torch
import numpy as np
from accelerate.logging import get_logger

log = get_logger(__name__)

def set_random_seed(seed):
    """Sets the random seed for reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    log.info(f"Set random seed to {seed} for random, numpy, and torch.")


def build_string_from_dict(data_dict, sep=', ', precision=4):
    """Formats a dictionary into a string."""
    ret_str = []
    for key, value in data_dict.items():
        if isinstance(value, float):
            formatted_value = f"{value:.{precision}f}"
        elif isinstance(value, (int, str)):
            formatted_value = str(value)
        elif isinstance(value, torch.Tensor):
             formatted_value = f"Tensor(shape={value.shape}, dtype={value.dtype}, device={value.device})"
        else:
             formatted_value = str(value)
        ret_str.append(f"{key}: {formatted_value}")
    return sep.join(ret_str)


def find_and_kill_accelerate_processes(confirm=True):
    """
    Finds and optionally terminates processes running 'accelerate launch'.
    USE WITH EXTREME CAUTION. Consider running as a separate cleanup script.

    Args:
        confirm (bool): If True, requires user confirmation before killing.
    """
    target_processes = []
    current_pid = os.getpid()
    script_path = os.path.abspath(sys.argv[0])

    log.warning("Scanning for processes launched via 'accelerate launch'...")

    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            pinfo = proc.info
            cmdline = pinfo.get('cmdline')

            if cmdline and 'accelerate' in cmdline and 'launch' in cmdline:
                pid = pinfo['pid']

                if pid == current_pid:
                    log.debug(f"  Skipping current process PID={pid}")
                    continue
                try:
                    parent = psutil.Process(pid).parent()
                    if parent and parent.pid == current_pid:
                         log.debug(f"  Skipping child process PID={pid} (parent is current process)")
                         continue
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass


                command_str = ' '.join(cmdline)
                log.info(f"  Found potential Accelerate process: PID={pid}, Command='{command_str}'")
                target_processes.append(proc)

        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        except Exception as e:
            log.error(f"  Error accessing info for a process: {e}", exc_info=True)

    if not target_processes:
        log.info("No other 'accelerate launch' processes found to terminate.")
        return

    log.warning(f"\nFound {len(target_processes)} potential process(es) to terminate.")

    if confirm:
        try:
            response = input("Proceed with termination? (yes/no): ").lower()
        except EOFError: # Handle non-interactive environments
             log.warning("Cannot confirm in non-interactive mode. Aborting termination.")
             return

        if response != 'yes':
            log.info("Termination aborted by user.")
            return

    log.info("Attempting termination...")
    killed_count = 0
    for proc in target_processes:
        try:
            pid_to_kill = proc.pid
            log.info(f"  Terminating PID {pid_to_kill} ({' '.join(proc.cmdline())})...")
            proc.terminate()
            log.info(f"    Sent SIGTERM to PID {pid_to_kill}.")
            killed_count += 1

        except psutil.NoSuchProcess:
            log.warning(f"  Process with PID {pid_to_kill} no longer exists.")
        except psutil.AccessDenied:
            log.error(f"  Permission denied to terminate PID {pid_to_kill}. Try sudo?", exc_info=False)
        except Exception as e:
            log.error(f"  An error occurred while trying to terminate PID {pid_to_kill}: {e}", exc_info=True)

    log.info(f"Termination attempt finished. Killed {killed_count} process(es).")


def format_multiple_choice(input_text, choices, label_index, *args, **kwargs):
    """Formats a multiple-choice question (moved from original)."""
    out_str = [str(input_text)]
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    if not isinstance(choices, list) or not (0 <= label_index < len(choices)):
         log.warning(f"Invalid choices or label_index for input: {str(input_text)[:50]}...")
         return None

    for idx, choice in enumerate(choices):
        if idx >= len(alphabet):
             log.warning(f"Too many choices ({len(choices)}) for standard alphabet labeling. Stopping at Z.")
             break
        letter = alphabet[idx]
        out_str.append(f"({letter}) {choice}")

    label_str = out_str[label_index + 1]
    full_question_str = "\n".join(out_str)

    return dict(
        input=full_question_str,
        choices=choices,
        label=label_str,
        label_index=label_index
    )