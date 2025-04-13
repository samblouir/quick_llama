import os
import json
import threading
from accelerate.logging import get_logger

log = get_logger(__name__, log_level="INFO") # Default level

class TrainingLogger:
    def __init__(self, log_dir, accelerator):
        self.log_dir = log_dir
        self.accelerator = accelerator
        self.log_path = os.path.join(log_dir, f"log_rank_{accelerator.process_index}.txt")
        self.config_path = os.path.join(log_dir, "config.json")
        self.results_path = os.path.join(log_dir, "results.jsonl")

        if self.accelerator.is_main_process:
            os.makedirs(self.log_dir, exist_ok=True)
            open(self.results_path, "w").close()

        self.accelerator.wait_for_everyone()

    def log_message(self, message, level="info", main_process_only=False):
        """Logs a message using Accelerate's logger."""
        if main_process_only and not self.accelerator.is_main_process:
            return

        if level == "info":
            log.info(message)
        elif level == "warning":
            log.warning(message)
        elif level == "error":
            log.error(message)
        else:
            log.debug(message)

        if self.accelerator.is_main_process:
             self._write_to_file(str(message) + "\n")

    def _write_to_file(self, message):
        try:
            with open(self.log_path, "a") as f:
                f.write(message)
        except IOError as e:
            log.error(f"Failed to write to log file {self.log_path}: {e}")


    def log_config(self, config):
        """Saves the configuration dictionary to a JSON file."""
        if self.accelerator.is_main_process:
            log.info(f"Saving configuration to {self.config_path}")
            config_to_save = self._clean_dict_serializable(config)
            try:
                with open(self.config_path, "w") as f:
                    json.dump(config_to_save, f, indent=4)
            except TypeError as e:
                log.error(f"Failed to serialize config to JSON: {e}")
            except IOError as e:
                log.error(f"Failed to write config file {self.config_path}: {e}")

    def log_metrics(self, metrics_dict, step):
        """Logs metrics using accelerator.log and saves to results file."""
        if not isinstance(metrics_dict, dict):
            log.warning("log_metrics expects a dictionary.")
            return

        try:
             self.accelerator.log(metrics_dict, step=step)
        except Exception as e:
             log.error(f"Accelerator failed to log metrics: {e}")


        if self.accelerator.is_main_process:
            metrics_dict['step'] = step
            try:
                with open(self.results_path, "a") as f:
                    f.write(json.dumps(metrics_dict) + "\n")
            except IOError as e:
                 log.error(f"Failed to write results file {self.results_path}: {e}")


    def _clean_dict_serializable(self, d):
        """Creates a copy of the dict with non-serializable values removed."""
        out_dict = {}
        for key, value in d.items():
            try:
                json.dumps({key: value})
                out_dict[key] = value
            except TypeError:
                log.warning(f"Removing non-serializable key '{key}' from config log.")
                out_dict[key] = f"NOT_SERIALIZABLE: {str(type(value))}"
        return out_dict

    def log_dataset_stats(self, stats, name):
        """Logs dataset statistics."""
        if self.accelerator.is_main_process:
            stat_str = "\n".join([f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}"
                                  for key, value in stats.items()])
            self.log_message(f"\n--- {name} Dataset Stats --- \n{stat_str}\n---------------------------", main_process_only=True)
            stats_path = os.path.join(self.log_dir, f"{name}_stats.json")
            try:
                with open(stats_path, "w") as f:
                    json.dump(stats, f, indent=4)
            except IOError as e:
                log.error(f"Failed to write stats file {stats_path}: {e}")