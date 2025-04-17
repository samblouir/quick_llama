from accelerate.logging import get_logger
from tqdm import tqdm
import config as cfg
import data_utils
import evaluate
import logger as training_logger
import model_utils
import os
import random
import time
import torch
import utils

log = get_logger(__name__)


def main():
    config = cfg.get_config()
    accelerator = cfg.setup_accelerator(config)

    train_logger = training_logger.TrainingLogger(config["log_dir"], accelerator)

    train_logger.log_message(
        f"Starting run: {config.get('run_name', 'N/A')}", main_process_only=True
    )
    train_logger.log_message(
        f"Using {accelerator.num_processes} processes.", main_process_only=True
    )
    train_logger.log_message(
        f"Configuration:\n{utils.build_string_from_dict(config, sep='\\n')}",
        main_process_only=True,
    )
    train_logger.log_config(config)

    utils.set_random_seed(config["seed"])

    tokenizer = data_utils.initialize_tokenizer(config["model_name"])
    dataset = data_utils.load_and_prepare_dataset(config, accelerator)

    data_prep_config = {
        "sequence_length": config["sequence_length"],
        "batch_size_per_device": config["batch_size_per_device"],
        "dataset_name": config["dataset_name"],
        "model_name": config["model_name"],
        "minimum_sequence_length": config["minimum_sequence_length"],
    }

    train_batches, train_stats = data_utils.prepare_dataset_split(
        dataset["train"], data_prep_config, "train", accelerator
    )
    valid_batches, valid_stats = data_utils.prepare_dataset_split(
        dataset["validation"], data_prep_config, "validation", accelerator
    )

    train_logger.log_dataset_stats(train_stats, "Train")
    train_logger.log_dataset_stats(valid_stats, "Validation")

    global_batch_size = config["batch_size_per_device"] * accelerator.num_processes

    num_update_steps_per_epoch = (
        len(train_batches) // config["gradient_accumulation_steps"]
    )
    num_training_steps = num_update_steps_per_epoch * config["num_epochs"]
    config["num_training_steps"] = num_training_steps

    train_logger.log_message(
        f"Estimated training steps: {num_training_steps} ({len(train_batches)} batches/epoch, {config['num_epochs']} epochs, {config['gradient_accumulation_steps']} grad accum steps)",
        main_process_only=True,
    )

    model = model_utils.create_model(config)
    optimizer, lr_scheduler = model_utils.create_optimizer_and_scheduler(
        model, config, num_training_steps
    )

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    resume_step = 0
    if config.get("resume_from_checkpoint"):
        checkpoint_to_load = config["resume_from_checkpoint"]
        if checkpoint_to_load == "latest":
            checkpoint_to_load = config["checkpoint_dir"]

        if os.path.exists(checkpoint_to_load):
            train_logger.log_message(
                f"Attempting to resume from checkpoint: {checkpoint_to_load}"
            )
            # model, optimizer, lr_scheduler, are updated in-place
            loaded_step = model_utils.load_checkpoint(
                accelerator, checkpoint_to_load, model, optimizer, lr_scheduler
            )
            if loaded_step is not None:
                resume_step = loaded_step
                train_logger.log_message(
                    f"Resumed successfully from step {resume_step}."
                )
            else:
                train_logger.log_warning(
                    f"Failed to load checkpoint from {checkpoint_to_load}. Starting from scratch."
                )
        else:
            train_logger.log_warning(
                f"Resume checkpoint path not found: {checkpoint_to_load}. Starting from scratch."
            )

    def create_infinite_generator(batches):
        while True:
            if not batches:
                yield None
                continue
            indices = list(range(len(batches)))
            random.shuffle(indices)
            for i in indices:
                try:
                    batch_dict = batches[i]
                    tensor_batch = {
                        k: torch.tensor(v, dtype=torch.long).to(accelerator.device)
                        for k, v in batch_dict.items()
                    }
                    yield tensor_batch
                except Exception as e:
                    log.error(f"Error creating tensor batch: {e}", exc_info=True)
                    continue
            log.info("Data generator looped.")

    train_data_generator = create_infinite_generator(train_batches)

    start_time = time.time()
    completed_steps = resume_step
    initial_validation_loss = None
    best_validation_loss = float("inf")
    best_validation_step = 0

    config_softmax_temperature = torch.tensor(
        [config["softmax_temperature"]], device=accelerator.device
    ).to(torch.float16)

    train_logger.log_message(
        f"--- Starting Training (Step {completed_steps}/{num_training_steps}) ---"
    )

    progress_bar = tqdm(
        initial=completed_steps,
        total=num_training_steps,
        desc="Training",
        disable=not accelerator.is_main_process,
    )

    model.train()
    for step in range(completed_steps, num_training_steps):
        batch = next(train_data_generator)
        if batch is None:
            log.error("Data generator yielded None. Stopping training.")
            break

        batch["softmax_temperature"] = config_softmax_temperature

        with accelerator.accumulate(model):
            try:
                outputs = model(**batch, return_loss=True)

                if isinstance(outputs, dict):
                    loss = outputs.get("loss")
                elif isinstance(outputs, torch.Tensor):
                    loss = outputs
                else:
                    log.error(f"Unexpected output type from model: {type(outputs)}")
                    continue

                if loss is None:
                    log.error("Model did not return a 'loss'.")
                    continue

                avg_loss = accelerator.gather(loss.mean()).item()

                accelerator.backward(loss)

                if accelerator.sync_gradients and config.get("clip_grad_norm", None):
                    accelerator.clip_grad_norm_(
                        model.parameters(), config["clip_grad_norm"]
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            except Exception as e:
                log.error(f"Error during training step {step}: {e}", exc_info=True)

                continue

        progress_bar.update(1)
        progress_bar.set_postfix(
            {"loss": f"{avg_loss:.4f}", "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"}
        )
        completed_steps += 1

        if step % config.get("logging_steps", 50) == 0:
            elapsed_time = time.time() - start_time
            steps_per_second = (
                (step - resume_step + 1) / elapsed_time if elapsed_time > 0 else 0
            )
            metrics_to_log = {
                "train/loss": avg_loss,
                "train/learning_rate": lr_scheduler.get_last_lr()[0],
                "train/epoch": (step * config["gradient_accumulation_steps"])
                / len(train_batches)
                if len(train_batches) > 0
                else 0,
                "perf/steps_per_second": steps_per_second,
            }
            train_logger.log_metrics(metrics_to_log, step=step)

        if step > 0 and (
            step % config["steps_between_evals"] == 0 or step == num_training_steps - 1
        ):
            train_logger.log_message(f"\n--- Starting Evaluation (Step {step}) ---")
            eval_start_time = time.time()

            validation_batch_tensors = []
            for vb in valid_batches:
                try:
                    tensor_batch = {
                        k: torch.tensor(v, dtype=torch.long).to(accelerator.device)
                        for k, v in vb.items()
                    }
                    validation_batch_tensors.append(tensor_batch)
                except Exception as e:
                    log.error(
                        f"Error converting validation batch to tensor: {e}",
                        exc_info=True,
                    )

            if validation_batch_tensors:
                eval_loss = evaluate.run_validation_loss(
                    model, accelerator, validation_batch_tensors
                )
                initial_validation_loss = initial_validation_loss or eval_loss
                train_logger.log_metrics({"eval/loss": eval_loss}, step=step)
                train_logger.log_message(f"  Validation Loss: {eval_loss:.4f}")

                if eval_loss < best_validation_loss:
                    best_validation_loss = eval_loss
                    best_validation_step = step
                    train_logger.log_message(
                        f"  New best validation loss found at step {step}. Saving checkpoint."
                    )
                    model_utils.save_checkpoint(
                        accelerator,
                        model,
                        optimizer,
                        lr_scheduler,
                        step,
                        config["checkpoint_dir"],
                        config.get("save_limit", 3),
                    )

                    best_ckpt_dir = os.path.join(config["log_dir"], "best_model")

                else:
                    train_logger.log_message(
                        f"  Validation loss did not improve ({eval_loss:.4f} vs best {best_validation_loss:.4f} at step {best_validation_step})."
                    )

                    if config.get("save_checkpoints_periodically", True):
                        model_utils.save_checkpoint(
                            accelerator,
                            model,
                            optimizer,
                            lr_scheduler,
                            step,
                            config["checkpoint_dir"],
                            config.get("save_limit", 3),
                        )

            else:
                log.warning(
                    "No valid validation batches found. Skipping validation loss calculation."
                )

                model_utils.save_checkpoint(
                    accelerator,
                    model,
                    optimizer,
                    lr_scheduler,
                    step,
                    config["checkpoint_dir"],
                    config.get("save_limit", 3),
                )

            if config.get("run_accuracy_eval", False):
                accuracy = evaluate.evaluate_multiple_choice_accuracy(
                    model, accelerator, config, step
                )
                if accuracy >= 0:
                    train_logger.log_metrics({"eval/accuracy": accuracy}, step=step)
                    train_logger.log_message(f"  Accuracy Eval: {accuracy:.4f}")
                else:
                    train_logger.log_warning(
                        "Accuracy evaluation failed or was skipped."
                    )

            eval_duration = time.time() - eval_start_time
            train_logger.log_message(
                f"--- Evaluation Finished (Duration: {eval_duration:.2f}s) ---"
            )
            model.train()

    progress_bar.close()
    total_training_time = time.time() - start_time
    train_logger.log_message(f"\n--- Training Finished ---")
    train_logger.log_message(f"Total training time: {total_training_time:.2f} seconds")
    train_logger.log_message(
        f"Best validation loss: {best_validation_loss:.4f} at step {best_validation_step}"
    )
    train_logger.log_message(
        f"Final logs and checkpoints saved in: {config['log_dir']}"
    )

    train_logger.log_message("Saving final model state...")
    model_utils.save_checkpoint(
        accelerator,
        model,
        optimizer,
        lr_scheduler,
        num_training_steps,
        config["checkpoint_dir"],
        config.get("save_limit", 3),
    )

    accelerator.wait_for_everyone()
    accelerator.end_training()

    train_logger.log_message("Run completed.")


if __name__ == "__main__":
    main()
