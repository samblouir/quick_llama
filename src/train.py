import os
import time
from tqdm import tqdm
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed # Use Accelerate's set_seed

# Import from our refactored modules
import config as cfg
import data_utils
import model_utils
import evaluate
import logger as training_logger # Rename to avoid conflict with accelerate logger
import utils

log = get_logger(__name__) # Use Accelerate's logger

def main():
    # --- 1. Configuration & Initialization ---
    config = cfg.get_config()
    accelerator = cfg.setup_accelerator(config)

    # Setup logging
    train_logger = training_logger.TrainingLogger(config['log_dir'], accelerator)
    log_level = "INFO" if accelerator.is_main_process else "ERROR" # Reduce verbosity on non-main processes
    # configure_logging(log_level) # You might need a function to set log levels if not done by Accelerate

    train_logger.log_message(f"Starting run: {config.get('run_name', 'N/A')}", main_process_only=True)
    train_logger.log_message(f"Using {accelerator.num_processes} processes.", main_process_only=True)
    train_logger.log_message(f"Configuration:\n{utils.build_string_from_dict(config, sep='\\n')}", main_process_only=True)
    train_logger.log_config(config) # Save config JSON

    # Set random seed for reproducibility
    # set_seed(config['seed']) # Accelerate's set_seed handles synchronization
    utils.set_random_seed(config['seed']) # Using our utility ensures torch/numpy seeds are also set


    # --- 2. Load Tokenizer and Data ---
    tokenizer = data_utils.initialize_tokenizer(config['model_name'])
    dataset = data_utils.load_and_prepare_dataset(config, accelerator)

    # Prepare each split (tokenization, batching, caching)
    # Pass accelerator and relevant config parts
    data_prep_config = {
        'sequence_length': config['sequence_length'],
        'batch_size_per_device': config['batch_size_per_device'],
        'dataset_name': config['dataset_name'],
        'model_name': config['model_name'], # For cache key
         'minimum_sequence_length': config['minimum_sequence_length'], # For packer_batcher
        # Add other packer_batcher relevant keys if needed
    }

    train_batches, train_stats = data_utils.prepare_dataset_split(
        dataset['train'], data_prep_config, 'train', accelerator
    )
    valid_batches, valid_stats = data_utils.prepare_dataset_split(
        dataset['validation'], data_prep_config, 'validation', accelerator
    )
    # test_batches, test_stats = data_utils.prepare_dataset_split(
    #     dataset['test'], data_prep_config, 'test', accelerator
    # ) # Prepare test set if needed for final eval

    train_logger.log_dataset_stats(train_stats, "Train")
    train_logger.log_dataset_stats(valid_stats, "Validation")
    # train_logger.log_dataset_stats(test_stats, "Test")

    global_batch_size = config['batch_size_per_device'] * accelerator.num_processes

    # Estimate total training steps
    # Factor in gradient accumulation
    num_update_steps_per_epoch = len(train_batches) // config['gradient_accumulation_steps']
    num_training_steps = num_update_steps_per_epoch * config['num_epochs']
    config['num_training_steps'] = num_training_steps # Store for reference

    train_logger.log_message(f"Estimated training steps: {num_training_steps} ({len(train_batches)} batches/epoch, {config['num_epochs']} epochs, {config['gradient_accumulation_steps']} grad accum steps)", main_process_only=True)

    # --- 3. Initialize Model, Optimizer, Scheduler ---
    model = model_utils.create_model(config)
    optimizer, lr_scheduler = model_utils.create_optimizer_and_scheduler(
        model, config, num_training_steps
    )

    # --- 4. Prepare with Accelerator ---
    # This prepares the model, optimizer, and dataloaders for distributed training
    # Note: We prepare the *generators* or lists of batches here. Accelerate doesn't
    # require formal DataLoader objects if you manage batching yourself.
    model, optimizer, lr_scheduler = accelerator.prepare(
        model, optimizer, lr_scheduler
    )
    # We don't prepare train_batches/valid_batches directly, but we will iterate them.

    # --- 5. Checkpoint Loading / Resume Logic ---
    resume_step = 0
    if config.get("resume_from_checkpoint"): # Add this flag to config/args if needed
         checkpoint_to_load = config["resume_from_checkpoint"] # Specify path or 'latest'
         if checkpoint_to_load == 'latest':
              checkpoint_to_load = config['checkpoint_dir'] # Load latest from run's dir

         if os.path.exists(checkpoint_to_load):
              train_logger.log_message(f"Attempting to resume from checkpoint: {checkpoint_to_load}")
              loaded_step = model_utils.load_checkpoint(
                   accelerator, checkpoint_to_load, model, optimizer, lr_scheduler
              )
              if loaded_step is not None:
                   resume_step = loaded_step
                   train_logger.log_message(f"Resumed successfully from step {resume_step}.")
              else:
                   train_logger.log_warning(f"Failed to load checkpoint from {checkpoint_to_load}. Starting from scratch.")
         else:
              train_logger.log_warning(f"Resume checkpoint path not found: {checkpoint_to_load}. Starting from scratch.")


    # --- 6. Data Iterators ---
    # Use simple infinite generators for training data
    def create_infinite_generator(batches):
         while True:
              if not batches: # Handle empty list case
                   yield None # Or raise an error
                   continue
              indices = list(range(len(batches)))
              random.shuffle(indices)
              for i in indices:
                  # Convert numpy arrays to tensors just-in-time
                  try:
                       batch_dict = batches[i]
                       tensor_batch = {
                           k: torch.tensor(v, dtype=torch.long).to(accelerator.device)
                           for k, v in batch_dict.items()
                       }
                       yield tensor_batch
                  except Exception as e:
                       log.error(f"Error creating tensor batch: {e}", exc_info=True)
                       continue # Skip problematic batch
              log.info("Data generator looped.") # Log when it restarts

    train_data_generator = create_infinite_generator(train_batches)
    # For validation, we don't need an infinite generator, just iterate the list
    # Convert validation batches to tensors on the fly during evaluation

    # --- 7. Training Loop ---
    start_time = time.time()
    completed_steps = resume_step
    initial_validation_loss = None
    best_validation_loss = float('inf')
    best_validation_step = 0

    # Add the softmax temperature tensor to config for easy access in the loop
    config_softmax_temperature = torch.tensor([config['softmax_temperature']], device=accelerator.device).to(torch.float16) # Or model's dtype

    train_logger.log_message(f"--- Starting Training (Step {completed_steps}/{num_training_steps}) ---")

    progress_bar = tqdm(
        initial=completed_steps,
        total=num_training_steps,
        desc="Training",
        disable=not accelerator.is_main_process,
    )

    model.train() # Ensure model is in training mode
    for step in range(completed_steps, num_training_steps):
        # --- Batch Fetching ---
        batch = next(train_data_generator)
        if batch is None:
             log.error("Data generator yielded None. Stopping training.")
             break

        # Add softmax temperature if model expects it
        batch['softmax_temperature'] = config_softmax_temperature

        # --- Forward & Backward Pass ---
        # The forward pass is executed under the accelerator's context manager
        # to handle gradient accumulation and synchronization.
        with accelerator.accumulate(model):
            try:
                # Assuming model returns loss directly or in a dict {'loss': ...}
                outputs = model(**batch, return_loss=True)

                if isinstance(outputs, dict):
                     loss = outputs.get('loss')
                elif isinstance(outputs, torch.Tensor):
                     loss = outputs
                else:
                     log.error(f"Unexpected output type from model: {type(outputs)}")
                     continue # Skip step if loss is invalid

                if loss is None:
                     log.error("Model did not return a 'loss'.")
                     continue

                avg_loss = accelerator.gather(loss).mean().item() # Gather loss across devices for logging

                # Backward pass
                accelerator.backward(loss)

                # Clip gradients (optional but recommended)
                if accelerator.sync_gradients and config.get("clip_grad_norm", None):
                    # Only clip when gradients are synchronized (after accumulation)
                    accelerator.clip_grad_norm_(model.parameters(), config["clip_grad_norm"])

                # Optimizer Step (happens automatically based on grad accum steps)
                optimizer.step()
                lr_scheduler.step() # Step the scheduler after the optimizer
                optimizer.zero_grad()

            except Exception as e:
                 log.error(f"Error during training step {step}: {e}", exc_info=True)
                 # Consider adding logic to skip batch or stop training on errors
                 # accelerator.print(f"Problematic Batch Keys: {batch.keys()}")
                 # accelerator.print(f"Batch Shapes: { {k:v.shape for k,v in batch.items()} }")
                 continue # Skip to next step


        # --- Progress Update ---
        progress_bar.update(1)
        progress_bar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"
        })
        completed_steps += 1

        # --- Logging ---
        if step % config.get("logging_steps", 50) == 0: # Log metrics periodically
            elapsed_time = time.time() - start_time
            steps_per_second = (step - resume_step + 1) / elapsed_time if elapsed_time > 0 else 0
            metrics_to_log = {
                "train/loss": avg_loss,
                "train/learning_rate": lr_scheduler.get_last_lr()[0],
                "train/epoch": (step * config['gradient_accumulation_steps']) / len(train_batches) if len(train_batches)>0 else 0,
                "perf/steps_per_second": steps_per_second,
            }
            train_logger.log_metrics(metrics_to_log, step=step)


        # --- Evaluation & Checkpointing ---
        if step > 0 and (step % config['steps_between_evals'] == 0 or step == num_training_steps - 1):
            train_logger.log_message(f"\n--- Starting Evaluation (Step {step}) ---")
            eval_start_time = time.time()

            # Get validation loss
            # Convert validation batches to tensors here
            # Avoid creating a generator if the list is manageable
            validation_batch_tensors = []
            for vb in valid_batches:
                 try:
                     tensor_batch = {k: torch.tensor(v, dtype=torch.long).to(accelerator.device) for k, v in vb.items()}
                     validation_batch_tensors.append(tensor_batch)
                 except Exception as e:
                     log.error(f"Error converting validation batch to tensor: {e}", exc_info=True)

            if validation_batch_tensors:
                 eval_loss = evaluate.run_validation_loss(model, accelerator, validation_batch_tensors)
                 initial_validation_loss = (initial_validation_loss or eval_loss) # Store the first one
                 train_logger.log_metrics({"eval/loss": eval_loss}, step=step)
                 train_logger.log_message(f"  Validation Loss: {eval_loss:.4f}")

                 # Save checkpoint based on validation loss improvement
                 if eval_loss < best_validation_loss:
                      best_validation_loss = eval_loss
                      best_validation_step = step
                      train_logger.log_message(f"  New best validation loss found at step {step}. Saving checkpoint.")
                      model_utils.save_checkpoint(
                           accelerator, model, optimizer, lr_scheduler, step, config['checkpoint_dir'], config.get('save_limit', 3)
                      )
                      # Save a specific "best_model" checkpoint?
                      best_ckpt_dir = os.path.join(config['log_dir'], "best_model")
                      # model_utils.save_checkpoint(
                      #      accelerator, model, optimizer, lr_scheduler, step, best_ckpt_dir, keep_limit=1 # Keep only the best
                      # )

                 else:
                      train_logger.log_message(f"  Validation loss did not improve ({eval_loss:.4f} vs best {best_validation_loss:.4f} at step {best_validation_step}).")
                      # Optionally save checkpoints periodically even if loss doesn't improve
                      if config.get("save_checkpoints_periodically", True):
                           model_utils.save_checkpoint(
                                accelerator, model, optimizer, lr_scheduler, step, config['checkpoint_dir'], config.get('save_limit', 3)
                           )

            else:
                 log.warning("No valid validation batches found. Skipping validation loss calculation.")
                 # Save checkpoint periodically if validation fails or is skipped
                 model_utils.save_checkpoint(
                     accelerator, model, optimizer, lr_scheduler, step, config['checkpoint_dir'], config.get('save_limit', 3)
                 )


            # Optional: Run other evaluations (e.g., accuracy)
            if config.get("run_accuracy_eval", False): # Add flag to config
                 accuracy = evaluate.evaluate_multiple_choice_accuracy(model, accelerator, config, step)
                 if accuracy >= 0: # Check for valid accuracy result
                      train_logger.log_metrics({"eval/accuracy": accuracy}, step=step)
                      train_logger.log_message(f"  Accuracy Eval: {accuracy:.4f}")
                 else:
                      train_logger.log_warning("Accuracy evaluation failed or was skipped.")

            eval_duration = time.time() - eval_start_time
            train_logger.log_message(f"--- Evaluation Finished (Duration: {eval_duration:.2f}s) ---")
            model.train() # Ensure model is back in training mode


        # Early stopping condition (optional)
        # if step - best_validation_step > config.get("patience", 20) * config['steps_between_evals']:
        #     train_logger.log_message(f"Validation loss hasn't improved for {config['patience']} evaluations. Stopping early.")
        #     break

        # Break near the end as in original code? (Be cautious with this)
        # if (num_training_steps - 3) < step:
        #    train_logger.log_warning(f"Approaching end of training ({step}/{num_training_steps}). Stopping slightly early.")
        #    break


    # --- End of Training ---
    progress_bar.close()
    total_training_time = time.time() - start_time
    train_logger.log_message(f"\n--- Training Finished ---")
    train_logger.log_message(f"Total training time: {total_training_time:.2f} seconds")
    train_logger.log_message(f"Best validation loss: {best_validation_loss:.4f} at step {best_validation_step}")
    train_logger.log_message(f"Final logs and checkpoints saved in: {config['log_dir']}")

    # Optional: Save final model explicitly
    train_logger.log_message("Saving final model state...")
    model_utils.save_checkpoint(
         accelerator, model, optimizer, lr_scheduler, num_training_steps, config['checkpoint_dir'], config.get('save_limit', 3)
    )

    # Optional: Run final evaluation on test set if available
    # if 'test' in dataset:
    #    test_batches, test_stats = data_utils.prepare_dataset_split(
    #       dataset['test'], data_prep_config, 'test', accelerator
    #    )
    #    # ... run evaluation on test_batches ...

    accelerator.wait_for_everyone()
    accelerator.end_training() # Clean up accelerator resources

    # Consider removing or commenting out the aggressive process killing.
    # If needed, run it as a separate manual step.
    # utils.find_and_kill_accelerate_processes(confirm=False) # DANGEROUS!

    train_logger.log_message("Run completed.")


if __name__ == "__main__":
    main()