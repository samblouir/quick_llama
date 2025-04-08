import os
import json
import numpy as np
import torch
from tqdm import tqdm
from accelerate.logging import get_logger
from accelerate import Accelerator

import cache # Your existing module
from data_utils import get_tokenizer

log = get_logger(__name__)

def run_validation_loss(model, accelerator: Accelerator, validation_batches):
    """Calculates the average validation loss across all processes."""
    if not validation_batches:
        log.warning("Validation set is empty. Skipping validation loss calculation.")
        return float('inf') # Return infinity if no validation data

    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    total_batches = 0

    progress_bar = tqdm(
        desc="Validation Loss",
        total=len(validation_batches), # Rough estimate, might process fewer/more based on packing
        disable=not accelerator.is_local_main_process,
        position=accelerator.process_index # Stagger progress bars
    )

    with torch.no_grad(): # Disable gradient calculations
        for batch in validation_batches: # Assuming validation_batches yields dicts of tensors
            if not batch: continue # Skip empty batches

            # Forward pass to get loss
            # Assuming model returns loss when return_loss=True or similar
            try:
                 # Adapt this call based on your model's forward signature for loss
                 outputs = model(**batch, return_loss=True)

                 # Check if output is dict or tensor
                 if isinstance(outputs, dict):
                      loss = outputs.get('loss')
                 elif isinstance(outputs, torch.Tensor):
                      loss = outputs # Assume direct loss output
                 else:
                      log.error(f"Unexpected output type from model during validation: {type(outputs)}")
                      continue

                 if loss is None:
                      log.error("Model output did not contain 'loss' during validation.")
                      continue

                 # Simple averaging: accumulate loss for batches processed by this rank
                 total_loss += loss.item()
                 total_batches += 1

            except Exception as e:
                 log.error(f"Error during validation forward pass: {e}", exc_info=True)
                 # Decide: skip batch or fail validation?

            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item() if loss is not None else "N/A"})


    progress_bar.close()

    # --- Aggregate results across all processes ---
    # Gather total loss and batch counts from all GPUs
    aggregated_data = accelerator.gather_for_metrics({
        "total_loss": total_loss,
        "total_batches": total_batches
    })

    # Calculate average loss on the main process
    if accelerator.is_main_process:
        global_total_loss = aggregated_data["total_loss"].sum().item()
        global_total_batches = aggregated_data["total_batches"].sum().item()

        if global_total_batches > 0:
            final_avg_loss = global_total_loss / global_total_batches
            log.info(f"Validation Complete: Average Loss = {final_avg_loss:.4f} (over {global_total_batches} batches)")
        else:
            log.warning("No batches were successfully processed during validation.")
            final_avg_loss = float('inf')
    else:
         final_avg_loss = float('inf') # Placeholder for non-main processes

    # --- Cleanup ---
    model.train() # Set model back to training mode
    accelerator.wait_for_everyone() # Ensure all processes finish before proceeding

    # Broadcast the final loss from main process to all processes
    final_avg_loss_tensor = torch.tensor(final_avg_loss, device=accelerator.device)
    final_avg_loss_tensor = accelerator.reduce(final_avg_loss_tensor, reduction='mean') # Simple way to broadcast/sync

    return final_avg_loss_tensor.item()


# --- Task-Specific Evaluation (Example: MMLU-like Accuracy) ---
# This consolidates the logic from the original `run_evals` and `has_model_run_eval`

def evaluate_multiple_choice_accuracy(model, accelerator: Accelerator, config, step):
    """Evaluates accuracy on a multiple-choice question set."""

    # --- Load Test Questions ---
    # Avoid hardcoding paths, get from config or arguments
    test_set_path = config.get("test_set_path", None) # Add "test_set_path" to your config.py
    if not test_set_path:
        # Try default relative path (adjust as needed)
        base_dir = config.get('script_dir', os.path.dirname(__file__)) # Get script dir from config
        up_one_dir = os.path.split(base_dir)[0]
        test_set_path = os.path.join(up_one_dir, "data", "TestSet", "testset.json")
        log.warning(f"test_set_path not in config, defaulting to: {test_set_path}")

    if not os.path.exists(test_set_path):
        log.error(f"Test set file not found: {test_set_path}. Skipping accuracy evaluation.")
        return -1.0 # Indicate failure

    try:
        with open(test_set_path, "r") as f:
            all_test_questions = json.load(f)
    except Exception as e:
        log.error(f"Failed to load or parse test set {test_set_path}: {e}")
        return -1.0

    if not all_test_questions or not isinstance(all_test_questions, list):
         log.error(f"Test set {test_set_path} is empty or not a list.")
         return -1.0

    # Distribute questions among processes
    questions_for_this_process = all_test_questions[accelerator.process_index::accelerator.num_processes]
    num_questions_local = len(questions_for_this_process)
    log.info(f"Process {accelerator.process_index} evaluating {num_questions_local} questions.")

    if num_questions_local == 0:
         log.warning(f"Process {accelerator.process_index} has no questions to evaluate.")
         # Fall through, results will be 0


    # --- Run Inference ---
    model.eval() # Set model to evaluation mode
    correct_count = 0
    total_evaluated = 0
    tokenizer = get_tokenizer() # Ensure tokenizer is initialized

    progress_bar = tqdm(
        questions_for_this_process,
        desc=f"Accuracy Eval (Proc {accelerator.process_index})",
        disable=not accelerator.is_local_main_process
    )

    config_hash = config.get("config_hash", "unknown_config") # For caching

    with torch.no_grad():
        for question_data in progress_bar:
            input_string = question_data.get('input')
            correct_answer = question_data.get('answer') # Assuming 'answer' is the correct choice letter/text

            if not input_string or not correct_answer:
                log.warning(f"Skipping question due to missing 'input' or 'answer': {question_data}")
                continue

            total_evaluated += 1

            # --- Caching for Generations (optional but helpful) ---
            cache_key = cache.quick_key(dict(eval_step=step, config_hash=config_hash, input_string=input_string))
            output = None
            try:
                if config.get("use_eval_cache", True): # Add flag to config
                    output = cache.quick_load(cache_key)
                    log.debug(f"Cache hit for question: {input_string[:50]}...")
            except (FileNotFoundError, Exception): # Broad exception for cache miss/errors
                 log.debug(f"Cache miss for question: {input_string[:50]}...")
                 output = None # Ensure output is None if cache fails

            # --- Generate Answer if not cached ---
            if output is None:
                # Format input using chat template
                # Adapt based on how `model.chat` expects input
                input_messages = [{"role": "user", "content": input_string}]

                try:
                    # Use the model's chat/generate method
                    # Ensure the model object passed here is the prepared one or handle unwrapping
                    # Needs try-except as original code suggests potential issues
                    try:
                         # Try direct call first (if model is already prepared/unwrapped appropriately)
                         # Pass generation args if needed (max_length, temp, etc.) from config
                         output = model.chat(input_messages)
                    except AttributeError:
                         # Fallback to unwrapped model if .chat isn't on the proxy
                         log.debug("Calling .chat on unwrapped model.")
                         output = accelerator.unwrap_model(model).chat(input_messages)

                    # Save to cache if generation was successful
                    if output and config.get("use_eval_cache", True):
                        try:
                            cache.quick_save(cache_key, output)
                        except Exception as e:
                            log.warning(f"Failed to save generation to cache for key {cache_key}: {e}")

                except Exception as e:
                    log.error(f"Error during model generation for question: {input_string[:50]}... Error: {e}", exc_info=True)
                    output = "" # Set empty output on error

            # --- Compare Answer ---
            # Original logic: check prefix, substring containment
            # This might be too lenient for rigorous evaluation. Consider stricter checks.
            # E.g., check if generated output *exactly* matches one of the choices and if it's the correct one.
            output_processed = output.strip() if output else ""
            answer_processed = correct_answer.strip()

            is_correct = False
            if output_processed and answer_processed:
                 # Original checks:
                 if output_processed.startswith(answer_processed) or \
                    output_processed in answer_processed or \
                    answer_processed in output_processed:
                     is_correct = True
                 # Add more robust checks if needed, e.g., extracting choice letter

            if is_correct:
                correct_count += 1

            progress_bar.set_postfix({"acc": f"{correct_count/total_evaluated:.2f}" if total_evaluated > 0 else "N/A"})


    # --- Aggregate Accuracy Across Processes ---
    # Gather counts from all GPUs
    aggregated_counts = accelerator.gather_for_metrics({
        "correct": correct_count,
        "evaluated": total_evaluated
    })

    # Calculate global accuracy on the main process
    if accelerator.is_main_process:
        global_correct = aggregated_counts["correct"].sum().item()
        global_evaluated = aggregated_counts["evaluated"].sum().item()

        if global_evaluated > 0:
            final_accuracy = global_correct / global_evaluated
            log.info(f"Accuracy Evaluation Complete (Step {step}): Accuracy = {final_accuracy:.4f} ({global_correct}/{global_evaluated})")
        else:
            log.warning("No questions were successfully evaluated across all processes.")
            final_accuracy = 0.0 # Default to 0 if nothing was evaluated
    else:
        final_accuracy = 0.0 # Placeholder

    # --- Cleanup ---
    model.train() # Set back to training mode
    accelerator.wait_for_everyone()

    # Broadcast final accuracy
    final_accuracy_tensor = torch.tensor(final_accuracy, device=accelerator.device)
    final_accuracy_tensor = accelerator.reduce(final_accuracy_tensor, reduction='mean')

    # Optional: Log results to file (main process only)
    if accelerator.is_main_process:
         results_file = os.path.join(config['log_dir'], "accuracy_results.csv")
         write_header = not os.path.exists(results_file)
         try:
              with open(results_file, "a") as f:
                   if write_header:
                        f.write("step,accuracy,correct_count,total_evaluated\n")
                   f.write(f"{step},{final_accuracy_tensor.item():.4f},{global_correct},{global_evaluated}\n")
         except IOError as e:
              log.error(f"Failed to write accuracy results to {results_file}: {e}")


    return final_accuracy_tensor.item()