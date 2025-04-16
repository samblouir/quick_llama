import numpy as np
import time
import random
import os
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
from accelerate.logging import get_logger

import packer_batcher
import cache

log = get_logger(__name__)

_tokenizer_instance = None
_tokenizer_name = None


def initialize_tokenizer(model_name=None):
    """Initializes the tokenizer instance."""
    if model_name is None:
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
    global _tokenizer_instance, _tokenizer_name
    if _tokenizer_instance is None or _tokenizer_name != model_name:
        log.info(f"Initializing tokenizer: {model_name}")
        try:
            _tokenizer_instance = AutoTokenizer.from_pretrained(model_name)
            _tokenizer_name = model_name
            if _tokenizer_instance.pad_token is None:
                log.warning(
                    "Tokenizer does not have a pad token. Adding EOS as pad token."
                )
                _tokenizer_instance.pad_token = _tokenizer_instance.eos_token

        except Exception as e:
            log.error(f"Failed to load tokenizer {model_name}: {e}", exc_info=True)
            raise

    if _tokenizer_instance.pad_token is None:
        log.warning("Set Tokenizer.pad_token to EOS since the tokenizer did not have a pad token.")
        _tokenizer_instance.pad_token = _tokenizer_instance.eos_token

    return _tokenizer_instance


def get_tokenizer():
    """Returns the initialized tokenizer instance."""
    if _tokenizer_instance is None:
        initialize_tokenizer()
    return _tokenizer_instance


def tokenize_text(text):
    """Tokenizes a single string of text."""
    tokenizer = get_tokenizer()
    return tokenizer(str(text), return_tensors="np", add_special_tokens=False)["input_ids"][0]


def tokenize_chat(messages):
    """Applies chat template and tokenizes."""
    tokenizer = get_tokenizer()
    if not isinstance(messages, list) or not all(
        "role" in m and "content" in m for m in messages
    ):
        log.error(f"Invalid chat format: {messages}")
        raise ValueError("Invalid chat format provided to tokenize_chat")
    try:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="np"
        )[0]
    except Exception as e:
        log.error(
            f"Error applying chat template: {e} \nMessages: {messages}", exc_info=True
        )
        raise


def tokenize_dict_keys(input_dict, keys_to_tokenize):
    """Tokenizes specified keys in a dictionary."""
    out_dict = {}
    for key in keys_to_tokenize:
        if key in input_dict:
            try:
                out_dict[f"{key}_ids"] = tokenize_text(input_dict[key])
            except Exception as e:
                log.warning(f"Failed to tokenize key '{key}': {e}. Skipping.")
        else:
            log.warning(f"Key '{key}' not found in input dict for tokenization.")
    return out_dict


def load_and_prepare_dataset(config, accelerator):
    """Loads, splits, and prepares the dataset."""
    dataset_name = config["dataset_name"]
    seed = config["seed"]

    log.info(f"Loading dataset: {dataset_name}")
    try:
        if dataset_name == "teknium/OpenHermes-2.5":
            ds = load_dataset(
                dataset_name, split="train"
            )  # Load only train split initially
            ds_splits = ds.train_test_split(test_size=0.01, seed=seed)
            dataset = DatasetDict(
                {
                    "train": ds_splits["train"],
                    "validation": ds_splits["test"],
                }
            )
        else:
            dataset = load_dataset(dataset_name)
            if "validation" not in dataset:
                log.warning("No 'validation' split found. Splitting 'train' (90/10).")
                train_testvalid = dataset["train"].train_test_split(
                    test_size=0.1, seed=seed
                )
                dataset = DatasetDict(
                    {
                        "train": train_testvalid["train"],
                        "validation": train_testvalid["test"],
                    }
                )

        log.info(f"Dataset loaded. Splits: {list(dataset.keys())}")
        log.info(
            f"Train samples: {len(dataset['train'])}, Validation samples: {len(dataset['validation'])}"
        )
        return dataset

    except Exception as e:
        log.error(
            f"Failed to load or process dataset {dataset_name}: {e}", exc_info=True
        )
        raise


def prepare_dataset_split(dataset_split, config, split_name, accelerator):
    """Tokenizes and batches a single dataset split using packer_batcher."""

    # Ensure required keys are in config for packer_batcher
    required_keys = ["sequence_length", "minimum_sequence_length"]
    if not all(key in config for key in required_keys):
        raise ValueError(
            f"Missing required keys for packer_batcher in config: {required_keys}"
        )

    batcher = packer_batcher.Batcher(config=config)
    process_id = accelerator.process_index
    num_processes = accelerator.num_processes

    cache_key_info = {
        "dataset_name": config.get("dataset_name", "unknown_dataset"),
        "split_name": split_name,
        "tokenizer": config.get("model_name", "unknown_tokenizer"),
        "sequence_length": config["sequence_length"],
        "batch_size_per_device": config.get("batch_size_per_device", "unknown_bs"),
        "num_processes": num_processes,
        "process_id": process_id,
        "processing_version": "1.1",  # Increment if logic changes
    }
    ds_cache_key = cache.quick_key(cache_key_info)

    try:
        log.info(
            f"Attempting to load cached batches for {split_name} (Process {process_id}) from key: {ds_cache_key}"
        )
        batches, lengths = cache.quick_load(ds_cache_key)
        log.info(
            f"Loaded {len(batches)} cached batches for {split_name} (Process {process_id})."
        )
        if not batches:
            log.warning(
                f"Cache key {ds_cache_key} found but contained no batches. Reprocessing."
            )
            raise FileNotFoundError
        if not isinstance(batches, list) or not isinstance(lengths, list):
            log.warning(
                f"Cache key {ds_cache_key} contained invalid data types. Reprocessing."
            )
            raise FileNotFoundError

    except (FileNotFoundError, EOFError, TypeError, Exception) as e:
        log.warning(
            f"Cache not found or failed to load for {split_name} (Process {process_id}): {e}. Processing data..."
        )
        batches = []
        lengths = []

        total_items_in_split = len(dataset_split)
        items_per_process = total_items_in_split // num_processes
        remainder = total_items_in_split % num_processes
        if process_id < remainder:
            items_per_process += 1
        start_index = process_id * (total_items_in_split // num_processes) + min(
            process_id, remainder
        )
        end_index = start_index + items_per_process

        process_dataset_slice = dataset_split.select(range(start_index, end_index))

        progress_bar = tqdm(
            total=items_per_process,
            desc=f"Tokenizing {split_name} (Proc {process_id})",
            disable=not accelerator.is_local_main_process,
            position=process_id,
        )

        tokenizer = get_tokenizer()

        for idx, item in enumerate(process_dataset_slice):
            try:
                if config.get("dataset_name") == "teknium/OpenHermes-2.5":
                    conversation = item.get("conversations")
                    if not conversation or len(conversation) < 2:
                        log.debug(
                            f"Skipping item {idx} in proc {process_id} due to short/invalid conversation."
                        )
                        continue

                    messages = []
                    for message in conversation:
                        role = message.get("from")
                        value = message.get("value")
                        if role == "human":
                            role = "user"
                        elif role == "gpt":
                            role = "assistant"
                        else:
                            continue
                        messages.append({"role": role, "content": value})

                    if len(messages) < 2 or messages[-1]["role"] != "assistant":
                        log.debug(
                            f"Skipping item {idx} in proc {process_id}: Needs user/assistant pair ending with assistant."
                        )
                        continue

                    input_ids = tokenizer.apply_chat_template(
                        messages[:-1],
                        add_generation_prompt=True,
                        tokenize=True,
                        return_tensors="np",
                    )[0]

                    label_content = messages[-1]["content"]
                    label_ids_raw = tokenizer(
                        label_content, add_special_tokens=False, return_tensors="np"
                    )["input_ids"][0]
                    if label_ids_raw.size == 0:  # Handle empty assistant response
                        log.debug(
                            f"Skipping item {idx} in proc {process_id}: Assistant response tokenized to empty sequence."
                        )
                        continue
                    label_ids = np.append(label_ids_raw, tokenizer.eos_token_id).astype(
                        np.int64
                    )

                else:
                    log.error(
                        f"Unsupported dataset format: {config.get('dataset_name')}",
                    )
                    if "text" in item:
                        log.warning("Attempting generic processing using 'text' field.")
                        full_text_ids = tokenize_text(item["text"])
                        input_ids = full_text_ids
                        label_ids = full_text_ids  # Causal LM target
                    else:
                        progress_bar.update(1)
                        continue

                if input_ids.size > 0 and label_ids.size > 0:
                    current_length = len(input_ids) + len(label_ids)
                    if current_length < config["minimum_sequence_length"]:
                        log.debug(
                            f"Skipping item {idx} in proc {process_id}: Sequence length {current_length} below minimum {config['minimum_sequence_length']}."
                        )
                        progress_bar.update(1)
                        continue

                    lengths.append(current_length)
                    status = batcher.add(input_ids, label_ids)

                    if status in ["ready", "full"]:
                        popped = batcher.pop()
                        if popped and isinstance(popped, dict):
                            batches.append(popped)
                        if status == "full":
                            status = batcher.add(input_ids, label_ids)
                            if status == "full":
                                log.warning(
                                    f"Single item sequence length ({current_length}) might be too long for packer. Skipping item {idx} in proc {process_id}."
                                )

                else:
                    log.debug(
                        f"Skipping item {idx} in proc {process_id} due to empty input or label after tokenization."
                    )

            except Exception as e:
                log.error(
                    f"Error processing item {idx} in {split_name} (Process {process_id}): {e}",
                    exc_info=True,
                )
            finally:
                progress_bar.update(1)

        if batcher.get_sample_count() > 0:
            popped = batcher.pop()
            if popped and isinstance(popped, dict):
                batches.append(popped)

        progress_bar.close()

        if batches:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    log.info(
                        f"Saving {len(batches)} processed batches for {split_name} (Process {process_id}) to cache key: {ds_cache_key}"
                    )
                    cache.quick_save(ds_cache_key, (batches, lengths))
                    log.info(
                        f"Successfully saved cache for {split_name} (Process {process_id})."
                    )
                    break
                except Exception as e:
                    log.error(
                        f"Attempt {attempt + 1}/{max_retries} failed to save cache for key {ds_cache_key}: {e}",
                        exc_info=False,
                    )
                    if attempt < max_retries - 1:
                        sleep_time = (attempt + 1) * 5 + random.random() * 5
                        log.info(f"Retrying cache save in {sleep_time:.1f} seconds...")
                        time.sleep(sleep_time)
                    else:
                        log.error(
                            f"Failed to save cache for {split_name} after {max_retries} attempts. Continuing without caching this run."
                        )
        else:
            log.warning(
                f"No batches were generated for {split_name} (Process {process_id}). Nothing to cache."
            )

    log.info(
        f"Process {process_id} finished processing {split_name}. Waiting for others..."
    )
    accelerator.wait_for_everyone()
    log.info(f"All processes finished processing {split_name}.")

    if lengths:
        stats = {
            "num_sequences_processed": len(lengths),
            "max_length": max(lengths) if lengths else 0,
            "min_length": min(lengths) if lengths else 0,
            "mean_length": np.mean(lengths) if lengths else 0.0,
            "std_length": np.std(lengths) if lengths else 0.0,
            "num_batches_created": len(batches),
        }
    else:
        stats = {
            "num_sequences_processed": 0,
            "max_length": 0,
            "min_length": 0,
            "mean_length": 0.0,
            "std_length": 0.0,
            "num_batches_created": 0,
        }

    log.info(f"Stats for {split_name} (Process {process_id}): {stats}")

    return batches, stats


def get_dataloader(batches, accelerator, shuffle=False):
    """Creates a simple generator to yield batches (already processed)."""
    if not batches:
        log.warning(
            f"Process {accelerator.process_index}: No batches provided to create dataloader."
        )
        return None

    process_batches = list(batches)  # Make a copy

    if shuffle:
        random.shuffle(process_batches)

    def batch_generator():
        for batch_dict in process_batches:
            if not batch_dict or not isinstance(batch_dict, dict):
                log.debug(f"Skipping invalid batch item: {batch_dict}")
                continue
            try:
                tensor_batch = {
                    k: torch.tensor(
                        v, dtype=torch.long
                    ).pin_memory()  # Pin memory for potential speedup
                    for k, v in batch_dict.items()
                    if isinstance(v, np.ndarray)  # Ensure it's array data
                }
                # Move to device just-in-time if pin_memory() doesn't require it earlier
                # tensor_batch = {k: v.to(accelerator.device, non_blocking=True) for k,v in tensor_batch.items()}
                yield tensor_batch
            except (TypeError, ValueError, RuntimeError) as e:
                log.error(
                    f"Error converting batch to tensor or moving to device: {e}. Batch keys: {batch_dict.keys()}",
                    exc_info=True,
                )
                continue

    return batch_generator()
