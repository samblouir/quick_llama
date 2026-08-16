#!/usr/bin/env python3
"""Measure OpenHermes-2.5 with its native OpenHermes/Mistral ChatML tokenizer.

The script deliberately counts the complete dataset rather than sampling. It emits
revision pins, serialization hashes, token totals, role totals, and length
statistics so that the result can be independently reproduced.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from array import array
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer

DATASET_ID = "teknium/OpenHermes-2.5"
TOKENIZER_ID = "teknium/OpenHermes-2.5-Mistral-7B"
EXPECTED_ROWS = 1_001_551
BATCH_SIZE = int(os.environ.get("TOKEN_COUNT_BATCH_SIZE", "512"))
OUTPUT_PATH = Path(os.environ.get("TOKEN_COUNT_OUTPUT", "openhermes-2.5-token-count.json"))
ROLE_MAP = {
    "system": "system",
    "human": "user",
    "gpt": "assistant",
    "user": "user",
    "assistant": "assistant",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def framed_hash_update(hasher: Any, text: str) -> int:
    data = text.encode("utf-8")
    hasher.update(len(data).to_bytes(8, "little", signed=False))
    hasher.update(data)
    return len(data)


def manual_chatml(conversation: list[dict[str, Any]]) -> tuple[str, list[dict[str, str]], Counter[str]]:
    pieces: list[str] = []
    messages: list[dict[str, str]] = []
    unknown = Counter()
    for turn in conversation:
        source_role = str(turn.get("from", ""))
        role = ROLE_MAP.get(source_role)
        if role is None:
            unknown[source_role] += 1
            role = source_role
        value = turn.get("value", "")
        if value is None:
            value = ""
        elif not isinstance(value, str):
            value = str(value)
        pieces.append(f"<|im_start|>{role}\n{value}<|im_end|>\n")
        messages.append({"role": role, "content": value})
    return "".join(pieces), messages, unknown


def encode_lengths(backend: Any, texts: list[str]) -> list[int]:
    if not texts:
        return []
    return [len(encoded.ids) for encoded in backend.encode_batch(texts, add_special_tokens=False)]


def percentile_summary(values: array) -> dict[str, float | int]:
    x = np.frombuffer(values, dtype=np.uint64)
    quantiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9, 100]
    result: dict[str, float | int] = {
        "count": int(x.size),
        "sum": int(x.sum(dtype=np.uint64)),
        "mean": float(x.mean()),
    }
    for q, value in zip(quantiles, np.percentile(x, quantiles)):
        result[f"p{str(q).replace('.', '_')}"] = int(round(float(value)))
    return result


def main() -> None:
    started_wall = utc_now()
    started = time.monotonic()
    print(f"measurement_started_utc={started_wall}", flush=True)
    print(f"dataset_id={DATASET_ID}", flush=True)
    print(f"tokenizer_id={TOKENIZER_ID}", flush=True)
    print(f"batch_size={BATCH_SIZE}", flush=True)

    api = HfApi()
    dataset_info = api.dataset_info(DATASET_ID, files_metadata=True)
    model_info = api.model_info(TOKENIZER_ID, files_metadata=True)
    print(f"dataset_revision={dataset_info.sha}", flush=True)
    print(f"tokenizer_revision={model_info.sha}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ID,
        revision=model_info.sha,
        use_fast=True,
        trust_remote_code=False,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("A fast tokenizer is required for the full-corpus measurement")
    backend = tokenizer.backend_tokenizer

    special_contract = {
        "tokenizer_class": tokenizer.__class__.__name__,
        "is_fast": bool(tokenizer.is_fast),
        "vocab_size_property": int(tokenizer.vocab_size),
        "effective_length": int(len(tokenizer)),
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token": tokenizer.unk_token,
        "unk_token_id": tokenizer.unk_token_id,
        "im_start_id": tokenizer.convert_tokens_to_ids("<|im_start|>"),
        "im_end_id": tokenizer.convert_tokens_to_ids("<|im_end|>"),
        "add_bos_token": getattr(tokenizer, "add_bos_token", None),
        "add_eos_token": getattr(tokenizer, "add_eos_token", None),
        "chat_template": getattr(tokenizer, "chat_template", None),
    }
    print("tokenizer_contract=" + json.dumps(special_contract, ensure_ascii=False, sort_keys=True), flush=True)

    for key, expected in {
        "bos_token_id": 1,
        "im_start_id": 32001,
        "im_end_id": 32000,
    }.items():
        if special_contract[key] != expected:
            raise RuntimeError(f"Tokenizer contract mismatch for {key}: {special_contract[key]!r} != {expected!r}")

    dataset = load_dataset(
        DATASET_ID,
        split="train",
        revision=dataset_info.sha,
        trust_remote_code=False,
    )
    row_count = len(dataset)
    print(f"loaded_rows={row_count:,}", flush=True)
    print(f"columns={dataset.column_names}", flush=True)
    print(f"features={dataset.features}", flush=True)
    if row_count != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS:,} rows, found {row_count:,}")
    if "conversations" not in dataset.column_names:
        raise RuntimeError("Dataset has no 'conversations' column")

    canonical_lengths = array("Q")
    no_final_newline_lengths = array("Q")
    turn_lengths = array("Q")
    content_lengths = array("Q")

    role_turn_counts: Counter[str] = Counter()
    role_content_tokens: Counter[str] = Counter()
    source_role_counts: Counter[str] = Counter()
    unknown_role_counts: Counter[str] = Counter()
    conversations_with_system = 0
    empty_messages = 0
    weighted_messages = 0
    total_chars = 0
    total_utf8_bytes = 0
    canonical_utf8_bytes = 0
    canonical_hash = hashlib.sha256()
    no_final_newline_hash = hashlib.sha256()
    running_canonical_total = 0

    template_probe: dict[str, Any] = {
        "available": bool(getattr(tokenizer, "chat_template", None)),
        "checked_rows": 0,
        "equal_to_manual": None,
        "first_mismatch_row": None,
        "manual_preview": None,
        "template_preview": None,
    }

    processed = 0
    for start in range(0, row_count, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, row_count)
        conversations = dataset[start:stop]["conversations"]

        canonical_texts: list[str] = []
        no_final_newline_texts: list[str] = []
        batch_content_texts: list[str] = []
        batch_content_roles: list[str] = []

        for local_index, conversation in enumerate(conversations):
            if conversation is None:
                conversation = []
            text, messages, unknown = manual_chatml(conversation)
            unknown_role_counts.update(unknown)
            canonical_texts.append(text)
            no_final = text[:-1] if text.endswith("\n") else text
            no_final_newline_texts.append(no_final)
            turn_lengths.append(len(conversation))

            if any(message["role"] == "system" for message in messages):
                conversations_with_system += 1

            for original_turn, message in zip(conversation, messages):
                source_role = str(original_turn.get("from", ""))
                role = message["role"]
                value = message["content"]
                source_role_counts[source_role] += 1
                role_turn_counts[role] += 1
                if value == "":
                    empty_messages += 1
                if original_turn.get("weight") is not None:
                    weighted_messages += 1
                total_chars += len(value)
                total_utf8_bytes += len(value.encode("utf-8"))
                batch_content_texts.append(value)
                batch_content_roles.append(role)

            canonical_utf8_bytes += framed_hash_update(canonical_hash, text)
            framed_hash_update(no_final_newline_hash, no_final)

            if template_probe["available"] and template_probe["checked_rows"] < 100:
                row_index = start + local_index
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                equal = rendered == text
                template_probe["checked_rows"] += 1
                if template_probe["equal_to_manual"] is None:
                    template_probe["equal_to_manual"] = True
                if not equal:
                    template_probe["equal_to_manual"] = False
                    if template_probe["first_mismatch_row"] is None:
                        template_probe["first_mismatch_row"] = row_index
                        template_probe["manual_preview"] = text[:1000]
                        template_probe["template_preview"] = rendered[:1000]

        batch_canonical_lengths = encode_lengths(backend, canonical_texts)
        batch_no_final_lengths = encode_lengths(backend, no_final_newline_texts)
        batch_content_lengths = encode_lengths(backend, batch_content_texts)

        canonical_lengths.extend(batch_canonical_lengths)
        no_final_newline_lengths.extend(batch_no_final_lengths)
        content_lengths.extend(batch_content_lengths)
        running_canonical_total += sum(batch_canonical_lengths)
        for role, length in zip(batch_content_roles, batch_content_lengths):
            role_content_tokens[role] += length

        processed = stop
        if processed == row_count or processed % 10_000 < BATCH_SIZE:
            elapsed = time.monotonic() - started
            print(
                f"progress rows={processed:,}/{row_count:,} "
                f"canonical_tokens={running_canonical_total:,} "
                f"rows_per_second={processed / max(elapsed, 1e-9):.1f} "
                f"elapsed_seconds={elapsed:.1f}",
                flush=True,
            )

    canonical_total = int(sum(canonical_lengths))
    no_final_newline_total = int(sum(no_final_newline_lengths))
    content_total = int(sum(content_lengths))
    turn_count = int(sum(turn_lengths))

    thresholds: dict[str, dict[str, int]] = {}
    canonical_np = np.frombuffer(canonical_lengths, dtype=np.uint64)
    for threshold in (2048, 4096, 8192, 16384, 32768):
        clipped = np.minimum(canonical_np, threshold)
        thresholds[str(threshold)] = {
            "conversations_over": int(np.count_nonzero(canonical_np > threshold)),
            "tokens_retained_if_each_conversation_truncated": int(clipped.sum(dtype=np.uint64)),
            "tokens_dropped_if_each_conversation_truncated": int((canonical_np - clipped).sum(dtype=np.uint64)),
        }

    elapsed = time.monotonic() - started
    result: dict[str, Any] = {
        "measurement": {
            "started_utc": started_wall,
            "completed_utc": utc_now(),
            "elapsed_seconds": elapsed,
            "host": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "batch_size": BATCH_SIZE,
        },
        "source": {
            "dataset_id": DATASET_ID,
            "dataset_revision": dataset_info.sha,
            "dataset_rows": row_count,
            "dataset_columns": list(dataset.column_names),
            "tokenizer_id": TOKENIZER_ID,
            "tokenizer_revision": model_info.sha,
        },
        "tokenizer_contract": special_contract,
        "serialization": {
            "name": "OpenHermes/Axolotl ChatML completed-conversation form",
            "per_message": "<|im_start|>{mapped_role}\\n{value}<|im_end|>\\n",
            "role_mapping": ROLE_MAP,
            "canonical_includes_bos": False,
            "canonical_has_final_newline": True,
            "bos_variant_definition": "canonical token count plus one BOS token per conversation",
            "content_only_definition": "each message value encoded independently with add_special_tokens=False",
        },
        "integrity": {
            "expected_rows": EXPECTED_ROWS,
            "row_count_matches": row_count == EXPECTED_ROWS,
            "processed_rows": processed,
            "canonical_framed_sha256": canonical_hash.hexdigest(),
            "canonical_no_final_newline_framed_sha256": no_final_newline_hash.hexdigest(),
            "canonical_utf8_payload_bytes": canonical_utf8_bytes,
            "unknown_role_counts": dict(sorted(unknown_role_counts.items())),
            "template_probe": template_probe,
        },
        "counts": {
            "conversations": row_count,
            "turns": turn_count,
            "conversations_with_system": conversations_with_system,
            "empty_messages": empty_messages,
            "messages_with_non_null_weight": weighted_messages,
            "content_characters": total_chars,
            "content_utf8_bytes": total_utf8_bytes,
            "source_role_turns": dict(sorted(source_role_counts.items())),
            "mapped_role_turns": dict(sorted(role_turn_counts.items())),
        },
        "tokens": {
            "canonical_chatml_no_bos": canonical_total,
            "canonical_chatml_one_bos_per_conversation": canonical_total + row_count,
            "canonical_chatml_no_final_newline_no_bos": no_final_newline_total,
            "independently_tokenized_message_content_only": content_total,
            "role_content_only": dict(sorted(role_content_tokens.items())),
            "canonical_minus_independent_content": canonical_total - content_total,
            "mean_canonical_tokens_per_conversation": canonical_total / row_count,
            "mean_canonical_tokens_per_turn": canonical_total / turn_count,
        },
        "distributions": {
            "canonical_tokens_per_conversation": percentile_summary(canonical_lengths),
            "no_final_newline_tokens_per_conversation": percentile_summary(no_final_newline_lengths),
            "content_tokens_per_message": percentile_summary(content_lengths),
            "turns_per_conversation": percentile_summary(turn_lengths),
        },
        "per_conversation_truncation": thresholds,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    print("RESULT_JSON_BEGIN", flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    print("RESULT_JSON_END", flush=True)
    print(f"result_path={OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
