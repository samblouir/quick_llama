#!/usr/bin/env python3
"""Exact full-corpus token count for teknium/OpenHermes-2.5.

The primary measurement is the complete-conversation ChatML serialization used by
OpenHermes/Axolotl, tokenized with the native OpenHermes-2.5-Mistral-7B fast
tokenizer. No sampling or byte-to-token extrapolation is used.
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
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

DATASET_ID = "teknium/OpenHermes-2.5"
TOKENIZER_ID = "teknium/OpenHermes-2.5-Mistral-7B"
PARQUET_REVISION = "refs/convert/parquet"
PARQUET_FILENAME = "default/train/0000.parquet"
EXPECTED_PARQUET_SHA256 = "9d83d1f964b536440458ababe98ce3792dde357b23c8183dc16fb927ef2eeec0"
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


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def framed_hash_update(hasher: Any, text: str) -> int:
    data = text.encode("utf-8")
    hasher.update(len(data).to_bytes(8, "little", signed=False))
    hasher.update(data)
    return len(data)


def render_chatml(conversation: list[dict[str, Any]]) -> tuple[str, Counter[str]]:
    pieces: list[str] = []
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
    return "".join(pieces), unknown


def encode_lengths(backend: Any, texts: list[str]) -> list[int]:
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
    parquet_info = api.dataset_info(DATASET_ID, revision=PARQUET_REVISION, files_metadata=True)
    model_info = api.model_info(TOKENIZER_ID, files_metadata=True)
    print(f"dataset_revision={dataset_info.sha}", flush=True)
    print(f"parquet_revision={parquet_info.sha}", flush=True)
    print(f"tokenizer_revision={model_info.sha}", flush=True)

    parquet_path = hf_hub_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=PARQUET_REVISION,
        filename=PARQUET_FILENAME,
    )
    parquet_sha256 = sha256_file(parquet_path)
    print(f"parquet_path={parquet_path}", flush=True)
    print(f"parquet_sha256={parquet_sha256}", flush=True)
    if parquet_sha256 != EXPECTED_PARQUET_SHA256:
        raise RuntimeError(
            f"Parquet SHA-256 mismatch: {parquet_sha256} != {EXPECTED_PARQUET_SHA256}"
        )

    parquet_file = pq.ParquetFile(parquet_path)
    row_count = int(parquet_file.metadata.num_rows)
    print(f"parquet_rows={row_count:,}", flush=True)
    print(f"parquet_schema={parquet_file.schema_arrow}", flush=True)
    if row_count != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS:,} rows, found {row_count:,}")
    if "conversations" not in parquet_file.schema_arrow.names:
        raise RuntimeError("Parquet has no 'conversations' column")

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ID,
        revision=model_info.sha,
        use_fast=True,
        trust_remote_code=False,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("A fast tokenizer is required for the full-corpus measurement")
    backend = tokenizer.backend_tokenizer

    tokenizer_contract = {
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
    print("tokenizer_contract=" + json.dumps(tokenizer_contract, ensure_ascii=False, sort_keys=True), flush=True)
    for key, expected in {
        "bos_token_id": 1,
        "im_start_id": 32001,
        "im_end_id": 32000,
    }.items():
        if tokenizer_contract[key] != expected:
            raise RuntimeError(
                f"Tokenizer contract mismatch for {key}: {tokenizer_contract[key]!r} != {expected!r}"
            )

    canonical_lengths = array("Q")
    turn_lengths = array("Q")
    canonical_hash = hashlib.sha256()
    canonical_utf8_bytes = 0
    role_turn_counts: Counter[str] = Counter()
    source_role_counts: Counter[str] = Counter()
    unknown_role_counts: Counter[str] = Counter()
    conversations_with_system = 0
    empty_messages = 0
    weighted_messages = 0
    content_characters = 0
    content_utf8_bytes = 0
    running_token_total = 0
    processed = 0

    for record_batch in parquet_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=["conversations"],
        use_threads=True,
    ):
        conversations = record_batch.column(0).to_pylist()
        rendered_batch: list[str] = []

        for conversation in conversations:
            if conversation is None:
                conversation = []
            text, unknown = render_chatml(conversation)
            rendered_batch.append(text)
            unknown_role_counts.update(unknown)
            turn_lengths.append(len(conversation))
            canonical_utf8_bytes += framed_hash_update(canonical_hash, text)

            has_system = False
            for turn in conversation:
                source_role = str(turn.get("from", ""))
                role = ROLE_MAP.get(source_role, source_role)
                value = turn.get("value", "")
                if value is None:
                    value = ""
                elif not isinstance(value, str):
                    value = str(value)
                source_role_counts[source_role] += 1
                role_turn_counts[role] += 1
                has_system = has_system or role == "system"
                empty_messages += int(value == "")
                weighted_messages += int(turn.get("weight") is not None)
                content_characters += len(value)
                content_utf8_bytes += len(value.encode("utf-8"))
            conversations_with_system += int(has_system)

        lengths = encode_lengths(backend, rendered_batch)
        canonical_lengths.extend(lengths)
        running_token_total += sum(lengths)
        processed += len(conversations)

        if processed == row_count or processed % 10_000 < BATCH_SIZE:
            elapsed = time.monotonic() - started
            print(
                f"progress rows={processed:,}/{row_count:,} "
                f"canonical_tokens={running_token_total:,} "
                f"rows_per_second={processed / max(elapsed, 1e-9):.1f} "
                f"elapsed_seconds={elapsed:.1f}",
                flush=True,
            )

    if processed != row_count:
        raise RuntimeError(f"Processed {processed:,} rows, expected {row_count:,}")

    canonical_total = int(sum(canonical_lengths))
    turn_count = int(sum(turn_lengths))
    canonical_np = np.frombuffer(canonical_lengths, dtype=np.uint64)
    truncation: dict[str, dict[str, int]] = {}
    for threshold in (1024, 2048, 4096, 8192, 16384, 32768):
        clipped = np.minimum(canonical_np, threshold)
        truncation[str(threshold)] = {
            "conversations_over": int(np.count_nonzero(canonical_np > threshold)),
            "tokens_retained_if_each_conversation_truncated": int(clipped.sum(dtype=np.uint64)),
            "tokens_dropped_if_each_conversation_truncated": int(
                (canonical_np - clipped).sum(dtype=np.uint64)
            ),
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
            "method": "full corpus; no sampling or extrapolation",
        },
        "source": {
            "dataset_id": DATASET_ID,
            "dataset_revision": dataset_info.sha,
            "parquet_revision": parquet_info.sha,
            "parquet_filename": PARQUET_FILENAME,
            "parquet_sha256": parquet_sha256,
            "parquet_rows": row_count,
            "tokenizer_id": TOKENIZER_ID,
            "tokenizer_revision": model_info.sha,
        },
        "serialization": {
            "name": "OpenHermes/Axolotl ChatML completed-conversation form",
            "per_message": "<|im_start|>{mapped_role}\\n{value}<|im_end|>\\n",
            "role_mapping": ROLE_MAP,
            "canonical_includes_bos": False,
            "canonical_has_final_newline": True,
            "one_bos_variant": "canonical token count plus exactly one BOS token per conversation",
        },
        "tokenizer_contract": tokenizer_contract,
        "integrity": {
            "expected_rows": EXPECTED_ROWS,
            "processed_rows": processed,
            "row_count_matches": processed == EXPECTED_ROWS,
            "canonical_framed_sha256": canonical_hash.hexdigest(),
            "canonical_utf8_payload_bytes": canonical_utf8_bytes,
            "unknown_role_counts": dict(sorted(unknown_role_counts.items())),
        },
        "counts": {
            "conversations": row_count,
            "turns": turn_count,
            "conversations_with_system": conversations_with_system,
            "empty_messages": empty_messages,
            "messages_with_non_null_weight": weighted_messages,
            "content_characters": content_characters,
            "content_utf8_bytes": content_utf8_bytes,
            "source_role_turns": dict(sorted(source_role_counts.items())),
            "mapped_role_turns": dict(sorted(role_turn_counts.items())),
        },
        "tokens": {
            "canonical_chatml_no_bos": canonical_total,
            "canonical_chatml_one_bos_per_conversation": canonical_total + row_count,
            "mean_canonical_tokens_per_conversation_no_bos": canonical_total / row_count,
            "mean_canonical_tokens_per_turn_no_bos": canonical_total / turn_count,
        },
        "distributions": {
            "canonical_tokens_per_conversation_no_bos": percentile_summary(canonical_lengths),
            "turns_per_conversation": percentile_summary(turn_lengths),
        },
        "per_conversation_truncation_no_bos": truncation,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print("RESULT_JSON_BEGIN", flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    print("RESULT_JSON_END", flush=True)
    print(f"result_path={OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
