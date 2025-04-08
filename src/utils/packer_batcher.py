"""
packer_batcher.py

Implements sequence packing logic for efficient training with minimal padding.
"""

import copy
import heapq
import numpy as np
from typing import Dict, List, Any, Optional

class Packer:
    """
    A helper class to pack multiple sequences into a fixed-size buffer.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = copy.deepcopy(config)
        self.minimum_sequence_length = self.config.get("minimum_sequence_length", 64)
        self.sequence_length = self.config.get("sequence_length", 1024)
        self.reset(self.sequence_length)

    def reset(self, sequence_length: Optional[int] = None) -> None:
        if sequence_length is None:
            sequence_length = self.sequence_length

        self.current_dict = {
            "input_ids": np.zeros((sequence_length,), dtype=np.int32),
            "attention_mask": np.zeros((sequence_length,), dtype=np.int32),
            "label_ids": np.full((sequence_length,), -100, dtype=np.int32),
            "segment_ids": np.zeros((sequence_length,), dtype=np.int32),
        }
        self.remaining_space = sequence_length
        self.sequence_length = sequence_length
        self.data_index = 0
        self.segment_counter = 0

    def get_remaining_space(self) -> int:
        return self.remaining_space

    def is_ready(self) -> bool:
        return self.remaining_space <= self.minimum_sequence_length

    def can_accept(self, input_ids: np.ndarray, label_ids: np.ndarray) -> bool:
        length_to_add = len(input_ids) + len(label_ids) - 1
        return length_to_add <= self.remaining_space

    def add(self, input_ids: np.ndarray, label_ids: np.ndarray) -> bool:
        length_to_add = len(input_ids) + len(label_ids) - 1
        if not self.can_accept(input_ids, label_ids):
            raise ValueError(
                f"Not enough space to add {length_to_add} tokens. "
                f"Remaining: {self.remaining_space}"
            )
        self.segment_counter += 1

        input_start = self.data_index
        input_end = input_start + len(input_ids)
        label_start = input_end - 1
        label_end = label_start + len(label_ids)

        self.current_dict["input_ids"][input_start:input_end] = input_ids
        self.current_dict["attention_mask"][input_start:label_start] = 1

        # "Teacher forcing": input tokens from label minus last
        self.current_dict["input_ids"][label_start:label_end - 1] = label_ids[:-1]

        self.current_dict["label_ids"][label_start:label_end] = label_ids
        self.current_dict["segment_ids"][input_start:label_end] = self.segment_counter

        self.data_index += length_to_add
        self.remaining_space -= length_to_add
        return self.is_ready()

class Batcher:
    """
    A container of Packets for a mini-batch. Uses a priority queue based on remaining space.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = copy.deepcopy(config)
        self.batch_size = self.config.get("batch_size", 8)
        self.sequence_length = self.config.get("sequence_length", 1024)
        self.packers = []
        self.pq = []
        self.reset(self.sequence_length, self.batch_size)

    def reset(self, sequence_length: int, batch_size: int) -> None:
        self.packers.clear()
        self.pq.clear()

        for i in range(batch_size):
            packer = Packer(self.config)
            self.packers.append(packer)
            heapq.heappush(self.pq, (packer.get_remaining_space(), i, packer))

    def add(self, input_ids: np.ndarray, label_ids: np.ndarray) -> str:
        buffer = []
        packer_found = False
        while self.pq:
            remaining_space, idx, packer = heapq.heappop(self.pq)
            if packer.can_accept(input_ids, label_ids):
                packer.add(input_ids, label_ids)
                heapq.heappush(self.pq, (packer.get_remaining_space(), idx, packer))
                packer_found = True
                break
            else:
                buffer.append((remaining_space, idx, packer))
        while buffer:
            heapq.heappush(self.pq, buffer.pop())

        if not packer_found:
            return "full"
        return "ready" if all(p[2].is_ready() for p in self.pq) else "not ready"

    def pop(self, peek: bool = False) -> Dict[str, np.ndarray]:
        sorted_packers = sorted(self.pq, key=lambda x: x[1])
        keys = sorted_packers[0][2].current_dict.keys()

        # Combine
        stacked_dict = {
            key: np.stack([p[2].current_dict[key] for p in sorted_packers], axis=0)
            for key in keys
        }
        if not peek:
            self.reset(self.sequence_length, self.batch_size)
        return stacked_dict

    def get_sample_count(self) -> int:
        total = 0
        for (_, _, packer) in self.pq:
            total += packer.segment_counter
        return total