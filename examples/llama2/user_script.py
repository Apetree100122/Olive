# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import random

from typing import Union

import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer

from olive.data.registry import Registry

# -----------------------------------------------------------------------------
#  QLoRA load_dataset component
# -----------------------------------------------------------------------------


@Registry.register_dataset()
def tiny_code_dataset(
    data_name: str, split: str, language: str, token: Union[bool, str] = True, trust_remote_code=None
):
    # TODO(anyone): build-in tiny code example dataset
    dataset = load_dataset(data_name, split=split, token=token, trust_remote_code=trust_remote_code)
    return dataset.filter(lambda x: x["programming_language"] == language)


# -----------------------------------------------------------------------------
# Quantization calibration
# -----------------------------------------------------------------------------


def tokenize_function(examples):
    # There's a bug that makes the rust-based fast tokenizer hang randomly (probably due to a deadlock),
    # so use the "slow" python one instead
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False)
    return tokenizer(examples["text"])


class PileDataloader:
    def __init__(self, batch_size=1, seqlen=2048, max_seq_len=2080, sub_folder="train"):
        random.seed(0)
        self.seqlen = seqlen
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.dataset = load_dataset("NeelNanda/pile-10k", split=sub_folder)
        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    def __iter__(self):
        length = len(self.dataset)
        counter = 0

        while counter < length:
            # Pick a random sample from the dataset that has at least 2048 tokens
            sample_index = random.randint(0, len(self.dataset) - 1)
            sample = self.dataset[sample_index]["input_ids"]
            while sample.shape[0] <= self.seqlen:
                sample_index = random.randint(0, len(self.dataset) - 1)
                sample = self.dataset[sample_index]["input_ids"]

            # Randomly pick a subsequence of 2048 tokens in the middle of the dataset
            token_start = random.randint(0, sample.shape[0] - self.seqlen - 1)
            token_end = token_start + self.seqlen
            input_ids = sample[token_start:token_end].unsqueeze(0).cpu().numpy().astype("int64")

            initial_position_ids = np.arange(self.seqlen, dtype=np.int64).reshape((1, self.seqlen))
            attention_mask = np.pad(
                np.ones((1, self.seqlen), dtype=np.int64), ((0, 0), (0, self.max_seq_len - self.seqlen))
            )

            initial_inputs = {
                "input_ids": input_ids,
                "position_ids": initial_position_ids,
                "attention_mask": attention_mask,
            }

            for layer_index in range(32):
                initial_inputs[f"past_key_values.{layer_index}.key"] = np.zeros(
                    (1, 32, self.max_seq_len, 128), dtype=np.float32
                )
                initial_inputs[f"past_key_values.{layer_index}.value"] = np.zeros(
                    (1, 32, self.max_seq_len, 128), dtype=np.float32
                )

            yield initial_inputs, 0

class PileDataloader2:
    def __init__(self, batch_size=2, seqlen=8, max_seq_len=8, sub_folder="train"):
        random.seed(0)
        self.seqlen = seqlen
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.dataset = load_dataset("NeelNanda/pile-10k", split=sub_folder)
        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    def __iter__(self):
        length = len(self.dataset)
        counter = 0

        while counter < length:
            batch_input_ids = []
            batch_position_ids = []
            batch_attention_mask = []

            for _ in range(self.batch_size):
                # Pick a random sample from the dataset that has at least seqlen tokens
                sample_index = random.randint(0, len(self.dataset) - 1)
                sample = self.dataset[sample_index]["input_ids"]
                while sample.shape[0] < self.seqlen:
                    sample_index = random.randint(0, len(self.dataset) - 1)
                    sample = self.dataset[sample_index]["input_ids"]

                # Randomly pick a subsequence of seqlen tokens in the middle of the dataset
                token_start = random.randint(0, sample.shape[0] - self.seqlen)
                token_end = token_start + self.seqlen
                input_ids = sample[token_start:token_end].cpu().numpy().astype("int64")
                position_ids = np.arange(self.seqlen, dtype=np.int64)
                attention_mask = np.pad(
                    np.ones(self.seqlen, dtype=np.int64), (0, self.max_seq_len - self.seqlen)
                )

                batch_input_ids.append(input_ids)
                batch_position_ids.append(position_ids)
                batch_attention_mask.append(attention_mask)

            batch_input_ids = np.stack(batch_input_ids)
            batch_position_ids = np.stack(batch_position_ids)
            batch_attention_mask = np.stack(batch_attention_mask)

            initial_inputs = {
                "input_ids": batch_input_ids,
                "position_ids": batch_position_ids,
                "attention_mask": batch_attention_mask,
            }

            for layer_index in range(32):
                initial_inputs[f"past_key_values.{layer_index}.key"] = np.zeros(
                    (self.batch_size, 32, self.max_seq_len, 128), dtype=np.float32
                )
                initial_inputs[f"past_key_values.{layer_index}.value"] = np.zeros(
                    (self.batch_size, 32, self.max_seq_len, 128), dtype=np.float32
                )

            yield initial_inputs, 0
            counter += self.batch_size

def calib_dataloader(data_dir, batch_size, *args, **kwargs):
    return PileDataloader2(batch_size=batch_size)
