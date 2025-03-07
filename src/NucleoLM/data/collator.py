# modified from https://github.com/CatIIIIIIII/RNAErnie2

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Dict, List, Union, Tuple, Optional

import numpy as np
import torch
from transformers import DataCollatorForLanguageModeling
from transformers.data.data_collator import pad_without_fast_tokenizer_warning, _torch_collate_batch


@dataclass
class PretrainDataCollatorWithStructure(DataCollatorForLanguageModeling):

    motif_tree: Dict[str, List] = None
    step_cnt: int = 0

    r"""
    Data collator for pretraining data.
    """

    def set_ratio(self, step: int, total_steps: int = 167789):
        ratio = None
        if step < total_steps * 0.15:
            ratio = 0.
        elif step < total_steps * 0.85:
            # linear increase from 0 to 0.5
            ratio = 0.5 * (step - total_steps * 0.15) / (total_steps * 0.7)
        else:
            ratio = 0.5
        return ratio

    def __call__(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        self.step_cnt += 1
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer, examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of
            )
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        # randomly take one of the three masking strategies
        ratio = self.set_ratio(self.step_cnt)
        mask_strategy = np.random.choice(["tokens", "subseq", "motif", "structure"], p=[1-ratio, 0., 0., ratio])
        if mask_strategy == "tokens":
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        elif mask_strategy == "subseq":
            batch["input_ids"], batch["labels"] = self.torch_mask_subseq(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        elif mask_strategy == "motif":
            batch["input_ids"], batch["labels"] = self.torch_mask_motif(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        elif mask_strategy == "structure":
            batch["input_ids"], batch["labels"] = self.torch_mask_structure(
                batch["input_ids"], batch["connects"], special_tokens_mask=special_tokens_mask
            )
        else:
            raise ValueError("Unknown masking strategy.")

        return batch

    def mask_transform(self, inputs: Any, labels: Any, masked_indices: Any) -> Tuple[Any, Any]:
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(
                special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        return self.mask_transform(inputs, labels, masked_indices)

    def torch_mask_subseq(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked sub-sequence inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        subseq_ranges = (3, 6)

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(
            labels.shape,
            self.mlm_probability / (subseq_ranges[0] + subseq_ranges[1]) * 2)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(
                special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()
        # special_tokens_mask[:, 1:1+subseq_ranges[-1]] = True

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # randomly extend the masked token to masked sub-sequence
        rows, cols = masked_indices.shape
        extend_lengths = torch.randint(
            subseq_ranges[0], subseq_ranges[1]+1, (rows, cols))

        true_indices = torch.nonzero(masked_indices, as_tuple=True)
        start_indices = true_indices[1] - extend_lengths[true_indices]
        mask = torch.zeros_like(masked_indices, dtype=torch.bool)
        for i, (row, col) in enumerate(zip(*true_indices)):
            # Ensure start index is not negative
            start_idx = max(start_indices[i], 0)
            mask[row, start_idx:col+1] = True
        masked_indices = mask & ~special_tokens_mask
        return self.mask_transform(inputs, labels, masked_indices)

    def torch_mask_motif(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked motif inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        motif_tree_db = self.motif_tree["DataBases"]
        motif_tree_stats = self.motif_tree["Statistics"]

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(
                special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()
        masked_num = (~special_tokens_mask).sum(axis=1) * self.mlm_probability
        # labels = -100 * torch.ones_like(labels)
        masked_indices = torch.zeros_like(labels, dtype=torch.bool)

        # search for motifs
        np_rng = np.random.default_rng()
        inputs_list = inputs.cpu().tolist()
        for i, input_list in enumerate(inputs_list):
            ngram_indexes = []
            # motif_name = random.sample(motif_trees.keys(), 1)
            search_results = motif_tree_db.search_all(input_list)
            for result in search_results:
                ngram_index = list(range(result[1], result[1] + len(result[0])))
                ngram_indexes.append(ngram_index)
            np_rng.shuffle(ngram_indexes)

            # mask reminding tokens with statistics if former not full
            search_results = motif_tree_stats.search_all(input_list)
            for result in search_results:
                ngram_index = list(range(result[1], result[1] + len(result[0])))
                ngram_indexes.append(ngram_index)

            # merge ngram indexes to a set
            covered_indexes = set()
            for index_set in ngram_indexes:
                if len(covered_indexes) >= masked_num[i]:
                    break
                for index in index_set:
                    if index in covered_indexes:
                        continue
                    covered_indexes.add(index)
            masked_indices[i, list(covered_indexes)] = True

        labels[~masked_indices] = -100

        return self.mask_transform(inputs, labels, masked_indices)


    def torch_mask_structure(self, inputs: Any, connects: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked SS inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        print('in mask_structure', inputs.shape, connects.shape)
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        rng = torch.arange(inputs.shape[1])
        left_connects = torch.where(connects > rng, connects, 0)
        # currently left: masked_left_connects_indices
        connects_indices = connects!=0
        masked_left_fill_right_connects_indices = masked_indices & (left_connects!=0)
        print('before fill', masked_left_fill_right_connects_indices)
        fill_right_indices = torch.nonzero(masked_left_fill_right_connects_indices, as_tuple=True)
        # currently left and right: fill right, masked left_right connects_indices
        for i, (row, col) in enumerate(zip(*fill_right_indices)):
            masked_left_fill_right_connects_indices[row, left_connects[row, col]] = True

        final_masked_indices = masked_left_fill_right_connects_indices | (masked_indices & ~connects_indices)
        print('after  fill', masked_left_fill_right_connects_indices)
        print('after  ori ', masked_indices)
        print('after final', final_masked_indices)
        ## num_masked_token of:  labels: masked_indices, inputs: final_masked_indices
        ## using local various context of loop bases to predict relatively stable helix (SS).
        labels[~final_masked_indices] = -100  # We only compute loss on masked tokens
        return self.mask_transform(inputs, labels, final_masked_indices)
