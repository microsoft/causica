import itertools
import random
import warnings
from itertools import zip_longest
from typing import Iterable, Iterator

import torch
from torch.utils.data import BatchSampler, Sampler, SubsetRandomSampler


class SubsetBatchSampler(Sampler[list[int]]):
    """A Pytorch batch sampler that samples batches from a list of subsets.

    Each batch will be sampled from a single subset. The subsets are sampled in random order if shuffle is True.
    """

    def __init__(
        self, subset_lengths: list[int], batch_size: int, shuffle: bool = True, drop_last: bool = False
    ) -> None:
        """
        Args:
            subset_lengths: The lengths of each subset
            batch_size: The batch size to use
            shuffle: Whether to shuffle the indices
            drop_last: Whether to drop the last batch of each subset if it is smaller than the batch size.
        """
        if batch_size > min(subset_lengths):
            warnings.warn("Batch size is larger than the smallest subset length")
        if any(length < 1 for length in subset_lengths):
            raise ValueError("Subset lengths must be at least 1")

        self.subset_lengths = subset_lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.start_indices = torch.cumsum(torch.tensor([0] + self.subset_lengths[:-1]), 0)

        self.batch_samplers: list[Iterator | BatchSampler]
        self.indices_lists = [
            (torch.arange(length) + offset).tolist() for length, offset in zip(self.subset_lengths, self.start_indices)
        ]
        if self.shuffle:
            self.batch_samplers = [
                BatchSampler(SubsetRandomSampler(indices), batch_size, drop_last=drop_last)
                for indices in self.indices_lists
            ]
        else:
            # Replace with `itertools.batched` when Python >= 3.12
            self.batch_samplers = []
            for indices in self.indices_lists:
                args = [iter(indices)] * batch_size
                self.batch_samplers.append(zip_longest(*args, fillvalue=None))

    def __iter__(self):
        # Support function to truncate batches to the correct size when not shuffling
        if not self.shuffle:
            yield from itertools.chain.from_iterable(
                yield_truncated_batch_from_zip(iterator, self.batch_size, self.drop_last)
                for iterator in self.batch_samplers
            )

        iterators = [iter(batch_sampler) for batch_sampler in self.batch_samplers]
        while iterators:
            permutation = list(enumerate(iterators))
            random.shuffle(permutation)
            stopped = []
            for i, iterator in permutation:
                try:
                    yield next(iterator)
                except StopIteration:
                    stopped.append(i)

            # Delete in reverse order to preserve indices
            for i in sorted(stopped, reverse=True):
                del iterators[i]

    def __len__(self) -> int:
        if self.drop_last:
            return sum(length // self.batch_size for length in self.subset_lengths)
        return sum((length + self.batch_size - 1) // self.batch_size for length in self.subset_lengths)


def yield_truncated_batch_from_zip(iterator: Iterable, batch_size: int, drop_last: bool = False):
    """Yield truncated batches from an iterator (dropping None elements), skipping the last batch if necessary."""
    for batch in iterator:
        batch = list(b for b in batch if b is not None)
        if not drop_last or len(batch) == batch_size:
            yield batch
        else:
            return
