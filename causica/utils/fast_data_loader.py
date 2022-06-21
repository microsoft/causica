import torch
from torch.utils.data import DataLoader


class FastTensorDataLoader(DataLoader):
    """
    Fast data loader for in-memory tensors. This loader avoids any calls to `torch.stack`
    and is the fastest choice when datasets can be held in memory.
    """

    def __init__(self, *tensors, batch_size, shuffle=False, drop_last=False):  # pylint: disable=super-init-not-called
        """
        Args:
            *tensors (torch.Tensor): the tensors that form the dataset. Dimension 0 is regarded as the
                      dimension to draw batches from.
            batch_size (int): the batch size for this data loader.
            shuffle (bool): whether to shuffle the dataset.
            drop_last (bool): whether to neglect the final batch (ensures every batch has the same size).
        """
        self.tensors = tensors
        self.n_rows = self.tensors[0].shape[0]

        # Check all tensors have the same number of rows
        assert all(a.shape[0] == self.n_rows for a in tensors)
        # Checkk all tensors on same device
        assert all(a.device == self.tensors[0].device for a in tensors)

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        if self.shuffle:
            self.idxs = torch.randperm(self.n_rows, device=self.tensors[0].device)
        else:
            self.idxs = torch.arange(self.n_rows, device=self.tensors[0].device)
        self.batch_start = 0
        return self

    def __next__(self):
        if self.drop_last and (self.batch_start + self.batch_size > self.n_rows):
            raise StopIteration
        if self.batch_start >= self.n_rows:
            raise StopIteration
        idxs = self.idxs[self.batch_start : self.batch_start + self.batch_size]
        batch = tuple(a[idxs] for a in self.tensors)
        self.batch_start += self.batch_size
        return batch

    def __len__(self):
        if self.n_rows % self.batch_size == 0 or self.drop_last:
            return self.n_rows // self.batch_size
        else:
            return 1 + self.n_rows // self.batch_size
