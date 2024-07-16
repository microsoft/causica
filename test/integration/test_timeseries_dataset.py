import os

import torch
from torch.utils.data import DataLoader

from causica.datasets.causica_dataset_format import CAUSICA_DATASETS_PATH
from causica.datasets.timeseries_dataset import IndexedTimeseriesDataset


def test_ecoli_100():
    dataset_root = os.path.join(CAUSICA_DATASETS_PATH, "Ecoli1_100")
    data_path = os.path.join(dataset_root, "train.csv")
    adj_path = os.path.join(dataset_root, "adj_matrix.npy")

    dataset = IndexedTimeseriesDataset(0, data_path, adj_path)
    dataloader = DataLoader(dataset, batch_size=46, collate_fn=torch.stack)
    (full_batch,) = list(dataloader)
    assert full_batch.batch_size == (46, 21)  # The dataset is known to contain 46 timeseries with 21 steps each
    assert len(full_batch.keys()) == 100  # And 100 features
    assert all(value.shape == (46, 21, 1) for value in full_batch.values())  # Each with 1 dimension per sample and step
