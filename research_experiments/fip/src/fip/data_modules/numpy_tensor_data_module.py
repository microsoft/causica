import os
from typing import Optional

import fsspec
import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data as data_utils
from torch.utils.data import Dataset


class DatasetCounterFactual(Dataset):
    """A custom Dataset class for counterfactual data."""

    def __init__(
        self,
        data_dir: str,
        num_interventions: int,
    ):
        """
        Args:
            data_dir: Path with all the data
            num_interventions: Number of interventions
        """

        self.true_graph: Optional[torch.Tensor]
        graph_path = os.path.join(data_dir, "true_graph" + ".npy")
        if os.path.exists(graph_path):
            with fsspec.open(graph_path, "rb") as f:
                self.true_graph = torch.tensor(np.load(f), dtype=torch.long)
        else:
            print("Error loading file " + graph_path)
            self.true_graph = None

        assert num_interventions > 0, "Please provide a positive number of interventions"

        self.factual_data = []
        self.counterfactual_data = []
        self.intervention_indices = []
        self.intervention_values = []
        for k in range(num_interventions):
            curr_counterfactual_data_path = os.path.join(data_dir, "x_cf_" + str(k) + ".npy")

            if os.path.exists(curr_counterfactual_data_path):
                with fsspec.open(curr_counterfactual_data_path, "rb") as f:
                    curr_counterfactual_data = torch.tensor(np.load(f), dtype=torch.float)
                assert (
                    curr_counterfactual_data.shape[1] % 2 == 0
                ), "The data should be of shape (batch_size, num_nodes * 2 + 2)"
                self.num_nodes = curr_counterfactual_data.shape[1] // 2 - 1

                x_f = curr_counterfactual_data[:, : self.num_nodes]
                self.factual_data.append(x_f)

                x_cf = curr_counterfactual_data[:, self.num_nodes : self.num_nodes * 2]
                self.counterfactual_data.append(x_cf)

                intervention_indices = curr_counterfactual_data[:, -2].long()
                self.intervention_indices.append(
                    intervention_indices[0].item()
                )  # here we assume that the intervention indices are the same for all the samples per file

                intervention_values = curr_counterfactual_data[:, -1]
                self.intervention_values.append(intervention_values)
            else:
                print("Error loading file " + curr_counterfactual_data_path)
                continue

    def __len__(self):
        return len(self.factual_data)

    def __getitem__(self, idx):
        res = (
            self.factual_data[idx],
            self.counterfactual_data[idx],
            self.intervention_indices[idx],
            self.intervention_values[idx],
        )
        if self.true_graph is not None:
            return res + (self.true_graph,)
        return res


class DatasetSingleTask(Dataset):
    """A custom Dataset class to uniformize the training data."""

    def __init__(
        self,
        data_path: str,
        standardize: bool,
        mean_data: Optional[torch.Tensor] = None,
        std_data: Optional[torch.Tensor] = None,
        graph_path: Optional[str] = None,
        split_data_noise: bool = False,
    ):
        """
        Args:
            data_path (string): Path with all the data
            standardize (bool): Whether to standardize the data
            mean_data (torch.Tensor): Mean of the data
            std_data (torch.Tensor): Standard deviation of the data
            graph_path (string): Path to the the graph.
            split_data_noise (bool): Whether to split the data into observations and noise
        """
        self.true_graph: Optional[torch.Tensor]
        if graph_path is not None:
            with fsspec.open(graph_path, "rb") as f:
                self.true_graph = torch.tensor(np.load(f), dtype=torch.long)
        else:
            self.true_graph = None

        with fsspec.open(data_path, "rb") as f:
            self.data = torch.tensor(np.load(f), dtype=torch.float)

        self.num_samples = self.data.shape[0]

        self.split_data_noise = split_data_noise
        if self.split_data_noise:
            assert (
                self.data.shape[1] % 2 == 0
            ), "The data should be of shape (batch_size, num_nodes * 2) if split_data_noise is True"
            self.num_nodes = self.data.shape[1] // 2

            if mean_data is not None and std_data is not None:
                self.mean_data = mean_data
                self.std_data = std_data
            else:
                self.mean_data = self.data[:, : self.num_nodes].mean(dim=0, keepdim=True)
                self.std_data = self.data[:, : self.num_nodes].std(dim=0, keepdim=True)

            self.standardize = standardize
            if self.standardize:
                # we standardizes both noise and observations using the same mean and std (of the original observations)
                self.data[:, : self.num_nodes] = (self.data[:, : self.num_nodes] - self.mean_data) / self.std_data
                self.data[:, self.num_nodes :] = self.data[:, self.num_nodes :] / self.std_data
        else:
            self.num_nodes = self.data.shape[1]

            if mean_data is not None and std_data is not None:
                self.mean_data = mean_data
                self.std_data = std_data
            else:
                self.mean_data = self.data.mean(dim=0, keepdim=True)
                self.std_data = self.data.std(dim=0, keepdim=True)

            self.standardize = standardize
            if self.standardize:
                self.data = (self.data - self.mean_data) / self.std_data

        if self.true_graph is not None:
            assert (
                self.num_nodes == self.true_graph.shape[-1]
            ), "The number of nodes in the graph and the data should match"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_data = self.data[idx, ...]

        if self.split_data_noise:
            res = sample_data[: self.num_nodes], sample_data[self.num_nodes :]
        else:
            res = (sample_data,)

        if self.true_graph is not None:
            res = res + (self.true_graph,)

        return res


def sample_base_datasets(
    data_dir: str = "",
    standardize: bool = True,
    with_true_graph: bool = True,
    split_data_noise: bool = True,
) -> tuple:
    """Provides the datasets for training, validation and testing.

    Args:
        data_dir: Directory storing the dataset
        standardize: Whether to standardize the data
        with_true_graph: Whether to load the true graph
        split_data_noise: Whether to split the data into observations and noise
    """

    train_data_path = os.path.join(data_dir, "train_x.npy")
    val_data_path = os.path.join(data_dir, "val_x.npy")
    test_data_path = os.path.join(data_dir, "test_x.npy")

    if with_true_graph:
        graph_path = os.path.join(data_dir, "true_graph" + ".npy")
    else:
        graph_path = None

    data_obj_train = DatasetSingleTask(
        train_data_path,
        standardize,
        graph_path=graph_path,
        split_data_noise=split_data_noise,
    )
    if data_obj_train.standardize:
        mean_data = data_obj_train.mean_data
        std_data = data_obj_train.std_data
    else:
        mean_data = None
        std_data = None

    data_obj_val = DatasetSingleTask(
        val_data_path,
        standardize,
        mean_data=mean_data,
        std_data=std_data,
        graph_path=graph_path,
        split_data_noise=split_data_noise,
    )

    data_obj_test = DatasetSingleTask(
        test_data_path,
        standardize,
        mean_data=mean_data,
        std_data=std_data,
        graph_path=graph_path,
        split_data_noise=split_data_noise,
    )

    return data_obj_train, data_obj_val, data_obj_test


def sample_base_data_loaders(
    data_dir: str = "",
    standardize: bool = True,
    with_true_graph: bool = True,
    split_data_noise: bool = True,
    train_batch_size: int = 64,
    test_batch_size: int = 1024,
    only_observational_data: bool = False,
) -> tuple:
    """Provides the data loaders for different train and test batch sizes.

    First samples the dataset using the get_dataset function
    Then creates data loader for train, validation, and test splits of the dataset

    Args:
        data_dir: Directory storing the dataset
        standardize: Whether to standardize the data
        with_true_graph: Whether to load the true graph
        split_data_noise: Whether to split the data into observations and noise
        train_batch_size: Batch size used for training a model
        test_batch_size: Batch size used for evaluating a model
        only_observational_data: Whether to return only the observational data
    """

    data_obj_train, data_obj_val, data_obj_test = sample_base_datasets(
        data_dir,
        standardize=standardize,
        with_true_graph=with_true_graph,
        split_data_noise=split_data_noise,
    )

    if only_observational_data:
        # remove the noise from the data
        train_data = data_obj_train.data[:, : data_obj_train.num_nodes]
        val_data = data_obj_val.data[:, : data_obj_val.num_nodes]
        test_data = data_obj_test.data[:, : data_obj_test.num_nodes]

    # create dataloader with only observational data
    train_loader = data_utils.DataLoader(train_data, batch_size=train_batch_size)
    val_loader = data_utils.DataLoader(val_data, batch_size=test_batch_size)
    test_loader = data_utils.DataLoader(test_data, batch_size=test_batch_size)

    if with_true_graph:
        true_graph = data_obj_train.true_graph
        return train_loader, val_loader, test_loader, true_graph
    return train_loader, val_loader, test_loader


class NumpyTensorDataModule(pl.LightningDataModule):
    """Provides the data loaders for different data splits and batch size."""

    def __init__(
        self,
        data_dir: str = "",
        train_batch_size: int = 64,
        test_batch_size: int = 1024,
        standardize: bool = True,
        with_true_graph: bool = True,
        split_data_noise: bool = True,
        dod: bool = False,
        num_workers: int = 0,
        shuffle: bool = False,
        num_interventions: int = 0,
    ):
        """
        Args:
            data_dir: Directory storing the dataset
            train_batch_size: Batch size used for training a model
            test_batch_size: Batch size used for evaluating a model
            standardize: Whether to standardize the data
            with_true_graph: Whether to load the true graph
            split_data_noise: Whether to split the data into observations and noise
            dod: Whether to use the DoD approach
            num_workers: Number of workers to use for data loading
            shuffle: Whether to shuffle the data
            num_interventions: Number of interventions, if it is set to 0 no counterfactual data is loaded
        """
        super().__init__()
        self.data_dir = data_dir
        self.train_batch_size_init = train_batch_size
        self.test_batch_size_init = test_batch_size
        self.standardize = standardize
        self.with_true_graph = with_true_graph
        self.split_data_noise = split_data_noise
        self.dod = dod
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.num_interventions = num_interventions

        if self.dod:
            assert (
                self.train_batch_size_init == self.test_batch_size_init
            ), "If DoD, train and test batch sizes should be the same to ensure that the same number of samples per dataset are used for training and testing"

    def create_batch_view(self, x: torch.Tensor, batch_size: int):
        # Create batches to predict the leaves
        num_batches = x.shape[0] // batch_size
        num_samples = num_batches * batch_size
        x = x[:num_samples, :]

        # Reshape observational data for leaf predictions
        total_nodes = x.shape[-1]
        return x.view(num_batches, batch_size, total_nodes)

    def get_output_uniform_shape(self, batch):
        if len(batch) == 3:
            x, n, true_graph = batch
        elif len(batch) == 2 and self.with_true_graph:
            x, true_graph = batch
            n = None
        elif len(batch) == 2 and self.split_data_noise:
            x, n = batch
            true_graph = None
        elif len(batch) == 1:
            x = batch[0]
            n = None
            true_graph = None
        else:
            raise ValueError("Wrong batch size")

        return x, n, true_graph

    def on_before_batch_transfer(self, batch, dataloader_idx):
        if dataloader_idx == 0:
            x, n, true_graph = self.get_output_uniform_shape(batch)

            if true_graph is not None and len(true_graph.shape) == 3:
                true_graph = true_graph[0]

            if self.dod:
                x = self.create_batch_view(x, self.train_batch_size_init)
                if n is not None:
                    n = self.create_batch_view(n, self.train_batch_size_init)
                if true_graph is not None:
                    true_graph = true_graph.repeat(x.shape[0], 1, 1)

            if n is None:
                n = torch.zeros_like(x)

            if true_graph is None:
                num_nodes = x.shape[-1]
                true_graph = -torch.ones(num_nodes, num_nodes)

            return x, n, true_graph

        if self.num_interventions > 0 and dataloader_idx == 1:
            x_f, x_cf, intervention_indices, intervention_values, *optional_graph = batch
            x_f = x_f.squeeze(0)
            x_cf = x_cf.squeeze(0)
            intervention_indices = intervention_indices.squeeze(0)
            intervention_values = intervention_values.squeeze(0)
            if optional_graph:
                true_graph = optional_graph[0].squeeze(0)
            else:
                true_graph = None

            return x_f, x_cf, intervention_indices, intervention_values, true_graph

        return None

    def setup(self, stage: str):

        self.train_data, self.val_data, self.test_data = sample_base_datasets(
            self.data_dir,
            standardize=self.standardize,
            with_true_graph=self.with_true_graph,
            split_data_noise=self.split_data_noise,
        )

        tot_num_samples_train = len(self.train_data)
        tot_num_samples_test = len(self.test_data)

        if self.dod:
            self.train_batch_size = tot_num_samples_train
            self.test_batch_size = tot_num_samples_test
        else:
            self.train_batch_size = self.train_batch_size_init
            self.test_batch_size = self.test_batch_size_init

        if self.num_interventions > 0:
            self.cf_dataset = DatasetCounterFactual(
                data_dir=self.data_dir,
                num_interventions=self.num_interventions,
            )

    def train_dataloader(self):
        return data_utils.DataLoader(
            self.train_data, batch_size=self.train_batch_size, shuffle=self.shuffle, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return data_utils.DataLoader(self.val_data, batch_size=self.test_batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return data_utils.DataLoader(self.test_data, batch_size=self.test_batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        test_dataloader = data_utils.DataLoader(
            self.test_data, batch_size=self.test_batch_size, num_workers=self.num_workers
        )
        if self.num_interventions > 0:
            cf_dataloader = data_utils.DataLoader(self.cf_dataset, batch_size=1, num_workers=self.num_workers)
            return [test_dataloader, cf_dataloader]

        return test_dataloader
