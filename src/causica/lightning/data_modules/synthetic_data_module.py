from itertools import product
from typing import Iterable

import pytorch_lightning as pl
import torch
import torch.distributions as td
from tensordict import TensorDict
from torch.utils.data import ChainDataset, DataLoader, Dataset

from causica.data_generation.samplers.functional_relationships_sampler import (
    LinearRelationshipsSampler,
    RFFFunctionalRelationshipsSampler,
)
from causica.data_generation.samplers.noise_dist_sampler import (
    JointNoiseModuleSampler,
    UnivariateNormalNoiseModuleSampler,
)
from causica.data_generation.samplers.sampler import Sampler
from causica.data_generation.samplers.sem_sampler import SEMSampler
from causica.data_generation.synthetic_dataset import CausalDataset
from causica.distributions import ErdosRenyiDAGDistribution
from causica.functional_relationships.functional_relationships import FunctionalRelationships


class SyntheticDataModule(pl.LightningDataModule):
    """A datamodule to produce datasets and their underlying causal graphs and interventions."""

    def __init__(
        self,
        node_nums: list[int],
        probs: list[float],
        train_batch_size: int,
        test_batch_size: int,
        dataset_size: int,
        num_interventions: int = 0,
        num_sems: int = 0,
        num_workers: int = 0,
    ) -> None:
        """
        Args:
            node_nums: List of number of nodes for generated graphs
            probs: List of the probabilities of edges in each graph when using Erdos Renyi
            train_batch_size: The training batch size to use
            test_batch_size: The testing batch size to use
            dataset_size: The size of dataset to generate
            num_interventions: The number of interventions to generate (0 for no interventions)
            num_sems: The number of SEMs to generate (0 for infinite SEMs)
            num_workers: The number of workers to use for the dataloader
        """
        super().__init__()
        self.node_nums = node_nums
        self.probs = probs
        self.dataset_size = dataset_size
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.sems_dict: dict[tuple[int, float, bool], SEMSampler] = {}
        self.num_interventions = num_interventions
        self.num_workers = num_workers
        self.num_sems = num_sems

        self.dataloader_args = {
            "collate_fn": _tuple_collate_fn,
            "num_workers": self.num_workers,
            "worker_init_fn": worker_init_fn if self.num_workers > 0 else None,
            "persistent_workers": self.num_workers > 0,
            "pin_memory": True,
            "prefetch_factor": 16 if self.num_workers > 0 else None,
        }

        self.val_dataloader_args = {
            "collate_fn": _tuple_collate_fn,
            "pin_memory": True,
        }

        self.train_dataset: Dataset
        self.val_dataset: Dataset

        self.save_hyperparameters()

    def setup(self, stage: str):
        for num_nodes, prob, is_linear in product(self.node_nums, self.probs, (True, False)):
            shapes_dict = {f"x_{i}": torch.Size([1]) for i in range(num_nodes)}
            noise_dist_samplers = {
                f"x_{i}": UnivariateNormalNoiseModuleSampler(std_dist=td.Uniform(low=0.2, high=2.0), dim=1)
                for i in range(num_nodes)
            }
            joint_noise_module_sampler = JointNoiseModuleSampler(noise_dist_samplers)
            adjacency_dist = ErdosRenyiDAGDistribution(num_nodes=num_nodes, probs=torch.tensor(prob))

            dim = sum(shape.numel() for shape in shapes_dict.values())

            functional_relationships_sampler: Sampler[FunctionalRelationships]

            if is_linear:
                ones_matrix = torch.ones((dim, dim), dtype=torch.float32)
                functional_relationships_sampler = LinearRelationshipsSampler(
                    scale_dist=td.Uniform(
                        low=ones_matrix,
                        high=3.0 * ones_matrix,
                    ),
                    shapes_dict=shapes_dict,
                )
            else:
                num_rf = 100
                ones_matrix = torch.ones((num_rf, dim), dtype=torch.float32)
                ones = torch.ones((num_rf,), dtype=torch.float32)
                functional_relationships_sampler = RFFFunctionalRelationshipsSampler(
                    rf_dist=td.Uniform(
                        low=7.0 * ones_matrix,
                        high=10.0 * ones_matrix,
                    ),
                    coeff_dist=td.Uniform(
                        low=10.0 * ones,
                        high=20.0 * ones,
                    ),
                    shapes_dict=shapes_dict,
                )

            # store each of the sem samplers in a dictionary
            self.sems_dict[(num_nodes, prob, is_linear)] = SEMSampler(
                adjacency_dist=adjacency_dist,
                joint_noise_module_sampler=joint_noise_module_sampler,
                functional_relationships_sampler=functional_relationships_sampler,
            )

        self.train_dataset = self._get_dataset(self.train_batch_size)
        self.val_dataset = self._get_dataset(self.test_batch_size)

    def _get_dataset(self, dataset_size: int) -> Dataset:
        """Builds causal datasets given the SEM samplers.

        Args:
            dataset_size: Number of samples of the causal dataset (ie number of datasets generated).

        Returns:
            dataset object
        """
        dataset_fraction = self.num_workers if self.num_workers > 0 else 1
        factor = self.num_sems if self.num_sems > 0 else 1
        dataset = ChainDataset(
            [
                CausalDataset(
                    sampler,
                    sample_dataset_size=self.dataset_size,
                    dataset_size=dataset_size * factor * 16 // dataset_fraction,
                    num_interventions=self.num_interventions,
                    num_sems=self.num_sems,
                )
                for sampler in self.sems_dict.values()
            ]
        )

        return dataset

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.train_batch_size, **self.dataloader_args)

    def val_dataloader(self):
        datasets = [self.train_dataset, self.val_dataset] if self.num_sems > 0 else [self.val_dataset]

        return [
            DataLoader(dataset=dataset, batch_size=self.test_batch_size, **self.val_dataloader_args)
            for dataset in datasets
        ]


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset  # the dataset copy in this worker process
    if isinstance(dataset, ChainDataset):
        # Split chained datasets across multiple workers.
        dataset.datasets = [dataset.datasets[worker_info.id % len(dataset.datasets)]]


def _tuple_collate_fn(data: Iterable[tuple[torch.Tensor, ...]]) -> tuple[torch.Tensor, ...]:
    """Collates a list of tuples of tensors into a tuple of tensors.

    The dataloader returns batch_shape tuples of (X, y, ...), so we stack them all
    to get tensors of shapes [batch_shape, *X.shape] and [batch_shape, *y.shape] and so on.

    Args:
        data: list of tuple of tensors to collate. Assumes the dimensions of the tensors in the tuples match.

    Returns:
        collated data
    """

    def _nested_stack(x: list):
        """Stacks a tuple of tensors, returns None if an element is None, or returns lists of lists."""
        if isinstance(x[0], (torch.Tensor, TensorDict)):
            return torch.stack(x, dim=0)
        if isinstance(x[0], list):
            return list(x)
        if x[0] is None:
            return None

        raise ValueError(f"Unexpected type {type(x[0])}")

    return tuple(_nested_stack(list(x)) for x in zip(*data))
