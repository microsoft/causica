import random
from typing import Callable, Iterable, Optional, Union

import pytorch_lightning as pl
import torch
from tensordict import TensorDict
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from causica.data_generation.samplers.sem_sampler import SEMSampler
from causica.datasets.interventional_data import CounterfactualData
from causica.distributions.transforms import TensorToTensorDictTransform
from causica.sem.structural_equation_model import SEM


def sample_intervention_dict(
    tensordict_data: TensorDict, treatment: str | None = None, proportion_treatment: Optional[float] = None
) -> TensorDict:
    """Sample an intervention from a given SEM.

    This samples a random value for the treatment variable from the data. The value is sampled uniformly from the
    range of the treatment variable in the data.

    The treatment variable is chosen randomly across all nodes if not specified.

    Args:
        tensordict_data: Base data for sampling an intervention value.
        treatment: The name of the treatment variable. If None, a random variable is chosen across the tensordict keys.
        proportion_treatment: the proportion of the intervened nodes.

    Returns:
        A TensorDict holding the intervention value.
    """
    if treatment is None:
        if proportion_treatment is None:
            num_nodes_intervened = 1
        else:
            num_nodes = len(tensordict_data.keys())
            num_nodes_intervened = max(int(num_nodes * proportion_treatment), 1)

        treatment_var = random.sample(list(tensordict_data.keys()), num_nodes_intervened)
    else:
        treatment_var = [treatment]

    batch_axes = tuple(range(tensordict_data.batch_dims))
    res = TensorDict({}, batch_size=torch.Size())
    for curr_treat in treatment_var:
        treatment_shape = tensordict_data[curr_treat].shape[tensordict_data.batch_dims :]
        treatment_max = torch.amax(tensordict_data[curr_treat], dim=batch_axes)
        treatment_min = torch.amin(tensordict_data[curr_treat], dim=batch_axes)
        treatment_curr = torch.rand(treatment_shape) * (treatment_max - treatment_min) + treatment_min
        res[curr_treat] = treatment_curr
    return res


def sample_counterfactual(
    sem: SEM,
    factual_data: TensorDict,
    noise: TensorDict,
    treatment: str | None = None,
    proportion_treatment: Optional[float] = None,
) -> CounterfactualData:
    """Sample an intervention and it's sample mean from a given SEM.

    Args:
        sem: SEM to sample counterfactual data from.
        factual_data: Base data for sampling an counterfactual value.
        noise: Base noise for sampling an counterfactual value.
        treatment: The name of the treatment variable. If None, a random variable is chosen.
        proportion_treatment: the proportion of the intervened nodes.

    Returns:
        an counterfactual data object
    """
    intervention_a = sample_intervention_dict(
        factual_data, treatment=treatment, proportion_treatment=proportion_treatment
    )

    counterfactuals = sem.do(intervention_a).noise_to_sample(noise)

    return CounterfactualData(counterfactuals, intervention_a, factual_data)


def sample_dataset(
    sem: SEM,
    sample_dataset_size: torch.Size,
    td_to_tensor_transform: TensorToTensorDictTransform,
    num_interventions: int = 0,
    num_intervention_samples: int = 1,
    proportion_treatment: float = 0.5,
    sample_counterfactuals: bool = False,
):
    """Sample a new dataset and returns the data, graph and interventions if num_interventions > 0.

    Args:
        sem: The SEM to sample the data from.
        sample_dataset_size: The size of the dataset to sample from the SEM.
        td_to_tensor_transform: The transform to convert the TensorDicts to tensors.
        num_interventions: The number of interventions to sample per dataset. If 0, no interventions are sampled.
        num_intervention_samples: The number of samples to use per intervention.
        proportion_treatment: The proportion of treatment variables to sample.
        sample_counterfactuals: Whether to sample counterfactuals.
    """

    noise = sem.sample_noise(sample_dataset_size)
    observations = sem.noise_to_sample(noise)

    counterfactuals: list[CounterfactualData] | None = [] if sample_counterfactuals and num_interventions > 0 else None
    for _ in range(num_interventions):
        if isinstance(counterfactuals, list):
            cf_noise = sem.sample_noise(torch.Size([num_intervention_samples]))
            cf_observations = sem.noise_to_sample(cf_noise)
            counterfactuals.append(
                sample_counterfactual(sem, cf_observations, cf_noise, proportion_treatment=proportion_treatment)
            )

    noise = td_to_tensor_transform.inv(noise)
    observations = td_to_tensor_transform.inv(observations)

    return (observations, noise, sem.graph, counterfactuals)


class MyCausalDataset(Dataset):
    """A dataset that returns data from SEM samples.

    SEM samples consist of (dataset, noise, graph, interventions)

    The dataset holds samples from a SEM in a TensorDict as {node_name: [num_samples, *node_shape]}
    The noise holds samples from a SEM in a TensorDict as {node_name: [num_samples, *node_shape]}
    The graph is the causal graph of the SEM [num_nodes, num_nodes]
    The interventions are a list of InterventionData objects
    """

    def __init__(
        self,
        sem_sampler: SEMSampler,
        sample_dataset_size: int,
        dataset_size: int,
        num_sems: int = 0,
        num_interventions: int = 0,
        num_intervention_samples: int = 1,
        proportion_treatment: float = 0.5,
        sample_counterfactuals: bool = False,
    ):
        """
        Args:
            sem_sampler: The sampler for SEMs
            sample_dataset_size: The size of the dataset to sample from the SEM
            dataset_size: The size of this dataset.
            num_interventions: The number of interventions to sample per dataset. If 0, no interventions are sampled.
            num_intervention_samples: The number of samples to use to estimate the mean.
            num_sems: The number of sems to sample the data from. If 0, each data sample is generated from a new SEM.
            sample_counterfactuals: Whether to sample counterfactuals.
        """
        self.sem_sampler = sem_sampler
        self.sample_dataset_size = torch.Size([sample_dataset_size])
        self.num_sems = num_sems
        self.num_interventions = num_interventions
        self.num_intervention_samples = num_intervention_samples
        self.proportion_treatment = proportion_treatment
        self.sample_counterfactuals = sample_counterfactuals
        self.dataset_size = dataset_size

        self.sems = [sem_sampler.sample() for _ in range(num_sems)]
        self.td_to_tensor_transform = TensorToTensorDictTransform(self.sem_sampler.shapes_dict)
        self.cur_iter = 0

    def __len__(self):
        if self.num_sems == 0:
            return self.dataset_size
        return self.num_sems * self.dataset_size

    def __getitem__(self, idx):
        """Sample a new dataset and returns the data, graph and interventions if num_interventions > 0."""
        if self.num_sems > 0:
            sem = self.sems[self.cur_iter % self.num_sems]
            self.cur_iter += 1
        else:
            sem = self.sem_sampler.sample()

        return sample_dataset(
            sem,
            self.sample_dataset_size,
            self.td_to_tensor_transform,
            self.num_interventions,
            self.num_intervention_samples,
            self.proportion_treatment,
            self.sample_counterfactuals,
        )


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


class SyntheticDataModule(pl.LightningDataModule):
    """A datamodule to produce datasets and their underlying causal graphs and interventions."""

    def __init__(
        self,
        sem_samplers: Union[list[SEMSampler], Callable[[], list[SEMSampler]]],
        train_batch_size: int,
        test_batch_size: int,
        sample_dataset_size: int,
        standardize: bool,
        num_samples_used: int,
        num_interventions: int = 0,
        num_intervention_samples: int = 1000,
        proportion_treatment: float = 0.5,
        num_sems: int = 0,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        factor_epoch: int = 1,
        shuffle: bool = False,
        sample_counterfactuals: bool = False,
    ) -> None:
        """
        Args:
            sem_samplers: Either a list of sem samplers or a function that returns a list of SEM samplers
            train_batch_size: The training batch size to use
            test_batch_size: The testing batch size to use
            sample_dataset_size: The size of dataset to generate
            standardize: Whether to standardize the data
            num_samples_used: The number of samples to use from the dataset
            num_interventions: The number of interventions to generate (0 for no interventions)
            num_intervention_samples: The number of samples to use per intervention
            proportion_treatment: The proportion of treatment variables to sample
            num_sems: The number of SEMs to generate (0 for infinite SEMs)
            num_workers: The number of workers to use for the dataloader
            pin_memory: Whether to pin the memory
            persistent_workers: Whether to use persistent workers
            prefetch_factor: The prefetch factor to use
            factor_epoch: The factor to multiply the number of SEMs by
            shuffle: Whether to shuffle the data
            sample_counterfactuals: Whether to sample counterfactuals
        """
        super().__init__()
        self.sample_dataset_size = sample_dataset_size
        self.num_samples_used = num_samples_used
        self.standardize = standardize
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_interventions = num_interventions
        self.num_intervention_samples = num_intervention_samples
        self.proportion_treatment = proportion_treatment
        self.num_sems = num_sems
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.sample_counterfactuals = sample_counterfactuals

        self.sem_samplers = sem_samplers if isinstance(sem_samplers, list) else sem_samplers()
        if factor_epoch > 0:
            self.sem_samplers = factor_epoch * self.sem_samplers
            if len(self.sem_samplers) < self.num_workers:
                repeat_factor = max(self.num_workers // len(self.sem_samplers), 1)
                self.sem_samplers = repeat_factor * self.sem_samplers

        self.dataloader_args = {
            "collate_fn": _tuple_collate_fn,
            "num_workers": self.num_workers,
            "persistent_workers": self.num_workers > 0 and persistent_workers,
            "pin_memory": pin_memory,
            "prefetch_factor": prefetch_factor if self.num_workers > 0 else None,
        }

        self.val_dataloader_args = {
            "collate_fn": _tuple_collate_fn,
            "pin_memory": pin_memory,
        }

        self.train_dataset: Dataset
        self.val_dataset: Dataset

        self.train_dataset = self._get_dataset(self.train_batch_size)
        self.val_dataset = self._get_dataset(self.test_batch_size)

    def on_before_batch_transfer(self, batch, dataloader_idx):
        (val_X, val_y, graph, cf_batch) = batch
        graph = graph.transpose(-2, -1)  # because the parents are in the columns!

        if self.standardize:
            mean_data, var_data = self.get_mean_and_var(val_X)
            val_X = self.normalize_data(val_X, mean=mean_data, std=var_data)
            val_y = self.normalize_data(val_y, std=var_data)

        return (val_X[:, : self.num_samples_used, :], val_y[:, : self.num_samples_used, :], graph, cf_batch)

    def get_mean_and_var(self, data, dim=1):
        """Get the mean and variance of the data"""
        mean = data.mean(dim=dim, keepdim=True)
        std = data.std(dim=dim, keepdim=True)
        # if std is zero, replace it with one
        std[std == 0.0] = 1

        return mean, std

    def normalize_data(self, data, mean=None, std=None):
        """Normalize the data with specific mean and variance"""

        mean = torch.zeros_like(data) if mean is None else mean
        std = torch.ones_like(data) if std is None else std

        return (data - mean) / std

    def _get_dataset(self, dataset_size: int):
        """Builds causal datasets given the SEM samplers.

        Args:
            dataset_size: Number of samples of the causal dataset (ie number of datasets generated).

        Returns:
            dataset object
        """
        ## concatenate the dataset into a single dataset:
        dataset: Dataset
        dataset = ConcatDataset(
            [
                MyCausalDataset(
                    sem_sampler=sampler,
                    sample_dataset_size=self.sample_dataset_size,
                    dataset_size=dataset_size,
                    num_sems=self.num_sems,
                    num_interventions=self.num_interventions,
                    num_intervention_samples=self.num_intervention_samples,
                    proportion_treatment=self.proportion_treatment,
                    sample_counterfactuals=self.sample_counterfactuals,
                )
                for sampler in self.sem_samplers
            ]
        )
        return dataset

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle, **self.dataloader_args
        )

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, batch_size=self.test_batch_size, **self.dataloader_args)
