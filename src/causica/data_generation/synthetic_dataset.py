from itertools import repeat, starmap
from typing import Optional

import numpy as np
import torch
from tensordict import TensorDict
from torch.utils.data import IterableDataset

from causica.data_generation.samplers.sem_sampler import SEMSampler
from causica.datasets.interventional_data import InterventionData
from causica.distributions.transforms import TensorToTensorDictTransform


class CausalDataset(IterableDataset):
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
        dataset_size: Optional[int] = None,
        num_interventions: int = 0,
        num_intervention_samples: int = 1000,
        num_sems: int = 0,
    ):
        """
        Args:
            sem_sampler: The sampler for SEMs
            sample_dataset_size: The size of the dataset to sample from the SEM
            dataset_size: The size of this dataset, if not supplied it will be infinitely long
                It is useful to set this value to a finite size so it can be used with `ChainDataset`,
                which relies on the iterator terminating to chain the next one.
            num_interventions: The number of interventions to sample per dataset. If 0, no interventions are sampled.
            num_intervention_samples: The number of samples to use to estimate the mean.
            num_sems: The number of sems to sample the data from. If 0, each data sample is generated from a new SEM.
        """
        self.sem_sampler = sem_sampler
        self.sample_dataset_size = torch.Size([sample_dataset_size])
        self.dataset_size = dataset_size
        self.num_interventions = num_interventions
        self.num_intervention_samples = num_intervention_samples
        self.num_sems = num_sems

        self.sems = [sem_sampler.sample() for _ in range(num_sems)]
        self.td_to_tensor_transform = TensorToTensorDictTransform(self.sem_sampler.shapes_dict)
        self.cur_iter = 0

    def __iter__(self):
        """Return an iterator over samples in the dataset.

        See Also:
            CausalDataset: For a description of the format of the samples.

        Note:
            We use starmap as in `repeatfunc` https://docs.python.org/3/library/itertools.html#itertools-recipes
            This creates a generator applying `_sample `dataset_size` times, or an infinite
            generator if `dataset_size` is `None`
        """
        return starmap(self._sample, repeat(tuple(), times=self.dataset_size))

    def _sample_intervention(self, sem, tensordict_data) -> InterventionData:
        """Sample an intervention and it's sample mean from a given SEM.

        Args:
            sem: SEM to sample interventional data from.
            tensordict_data: Base data for sampling an intervention value.

        Returns:
            an intervention data object
        """
        # sample the treatment and effect variable
        treatment = np.random.choice(sem.node_names, size=1, replace=False).item()

        batch_axes = tuple(range(tensordict_data.batch_dims))
        treatment_shape = tensordict_data[treatment].shape[tensordict_data.batch_dims :]
        treatment_max = torch.amax(tensordict_data[treatment], dim=batch_axes)
        treatment_min = torch.amin(tensordict_data[treatment], dim=batch_axes)

        treatment_a = torch.rand(treatment_shape) * (treatment_max - treatment_min) + treatment_min

        intervention_a = TensorDict({treatment: treatment_a}, batch_size=torch.Size())

        intervention_a_samples = sem.do(intervention_a).sample((self.num_intervention_samples,))

        intervention_data = InterventionData(
            intervention_a_samples,
            intervention_a,
            TensorDict({}, batch_size=torch.Size()),
        )
        return intervention_data

    def _sample(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[list[InterventionData]]]:
        """Sample a new dataset and returns the data, graph and interventions if num_interventions > 0."""
        if self.num_sems > 0:
            sem = self.sems[self.cur_iter % self.num_sems]
            self.cur_iter += 1
        else:
            sem = self.sem_sampler.sample()
        noise = sem.sample_noise(self.sample_dataset_size)
        observations = sem.noise_to_sample(noise)

        interventions = None
        if self.num_interventions > 0:
            interventions = [self._sample_intervention(sem, observations) for _ in range(self.num_interventions)]

        return (observations, noise, sem.graph, interventions)
