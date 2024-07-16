import torch
from torch.utils.data import Dataset

from causica.data_generation.generate_data import sample_dataset
from causica.data_generation.samplers.sem_sampler import SEMSampler
from causica.datasets.causal_dataset import CausalDataset
from causica.distributions.transforms import TensorToTensorDictTransform


class CausalMetaset(Dataset):
    """A metaset (dataset of datasets) that returns datasets from SEM samples.

    SEM samples are returned as a CausalDataset object.

    See Also:
        CausalDataset: For a description of the format of the samples.
    """

    def __init__(
        self,
        sem_sampler: SEMSampler,
        sample_dataset_size: int,
        dataset_size: int,
        num_interventions: int = 0,
        num_intervention_samples: int = 1000,
        num_sems: int = 0,
        sample_interventions: bool = False,
        sample_counterfactuals: bool = False,
        treatment_variable: str | None = None,
        effect_variables: list[str] | None = None,
    ):
        """
        Args:
            sem_sampler: The sampler for SEMs
            sample_dataset_size: The size of the dataset to sample from the SEM
            dataset_size: The size of this dataset, if not supplied it will be infinitely long
                It is useful to set this value to a finite size so it can be used with `ChainDataset`,
                which relies on the iterator terminating to chain the next one.
            num_interventions: The number of interventions to sample per dataset. If 0, no interventions are sampled.
            num_intervention_samples: The number of interventional samples to sample.
            num_sems: The number of sems to sample the data from. If 0, each data sample is generated from a new SEM.
            sample_interventions: Whether to sample interventions.
            sample_counterfactuals: Whether to sample counterfactuals.
            treatment_variable: This specify the name of the nodes to be taken as treatment from the generated sems.
                The sem sampler must always generate this treatment.
            effet_variables: This specify the names of the nodes to be taken as effects from the generated sems.
                The sem sampler must always generate this effect.
        """
        self.sem_sampler = sem_sampler
        self.sample_dataset_size = torch.Size([sample_dataset_size])
        self.dataset_size = dataset_size
        self.num_interventions = num_interventions
        self.num_intervention_samples = num_intervention_samples
        self.num_sems = num_sems
        self.sample_interventions = sample_interventions
        self.sample_counterfactuals = sample_counterfactuals
        self.treatment_variable = treatment_variable
        self.effect_variables = effect_variables

        self.sems = [sem_sampler.sample() for _ in range(num_sems)]
        self.td_to_tensor_transform = TensorToTensorDictTransform(self.sem_sampler.shapes_dict)

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return self.dataset_size

    def __getitem__(self, index) -> CausalDataset:
        """Return a sample from the dataset.

        See Also:
            CausalDataset: For a description of the format of the samples.
        """
        if index >= len(self) or index < -len(self):
            raise IndexError(f"index {index} out of range for dataset of size {len(self)}")
        return self._sample(index)

    def _sample(self, index: int = 0) -> CausalDataset:
        """Sample a new dataset and returns the data, graph and potentially interventions and counterfactuals."""
        if self.num_sems > 0:
            sem = self.sems[index % self.num_sems]
        else:
            sem = self.sem_sampler.sample()

        return sample_dataset(
            sem,
            self.sample_dataset_size,
            self.num_interventions,
            self.num_intervention_samples,
            self.sample_interventions,
            self.sample_counterfactuals,
            self.treatment_variable,
            self.effect_variables,
        )
