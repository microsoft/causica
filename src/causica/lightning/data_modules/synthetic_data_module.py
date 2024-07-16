import inspect
from collections import defaultdict
from typing import Callable

import pytorch_lightning as pl
from torch.utils.data import ConcatDataset, DataLoader

from causica.data_generation.samplers.sem_sampler import SEMSampler
from causica.datasets.samplers import SubsetBatchSampler
from causica.datasets.synthetic_dataset import CausalMetaset


class SyntheticDataModule(pl.LightningDataModule):
    """A datamodule to produce datasets and their underlying causal graphs and interventions.

    This datamodule samples synthetic datasets from a list of SEM samplers and returns batches of CausalDataset objects.
    This means, that the underlying tensors are not yet stacked and will need to be processed by the module using the
    data. Currently, the data module uses the `SubsetBatchSampler` to sample batches from the different sampled SEMs,
    ensuring that each batch is sampled from SEMs with the same number of nodes. This is important, as it allows for
    easy batching of the data, as the tensors are already of the same shape.

    Note:
        This data module does currently not support DDP, because the batch sampler does not simply consume indices
        from a a regular sampler. In future, DDP support will be added by ensuring that each compute node is individually
        seeded to generate different data samples and adapting the batch sampler to ensure that an epoch has the same length
        regardless of the number of compute nodes.

        This data module might generate duplicate data samples, when num_workers > 1. This is because the individual
        workers are not individually seeded.
    """

    def __init__(
        self,
        sem_samplers: list[SEMSampler] | Callable[[], list[SEMSampler]],
        train_batch_size: int,
        test_batch_size: int,
        dataset_size: int,
        num_interventions: int = 0,
        num_intervention_samples: int = 100,
        num_sems: int = 0,
        batches_per_metaset: int = 1,
        sample_interventions: bool = False,
        sample_counterfactuals: bool = False,
        treatment_variable: str | None = None,
        effect_variables: list[str] | None = None,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 16,
    ) -> None:
        """
        Args:
            sem_samplers: Either a list of sem samplers or a function that returns a list of SEM samplers
            train_batch_size: The training batch size to use
            test_batch_size: The testing batch size to use
            dataset_size: The size of dataset to generate
            num_interventions: The number of interventions to generate (0 for no interventions)
            num_sems: The number of SEMs to generate (0 for infinite SEMs)
            batches_per_metaset: The number of batches per epoch per dataset
            sample_interventions: Whether to sample interventions
            sample_counterfactuals: Whether to sample counterfactuals
            treatment_variable: This specify the name of the nodes to be taken as treatment from the generated sems.
                The sem sampler must always generate this treatment.
            effet_variables: This specify the names of the nodes to be taken as effects from the generated sems.
                The sem sampler must always generate this effect.
            num_workers: The number of workers to use for the dataloader
            pin_memory: Whether to pin memory for the dataloader
            persistent_workers: Whether to use persistent workers for the dataloader
            prefetch_factor: The prefetch factor to use for the dataloader
        """
        super().__init__()
        self.dataset_size = dataset_size
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        if callable(sem_samplers) and inspect.isclass(type(sem_samplers)):
            self.sem_samplers_hparams = vars(sem_samplers)
        elif callable(sem_samplers):
            self.sem_samplers_hparams = {k: v.default for k, v in inspect.signature(sem_samplers).parameters.items()}
        else:
            self.sem_samplers_hparams = {}

        self.sem_sampler_list = sem_samplers if isinstance(sem_samplers, list) else sem_samplers()
        self.num_interventions = num_interventions
        self.num_intervention_samples = num_intervention_samples
        self.num_workers = num_workers
        self.num_sems = num_sems
        self.batches_per_metaset = batches_per_metaset
        self.sample_interventions = sample_interventions
        self.sample_counterfactuals = sample_counterfactuals
        self.treatment_variable = treatment_variable
        self.effect_variables = effect_variables

        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

        self.save_hyperparameters(logger=False)
        self.train_dataset = self._get_dataset(self.train_batch_size)
        self.val_dataset = self._get_dataset(self.test_batch_size)

    def prepare_data(self) -> None:
        # Log hyperparameters if trainer is available
        if self.trainer and self.trainer.logger:
            hyperparams = {k: v for k, v in self.hparams.items() if k != "sem_samplers"}
            hyperparams["sem_samplers_hparams"] = self.sem_samplers_hparams

            self.trainer.logger.log_hyperparams(hyperparams)

    def _get_dataset(self, dataset_size: int) -> ConcatDataset:
        """Builds causal datasets given the SEM samplers.

        Args:
            dataset_size: Number of samples of the causal dataset (ie number of datasets generated).

        Returns:
            dataset object
        """
        cur_dataset_size = dataset_size * self.batches_per_metaset
        if self.num_sems > 0 and cur_dataset_size < self.num_sems:
            raise ValueError(
                f"Dataset size must be at least the number of SEMs. Got {cur_dataset_size} < {self.num_sems}"
            )
        datasets_by_nodes = defaultdict(list)
        for sampler in self.sem_sampler_list:
            datasets_by_nodes[sampler.adjacency_dist.num_nodes].append(
                CausalMetaset(
                    sampler,
                    sample_dataset_size=self.dataset_size,
                    dataset_size=cur_dataset_size,
                    num_interventions=self.num_interventions,
                    num_intervention_samples=self.num_intervention_samples,
                    num_sems=self.num_sems,
                    sample_interventions=self.sample_interventions,
                    sample_counterfactuals=self.sample_counterfactuals,
                    treatment_variable=self.treatment_variable,
                    effect_variables=self.effect_variables,
                )
            )

        return ConcatDataset(ConcatDataset(ds) for ds in datasets_by_nodes.values())

    def train_dataloader(self):
        batch_sampler = SubsetBatchSampler([len(d) for d in self.train_dataset.datasets], self.train_batch_size)
        return DataLoader(
            dataset=self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=lambda x: x,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self):
        batch_sampler = SubsetBatchSampler([len(d) for d in self.val_dataset.datasets], self.test_batch_size)
        return DataLoader(
            dataset=self.val_dataset, batch_sampler=batch_sampler, collate_fn=lambda x: x, pin_memory=self.pin_memory
        )

    def test_dataloader(self):
        return self.val_dataloader()
