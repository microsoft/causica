import logging
from typing import Any, Optional, Sequence, Union

import fsspec
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.types import STEP_OUTPUT
from tensordict import TensorDict

from causica.datasets.causica_dataset_format import CounterfactualWithEffects, InterventionWithEffects
from causica.distributions import (
    AdjacencyDistribution,
    ConstrainedAdjacency,
    DistributionModule,
    ENCOAdjacencyDistributionModule,
    ExpertGraphContainer,
    GibbsDAGPrior,
    JointNoiseModule,
    create_noise_modules,
)
from causica.distributions.noise.joint import ContinuousNoiseDist
from causica.fsspec_helpers import get_storage_options_for_path
from causica.functional_relationships import ICGNN
from causica.graph.dag_constraint import calculate_dagness
from causica.graph.evaluation_metrics import adjacency_f1, orientation_f1
from causica.lightning.callbacks import AuglagLRCallback
from causica.lightning.data_modules.deci_data_module import DECIDataModule
from causica.lightning.modules.variable_spec_module import VariableSpecModule
from causica.sem.sem_distribution import SEMDistributionModule
from causica.sem.structural_equation_model import SEM
from causica.training.auglag import AugLagLossCalculator, AugLagLR, AugLagLRConfig
from causica.training.evaluation import (
    eval_ate_rmse,
    eval_intervention_likelihoods,
    eval_ite_rmse,
    list_logsumexp,
    list_mean,
)

logging.basicConfig(level=logging.INFO)

NUM_GRAPH_SAMPLES = 100
NUM_ATE_ITE_SEMS = 10


class DECIModule(VariableSpecModule):
    """A PyTorch Lightning Module for running a DECI model."""

    def __init__(
        self,
        noise_dist: ContinuousNoiseDist = ContinuousNoiseDist.SPLINE,
        embedding_size: int = 32,
        out_dim_g: int = 32,
        norm_layer: bool = True,
        res_connection: bool = True,
        init_alpha: float = 0.0,
        init_rho: float = 1.0,
        prior_sparsity_lambda: float = 0.05,
        gumbel_temp: float = 0.25,
        auglag_config: Optional[AugLagLRConfig] = None,
        expert_graph_container: Optional[ExpertGraphContainer] = None,
        constraint_matrix_path: Optional[str] = None,
    ):
        super().__init__()
        self.auglag_config = auglag_config if auglag_config is not None else AugLagLRConfig()
        self.constraint_matrix_path = constraint_matrix_path
        self.constraint_matrix: Optional[torch.Tensor] = None

        self.embedding_size = embedding_size
        self.expert_graph_container: Optional[ExpertGraphContainer] = expert_graph_container

        self.gumbel_temp = gumbel_temp
        self.init_alpha = init_alpha
        self.init_rho = init_rho
        self.is_setup = False
        self.noise_dist = noise_dist
        self.norm_layer = norm_layer
        self.out_dim_g = out_dim_g
        self.prior_sparsity_lambda = prior_sparsity_lambda
        self.res_connection = res_connection

        # Inferred once the datamodule is available using `self.infer_missing_state_from_dataset()`
        self.dataset_name = None
        self.num_samples = None
        self.variable_group_shapes = None
        self.variable_types = None
        self.variable_names = None

    def prepare_data(self) -> None:
        """Set the constraint matrix (if necessary)."""
        if self.constraint_matrix_path:
            storage_options = get_storage_options_for_path(self.constraint_matrix_path)

            with fsspec.open(self.constraint_matrix_path, **storage_options) as f:
                if self.constraint_matrix_path.endswith("npy"):
                    constraint_matrix = np.load(f)
                else:
                    raise ValueError(f"Unsupported constraint matrix format: {self.constraint_matrix_path}")
            self.constraint_matrix = torch.tensor(constraint_matrix)
        return super().prepare_data()

    def infer_missing_state_from_dataset(self):
        dataset_defined_members = [
            self.dataset_name,
            self.num_samples,
            self.variable_group_shapes,
            self.variable_types,
            self.variable_names,
        ]
        if any(member is None for member in dataset_defined_members):
            datamodule = getattr(self.trainer, "datamodule", None)
            if not isinstance(datamodule, DECIDataModule):
                raise TypeError(
                    f"Incompatible data module {datamodule}, requires a DECIDataModule but is "
                    f"{type(datamodule).mro()}"
                )
            if self.dataset_name is None:
                self.dataset_name = datamodule.dataset_name
            if self.num_samples is None:
                self.num_samples = len(datamodule.dataset_train)
            if self.variable_group_shapes is None:
                self.variable_group_shapes = datamodule.variable_shapes
            if self.variable_types is None:
                self.variable_types = datamodule.variable_types
            if self.variable_names is None:
                self.variable_names = datamodule.column_names

    def setup(self, stage: Optional[str] = None):
        if self.is_setup:
            return  # Already setup
        if stage not in {TrainerFn.TESTING, TrainerFn.FITTING}:
            raise ValueError(f"Model can only be setup during the {TrainerFn.FITTING} and {TrainerFn.TESTING} stages.")

        self.infer_missing_state_from_dataset()
        assert self.dataset_name is not None
        assert self.num_samples is not None
        assert self.variable_group_shapes is not None
        assert self.variable_types is not None
        assert self.variable_names is not None

        self.node_names = list(self.variable_group_shapes)
        num_nodes = len(self.variable_group_shapes)

        adjacency_dist: DistributionModule[AdjacencyDistribution] = ENCOAdjacencyDistributionModule(num_nodes)
        if self.constraint_matrix is not None:
            adjacency_dist = ConstrainedAdjacency(adjacency_dist, self.constraint_matrix)

        icgnn = ICGNN(
            shapes=self.variable_group_shapes,
            embedding_size=self.embedding_size,
            out_dim_g=self.out_dim_g,
            norm_layer=None if self.norm_layer is False else torch.nn.LayerNorm,
            res_connection=self.res_connection,
        )
        noise_submodules = create_noise_modules(self.variable_group_shapes, self.variable_types, self.noise_dist)
        noise_module = JointNoiseModule(noise_submodules)
        self.sem_module: SEMDistributionModule = SEMDistributionModule(adjacency_dist, icgnn, noise_module)
        self.auglag_loss: AugLagLossCalculator = AugLagLossCalculator(
            init_alpha=self.init_alpha, init_rho=self.init_rho
        )
        self.prior: GibbsDAGPrior = GibbsDAGPrior(
            num_nodes=num_nodes,
            sparsity_lambda=self.prior_sparsity_lambda,
            expert_graph_container=self.expert_graph_container,
        )
        self.is_setup = True

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        _ = kwargs
        batch, *_ = args
        batch = batch.apply(lambda t: t.to(torch.float32, non_blocking=True))

        sem_distribution = self.sem_module()
        sem, *_ = sem_distribution.relaxed_sample(torch.Size([]), temperature=self.gumbel_temp)  # soft sample

        batch_log_prob = sem.log_prob(batch).mean()
        sem_distribution_entropy = sem_distribution.entropy()
        prior_term = self.prior.log_prob(sem.graph)
        objective = (-sem_distribution_entropy - prior_term) / self.num_samples - batch_log_prob
        constraint = calculate_dagness(sem.graph)
        step_output = {
            "alpha": self.auglag_loss.alpha,
            "rho": self.auglag_loss.rho,
            "loss": self.auglag_loss(objective, constraint / self.num_samples),
            "batch_log_prob": batch_log_prob,
            "constraint": constraint,
            "num_edges": (sem.graph > 0.0).count_nonzero(),
            "vardist_entropy": sem_distribution_entropy,
            "prior_term": prior_term,
        }
        self.log_dict(step_output, on_epoch=True)  # Only log this on epoch end
        return step_output

    def configure_optimizers(self):
        """Set the learning rates for different sets of parameters."""
        modules = {
            "icgnn": self.sem_module.functional_relationships,
            "vardist": self.sem_module.adjacency_module,
            "noise_dist": self.sem_module.noise_module,
        }
        parameter_list = [
            {"params": module.parameters(), "lr": self.auglag_config.lr_init_dict[name], "name": name}
            for name, module in modules.items()
        ]

        # Check that all modules are added to the parameter list
        check_modules = set(modules.values())
        for module in self.parameters(recurse=False):
            assert module in check_modules, f"Module {module} not in module list"

        return torch.optim.Adam(parameter_list)

    def configure_callbacks(self) -> Union[Sequence[pl.Callback], pl.Callback]:
        """Create a callback for the auglag callback."""
        lr_scheduler = AugLagLR(config=self.auglag_config)
        return [AuglagLRCallback(lr_scheduler, log_auglag=True)]

    def test_step_observational(self, batch: TensorDict, *args, **kwargs):
        """Evaluate the log prob of the model on the test set using multiple graph samples."""
        batch = batch.apply(lambda t: t.to(torch.float32, non_blocking=True))
        sems = self.sem_module().sample(torch.Size([NUM_GRAPH_SAMPLES]))
        dataset_size = self.trainer.datamodule.dataset_test.batch_size  # type: ignore
        assert len(dataset_size) == 1, "Only one batch size is supported"

        # Estimate log prob for each sample using graph samples and report the mean over graphs
        log_prob_test = list_logsumexp([sem.log_prob(batch) for sem in sems]) - np.log(NUM_GRAPH_SAMPLES)
        self.log(
            "eval/test_LL",
            torch.sum(log_prob_test, dim=-1).item() / dataset_size[0],
            reduce_fx=sum,
            add_dataloader_idx=False,
        )

    def test_step_graph(self, true_adj_matrix: torch.Tensor, *args, **kwargs):
        """Evaluate the graph reconstruction performance of the model against the true graph."""
        _, _ = args, kwargs
        sem_samples = self.sem_module().sample(torch.Size([NUM_GRAPH_SAMPLES]))
        graph_samples = [sem.graph for sem in sem_samples]

        adj_f1 = list_mean([adjacency_f1(true_adj_matrix, graph) for graph in graph_samples]).item()
        orient_f1 = list_mean([orientation_f1(true_adj_matrix, graph) for graph in graph_samples]).item()
        self.log("eval/adjacency.f1", adj_f1, add_dataloader_idx=False)
        self.log("eval/orientation.f1", orient_f1, add_dataloader_idx=False)

    def test_step_interventions(self, interventions: InterventionWithEffects, *args, **kwargs):
        """Evaluate the ATE and Interventional log prob performance of the model"""
        _, _ = args, kwargs
        sems_list: list[SEM] = list(self.sem_module().sample(torch.Size([NUM_ATE_ITE_SEMS])))
        interventional_log_prob = eval_intervention_likelihoods(sems_list, interventions)
        self.log("eval/Interventional_LL", torch.mean(interventional_log_prob).item(), add_dataloader_idx=False)

        mean_ate_rmse = eval_ate_rmse(sems_list, interventions)
        self.log("eval/ATE_RMSE", list_mean(list(mean_ate_rmse.values())).item(), add_dataloader_idx=False)

    def test_step_counterfactuals(self, counterfactuals: CounterfactualWithEffects, *args, **kwargs):
        """Evaluate the ITE performance of the model"""
        _, _ = args, kwargs
        sems_list = list(self.sem_module().sample(torch.Size([NUM_ATE_ITE_SEMS])))
        ite_rmse = eval_ite_rmse(sems_list, counterfactuals)
        self.log("eval/ITE_RMSE", list_mean(list(ite_rmse.values())).item(), add_dataloader_idx=False)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["sem_module"] = self.sem_module
