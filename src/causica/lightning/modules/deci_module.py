import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, Optional, Sequence, Union

import fsspec
import numpy as np
import pytorch_lightning as pl
import torch
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
from causica.functional_relationships import DECIEmbedFunctionalRelationships
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
        num_layers_g: int = 2,
        num_layers_zeta: int = 2,
        init_alpha: float = 0.0,
        init_rho: float = 1.0,
        prior_sparsity_lambda: float = 0.05,
        gumbel_temp: float = 0.25,
        auglag_config: Optional[AugLagLRConfig] = None,
        expert_graph_container: Optional[ExpertGraphContainer] = None,
        constraint_matrix_path: Optional[str] = None,
        disable_auglag_epochs: Optional[int] = None,
        test_metric_prefix: Optional[str] = "eval",
    ):
        """DECI Module
        Args:
            noise_dist: The type of the noise distribution to use. The noise type will be used for each variable in the
                SEM.
            embedding_size: The size of the embedding used in the functional relationships. This specifies the
                flexibility of the trainable embeddings for each node.
            out_dim_g: The size of the output of the functional relationships encoder g.
            num_layers_g: The number of layers in the functional relationships encoder g.
            num_layers_zeta: The number of layers in the decoder zeta.
            init_alpha: The initial value of the augmented lagrangian parameter alpha. The augmented lagrangian is
                \rho h(G)^2 + \alpha h(G), where h(G) is the DAG constraint.
            init_rho: The initial value of the augmented lagrangian parameter rho.
            prior_sparsity_lambda: The sparsity coefficient in the graph prior. Larger value prefers sparser graphs.
            gumbel_temp: The temperature used in the gumbel softmax relaxation.
            auglag_config: The configuration for the augmented lagrangian scheduler.
            expert_graph_container: The expert graph container to use. It contains the prior knowledge graph that
                the inferred graph should be close to.
            constraint_matrix_path: The path to the constraint matrix. The constraint matrix specifies the edge
                directions that are prohibited or required.
            disable_auglag_epochs: The number of epochs to disable the augmented lagrangian for.
            test_metric_prefix: Prefix used for logging. The logged metric name will be
                {test_metric_prefix}/{metric_name}.
        """
        super().__init__()
        self.auglag_config = auglag_config if auglag_config is not None else AugLagLRConfig()
        self.disable_auglag_epochs = disable_auglag_epochs
        self.lr_scheduler = AugLagLR(config=self.auglag_config)
        self.constraint_matrix_path = constraint_matrix_path
        self.constraint_matrix: Optional[torch.Tensor] = None

        self.embedding_size = embedding_size
        self.out_dim_g = out_dim_g
        self.num_layers_g = num_layers_g
        self.num_layers_zeta = num_layers_zeta

        self.expert_graph_container: Optional[ExpertGraphContainer] = expert_graph_container

        self.gumbel_temp = gumbel_temp
        self.is_setup = False
        self.noise_dist = noise_dist
        self.prior_sparsity_lambda = prior_sparsity_lambda
        self.test_metric_prefix = test_metric_prefix

        self.auglag_loss: AugLagLossCalculator = AugLagLossCalculator(init_alpha=init_alpha, init_rho=init_rho)

        # Inferred once the datamodule is available using `self.infer_missing_state_from_dataset()`
        self.num_samples = None
        self.variable_group_shapes = None
        self.variable_types = None

        self.save_hyperparameters(logger=False)

    def on_train_start(self) -> None:
        super().on_train_start()

        # Log a version of hparams that are easier to handle for loggers, by converting dataclasses to dicts
        if self.logger is not None:
            hyperparams = {
                k: asdict(v) if is_dataclass(v) and not isinstance(v, type) else v
                for k, v in self.hparams.items()
                if not isinstance(v, torch.nn.Module)
            }
            self.logger.log_hyperparams(hyperparams)

    def prepare_data(self) -> None:
        """Set the constraint matrix (if necessary)."""
        if self.constraint_matrix_path:
            with fsspec.open(self.constraint_matrix_path) as f:
                if self.constraint_matrix_path.endswith("npy"):
                    constraint_matrix = np.load(f)
                else:
                    raise ValueError(f"Unsupported constraint matrix format: {self.constraint_matrix_path}")
            self.constraint_matrix = torch.tensor(constraint_matrix)
        return super().prepare_data()

    def infer_missing_state_from_dataset(self):
        dataset_defined_members = [
            self.num_samples,
            self.variable_group_shapes,
            self.variable_types,
        ]
        if any(member is None for member in dataset_defined_members):
            datamodule = getattr(self.trainer, "datamodule", None)
            if not isinstance(datamodule, DECIDataModule):
                raise TypeError(
                    f"Incompatible data module {datamodule}, requires a DECIDataModule but is "
                    f"{type(datamodule).mro()}"
                )
            if self.num_samples is None:
                self.num_samples = len(datamodule.dataset_train)
            if self.variable_group_shapes is None:
                self.variable_group_shapes = datamodule.variable_shapes
            if self.variable_types is None:
                self.variable_types = datamodule.variable_types

    def setup(self, stage: Optional[str] = None):

        _ = stage
        if self.is_setup:
            return  # Already setup

        self.infer_missing_state_from_dataset()
        assert self.num_samples is not None
        assert self.variable_group_shapes is not None
        assert self.variable_types is not None

        num_nodes = len(self.variable_group_shapes)

        adjacency_dist: DistributionModule[AdjacencyDistribution] = ENCOAdjacencyDistributionModule(num_nodes)
        if self.constraint_matrix is not None:
            adjacency_dist = ConstrainedAdjacency(adjacency_dist, self.constraint_matrix)

        functional_relationships = DECIEmbedFunctionalRelationships(
            shapes=self.variable_group_shapes,
            embedding_size=self.embedding_size,
            out_dim_g=self.out_dim_g,
            num_layers_g=self.num_layers_g,
            num_layers_zeta=self.num_layers_zeta,
        )
        noise_submodules = create_noise_modules(self.variable_group_shapes, self.variable_types, self.noise_dist)
        noise_module = JointNoiseModule(noise_submodules)
        self.sem_module: SEMDistributionModule = SEMDistributionModule(
            adjacency_dist, functional_relationships, noise_module
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
            "functional_relationships": self.sem_module.functional_relationships,
            "vardist": self.sem_module.adjacency_module,
            "noise_dist": self.sem_module.noise_module,
        }
        parameter_list = [
            {
                "params": module.parameters(),
                "lr": self.auglag_config.lr_init_dict[name],
                "name": name,
            }
            for name, module in modules.items()
        ]

        # Check that all modules are added to the parameter list
        check_modules = set(modules.values())
        for module in self.parameters(recurse=False):
            assert module in check_modules, f"Module {module} not in module list"

        return torch.optim.Adam(parameter_list)

    def configure_callbacks(self) -> Union[Sequence[pl.Callback], pl.Callback]:
        """Create a callback for the auglag callback."""
        disabled_epochs = set(range(self.disable_auglag_epochs)) if self.disable_auglag_epochs else None
        return [AuglagLRCallback(self.lr_scheduler, log_auglag=True, disabled_epochs=disabled_epochs)]

    def test_step_observational(self, batch: TensorDict, *args, **kwargs):
        """Evaluate the log prob of the model on the test set using multiple graph samples."""
        _, _ = args, kwargs
        batch = batch.apply(lambda t: t.to(torch.float32, non_blocking=True))
        sems = self.sem_module().sample(torch.Size([NUM_GRAPH_SAMPLES]))
        dataset_size = self.trainer.datamodule.dataset_test.batch_size  # type: ignore
        assert len(dataset_size) == 1, "Only one batch size is supported"

        # Estimate log prob for each sample using graph samples and report the mean over graphs
        log_prob_test = list_logsumexp([sem.log_prob(batch) for sem in sems]) - np.log(NUM_GRAPH_SAMPLES)
        self.log(
            f"{self.test_metric_prefix}/test_LL",
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
        self.log(f"{self.test_metric_prefix}/adjacency.f1", adj_f1, add_dataloader_idx=False)
        self.log(f"{self.test_metric_prefix}/orientation.f1", orient_f1, add_dataloader_idx=False)

    def test_step_interventions(self, interventions: InterventionWithEffects, *args, **kwargs):
        """Evaluate the ATE and Interventional log prob performance of the model"""
        _, _ = args, kwargs
        sems_list: list[SEM] = list(self.sem_module().sample(torch.Size([NUM_ATE_ITE_SEMS])))
        interventional_log_prob = eval_intervention_likelihoods(sems_list, interventions)
        self.log(
            f"{self.test_metric_prefix}/Interventional_LL",
            torch.mean(interventional_log_prob).item(),
            add_dataloader_idx=False,
        )

        mean_ate_rmse = eval_ate_rmse(sems_list, interventions)
        self.log(
            f"{self.test_metric_prefix}/ATE_RMSE",
            list_mean(list(mean_ate_rmse.values())).item(),
            add_dataloader_idx=False,
        )

    def test_step_counterfactuals(self, counterfactuals: CounterfactualWithEffects, *args, **kwargs):
        """Evaluate the ITE performance of the model"""
        _, _ = args, kwargs
        sems_list = list(self.sem_module().sample(torch.Size([NUM_ATE_ITE_SEMS])))
        ite_rmse = eval_ite_rmse(sems_list, counterfactuals)
        self.log(
            f"{self.test_metric_prefix}/ITE_RMSE",
            list_mean(list(ite_rmse.values())).item(),
            add_dataloader_idx=False,
        )

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        # initialise all the parameters, we can
        super().load_state_dict(state_dict, strict=False)
        # setup the model
        self.setup()
        # load the state dict again to fill in the parameters
        return super().load_state_dict(state_dict, strict=strict)

    def set_extra_state(self, state: Any):
        self.variable_group_shapes = state["shapes"]
        self.variable_types = state["types"]
        self.constraint_matrix = state["constraint_matrix"]
        self.num_samples = state["num_samples"]

    def get_extra_state(self) -> Any:
        return {
            "shapes": self.variable_group_shapes,
            "types": self.variable_types,
            "constraint_matrix": self.constraint_matrix,
            "num_samples": self.num_samples,
        }
