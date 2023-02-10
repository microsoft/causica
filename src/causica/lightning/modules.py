import logging
from typing import Any, Dict, Optional, Sequence, Union

import fsspec
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.nn import Parameter, ParameterDict

from causica.distributions import (
    AdjacencyDistribution,
    ENCOAdjacencyDistribution,
    ExpertGraphContainer,
    GibbsDAGPrior,
    ParametrizedDistribution,
    constrained_adjacency,
)
from causica.functional_relationships import ICGNN
from causica.graph.dag_constraint import calculate_dagness
from causica.lightning.callbacks import AuglagLRCallback
from causica.lightning.data_modules import DECIDataModule
from causica.sem.distribution_parameters_sem import DistributionParametersSEM
from causica.training.auglag import AugLagLossCalculator, AugLagLR, AugLagLRConfig
from causica.training.trainable_container import NoiseDist, TrainableContainer, create_noise_dists
from causica.training.training_callbacks import AverageMetricTracker

logging.basicConfig(level=logging.INFO)

DATASET_PATH = "https://azuastoragepublic.blob.core.windows.net/datasets"


class DECIModule(pl.LightningModule):
    """A PyTorch Lightning Module for running a DECI model."""

    def __init__(
        self,
        noise_dist: NoiseDist = NoiseDist.SPLINE,
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
        self.expert_graph_container = expert_graph_container
        self.gumbel_temp = gumbel_temp
        self.init_alpha = init_alpha
        self.init_rho = init_rho
        self.is_setup = False
        self.noise_dist = noise_dist
        self.norm_layer = norm_layer
        self.out_dim_g = out_dim_g
        self.prior_sparsity_lambda = prior_sparsity_lambda
        self.res_connection = res_connection

    def prepare_data(self) -> None:
        if self.constraint_matrix_path:
            with fsspec.open(self.constraint_matrix_path) as f:
                constraint_matrix = np.loadtxt(f, dtype=int, delimiter=",")
            self.constraint_matrix = torch.tensor(constraint_matrix)
        return super().prepare_data()

    def setup(self, stage: Optional[str] = None):
        if self.is_setup:
            return  # Already setup
        elif stage != TrainerFn.FITTING:
            raise ValueError(f"Model can only be setup during the {TrainerFn.FITTING} stage.")

        datamodule = self.trainer.datamodule  # type: ignore
        if not isinstance(datamodule, DECIDataModule):
            raise TypeError(
                f"Incompatible data module {datamodule}, requires a DECIDataModule but is " f"{type(datamodule).mro()}"
            )
        dataset_train = datamodule.dataset_train
        variable_group_shapes = datamodule.get_variable_shapes()

        self.node_names = list(variable_group_shapes)
        num_nodes = len(variable_group_shapes)
        vardist_logits_orient = Parameter(torch.zeros(int(num_nodes * (num_nodes - 1) / 2)), requires_grad=True)
        vardist_logits_exist = Parameter(torch.zeros(num_nodes, num_nodes), requires_grad=True)

        adjacency_distribution = ENCOAdjacencyDistribution
        if self.constraint_matrix is not None:
            adjacency_distribution = constrained_adjacency(
                adjacency_distribution, graph_constraint_matrix=self.constraint_matrix
            )

        param_vardist = ParametrizedDistribution(
            adjacency_distribution,
            ParameterDict({"logits_orient": vardist_logits_orient, "logits_exist": vardist_logits_exist}),
        )

        # Define ICGNN
        icgnn = ICGNN(
            variables=variable_group_shapes,
            embedding_size=self.embedding_size,
            out_dim_g=self.out_dim_g,
            norm_layer=None if self.norm_layer is False else torch.nn.LayerNorm,
            res_connection=self.res_connection,
        )

        self.num_samples = len(dataset_train)
        self.noise_dist_funcs, param_noise_dist = create_noise_dists(
            variable_group_shapes, datamodule.get_variable_types(), self.noise_dist
        )

        self.container = TrainableContainer(
            dataset_name=datamodule.dataset_name,
            icgnn=icgnn,
            vardist=param_vardist,
            noise_dist_params=param_noise_dist,
            noise_dist_type=self.noise_dist,
        )
        self.auglag_loss = AugLagLossCalculator(init_alpha=self.init_alpha, init_rho=self.init_rho)
        self.prior = GibbsDAGPrior(
            num_nodes=num_nodes,
            sparsity_lambda=self.prior_sparsity_lambda,
            expert_graph_container=self.expert_graph_container,
        )
        # TODO: Set a more reasonable averaging period
        self.average_batch_log_prob_tracker = AverageMetricTracker(averaging_period=10)
        self.is_setup = True

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        # sample graph
        _ = kwargs
        batch, *_ = args
        batch = batch.apply(lambda t: t.to(torch.float32))

        vardist = self.container.vardist()
        assert isinstance(vardist, AdjacencyDistribution)
        cur_graph = vardist.relaxed_sample(torch.Size([]), temperature=self.gumbel_temp)  # soft sample

        # Create SEM
        sem = DistributionParametersSEM(
            graph=cur_graph,
            node_names=self.node_names,
            noise_dist=self.noise_dist_funcs,
            func=self.container.icgnn,
        )

        batch_log_prob = sem.log_prob(batch).mean()
        objective = (-vardist.entropy() - self.prior.log_prob(cur_graph)) / self.num_samples - batch_log_prob
        constraint = calculate_dagness(cur_graph)
        self.average_batch_log_prob_tracker.step(batch_log_prob.item())
        step_output = {
            "loss": self.auglag_loss(objective, constraint / self.num_samples),
            "batch_log_prob": batch_log_prob,
            "constraint": constraint,
            "num_edges": (cur_graph > 0.0).count_nonzero(),
            "average_batch_log_prob": self.average_batch_log_prob_tracker.average,
        }
        self.log_dict({"alpha": self.auglag_loss.alpha, "rho": self.auglag_loss.rho, **step_output}, prog_bar=True)
        return step_output

    def configure_optimizers(self):
        # Define Optimizer
        parameter_list = [
            {
                "params": module if isinstance(module, Parameter) else module.parameters(),
                "lr": self.auglag_config.lr_init_dict[name],
                "name": name,
            }
            for name, module in self.container.named_children()
        ]
        return torch.optim.Adam(parameter_list)

    def configure_callbacks(self) -> Union[Sequence[pl.Callback], pl.Callback]:
        # TODO: Integrate with Lightning lr scheduler and set up in `configure_optimizers` instead
        auglag_callback = AuglagLRCallback(AugLagLR(config=self.auglag_config))
        return [auglag_callback]

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["container"] = self.container
