from typing import Optional, Union

import numpy as np
import torch
import torchsde
from pytorch_lightning import LightningModule
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.types import STEP_OUTPUT
from scotch.latent_learning.functional_relationship import SCOTCHFunctionalRelationships
from scotch.latent_learning.graph_distribution import BernoulliDigraphDistributionModule
from scotch.latent_learning.scotch_nns import (
    DECIEmbedNNCoefficient,
    NeuralContextualDriftCoefficient,
    NeuralDiffusionCoefficient,
    NeuralTrajectoryGraphEncoder,
)
from scotch.scotch_utils.graph_metrics import (
    confusion_matrix_batched,
    f1_score,
    false_discovery_rate,
    true_positive_rate,
)
from scotch.sdes.scotch_sdes import AugmentedSCOTCHSDE, SCOTCHPriorSDE, swap_t_and_batch_dimensions
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn

from causica.distributions import GibbsDAGPrior
from causica.distributions.transforms import TensorToTensorDictTransform, shapes_to_slices


class LinearScheduler:
    def __init__(self, iters, maxval=1.0):
        self._iters = max(1, iters)
        self._val = maxval / self._iters
        self._maxval = maxval

    def step(self):
        self._val = min(self._maxval, self._val + self._maxval / self._iters)

    @property
    def val(self):
        return self._val


class SCOTCHModule(LightningModule):
    """PyTorch Lightning module implementing the graph-based continuous time series model."""

    def __init__(
        self,
        embedding_size: int = 32,
        out_dim_g: int = 32,
        prior_sparsity_lambda: float = 0.05,
        gumbel_temp: float = 0.25,
        context_size: int = 64,
        hidden_size: int = 128,
        kl_anneal_iters: int = 1000,
        learning_rates: Optional[dict] = None,
        use_adjoint: bool = False,
        sde_method: str = "euler",
        dt: float = 1e-2,
        noise_scale: Union[float, Tensor] = 0.01,
        record_graph_logits: bool = True,
        lr_warmup_iters: int = 1,
        ignore_self_connections: bool = False,
        layer_norm: bool = False,
        res_connections: bool = False,
        deci_diffusion: bool = False,
        compute_auroc=True,
        add_diffusion_self_connections: bool = False,
        sigmoid_output: bool = False,
    ):
        """Initialize the SCOTCH module. Some of the arguments are similar to arguments of the DeciModule constructor,
        while others are specific to SCOTCH.

        Args:
            embedding_size, out_dim_g: Hyperparameters of the prior diffusion coefficient
                network, implemented by a DECIEmbedNN.
            prior_sparsity_lambda: Sparsity parameter for the prior directed graph distribution.
            gumbel_temp: Temperature for the Gumbel-Softmax distribution used to sample directed graphs during training.
            context_size: Dimension of the context vector. Given an observed trajectory, an encoder is used to predict
                this vector, which is then used as input to the NN modelling the posterior SDE (given trajectory).
            hidden_size: Size of hidden NN layers in the encoder and posterior SDE NN.
            kl_anneal_iters: Number of iterations to anneal the KL term for the SDE ELBO.
            learning_rates: Learning rates for different parts of the components of the model. Keys are ["graph",
                "qz0_mean_net", "qz0_logstd_net", "pz0_mean", "pz0_logstd", "prior_drift_fn", "diffusion_fn",
                "posterior_drift_fn", "trajectory_encoder"].
            use_adjoint: Whether to use the adjoint method for computing gradients through the (augmented) SDE during
                training.
            sde_method: Method to use for solving SDE.
            dt: Time step to use for solving SDE.
            noise_scale: Noise scale of the (Laplace) noise distribution for the observed trajectory given the latent
                trajectory.
            graph_constrained_posterior: Whether to use the constrain the posterior distribution to respect graphs (in
                the same manner as the prior); if True, we use a DECIEmbedNN for the posterior
            record_graph_logits: Whether to record the logits of the graph distribution in the metrics.
            lr_warmup_iters: Number of iterations to linearly increase the learning rate from 0 to the specified value.
            ignore_self_connections: If True, the diagonal logits of the predicted adjacency matrix are set to -inf
                during computation of graph metrics (auroc, f1, tpr, fdr) only.
            layer_norm: Whether to activate layer_norm for DECI neural networks
            res_connections: Whether to add residual connections for DECI neural networks
            deci_diffusion: whether to use DECI for diffusion coefficient (True), or independent nns for each dimension (False)
            compute_auroc: whether to compute auroc
            add_diffusion_self_connections: whether to add self-connections to the diffusion coefficient graph. only used if
                deci_diffusion is True.
            sigmoid_output: whether to apply sigmoid to the output of the diffusion coefficient nn. Only used if deci_diffusion
        """
        super().__init__()

        self.embedding_size = embedding_size

        self.gumbel_temp = gumbel_temp
        self.is_setup = False
        self.out_dim_g = out_dim_g
        self.prior_sparsity_lambda = prior_sparsity_lambda

        self.context_size = context_size
        self.hidden_size = hidden_size
        self.kl_scheduler = LinearScheduler(kl_anneal_iters)
        if learning_rates is None:
            learning_rates = {"lambda": 3e-3}
        self.learning_rates = learning_rates
        self.use_adjoint = use_adjoint
        self.sde_method = sde_method
        self.dt = dt
        self.noise_scale = noise_scale

        self.record_graph_logits = record_graph_logits
        self.lr_warmup_iters = lr_warmup_iters
        self.ignore_self_connections = ignore_self_connections

        self.layer_norm = layer_norm
        self.res_connections = res_connections
        self.deci_diffusion = deci_diffusion
        self.add_diffusion_self_connections = add_diffusion_self_connections

        self.compute_auroc = compute_auroc
        self.sigmoid_output = sigmoid_output

    def setup(self, stage: Optional[str] = None):
        """Set up all components of the SCOTCH SDE model."""
        if self.is_setup:
            return  # Already setup
        if stage not in {TrainerFn.TESTING, TrainerFn.FITTING}:
            raise ValueError(f"Model can only be setup during the {TrainerFn.FITTING} and {TrainerFn.TESTING} stages.")

        # Replaces adjacency distribution in DECIModule, necessary to encode directed graph (with self-loop) distrn.

        variable_group_shapes = self.trainer.datamodule.variable_shapes  # type: ignore
        self.num_nodes = len(variable_group_shapes)
        self.adjacency_dist_module = BernoulliDigraphDistributionModule(self.num_nodes)

        self.graph_prior: GibbsDAGPrior = GibbsDAGPrior(
            num_nodes=self.num_nodes, sparsity_lambda=self.prior_sparsity_lambda
        )

        self.observed_size, _ = shapes_to_slices(variable_group_shapes)
        self.tensor_to_td = TensorToTensorDictTransform(variable_group_shapes)

        self.latent_size = self.observed_size  # in the SCOTCH model, we use the same dimension for observed/latents

        if self.layer_norm:
            deci_drift = SCOTCHFunctionalRelationships(
                shapes=variable_group_shapes,
                embedding_size=self.embedding_size,
                out_dim_g=self.out_dim_g,
                norm_layer=nn.LayerNorm,
                res_connection=self.res_connections,
            )
            deci_diffusion = SCOTCHFunctionalRelationships(
                shapes=variable_group_shapes,
                embedding_size=self.embedding_size,
                out_dim_g=self.out_dim_g,
                norm_layer=nn.LayerNorm,
                res_connection=self.res_connections,
                sigmoid_output=self.sigmoid_output,
            )
        else:
            deci_drift = SCOTCHFunctionalRelationships(
                shapes=variable_group_shapes,
                embedding_size=self.embedding_size,
                out_dim_g=self.out_dim_g,
                res_connection=self.res_connections,
            )
            deci_diffusion = SCOTCHFunctionalRelationships(
                shapes=variable_group_shapes,
                embedding_size=self.embedding_size,
                out_dim_g=self.out_dim_g,
                res_connection=self.res_connections,
                sigmoid_output=self.sigmoid_output,
            )
        # SDE components
        self.qz0_mean_net = nn.Linear(self.context_size, self.latent_size)
        self.qz0_logstd_net = nn.Linear(self.context_size, self.latent_size)
        self.pz0_mean = nn.Parameter(torch.zeros(1, self.latent_size))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, self.latent_size))

        self.prior_drift_fn = DECIEmbedNNCoefficient(deci_drift.nn)
        if self.deci_diffusion:
            self.diffusion_fn = DECIEmbedNNCoefficient(
                deci_diffusion.nn, add_self_connections=self.add_diffusion_self_connections
            )
        else:
            self.diffusion_fn = NeuralDiffusionCoefficient(latent_size=self.latent_size, hidden_size=self.hidden_size)
        self.posterior_drift_fn = NeuralContextualDriftCoefficient(
            latent_size=self.latent_size, hidden_size=self.hidden_size, context_size=self.context_size
        )
        self.trajectory_encoder = NeuralTrajectoryGraphEncoder(
            observed_size=self.observed_size, hidden_size=self.hidden_size, context_size=self.context_size
        )

        self.ts = self.trainer.datamodule.ts  # type: ignore

        self.is_setup = True

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        """Compute the loss function (ELBO) for SCOTCH for the batch; involves integrating an augmented SDE."""
        _ = kwargs
        batch, *_ = args
        # TensorDict mapping variable name to Tensor of shape (batch_size, variable_group_size, num_time_points)
        batch = batch.apply(lambda t: t.to(torch.float32, non_blocking=True))
        batch_unrolled = self.tensor_to_td.inv(batch)

        # Step 1: Sample a graph from variational posterior over graphs
        adjacency_distribution = self.adjacency_dist_module()
        sampled_graphs = adjacency_distribution.relaxed_sample(
            torch.Size([batch_unrolled.shape[0]]), temperature=self.gumbel_temp
        )  # hard gumbel-softmax samples, shape (batch_size, num_nodes, num_nodes)

        # Step 2: For each observed trajectory, compute context for the variational posterior (over latent trajectories)
        # Context vectors are Tensors of shape (batch_size, num_time_points, context_size)
        context_vectors = self.trajectory_encoder(
            torch.flip(batch_unrolled, dims=(1,)), sampled_graphs
        )  # feed into encoder with reversed time
        context_vectors = torch.flip(context_vectors, dims=(1,))  # unflip time dimension
        ts_context_vectors = (self.ts, context_vectors)

        # Step 3: Sample initial latent state
        qz0_mean, qz0_logstd = self.qz0_mean_net(context_vectors[:, 0, :]), self.qz0_logstd_net(
            context_vectors[:, 0, :]
        )
        qz0 = torch.distributions.Normal(loc=qz0_mean, scale=qz0_logstd.exp())
        z0 = qz0.rsample()  # shape (batch_size, latent_size)

        # Step 4: Compute loss components

        # Compute KL (t = 0)

        pz0 = torch.distributions.Normal(loc=self.pz0_mean, scale=self.pz0_logstd.exp())
        logqp0 = torch.distributions.kl_divergence(qz0, pz0).sum(dim=1)  # KL (t = 0)

        # Compute KL (path) and log-likelihood of observed trajectories
        aug_z0 = nn.functional.pad(z0, (0, 1), value=0.0)

        aug_sde = AugmentedSCOTCHSDE(
            posterior_drift_net=self.posterior_drift_fn,
            diffusion_net=self.diffusion_fn,
            prior_drift_net=self.prior_drift_fn,
            ts_context_vectors=ts_context_vectors,
            graphs=sampled_graphs,
        )
        # aug_zs_t_first has shape (num_time_points, batch_size, latent_size + 1)
        if self.use_adjoint:
            # Adjoint method requires parameters to be explicitly passed in.
            aug_zs_t_first = torchsde.sdeint_adjoint(
                aug_sde,
                aug_z0,
                self.ts,
                adjoint_params=self.parameters(),
                dt=self.dt,
                method=self.sde_method,
                names={"drift": "f", "diffusion": "g"},
            )
        else:
            aug_zs_t_first = torchsde.sdeint(
                aug_sde, aug_z0, self.ts, dt=self.dt, method=self.sde_method, names={"drift": "f", "diffusion": "g"}
            )
        aug_zs = swap_t_and_batch_dimensions(aug_zs_t_first)
        zs, logqp_path = aug_zs[:, :, :-1], aug_zs[:, -1, -1]

        if isinstance(self.noise_scale, Tensor):
            reshaped_noise_scale = self.noise_scale.repeat(zs.shape[0], zs.shape[1], 1)
            xs_dist = torch.distributions.Laplace(loc=zs, scale=reshaped_noise_scale)
        else:
            xs_dist = torch.distributions.Laplace(loc=zs, scale=self.noise_scale)

        log_pxs_tensor = xs_dist.log_prob(batch_unrolled)
        log_pxs = log_pxs_tensor.sum()  # sum over batch, time points, and variables

        logqp = (logqp0 + logqp_path).sum(dim=0)

        # Compute expected log-prior and entropy terms for posterior graph distribution
        log_graph_prior = (
            self.graph_prior.log_prob(sampled_graphs) / sampled_graphs.shape[0]
        )  # expected log-prior graph probability under posterior adjacency_distribution
        graph_entropy = adjacency_distribution.entropy()

        # Compute overall loss
        nll = -log_pxs + logqp * self.kl_scheduler.val - log_graph_prior - graph_entropy

        step_output = {
            "loss": nll,
            "log_pxs": log_pxs,
            "logqp": logqp,
            "log_graph_prior": log_graph_prior,
            "graph_entropy": graph_entropy,
        }
        if self.record_graph_logits:
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    step_output[f"graph_logits_{i}_{j}"] = adjacency_distribution.logits[i, j]
        self.log_dict(step_output, prog_bar=True)

        self.kl_scheduler.step()

        return step_output

    def validation_step(self, *args, **kwargs):
        """Compute validation metrics: test_mse, tpr (true positive rate), fdr (false discovery rate)."""

        _ = kwargs
        batch, *_ = args
        # TensorDict mapping variable name to Tensor of shape (batch_size, variable_group_size, num_time_points)
        batch = batch.apply(lambda t: t.to(torch.float32, non_blocking=True))
        batch_unrolled = self.tensor_to_td.inv(batch)

        # Extract fixed brownian motion and true graph from the data module
        bm_validation = self.trainer.datamodule.bm_validation
        true_graph = self.trainer.datamodule.true_graph

        # Step 1: Sample a graph from variational posterior over graphs
        adjacency_distribution = self.adjacency_dist_module()
        sampled_graphs = adjacency_distribution.sample(
            torch.Size([batch_unrolled.shape[0]])
        )  # samples (w/o gumbel-softmax), shape (batch_size, num_nodes, num_nodes)
        if self.ignore_self_connections:
            # remove self-connections for evaluation
            for i, _ in enumerate(sampled_graphs):
                sampled_graphs[i] = sampled_graphs[i] - torch.diag(torch.diag(sampled_graphs[i]))

        # Step 2: Compute metrics
        logits = adjacency_distribution.logits.cpu().numpy()
        if self.ignore_self_connections:
            np.fill_diagonal(logits, -100000)
        # print("Logits: ", logits)
        # print("True graph: ", true_graph)
        if self.compute_auroc:
            auroc = roc_auc_score(y_true=true_graph.flatten().cpu().numpy(), y_score=logits.flatten())
            print("AUROC: ", auroc)
        conf_matrix = confusion_matrix_batched(true_graph, sampled_graphs)
        tpr = true_positive_rate(conf_matrix[1, 1], conf_matrix[1, 0])
        fdr = false_discovery_rate(conf_matrix[1, 1], conf_matrix[0, 1])
        f1 = f1_score(conf_matrix[1, 1], conf_matrix[0, 1], conf_matrix[1, 0])

        step_output = {"tpr": tpr, "fdr": fdr, "f1": f1}
        if self.compute_auroc:
            step_output["auroc"] = auroc

        # If brownian motions provided for validation trajectories.
        if bm_validation is not None:
            # Initial latent states
            z0 = batch_unrolled[:, 0, :]

            prior_sde = SCOTCHPriorSDE(
                prior_drift_net=self.prior_drift_fn, diffusion_net=self.diffusion_fn, graphs=sampled_graphs
            )

            # Compute predicted trajectory, given prior SDE, initial latent state, and fixed brownian motion
            if self.use_adjoint:
                # Adjoint method requires parameters to be explicitly passed in.
                pred_zs_t_first = torchsde.sdeint_adjoint(
                    prior_sde,
                    z0,
                    self.ts,
                    adjoint_params=self.parameters(),
                    dt=self.dt,
                    bm=bm_validation,
                    method=self.sde_method,
                    names={"drift": "f", "diffusion": "g"},
                )
            else:
                pred_zs_t_first = torchsde.sdeint(
                    prior_sde,
                    z0,
                    self.ts,
                    dt=self.dt,
                    bm=bm_validation,
                    method=self.sde_method,
                    names={"drift": "f", "diffusion": "g"},
                )
            pred_zs = swap_t_and_batch_dimensions(pred_zs_t_first)
            step_output["test_mse"] = ((pred_zs[:, 1:, :] - batch_unrolled[:, 1:, :]) ** 2).mean()

        self.log_dict(step_output)

        return step_output

    def configure_optimizers(self):
        modules = {
            "graph": self.adjacency_dist_module,
            "qz0_mean_net": self.qz0_mean_net,
            "qz0_logstd_net": self.qz0_logstd_net,
            "prior_drift_fn": self.prior_drift_fn,
            "diffusion_fn": self.diffusion_fn,
            "posterior_drift_fn": self.posterior_drift_fn,
            "trajectory_encoder": self.trajectory_encoder,
        }
        other_parameters = {
            "pz0_mean": self.pz0_mean,
            "pz0_logstd": self.pz0_logstd,
        }
        parameter_list = [
            {"params": module.parameters(), "lr": self.learning_rates[name], "name": name}
            for name, module in modules.items()
        ] + [
            {"params": [parameter], "lr": self.learning_rates[name], "name": name}
            for name, parameter in other_parameters.items()
        ]

        optimizer = torch.optim.Adam(parameter_list)

        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda epoch: min(1, epoch / self.lr_warmup_iters)
            ),
        }

    def sample(self, batch_size, ts, bm=None, z0=None):
        adjacency_distribution = self.adjacency_dist_module()  # self.sem_module.adjacency_module()
        # hard sample
        sampled_graphs = adjacency_distribution.sample(torch.Size([batch_size]))

        # Sample initial positions
        if z0 is None:
            eps = torch.randn(size=(batch_size, *self.pz0_mean.shape[1:]), device=self.pz0_mean.device)
            z0 = self.pz0_mean + self.pz0_logstd.exp() * eps

        sde = SCOTCHPriorSDE(self.prior_drift_fn, self.diffusion_fn, sampled_graphs)
        return sde.sample(z0, ts, bm, dt=self.dt)
