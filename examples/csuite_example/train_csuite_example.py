"""
An example script showing how to train a DECI model on CSuite Data.

This demonstrates how to assemble the various components of the library and how to perform training.

This logs MLFlow logs to the environment variable location if it is set https://mlflow.org/docs/latest/tracking.html#where-runs-are-recorded
"""
import argparse
import logging
import pathlib
from dataclasses import dataclass

import fsspec
import mlflow
import pytorch_lightning as pl
import torch
from dataclasses_json import DataClassJsonMixin
from tensordict import TensorDict
from torch.nn import Parameter, ParameterDict
from torch.optim import Adam
from torch.utils.data import DataLoader

from causica.datasets.csuite_data import DataEnum, get_categorical_sizes, load_data
from causica.datasets.tensordict_utils import convert_one_hot, tensordict_shapes
from causica.distributions import (
    AdjacencyDistribution,
    ENCOAdjacencyDistribution,
    GibbsDAGPrior,
    ParametrizedDistribution,
)
from causica.functional_relationships import ICGNN
from causica.graph.dag_constraint import calculate_dagness
from causica.sem.distribution_parameters_sem import DistributionParametersSEM
from causica.training.auglag import AugLagLossCalculator, AugLagLR, AugLagLRConfig
from causica.training.trainable_container import NoiseDist, TrainableContainer, create_noise_dists
from causica.training.training_callbacks import AverageMetricTracker

logging.basicConfig(level=logging.INFO)

BEST_MODEL = "best_model.pt"
LAST_MODEL = "last_model.pt"


DATASET_PATH = "https://azuastoragepublic.blob.core.windows.net/datasets"


# Data class for config
@dataclass(frozen=True)
class TrainingConfig(DataClassJsonMixin):  # using mixin to prevent mypy errors
    noise_dist: NoiseDist = NoiseDist.GAUSSIAN
    batch_size: int = 128
    max_epoch: int = 1000
    gumbel_temp: float = 0.25
    averaging_period: int = 10
    prior_sparsity_lambda: float = 0.05
    # Auglag parameters
    init_alpha: float = 0.0
    init_rho: float = 1.0


@dataclass(frozen=True)
class ICGNNConfig(DataClassJsonMixin):
    embedding_size: int = 32
    out_dim_g: int = 32
    norm_layer: bool = True
    res_connection: bool = True


def train(
    dataset: str, seed: int, auglag_config: AugLagLRConfig, icgnn_config: ICGNNConfig, training_config: TrainingConfig
) -> TrainableContainer:
    """Run training on CSuite with a noise distribution given by `noise_dist_enum`."""
    pl.seed_everything(seed)  # set the random seed
    device = "cuda" if torch.cuda.is_available() else "cpu"

    variables_metadata = load_data(DATASET_PATH, dataset, DataEnum.VARIABLES_JSON)
    dataset_train: TensorDict = load_data(DATASET_PATH, dataset, DataEnum.TRAIN, variables_metadata)

    dataset_train = convert_one_hot(
        dataset_train, one_hot_sizes=get_categorical_sizes(variables_list=variables_metadata["variables"])
    )

    dataloader_train = DataLoader(
        dataset=dataset_train,
        collate_fn=lambda x: x,
        batch_size=training_config.batch_size,
        shuffle=True,
        drop_last=False,
    )

    num_nodes = len(dataset_train.keys())

    # Define vardist params
    vardist_logits_orient = Parameter(torch.zeros(int(num_nodes * (num_nodes - 1) / 2)), requires_grad=True)
    vardist_logits_exist = Parameter(torch.zeros(num_nodes, num_nodes), requires_grad=True)

    param_vardist = ParametrizedDistribution(
        ENCOAdjacencyDistribution,
        ParameterDict(dict(logits_orient=vardist_logits_orient, logits_exist=vardist_logits_exist)),
    )

    # Define ICGNN
    icgnn = ICGNN(
        variables=tensordict_shapes(dataset_train),
        embedding_size=icgnn_config.embedding_size,
        out_dim_g=icgnn_config.out_dim_g,
        norm_layer=torch.nn.LayerNorm if icgnn_config.norm_layer else None,
        res_connection=icgnn_config.res_connection,
    )
    # create the noise distributions
    variables = load_data(DATASET_PATH, dataset, DataEnum.VARIABLES_JSON)["variables"]
    types_dict = {var["group_name"]: var["type"] for var in variables}
    noise_dist_funcs, param_noise_dist = create_noise_dists(
        tensordict_shapes(dataset_train), types_dict, training_config.noise_dist
    )

    # Create container
    container = TrainableContainer(
        dataset_name=dataset,
        icgnn=icgnn,
        vardist=param_vardist,
        noise_dist_params=param_noise_dist,
        noise_dist_type=training_config.noise_dist,
    )
    container.to(device)

    # Create prior
    prior = GibbsDAGPrior(num_nodes=num_nodes, sparsity_lambda=training_config.prior_sparsity_lambda)

    # Define Optimizer
    # @TODO: Clean this up
    parameter_list = [
        {
            "params": module if isinstance(module, torch.nn.Parameter) else module.parameters(),
            "lr": auglag_config.lr_init_dict[name],
            "name": name,
        }
        for name, module in container.named_children()
    ]
    optimizer = Adam(parameter_list)

    # Define training helper classes
    scheduler = AugLagLR(config=auglag_config)
    avg_metric_tracker = AverageMetricTracker(averaging_period=training_config.averaging_period)
    auglag_loss = AugLagLossCalculator(init_alpha=training_config.init_alpha, init_rho=training_config.init_rho)

    assert len(dataset_train.batch_size) == 1, "Only 1D batch size is supported"
    num_samples = len(dataset_train)
    step_counter = 0
    for epoch in range(training_config.max_epoch):
        for batch in dataloader_train:
            batch = batch.apply(lambda t: t.to(dtype=torch.float32, device=device))
            optimizer.zero_grad()
            # sample graph
            vardist = param_vardist.forward()
            assert isinstance(vardist, AdjacencyDistribution)
            cur_graph = vardist.relaxed_sample(torch.Size([]), temperature=training_config.gumbel_temp)  # soft sample
            # Create SEM
            sem = DistributionParametersSEM(
                graph=cur_graph,
                node_names=dataset_train.keys(),
                noise_dist=noise_dist_funcs,
                func=container.icgnn,
            )

            batch_log_prob = sem.log_prob(batch).mean()  # average over graphs

            objective = (-vardist.entropy() - prior.log_prob(cur_graph)) / num_samples - batch_log_prob
            constraint = calculate_dagness(cur_graph)
            weighted_constraint = constraint / num_samples

            loss = auglag_loss(objective, weighted_constraint)  # we should use weighted constraint for the loss

            loss.backward()
            optimizer.step()
            scheduler.step(
                optimizer=optimizer,
                loss=auglag_loss,
                loss_value=loss.item(),
                lagrangian_penalty=constraint.item(),
            )
            if avg_metric_tracker.step(-batch_log_prob.item()):
                print(f"Saving model: {avg_metric_tracker.min_value}")
                torch.save({"container": container}, BEST_MODEL)

            # log metrics
            print(
                f"epoch:{epoch} loss:{loss.item():.5g} nll:{-batch_log_prob.detach().cpu().numpy():.5g} dagness:{constraint.item():.5f} "
                f"num_edges:{(cur_graph > 0.0).sum()} alpha:{auglag_loss.alpha:.5g} rho:{auglag_loss.rho:.5g} "
                f"step:{scheduler.outer_opt_counter}/{scheduler.step_counter} num_lr_updates:{scheduler.num_lr_updates}"
            )
            step_counter += 1

    # save the final model
    torch.save({"container": container}, LAST_MODEL)
    return container


def main():
    this_dir = pathlib.Path().resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", "-d", default="csuite_nonlingauss")
    parser.add_argument("--seed", "-s", default=123, type=int)

    parser.add_argument(
        "--config-auglag", type=str, default=f"{this_dir}/auglag_default.json", dest="auglag_config_path"
    )
    parser.add_argument("--config-icgnn", type=str, default=f"{this_dir}/icgnn_default.json", dest="icgnn_config_path")
    parser.add_argument(
        "--config-training", type=str, default=f"{this_dir}/training_default.json", dest="training_config_path"
    )

    args = parser.parse_args()

    # Load configs
    with fsspec.open(args.auglag_config_path, "r", encoding="utf-8") as f:
        auglag_config = AugLagLRConfig.from_json(f.read())

    with fsspec.open(args.icgnn_config_path, "r", encoding="utf-8") as f:
        icgnn_config = ICGNNConfig.from_json(f.read())

    with fsspec.open(args.training_config_path, "r", encoding="utf-8") as f:
        training_config = TrainingConfig.from_json(f.read())

    with mlflow.start_run(run_name=f"csuite_{args.dataset}", tags={"dataset": args.dataset, "seed": args.seed}) as run:
        if mlflow.get_tracking_uri().startswith("azureml://"):
            uri = f"https://ml.azure.com/experiments/id/{run.info.experiment_id}/runs/{run.info.run_id}"
            run_name = run.to_dictionary()["data"]["tags"].get("mlflow.runName", "")
            print(f"Started experiment {run_name}: {uri}")

        train(
            dataset=args.dataset,
            seed=args.seed,
            auglag_config=auglag_config,
            icgnn_config=icgnn_config,
            training_config=training_config,
        )


if __name__ == "__main__":
    main()
