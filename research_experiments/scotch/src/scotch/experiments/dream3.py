import argparse

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import MLFlowLogger
from scotch.latent_learning.scotch_data_module import SCOTCHDataModule
from scotch.latent_learning.scotch_module import SCOTCHModule
from tensordict import TensorDict

from causica.datasets.causica_dataset_format import Variable, VariablesMetadata
from causica.datasets.variable_types import VariableTypeEnum

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run DREAM3 experiments.")
    parser.add_argument("-d", "--dimension", type=int, help="dimension of dream3 dataset (10, 50, or 100)")
    parser.add_argument("-n", "--name", type=str, help="name of the dataset (Yeast1, Yeast2, Yeast3, Ecoli1, Ecoli2)")
    parser.add_argument("-e", "--epoch", type=int, help="max number of epochs", default=20000)
    parser.add_argument("-l", "--lr", type=float, help="learning rate", default=3e-3)
    parser.add_argument("-s", "--sparsity", type=float, help="sparsity penalty", default=500)
    parser.add_argument("-t", "--dt", type=float, help="dt", default=0.05)
    parser.add_argument("-nor", "--normalize", action="store_true", help="whether to normalize")
    parser.add_argument("-sd", "--seed", type=int, help="random seed", required=True)
    parser.add_argument("-en", "--experiment_name", type=str, help="experiment name", required=True)
    parser.add_argument("-res", "--res_connection", action="store_true", help="whether to use res_connection")
    parser.add_argument("-ln", "--layer_norm", action="store_true", help="whether to use layer_norm")
    parser.add_argument("-warm", "--lr_warmup", type=int, default=10000, help="warmup epochs")
    parser.add_argument("-deci", "--deci_diffusion", action="store_true", help="whether to use deci diffusion function")
    parser.add_argument(
        "-sig",
        "--sigmoid_output",
        action="store_true",
        help="whether to use sigmoid output for deci diffusion function",
    )

    args = parser.parse_args()
    # set seed
    seed_everything(args.seed)

    # HParams
    experiment_name = args.experiment_name
    max_epochs = args.epoch
    default_lr = args.lr
    lrs = {
        "graph": default_lr,  # changed from 1e-2
        "qz0_mean_net": default_lr,
        "qz0_logstd_net": default_lr,
        "pz0_mean": default_lr,
        "pz0_logstd": default_lr,
        "prior_drift_fn": default_lr,
        "diffusion_fn": default_lr,
        "posterior_drift_fn": default_lr,
        "trajectory_encoder": default_lr,
    }
    prior_sparsity_lambda = args.sparsity

    t_max = 1.05
    num_time_points = 21

    dt = args.dt  # sde solver dt = observation interval
    normalize = args.normalize
    lr_warmup_iters = 100
    res_connection = args.res_connection
    layer_norm = args.layer_norm
    deci_diffusion = args.deci_diffusion
    sigmoid_output = args.sigmoid_output

    hparams = {
        "seed": args.seed,
        "epoch": args.epoch,
        "normalize": args.normalize,
        "prior_sparsity_lambda": prior_sparsity_lambda,
        "t_max": t_max,
        "num_time_points": num_time_points,
        "dt": dt,
        "default_lr": default_lr,
        "lr_warmup_iters": lr_warmup_iters,
        "res_connection": res_connection,
        "layer_norm": layer_norm,
        "deci_diffusion": deci_diffusion,
        "sigmoid_output": sigmoid_output,
    }

    # DREAM3
    state_size = args.dimension
    variables_metadata = VariablesMetadata(
        [Variable(name=f"x{i}", type=VariableTypeEnum.CONTINUOUS, group_name=f"x{i}") for i in range(state_size)]
    )

    ts = torch.linspace(0, t_max, num_time_points)
    training_data_df = pd.read_csv(
        f"data/DREAM3 in silico challenge/Size{args.dimension}/DREAM3 data/InSilicoSize{args.dimension}-{args.name}-trajectories.tsv",
        sep="\t",
    ).drop("Time", axis=1)

    training_data = torch.tensor(training_data_df.values, device="cuda").view(
        -1, num_time_points, state_size
    )  # (batch, t, vars)

    training_data = (
        (training_data - training_data.mean(dim=(0, 1))) / training_data.std(dim=(0, 1)) if normalize else training_data
    )

    training_data = TensorDict(
        {f"x{i}": training_data[:, :, i].unsqueeze(dim=2) for i in range(state_size)},
        batch_size=[training_data.shape[0]],
    )

    edges_df = pd.read_csv(
        f"data/DREAM3 in silico challenge/Size{args.dimension}/Networks/InSilicoSize{args.dimension}-{args.name}.tsv",
        sep="\t",
        header=None,
    ).drop(2, axis=1)
    true_graph = torch.zeros((state_size, state_size), dtype=torch.int64, device="cuda")

    edges_df = edges_df.applymap(lambda x: int(x[1:]))
    edges = torch.tensor(edges_df.values, device="cuda")

    for edge in edges:
        true_graph[edge[0] - 1, edge[1] - 1] = 1  # 0-indexed

    validation_data = training_data
    #####

    scotch_data = SCOTCHDataModule(
        ts=ts,
        training_data=training_data,
        validation_data=validation_data,
        true_graph=true_graph,
        variables_metadata=variables_metadata,
        batch_size=1024,
    )

    # SCOTCH Module
    scotch = SCOTCHModule(
        learning_rates=lrs,
        prior_sparsity_lambda=prior_sparsity_lambda,
        dt=dt,
        record_graph_logits=args.dimension < 20,
        lr_warmup_iters=lr_warmup_iters,
        ignore_self_connections=True,
        res_connections=res_connection,
        layer_norm=layer_norm,
        deci_diffusion=deci_diffusion,
        add_diffusion_self_connections=True,
        sigmoid_output=sigmoid_output,
    )

    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri="file:./mlflow_logs/mlruns",
    )

    mlf_logger.log_hyperparams(hparams)

    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=max_epochs,
        fast_dev_run=False,
        callbacks=[
            TQDMProgressBar(refresh_rate=19),
            ModelCheckpoint(every_n_epochs=500, dirpath="./outputs", save_last=True),
        ],
        check_val_every_n_epoch=50,
        logger=mlf_logger,
    )

    trainer.fit(scotch, datamodule=scotch_data)
