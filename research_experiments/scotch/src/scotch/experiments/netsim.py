"""This file is for conducting the experiments on netsim dataset"""
import argparse

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
    parser = argparse.ArgumentParser(description="Run Netsim experiments.")
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
    # ADDED
    parser.add_argument("-p", "--missing_prob", type=float, help="missing probability", required=True)

    args = parser.parse_args()

    seed_everything(args.seed)
    # HParams
    experiment_name = args.experiment_name
    max_epochs = args.epoch
    default_lr = args.lr
    res_connection = args.res_connection
    layer_norm = args.layer_norm
    deci_diffusion = args.deci_diffusion
    sigmoid_output = args.sigmoid_output
    missing_prob = args.missing_prob

    lrs = {
        "graph": default_lr,
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
    train_size = 5
    val_size = -1
    t_max = 10.0
    num_time_points = 200

    dt = args.dt  # sde solver dt = observation interval
    normalize = args.normalize
    lr_warmup_iters = args.lr_warmup

    hparams = {
        "seed": args.seed,
        "epoch": args.epoch,
        "dt": args.dt,
        "default_lr": default_lr,
        "train_size": train_size,
        "val_size": val_size,
        "prior_sparsity_lambda": prior_sparsity_lambda,
        "t_max": t_max,
        "num_time_points": num_time_points,
        "normalize": normalize,
        "lr_warmup_iters": lr_warmup_iters,
        "res_connection": res_connection,
        "layer_rnorm": layer_norm,
        "deci_diffusion": deci_diffusion,
        "sigmoid_output": sigmoid_output,
        "missing_prob": missing_prob,
    }

    # netsim
    state_size = 15
    variables_metadata = VariablesMetadata(
        [Variable(name=f"x{i}", type=VariableTypeEnum.CONTINUOUS, group_name=f"x{i}") for i in range(state_size)]
    )

    subf = "norm" if args.normalize else "unnorm"
    ts = torch.load(f"data/netsim_processed/{subf}/times_{str(args.missing_prob)}_{args.seed}.pt")
    training_data = torch.load(f"data/netsim_processed/{subf}/data_{str(args.missing_prob)}_{args.seed}.pt")
    true_graph = torch.load(f"data/netsim_processed/{subf}/true_graph.pt")

    training_data = TensorDict(
        {f"x{i}": training_data[:, :, i].unsqueeze(dim=2) for i in range(state_size)},
        batch_size=[train_size],
    )

    validation_data = training_data

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
        layer_norm=layer_norm,
        res_connections=res_connection,
        deci_diffusion=True,
        add_diffusion_self_connections=True,
        sigmoid_output=sigmoid_output,
    )

    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri="file:./mlflow_logs/mlruns",
    )
    mlf_logger.log_hyperparams(hparams)
    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=max_epochs,
        fast_dev_run=False,
        callbacks=[
            TQDMProgressBar(refresh_rate=19),
            ModelCheckpoint(every_n_epochs=50),
        ],
        check_val_every_n_epoch=50,
        logger=mlf_logger,
    )

    trainer.fit(scotch, datamodule=scotch_data)
