import argparse

from fip.launchers.basic_commands import generate_path_logging, generate_time_name, launch_job


def generate_base_script(
    path_model_config: str,
    path_data_config: str,
    path_logging: str,
    batch_size: int,
    factor_epoch: int,
    accumulate_grad_batches: int,
):
    param_base_script = [
        (" --config ", path_model_config),
        (" --data ", path_data_config),
        (" --best_checkpoint_callback.dirpath ", path_logging),
        (" --data.train_batch_size ", batch_size),
        (" --data.test_batch_size ", batch_size),
        (" --data.factor_epoch ", factor_epoch),
        (" --trainer.accumulate_grad_batches ", accumulate_grad_batches),
    ]

    command_base_script = " python -m fip.entrypoint " + " ".join([f"{k}{v}" for k, v in param_base_script])

    return command_base_script


def command_python(
    dir_model: str,
    run_name: str,
    batch_size: int,
    factor_epoch: int,
    accumulate_grad_batches: int,
):
    path_model_config = "./src/cond_fip/config/encoder_training.yaml"
    path_data_config = "./src/cond_fip/config/synthetic_data_module.yaml"
    path_logging = generate_path_logging(run_name=run_name, dir_model=dir_model)
    return generate_base_script(
        path_model_config,
        path_data_config,
        path_logging,
        batch_size,
        factor_epoch,
        accumulate_grad_batches,
    )


def main(
    dir_model: str,
    batch_size: int,
    factor_epoch: int,
    accumulate_grad_batches: int,
    instance_count: int,
    gpus_per_node: int,
    local_resume: bool,
    run_name_resume: str,
    num_workers: int,
):
    run_name = generate_time_name()
    run_name = "encoder_training_" + run_name
    command_base_script = command_python(
        dir_model,
        run_name,
        batch_size,
        factor_epoch,
        accumulate_grad_batches,
    )

    launch_job(
        dir_model=dir_model,
        command_base_script=command_base_script,
        instance_count=instance_count,
        gpus_per_node=gpus_per_node,
        local_resume=local_resume,
        run_name_resume=run_name_resume,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # logging arguments
    parser.add_argument("--dir_model", type=str, default="./src/cond_fip/outputs", help="Where to store ckpts")

    # machine arguments
    parser.add_argument("--num_workers", type=int, default=23, help="Number of workers")

    # training arguments
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--factor_epoch", type=int, default=8, help="Factor epoch")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulate grad batches")

    # GPU argumnents
    parser.add_argument("--instance_count", type=int, default=-1, help="Number of instances")
    parser.add_argument("--gpus_per_node", type=int, default=1, help="Number of gpus per instance")

    # resume arguments
    parser.add_argument("--local_resume", action="store_true", help="")
    parser.add_argument("--run_name_resume", type=str, default="", help="Name of the run to be resumed")

    args = parser.parse_args()

    main(
        args.dir_model,
        args.batch_size,
        args.factor_epoch,
        args.accumulate_grad_batches,
        args.instance_count,
        args.gpus_per_node,
        args.local_resume,
        args.run_name_resume,
        args.num_workers,
    )
