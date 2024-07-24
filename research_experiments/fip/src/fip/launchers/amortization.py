import argparse

from fip.launchers.basic_commands import generate_path_logging, generate_time_name, launch_job

path_config: str = "src/fip/config/amortization/leaf_prediction.yaml"
path_data: str = "src/fip/config/synthetic_data_module.yaml"


def generate_base_script(path_logging: str, batch_size: int):
    param_base_script = [
        (" --config ", path_config),
        (" --data ", path_data),
        (" --best_checkpoint_callback.dirpath ", path_logging),
        (" --data.train_batch_size ", batch_size),
        (" --data.test_batch_size ", batch_size),
    ]

    command_base_script = " python -m fip.entrypoint " + " ".join([f"{k}{v}" for k, v in param_base_script])

    return command_base_script


def command_python(
    run_name: str,
    batch_size: int,
):
    path_logging = generate_path_logging(run_name=run_name)
    return generate_base_script(path_logging, batch_size)


def main(
    batch_size: int,
    instance_count: int,
    gpus_per_node: int,
    local_resume: bool,
    run_name_resume: str,
    num_workers: int,
):
    run_name = "leaf_amortization_" + generate_time_name()
    command_base_script = command_python(run_name, batch_size)
    launch_job(
        command_base_script=command_base_script,
        instance_count=instance_count,
        gpus_per_node=gpus_per_node,
        local_resume=local_resume,
        run_name_resume=run_name_resume,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # machine arguments
    parser.add_argument("--num_workers", type=int, default=23, help="Number of workers")

    # training arguments
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")

    # distributed arguments
    parser.add_argument("--instance_count", type=int, default=-1, help="Number of instances")
    parser.add_argument("--gpus_per_node", type=int, default=1, help="Number of gpus per instance")

    # resume arguments
    parser.add_argument("--local_resume", action="store_true", help="")
    parser.add_argument("--run_name_resume", type=str, default="", help="Name of the run to be resumed")

    args = parser.parse_args()

    main(
        args.batch_size,
        args.instance_count,
        args.gpus_per_node,
        args.local_resume,
        args.run_name_resume,
        args.num_workers,
    )
