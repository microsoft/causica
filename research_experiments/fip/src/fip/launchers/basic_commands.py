import datetime
import os


def generate_time_name():
    now = str(datetime.datetime.now().replace(second=0, microsecond=0)).replace(" ", "_")
    now = now.replace(":", "-")
    return f"{now}"


def generate_distributed_command(instance_count: int, gpus_per_node: int):
    param_distributed = [
        (" --model.distributed ", "true"),
        (" --trainer.devices ", str(gpus_per_node)),
        (" --trainer.num_nodes ", str(instance_count)),
        (" --trainer.strategy ", "ddp_find_unused_parameters_false"),
    ]
    return " ".join([f"{k}{v}" for k, v in param_distributed])


def generate_path_logging(run_name: str):
    path_logging = f"src/fip/outputs/{run_name}/outputs/"
    return path_logging


def generate_num_workers_command(num_workers: int):
    return f" --data.num_workers {str(num_workers)}"


def run_local(
    command_base_script: str,
    instance_count: int,
    gpus_per_node: int,
    num_workers: int,
):
    command_num_workers = generate_num_workers_command(num_workers)

    if instance_count != -1:
        assert instance_count > 0, "instance_count must be a positive integer"
        assert gpus_per_node > 0, "gpus_per_node must be a positive integer"
        command_distributed = generate_distributed_command(instance_count, gpus_per_node)
    else:
        command_distributed = ""
    script = command_base_script + command_distributed + command_num_workers
    os.system(script)


def resume_local(run_name: str):
    path_config = f"src/fip/outputs/{run_name}/outputs/config.yaml"
    command_base_script = "python -m fip.entrypoint --config " + path_config
    script = command_base_script
    os.system(script)


def launch_job(
    command_base_script: str,
    instance_count: int,
    gpus_per_node: int,
    local_resume: bool,
    run_name_resume: str,
    num_workers: int,
):
    if local_resume:
        resume_local(run_name_resume)
    else:
        run_local(
            command_base_script=command_base_script,
            instance_count=instance_count,
            gpus_per_node=gpus_per_node,
            num_workers=num_workers,
        )
