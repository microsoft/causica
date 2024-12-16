import argparse
import os

from fip.launchers.basic_commands import launch_job


def get_path_pretrained_cond_fip(run_id: str, dir_model: str):
    path_model_learned = f"{dir_model}/{run_id}/outputs/best_model.ckpt"
    if not os.path.exists(path_model_learned):
        raise FileNotFoundError(f"Model checkpoint not found at {path_model_learned}")
    return path_model_learned


def generate_base_script(
    path_data: str,
    path_model_learned: str,
):
    path_model_config = "./src/cond_fip/config/cond_fip_inference.yaml"
    path_data_config = "./src/cond_fip/config/numpy_tensor_data_module.yaml"
    param_base_script = [
        (" --config ", path_model_config),
        (" --data ", path_data_config),
        (" --model.enc_dec_model_path ", path_model_learned),
        (" --data.data_dir ", path_data),
    ]
    command_base_script = " python -m cond_fip.entrypoint_test " + " ".join([f"{k}{v}" for k, v in param_base_script])

    return command_base_script


def command_python(
    dir_model: str,
    run_id: str,
    path_data: str,
):
    path_model_learned = get_path_pretrained_cond_fip(run_id, dir_model)
    return generate_base_script(
        path_data,
        path_model_learned,
    )


def main(
    dir_model: str,
    run_id: str,
    path_data: str,
    num_workers: int,
):
    command_base_script = command_python(dir_model, run_id, path_data)

    launch_job(
        dir_model=dir_model,
        command_base_script=command_base_script,
        instance_count=-1,
        gpus_per_node=1,
        local_resume=False,
        run_name_resume="",
        num_workers=num_workers,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # pretrained argument
    parser.add_argument("--dir_model", type=str, default="./src/cond_fip/outputs", help="Where the ckpts are stored")
    parser.add_argument("--run_id", type=str, help="run id of the pre-trained cond-fip", required=True)

    # data argument
    parser.add_argument(
        "--path_data", type=str, default="./data/er_linear_gaussian_in/total_nodes_5/seed_1/", help="Path to the data"
    )

    # machine arguments
    parser.add_argument("--num_workers", type=int, default=23, help="Number of workers")

    args = parser.parse_args()

    main(
        args.dir_model,
        args.run_id,
        args.path_data,
        args.num_workers,
    )
