import argparse
import os

from fip.launchers.basic_commands import generate_path_logging, launch_job

path_config: str = "./src/fip/config/scm_learning_with_predicted_truth/scm_learning_predicted_leaf.yaml"
path_data: str = "./src/fip/config/numpy_tensor_data_module.yaml"


def get_info_pretrained_model(run_id: str):
    path_model_learned = f"./src/fip/outputs/{run_id}/outputs/best_model.ckpt"
    path_config_learned = f"./src/fip/outputs/{run_id}/outputs/config.yaml"

    return path_model_learned, path_config_learned


def generate_base_script(
    path_logging: str,
    data_dir: str,
    total_nodes: int,
    standardize: bool,
    path_model_learned: str,
    path_config_learned: str,
    batch_size: int,
    accumulate_grad_batches: int,
):
    param_base_script = [
        (" --config ", path_config),
        (" --data ", path_data),
        (" --best_checkpoint_callback.dirpath ", path_logging),
        (" --data.data_dir ", data_dir),
        (" --model.total_nodes ", str(total_nodes)),
        (" --data.standardize ", str(standardize)),
        (" --model.leaf_model_path ", path_model_learned),
        (" --model.leaf_config_path ", path_config_learned),
        (" --data.num_interventions ", str(total_nodes)),
        (" --data.train_batch_size ", batch_size),
        (" --data.test_batch_size ", batch_size),
        (" --trainer.accumulate_grad_batches ", accumulate_grad_batches),
    ]

    command_base_script = " python -m fip.entrypoint " + " ".join([f"{k}{v}" for k, v in param_base_script])

    return command_base_script


def command_python(
    run_name: str,
    dist_case: str,
    total_nodes: int,
    seed: int,
    standardize: bool,
    run_id: str,
    batch_size: int,
    accumulate_grad_batches: int,
):

    data_dir = os.path.join(
        ".",
        "data",
        dist_case,
        "total_nodes_" + str(total_nodes),
        "seed_" + str(seed),
    )

    path_model_learned, path_config_learned = get_info_pretrained_model(run_id)
    path_logging = generate_path_logging(run_name=run_name)
    return generate_base_script(
        path_logging,
        data_dir,
        total_nodes,
        standardize,
        path_model_learned,
        path_config_learned,
        batch_size,
        accumulate_grad_batches,
    )


def main(
    dist_case: str,
    total_nodes: int,
    seed: int,
    standardize: bool,
    run_id: str,
    instance_count: int,
    gpus_per_node: int,
    local_resume: bool,
    run_name_resume: str,
    batch_size: int,
    accumulate_grad_batches: int,
    num_workers: int,
):
    run_name = "predicted_leaf_" + dist_case + "_total_nodes_" + str(total_nodes) + "_seed_" + str(seed)
    command_base_script = command_python(
        run_name,
        dist_case,
        total_nodes,
        seed,
        standardize,
        run_id,
        batch_size,
        accumulate_grad_batches,
    )
    launch_job(
        command_base_script,
        instance_count,
        gpus_per_node,
        local_resume,
        run_name_resume,
        num_workers,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # pretrained argument
    parser.add_argument("--run_id", type=str, help="run id of the pre-trained model", required=True)

    # dataset argument
    parser.add_argument("--dist_case", type=str, default="er_linear_gaussian_in", help="")
    parser.add_argument("--total_nodes", type=int, default=5, help="")
    parser.add_argument("--seed", type=int, default=1, help="")
    parser.add_argument("--standardize", action="store_true", help="")

    # training argument
    parser.add_argument("--batch_size", type=int, default=2000, help="")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="")

    ## machine arguments
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers")

    # distributed arguments
    parser.add_argument("--instance_count", type=int, default=-1, help="Number of instances")
    parser.add_argument("--gpus_per_node", type=int, default=1, help="Number of gpus per node")

    # resume argument
    parser.add_argument("--local_resume", action="store_true", help="")
    parser.add_argument("--run_name_resume", type=str, help="")

    args = parser.parse_args()

    main(
        dist_case=args.dist_case,
        total_nodes=args.total_nodes,
        seed=args.seed,
        standardize=args.standardize,
        run_id=args.run_id,
        instance_count=args.instance_count,
        gpus_per_node=args.gpus_per_node,
        local_resume=args.local_resume,
        run_name_resume=args.run_name_resume,
        batch_size=args.batch_size,
        accumulate_grad_batches=args.accumulate_grad_batches,
        num_workers=args.num_workers,
    )
