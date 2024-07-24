import argparse
import os
from enum import Enum

from fip.launchers.basic_commands import generate_path_logging, launch_job


class GroundTruthEnum(Enum):
    """Different type of amortization."""

    PERM = "perm"
    GRAPH = "graph"


def generate_config_files(ground_truth_case: GroundTruthEnum):
    match ground_truth_case:
        case GroundTruthEnum.PERM:
            path_config = "src/fip/config/scm_learning_with_ground_truth/scm_learning_true_perm.yaml"
        case GroundTruthEnum.GRAPH:
            path_config = "src/fip/config/scm_learning_with_ground_truth/scm_learning_true_graph.yaml"
        case _:
            raise ValueError(f"{ground_truth_case} case not recognized")

    path_data = "src/fip/config/numpy_tensor_data_module.yaml"

    return path_config, path_data


def generate_base_script(
    path_config: str,
    path_data: str,
    path_logging: str,
    data_dir: str,
    total_nodes: int,
    standardize: bool,
    batch_size: int,
    accumulate_grad_batches: int,
):
    param_base_script = [
        (" --config ", path_config),
        (" --data ", path_data),
        (" --best_checkpoint_callback.dirpath ", path_logging),
        (" --data.data_dir ", data_dir),
        (" --data.train_batch_size ", batch_size),
        (" --data.test_batch_size ", batch_size),
        (" --trainer.accumulate_grad_batches ", accumulate_grad_batches),
        (" --model.total_nodes ", str(total_nodes)),
        (" --data.standardize ", str(standardize)),
        (" --data.num_interventions ", str(total_nodes)),
    ]

    command_base_script = " python -m fip.entrypoint " + " ".join([f"{k}{v}" for k, v in param_base_script])

    return command_base_script


def command_python(
    run_name: str,
    ground_truth_case: GroundTruthEnum,
    dist_case: str,
    total_nodes: int,
    seed: int,
    standardize: bool,
    batch_size: int,
    accumulate_grad_batches: int,
):
    data_dir = os.path.join(
        "src",
        "fip",
        "data",
        dist_case,
        "total_nodes_" + str(total_nodes),
        "seed_" + str(seed),
    )

    path_config, path_data = generate_config_files(ground_truth_case)
    path_logging = generate_path_logging(run_name=run_name)
    return generate_base_script(
        path_config,
        path_data,
        path_logging,
        data_dir,
        total_nodes,
        standardize,
        batch_size,
        accumulate_grad_batches,
    )


def main(
    ground_truth_case: GroundTruthEnum,
    dist_case: str,
    total_nodes: int,
    seed: int,
    standardize: bool,
    batch_size: int,
    accumulate_grad_batches: int,
    instance_count: int,
    gpus_per_node: int,
    local_resume: bool,
    run_name_resume: str,
    num_workers: int,
):
    run_name = (
        f"true_{ground_truth_case.value}_" + dist_case + "_total_nodes_" + str(total_nodes) + "_seed_" + str(seed)
    )
    command_base_script = command_python(
        run_name,
        ground_truth_case,
        dist_case,
        total_nodes,
        seed,
        standardize,
        batch_size,
        accumulate_grad_batches,
    )
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

    # type of ground truth
    parser.add_argument(
        "--ground_truth_case",
        type=GroundTruthEnum,
        help="Ground truth case",
        choices=GroundTruthEnum,
        required=True,
    )

    # dataset argument
    parser.add_argument(
        "--dist_case",
        type=str,
        default="er_linear_gaussian_in",
        help="",
    )
    parser.add_argument("--total_nodes", type=int, default=5, help="")
    parser.add_argument("--seed", type=int, default=1, help="")
    parser.add_argument("--standardize", action="store_true", help="")

    # training argument
    parser.add_argument("--batch_size", type=int, default=2000, help="")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="")

    ## machine arguments
    parser.add_argument("--num_workers", type=int, default=23, help="Number of workers")

    # distributed arguments
    parser.add_argument("--instance_count", type=int, default=-1, help="Number of instances")
    parser.add_argument("--gpus_per_node", type=int, default=1, help="Number of gpus per node")

    # resume arguments
    parser.add_argument("--local_resume", action="store_true", help="")
    parser.add_argument("--run_name_resume", type=str, default="", help="Name of the run to be resumed")

    args = parser.parse_args()
    main(
        ground_truth_case=args.ground_truth_case,
        dist_case=args.dist_case,
        total_nodes=args.total_nodes,
        seed=args.seed,
        standardize=args.standardize,
        batch_size=args.batch_size,
        accumulate_grad_batches=args.accumulate_grad_batches,
        instance_count=args.instance_count,
        gpus_per_node=args.gpus_per_node,
        local_resume=args.local_resume,
        run_name_resume=args.run_name_resume,
        num_workers=args.num_workers,
    )
