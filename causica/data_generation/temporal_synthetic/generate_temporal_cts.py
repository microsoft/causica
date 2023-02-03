import argparse
import json
import os

from causica.data_generation.temporal_synthetic.data_gen_util import (
    generate_cts_temporal_data,
    generate_name,
    set_random_seed,
)


def main(save_dir: str, num_nodes: int = 40, intervention_value: float = 10):
    series_length = 200 if num_nodes < 40 else 400
    burnin_length = 50 if num_nodes < 40 else 100
    num_train_samples = 50 if num_nodes < 40 else 100
    num_test_samples = 50
    num_intervention_samples = 5000
    connection_factor = 2

    disable_inst = False
    graph_type = ["ER", "ER"]
    graph_config = [
        {"m": num_nodes * 2 * connection_factor if not disable_inst else 0, "directed": True},
        {"m": num_nodes * connection_factor, "directed": True},
    ]

    # in this case, all inst and lagged adj are dags and have total edges per adj = 2*num_nodes
    # graph_type = ["SF", "SF"]
    # graph_config = [{"m":2, "directed":True}, {"m":2, "directed":True}]

    lag = 2
    is_history_dep = True

    noise_level = 0.5
    function_type = "mlp"
    noise_function_type = "spline_product"
    base_noise_type = "gaussian"
    num_interventions = 5

    seed = 5
    set_random_seed(seed)
    folder_name = generate_name(
        num_nodes,
        graph_type,
        lag,
        is_history_dep,
        noise_level,
        function_type=function_type,
        noise_function_type=noise_function_type,
        disable_inst=disable_inst,
        seed=seed,
        connection_factor=connection_factor,
        intervention_history=0,
    )

    path = os.path.join(save_dir, folder_name)

    generate_cts_temporal_data(
        path=path,
        series_length=series_length,
        burnin_length=burnin_length,
        num_train_samples=num_train_samples,
        num_test_samples=num_test_samples,
        num_nodes=num_nodes,
        graph_type=graph_type,
        graph_config=graph_config,
        lag=lag,
        is_history_dep=is_history_dep,
        noise_level=noise_level,
        function_type=function_type,
        noise_function_type=noise_function_type,
        base_noise_type=base_noise_type,
        num_interventions=num_interventions,
        num_intervention_samples=num_intervention_samples,
        intervention_value=intervention_value,
    )

    # Save graph config
    with open(os.path.join(path, "graph_config.json"), "w", encoding="utf-8") as fout:
        json.dump(graph_config, fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Temporal Synthetic Data Generator")
    parser.add_argument("--save_dir", "-s", default="data")
    parser.add_argument("--num_nodes", "-n", default=5, type=int)
    parser.add_argument("--intervention_value", "-i", default=10, type=float)

    args = parser.parse_args()

    main(save_dir=args.save_dir, num_nodes=args.num_nodes, intervention_value=args.intervention_value)
