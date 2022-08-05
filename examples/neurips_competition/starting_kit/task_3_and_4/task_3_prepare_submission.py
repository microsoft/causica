import argparse

import numpy as np
import pandas as pd

from causica.baselines.varlingam import VARLiNGAM
from causica.models.deci.fold_time_deci import FoldTimeDECI

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dir", "-l", type=str)
    parser.add_argument("--mapping_path", "-m", type=str)
    parser.add_argument("--request_path", "-r", type=str)
    parser.add_argument("--model_type", "-mt", type=str, choices=["varlingam", "ft_deci"], default="ft_deci")
    parser.add_argument("--out_dir", "-o", type=str, default="adj_matrices.npy")
    parser.add_argument("--num_samples", "-n", type=int, default=5)

    args = parser.parse_args()

    construct_mapping = np.load(args.mapping_path, allow_pickle=True).item()

    requested_constructs = pd.read_csv(args.request_path)

    construct_indices = []
    for c in requested_constructs.values[:, 0]:
        # Find the indices of the requested constructs in the model;
        # Add 1 to account for the bot action
        construct_indices.append(construct_mapping[c] + 1)

    if args.model_type == "ft_deci":
        model_ft_deci: FoldTimeDECI = FoldTimeDECI.load("", args.load_dir, "cpu")

        raw_adj_mat = model_ft_deci.get_adj_matrix(samples=args.num_samples)
        num_variables = model_ft_deci.variables_orig.num_groups

        adj_mat = raw_adj_mat.reshape([-1, model_ft_deci.lag + 1, num_variables, model_ft_deci.lag + 1, num_variables])
        adj_mat = adj_mat.sum(3).sum(1) > 0
    elif args.model_type == "varlingam":
        model_varlingam: VARLiNGAM = VARLiNGAM.load("", args.load_dir, "cpu")

        # Generate summary adjacency matrix by aggregating across time
        adj_mat = (model_varlingam.get_adj_matrix().sum(0) > 0)[np.newaxis, :, :]
    else:
        raise ValueError(f"model type {args.model_type} is unknown")

    sub_matrix = adj_mat[:, :, construct_indices][:, construct_indices]
    np.save(args.out_dir, sub_matrix[np.newaxis, :])
