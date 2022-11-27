import argparse

import numpy as np

from causica.baselines.varlingam import VARLiNGAM

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dirs", "-l", type=str, nargs="+")
    parser.add_argument("--out_dir", "-o", type=str, default="adj_matrices.npy")

    args = parser.parse_args()

    mats = []
    for load_dir in args.load_dirs:
        model = VARLiNGAM.load("", load_dir, "cpu")

        # Generate summary adjacency matrix by aggregating across time
        adj_mat = model.get_adj_matrix().sum(0) > 0

        # Delete bot action column
        mats.append(adj_mat[1:, 1:])

    np.save(args.out_dir, np.stack(mats))
