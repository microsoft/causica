import os

import numpy as np

# How to generate dataset in correct format
# Download https://github.com/kurowasan/GraN-DAG/blob/master/data/syntren_p20.zip
# Copy the 20 files (DAGi.py, datai.py, for i=1,...,10) to the directory where this file is.
# Run python generate.py. This will create all the datasets (standardized and non standardized, fully-observed and
# 30% of training set MCAR)


def save_data(savedir, adj_matrix, X, num_samples_train):
    np.savetxt(os.path.join(savedir, "adj_matrix.csv"), adj_matrix, delimiter=",", fmt="%i")
    np.savetxt(os.path.join(savedir, "all.csv"), X, delimiter=",")
    np.savetxt(os.path.join(savedir, "train.csv"), X[:num_samples_train, :], delimiter=",")
    np.savetxt(os.path.join(savedir, "test.csv"), X[num_samples_train:, :], delimiter=",")


def main():
    num_samples_train = 400

    for i, r in enumerate(["seed_1", "seed_2", "seed_3", "seed_4", "seed_5"]):
        index = i + 1
        adj_matrix = np.load(f"DAG{index}.npy")
        X = np.load(f"data{index}.npy")  # Shape (500, 20)
        mean = np.mean(X[:num_samples_train, :], axis=0)
        std = np.std(X[:num_samples_train, :], axis=0)
        X = X - mean  # Always remove mean
        X_std = X / std

        np.random.seed(index)

        # Save nonstd data
        savedir = f"syntren_{r}"
        os.system(f"mkdir {savedir}")
        print(savedir)
        save_data(savedir, adj_matrix, X, num_samples_train)

        # Save std data
        savedir = f"syntren_{r}_std"
        os.system(f"mkdir {savedir}")
        print(savedir)
        save_data(savedir, adj_matrix, X_std, num_samples_train)

        # Generate missing data in training set
        mask = np.random.rand(*X.shape)
        mask[num_samples_train:, :] = 1
        mask[mask >= 0.3] = 1
        mask[mask < 0.3] = np.nan

        X_miss = X * mask
        X_miss_std = X_std * mask

        # Save nonstd data
        savedir = f"miss_syntren_{r}"
        os.system(f"mkdir {savedir}")
        print(savedir)
        save_data(savedir, adj_matrix, X_miss, num_samples_train)

        # Save std data
        savedir = f"miss_syntren_{r}_std"
        os.system(f"mkdir {savedir}")
        print(savedir)
        save_data(savedir, adj_matrix, X_miss_std, num_samples_train)


if __name__ == "__main__":
    main()
