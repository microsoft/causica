import os

import numpy as np

# How to generate dataset in correct format
# 1. Download https://github.com/kurowasan/GraN-DAG/blob/master/data/sachs.zip
# 2. Extract, and move the files from the continue sub-directory (DAG1.npy, data1.npy) and place them
# in the directory where this file is.
# 3. Run python generate.py. This will create all the datasets (standardized and non standardized,
# fully-observed and 30% of training set MCAR)


def save_data(savedir, adj_matrix, X, indices_train, indices_test):
    np.savetxt(os.path.join(savedir, "adj_matrix.csv"), adj_matrix, delimiter=",", fmt="%i")
    np.savetxt(os.path.join(savedir, "all.csv"), X, delimiter=",")
    np.savetxt(os.path.join(savedir, "train.csv"), X[indices_train, :], delimiter=",")
    np.savetxt(os.path.join(savedir, "test.csv"), X[indices_test, :], delimiter=",")


def main():
    np.random.seed(1)

    adj_matrix = np.load("DAG1.npy")
    print(adj_matrix.shape)

    X = np.load("data1.npy")  # Shape (500, 20)
    print(X.shape)

    num_samples_train = 800

    for i in range(5):
        np.random.seed(i * 5)
        indices = np.arange(X.shape[0])
        indices = np.random.permutation(indices)
        indices_train = indices[:num_samples_train]
        indices_test = indices[num_samples_train:]

        mean = np.mean(X[:num_samples_train, :], axis=0)
        std = np.std(X[:num_samples_train, :], axis=0)
        X = X - mean  # Always remove mean
        X_std = X / std

        # Save data and std
        savedir = f"protein_cells_seed_{i + 1}"
        os.system(f"mkdir {savedir}")
        print(savedir)
        save_data(savedir, adj_matrix, X, indices_train, indices_test)

        # Save std data
        savedir = f"protein_cells_seed_{i + 1}_std"
        os.system(f"mkdir {savedir}")
        print(savedir)
        save_data(savedir, adj_matrix, X_std, indices_train, indices_test)

        # Generate missing data for training samples only
        mask = np.random.rand(*X.shape)
        mask[num_samples_train:, :] = 1
        mask[mask >= 0.3] = 1
        mask[mask < 0.3] = np.nan

        X_miss = X * mask
        X_miss_std = X_std * mask

        # Save non-std data
        savedir = f"miss_protein_cells_seed_{i + 1}"
        os.system(f"mkdir {savedir}")
        print(savedir)
        save_data(savedir, adj_matrix, X_miss, indices_train, indices_test)

        # Save std data
        savedir = f"miss_protein_cells_seed_{i + 1}_std"
        os.system(f"mkdir {savedir}")
        print(savedir)
        save_data(savedir, adj_matrix, X_miss_std, indices_train, indices_test)


if __name__ == "__main__":
    main()
