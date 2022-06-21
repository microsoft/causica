import argparse
import csv
import os

import numpy as np
from sklearn import datasets

DATASET_NAMES = ["boston", "wine"]


def get_args():
    parser = argparse.ArgumentParser(description="Downloads the dataset")

    parser.add_argument("dataset", choices=DATASET_NAMES, help="Dataset to download.")

    args = parser.parse_args()
    return args


def download_dataset(dataset_name: str):
    dataset = None
    if dataset_name == "boston":
        dataset = datasets.load_boston()
    elif dataset_name == "wine":
        dataset = datasets.load_wine()
    else:
        raise AttributeError("Not supported dataset for downloading")

    data = dataset["data"]
    target = dataset["target"]
    if target.ndim == 1:
        target = np.expand_dims(target, axis=1)  # Expand to give the same number of dims as data.

    combined = np.hstack((data, target))
    return combined


def save_dataset_to_csv(data, dataset_name: str):
    path = os.path.join("data", dataset_name, "all.csv")
    with open(path, "w", newline="\n", encoding="utf-8") as data_file:
        writer = csv.writer(data_file)
        writer.writerows(data)
    pass


def main():
    args = get_args()
    dataset_name = args.dataset

    dataset = download_dataset(dataset_name)
    save_dataset_to_csv(dataset, dataset_name)


if __name__ == "__main__":
    main()
