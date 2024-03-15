"""This script is to generate synthetic dataset for SCOTCH experiments."""
import os
import sys
from argparse import Namespace

import pandas as pd
import torch
from scipy.io import loadmat
from scotch.dataset_generation.example_sdes.lorenz_sde import Lorenz96SDE
from scotch.dataset_generation.example_sdes.yeast_glycolysis import YeastGlycolysisSDE
from scotch.dataset_generation.generate_trajectories import generate_and_return_trajectories


def subsample(ts, data, samp_every):
    subsamp_ts = ts[::samp_every]
    subsamp_data = data[:, ::samp_every, :]
    return subsamp_ts, subsamp_data


def generate_missing_data(ts, data, missing_probability):
    mask = torch.rand(data.shape[1]) > missing_probability
    missing_ts = ts[mask]
    missing_data = data[:, mask, :]
    return missing_ts, missing_data


def load_data_from_file(arguments):
    match arguments.dataset:
        case "lorenz":
            z0_train = torch.randn(size=(arguments.train_size, arguments.dimension), device="cuda")
            ts, training_data, _ = generate_and_return_trajectories(
                Lorenz96SDE,
                z0_train,
                num_time_points=arguments.num_time_points,
                t_max=arguments.t_max,
                noise_scale=arguments.noise_scale,
                F=arguments.forcing,
                return_raw=True,
                dt=0.005,
            )
        case "yeast":

            variable_ranges = torch.tensor(
                [[0.15, 1.60], [0.19, 2.16], [0.04, 0.20], [0.10, 0.35], [0.08, 0.30], [0.14, 2.67], [0.05, 0.10]],
                device="cuda",
            )
            unif_samples = torch.rand(size=(arguments.train_size, 7), device="cuda")
            z0_train = variable_ranges[:, 0] + (variable_ranges[:, 1] - variable_ranges[:, 0]) * unif_samples
            ts, training_data, _ = generate_and_return_trajectories(
                YeastGlycolysisSDE,
                z0_train,
                num_time_points=arguments.num_time_points,
                t_max=arguments.t_max,
                noise_scale=arguments.noise_scale,
                return_raw=True,
                dt=0.01,
            )
        case "netsim":
            training_data_dict = loadmat("<path to netsim data mat file>")
            ts = torch.linspace(0, 10, 200)
            training_data = torch.tensor(training_data_dict["ts"], device="cuda").view(50, 200, 15)[1:6, :, :]

    return ts, training_data


def load_graph_from_file(arguments):
    match arguments.dataset:
        case "netsim":
            training_data_dict = loadmat("<path to netsim data mat file>")
            true_graph = (torch.tensor(training_data_dict["net"], device="cuda")[0] != 0).long()
        case "lorenz":
            true_graph = Lorenz96SDE.graph(arguments.dimension).to("cuda")
        case "yeast":
            true_graph = YeastGlycolysisSDE.graph().to("cuda")

    return true_graph


def save_data_and_graph_to_file(arguments, ts, training_data, true_graph):
    match arguments.dataset:
        case "netsim":
            subf = "norm" if arguments.normalize else "unnorm"
            os.makedirs(f"data/netsim_processed_new/{subf}", exist_ok=True)
            torch.save(ts, f"data/netsim_processed_new/{subf}/times_{str(arguments.missing_prob)}_{arguments.seed}.pt")
            torch.save(
                training_data,
                f"data/netsim_processed_new/{subf}/data_{str(arguments.missing_prob)}_{arguments.seed}.pt",
            )
            torch.save(true_graph, f"data/netsim_processed_new/{subf}/true_graph.pt")
        case "lorenz":
            subf = "norm" if arguments.normalize else "unnorm"
            os.makedirs(f"data/lorenz96_processed/{arguments.dimension}/{subf}", exist_ok=True)
            torch.save(
                ts,
                f"data/lorenz96_processed/{arguments.dimension}/{subf}/times_{arguments.num_subsamp}_{str(arguments.missing_prob)}_{arguments.seed}.pt",
            )
            torch.save(
                training_data,
                f"data/lorenz96_processed/{arguments.dimension}/{subf}/data_{arguments.num_subsamp}_{str(arguments.missing_prob)}_{arguments.seed}.pt",
            )
            torch.save(true_graph, f"data/lorenz96_processed/{arguments.dimension}/{subf}/true_graph.pt")
        case "yeast":
            subf = "norm" if arguments.normalize else "unnorm"
            os.makedirs(f"data/yeast_processed/{arguments.dimension}/{subf}", exist_ok=True)
            torch.save(
                ts,
                f"data/yeast_processed/{arguments.dimension}/{subf}/times_{arguments.num_subsamp}_{str(arguments.missing_prob)}_{arguments.seed}.pt",
            )
            torch.save(
                training_data,
                f"data/yeast_processed/{arguments.dimension}/{subf}/data_{arguments.num_subsamp}_{str(arguments.missing_prob)}_{arguments.seed}.pt",
            )
            torch.save(true_graph, f"data/yeast_processed/{arguments.dimension}/{subf}/true_graph.pt")


def generate_data_standard(arguments, save_data=False):
    torch.manual_seed(arguments.seed)

    ts, training_data = load_data_from_file(arguments)
    _, num_time_points, _ = training_data.shape
    true_graph = load_graph_from_file(arguments)

    training_data = (
        (training_data - training_data.mean(dim=(0, 1))) / training_data.std(dim=(0, 1))
        if arguments.normalize
        else training_data
    )

    # create missing data
    if arguments.dataset in ("lorenz", "yeast"):
        ts, training_data = subsample(ts, training_data, num_time_points // arguments.num_subsamp)

    ts, training_data = generate_missing_data(ts, training_data, arguments.missing_prob)

    if save_data:
        save_data_and_graph_to_file(arguments, ts, training_data, true_graph)

    return ts, training_data, true_graph


if __name__ == "__main__":
    sys.setrecursionlimit(2000)
    gen_netsim_data = False
    print("Generating Yeast Glycolysis Dataset")
    for seed in [0, 1, 2, 3, 4]:
        for normalize in [False, True]:
            for num_subsamp in [100, 200, 500, 1000]:
                for missing_prob in [0.0, 0.1, 0.2, 0.5]:
                    print(
                        f"Seed: {seed}, Normalize: {normalize}, Num Subsamp: {num_subsamp}, Missing Prob: {missing_prob}"
                    )
                    args = Namespace(
                        normalize=normalize,
                        dataset="yeast",
                        missing_prob=missing_prob,
                        seed=seed,
                        train_size=100,
                        num_time_points=1000,
                        t_max=100.0,
                        noise_scale=0.01,
                        dimension=7,
                        num_subsamp=num_subsamp,
                    )
                    generate_data_standard(args, save_data=True)

    print("Generating Lorenz Dataset")
    for seed in [0, 1, 2, 3, 4]:
        for normalize in [False, True]:
            for num_subsamp in [1000, 500, 200, 100]:
                for missing_prob in [0.0, 0.3, 0.6]:
                    print(
                        f"Seed: {seed}, Normalize: {normalize}, Num Subsamp: {num_subsamp}, Missing Prob: {missing_prob}"
                    )
                    args = Namespace(
                        normalize=normalize,
                        dataset="lorenz",
                        missing_prob=missing_prob,
                        seed=seed,
                        forcing=10,
                        train_size=100,
                        num_time_points=1000,
                        t_max=100.0,
                        noise_scale=0.5,
                        dimension=10,
                        num_subsamp=num_subsamp,
                    )
                    generate_data_standard(args, save_data=True)
    if gen_netsim_data:
        print("Generating Netsim Dataset")
        for seed in [0, 1, 2, 3, 4]:
            for normalize in [False, True]:
                for missing_prob in [0.0, 0.1, 0.2]:
                    print(f"Seed: {seed}, Normalize: {normalize}, Missing Prob: {missing_prob}")
                    args = Namespace(normalize=normalize, dataset="netsim", missing_prob=missing_prob, seed=seed)
                    generate_data_standard(args, save_data=True)
