import os

import numpy as np
import pandas as pd
import pytest
import torch

from causica.datasets.causal_csv_dataset_loader import CausalCSVDatasetLoader
from causica.models.deci.ddeci import ADMGParameterisedDDECI, BowFreeDDECI

from ...unit_tests.models.test_deci import (
    cts_and_discrete_variables,
    five_cts_ungrouped_variables,
    mixed_type_group,
    six_cts_grouped_variables,
    two_cat_one_group,
)


@pytest.fixture(name="model_config")
def fixture_model_config():
    return {
        "tau_gumbel": 1.0,
        "lambda_dag": 100.0,
        "lambda_sparse": 1.0,
        "base_distribution_type": "gaussian",
        "var_dist_A_mode": "enco",
        "inference_network_layer_sizes": [20],
        "imputer_layer_sizes": [20],
        "random_seed": [0],
    }


@pytest.mark.parametrize(
    "variables",
    [
        five_cts_ungrouped_variables,
        six_cts_grouped_variables,
        cts_and_discrete_variables,
        mixed_type_group,
        two_cat_one_group,
    ],
)
def test_variable_construction(tmpdir_factory, model_config, variables):
    model = ADMGParameterisedDDECI.create(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        device="cpu",
        model_config_dict=model_config,
    )

    d = variables.num_groups
    assert model.latent_dim_all == (d * (d - 1) / 2)

    for latent_variable in model.latent_variables:
        assert latent_variable.is_latent

    for observed_variable in model.observed_variables:
        assert not observed_variable.is_latent


@pytest.mark.parametrize(
    "variables",
    [
        five_cts_ungrouped_variables,
        six_cts_grouped_variables,
        cts_and_discrete_variables,
        mixed_type_group,
        two_cat_one_group,
    ],
)
def test_get_adj_matrices(tmpdir_factory, model_config, variables):
    model = ADMGParameterisedDDECI.create(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        device="cpu",
        model_config_dict=model_config,
    )

    d = variables.num_groups
    num_latent = int(d * (d - 1) / 2)

    directed_adj, bidirected_adj = model.get_admg_matrices(samples=1, most_likely_graph=True)
    assert directed_adj.shape == (1, d, d)
    assert bidirected_adj.shape == (1, d, d)

    adj = model.get_adj_matrix(samples=1, most_likely_graph=True)
    assert adj.shape == (1, d + num_latent, d + num_latent)

    adj_constructed = model.var_dist_A.magnify_adj_matrices(
        torch.as_tensor(directed_adj).squeeze(0), torch.as_tensor(bidirected_adj).squeeze(0)
    )
    assert np.all(adj_constructed.detach().cpu().numpy().astype(np.float64) == adj)


@pytest.mark.parametrize(
    "variables",
    [
        five_cts_ungrouped_variables,
        six_cts_grouped_variables,
        cts_and_discrete_variables,
        mixed_type_group,
        two_cat_one_group,
    ],
)
def test_bowfree_dagness_factor(tmpdir_factory, model_config, variables):
    model = BowFreeDDECI.create(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        device="cpu",
        model_config_dict=model_config,
    )

    d = variables.num_groups

    directed_adj = torch.triu(torch.rand(d, d)) * (1 - torch.eye(d))
    directed_adj = directed_adj.round()
    bidirected_adj = torch.triu(torch.ones(d, d) * (1 - directed_adj))
    bidirected_adj = bidirected_adj * (1.0 - torch.eye(d, d))
    bidirected_adj = bidirected_adj + torch.transpose(bidirected_adj, 0, 1)

    assert model.dagness_factor(model.var_dist_A.magnify_adj_matrices(directed_adj, bidirected_adj)) == 0


@pytest.fixture(name="dataset")
def example_dataset(tmpdir_factory):
    # Dataset with latent variables.
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    data = np.random.randn(100, 8)  # [100, 8]
    train_data = data[:60]
    val_data = data[60:90]
    test_data = data[90:]
    pd.DataFrame(train_data).to_csv(os.path.join(dataset_dir, "train.csv"), header=None, index=None)
    pd.DataFrame(val_data).to_csv(os.path.join(dataset_dir, "val.csv"), header=None, index=None)
    pd.DataFrame(test_data).to_csv(os.path.join(dataset_dir, "test.csv"), header=None, index=None)
    dataset_loader = CausalCSVDatasetLoader(dataset_dir=dataset_dir)

    dataset = dataset_loader.load_predefined_dataset(max_num_rows=None, column_index=0)

    return dataset


@pytest.fixture(name="train_config")
def example_train_config():
    train_config = {
        "learning_rate": 1e-2,
        "batch_size": 3,
        "stardardize_data_mean": False,
        "stardardize_data_std": False,
        "rho": 1.0,
        "safety_rho": 1e13,
        "alpha": 0.0,
        "safety_alpha": 1e13,
        "tol_dag": 1e-5,
        "progress_rate": 0.65,
        "max_steps_auglag": 1,
        "max_auglag_inner_epochs": 2,
        "max_p_train_dropout": 0,
        "reconstruction_loss_factor": 1.0,
        "anneal_entropy": "noanneal",
    }
    return train_config


def test_admg_ddeci_run_train(tmpdir_factory, dataset, model_config, train_config):
    """Tests if run_train runs smoothly."""

    model = ADMGParameterisedDDECI.create(
        model_id="model_id",
        variables=dataset.variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        device="cpu",
        model_config_dict=model_config,
    )
    model.run_train(dataset, train_config_dict=train_config)


def test_bowfree_ddeci_run_train(tmpdir_factory, dataset, model_config, train_config):
    """Tests if run_train runs smoothly."""

    model = BowFreeDDECI.create(
        model_id="model_id",
        variables=dataset.variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        device="cpu",
        model_config_dict=model_config,
    )
    model.run_train(dataset, train_config_dict=train_config)
