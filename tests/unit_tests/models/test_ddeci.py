import os

import numpy as np
import pandas as pd
import pytest
import torch

from causica.datasets.causal_csv_dataset_loader import CausalCSVDatasetLoader
from causica.models.deci.ddeci import DDECI, _add_latent_variables

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
def test_log_prob(tmpdir_factory, model_config, variables):
    d = variables.num_processed_cols
    all_variables = _add_latent_variables(variables, 1)

    model = DDECI.create(
        model_id="model_id",
        variables=all_variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        device="cpu",
        model_config_dict=model_config,
    )

    assert (
        model.variables.num_processed_cols
        == model.observed_variables.num_processed_cols + model.latent_variables.num_processed_cols
    )

    x = torch.rand(10, d)
    for region, variable in zip(variables.processed_cols, variables):
        if variable.type_ == "categorical":
            maxes = x[:, region].max(-1, keepdim=True)[0]
            x[:, region] = (x[:, region] >= maxes).float()
        if variable.type_ == "binary":
            x[:, region] = (x[:, region] > 0.5).float()

    deterministic_log_p0 = model.log_prob(x, Nsamples_per_graph=100, most_likely_graph=True, most_likely_u=True)

    deterministic_log_p1 = model.log_prob(x, Nsamples_per_graph=1, most_likely_graph=True, most_likely_u=True)

    assert np.allclose(deterministic_log_p1, deterministic_log_p0)

    # This just tests that the stochastic mode runs
    _ = model.log_prob(x, Nsamples_per_graph=10, most_likely_graph=False)

    intervention_idxs = torch.tensor([0])
    if variables[0].group_name is not None:
        intervention_values = torch.tensor([0.0 for v in variables if v.group_name == variables[0].group_name])
    else:
        intervention_values = torch.tensor([0.0])

    deterministic_intervention_log_p = model.log_prob(
        x,
        most_likely_graph=True,
        intervention_idxs=intervention_idxs,
        intervention_values=intervention_values,
    )

    assert not np.allclose(deterministic_log_p1, deterministic_intervention_log_p)

    # Test the reconstruction dimensions are consistent.
    rec_x = model.reconstruct_x(x, most_likely_graph=True)
    assert rec_x.shape == x.shape

    all_continuous = np.all([var.type_ == "continuous" for var in variables])
    if all_continuous:
        # Test scale of imputer distribution.
        mask = torch.ones_like(x)
        _, qx_scale = model.get_params_variational_distribution(x, mask)
        assert np.all(qx_scale.detach().numpy() > 0.0)

    # Test modification of intervention indices and values.
    qu_x = model.inference_network(x)
    u = qu_x.sample()
    intervention_idxs_all, intervention_values_all = model.add_latents_to_intervention(
        u, intervention_idxs, intervention_values
    )

    assert intervention_idxs_all[: len(intervention_idxs)] == intervention_idxs
    assert intervention_idxs_all[-model.latent_dim_all :] == torch.as_tensor(
        list(range(model.num_nodes - model.latent_dim_all, model.num_nodes))
    )
    assert torch.all(intervention_values_all[:, : len(intervention_values)] == intervention_values)
    assert torch.all(intervention_values_all[:, -model.latent_dim_all :] == u)


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
def test_ate(tmpdir_factory, model_config, variables):
    d = variables.num_processed_cols
    all_variables = _add_latent_variables(variables, 1)

    model = DDECI.create(
        model_id="model_id",
        variables=all_variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        device="cpu",
        model_config_dict=model_config,
    )

    intervention_idxs = torch.tensor([0])
    if variables[0].group_name is not None:
        intervention_values = torch.tensor([0.0 for v in variables if v.group_name == variables[0].group_name])
    else:
        intervention_values = torch.tensor([0.0])

    cate0, cate0_norm = model.cate(
        intervention_idxs,
        intervention_values,
        conditioning_idxs=None,
        conditioning_values=None,
        Nsamples_per_graph=2,
        Ngraphs=1000,
        most_likely_graph=False,
    )

    cate1, cate1_norm = model.cate(
        intervention_idxs,
        intervention_values,
        conditioning_idxs=None,
        conditioning_values=None,
        Nsamples_per_graph=2,
        Ngraphs=1000,
        most_likely_graph=False,
    )

    #  Note that graph is deterministic here but samples are still random
    cate_det0, cate_det0_norm = model.cate(
        intervention_idxs,
        intervention_values,
        conditioning_idxs=None,
        conditioning_values=None,
        Nsamples_per_graph=2000,
        Ngraphs=1,
        most_likely_graph=True,
    )

    cate_det1, cate_det1_norm = model.cate(
        intervention_idxs,
        intervention_values,
        conditioning_idxs=None,
        conditioning_values=None,
        Nsamples_per_graph=2000,
        Ngraphs=1,
        most_likely_graph=True,
    )

    assert cate0.shape == (d,)
    assert cate0_norm.shape == (d,)

    # Test values for continuous only
    assert np.all((cate0 - cate1)[variables.continuous_idxs] != 0)
    assert np.all((cate0_norm - cate1_norm)[variables.continuous_idxs] != 0)

    assert np.all((cate_det0 - cate_det1)[variables.continuous_idxs] != 0)
    assert np.all((cate_det0_norm - cate_det1_norm)[variables.continuous_idxs] != 0)

    with pytest.raises(AssertionError, match="Nsamples_per_graph must be greater than 1"):
        model.cate(intervention_idxs, intervention_values, Nsamples_per_graph=1)


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
def test_ite(tmpdir_factory, model_config, variables):
    all_variables = _add_latent_variables(variables, 1)

    model = DDECI.create(
        model_id="model_id",
        variables=all_variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        device="cpu",
        model_config_dict=model_config,
    )

    base_samples = model.sample(Nsamples=10, most_likely_graph=False)

    intervention_idxs = torch.tensor([0])
    if variables[0].group_name is not None:
        intervention_values = torch.tensor([0.0 for v in variables if v.group_name == variables[0].group_name])
    else:
        intervention_values = torch.tensor([0.0])

    ite0, _ = model.ite(base_samples, intervention_idxs, intervention_values, Ngraphs=100, most_likely_graph=False)

    ite1, _ = model.ite(base_samples, intervention_idxs, intervention_values, Ngraphs=100, most_likely_graph=False)

    #  Note that graph is deterministic here but samples are still random
    ite_det0, _ = model.ite(base_samples, intervention_idxs, intervention_values, Ngraphs=1, most_likely_graph=True)

    ite_det1, _ = model.ite(base_samples, intervention_idxs, intervention_values, Ngraphs=1, most_likely_graph=True)

    assert ite0.shape == base_samples.shape

    # Test values for continuous only
    assert np.any(ite0[variables.continuous_idxs] != 0)
    assert np.any(ite1[variables.continuous_idxs] != 0)
    assert np.any(ite_det0[variables.continuous_idxs] != 0)
    assert np.any(ite_det1[variables.continuous_idxs] != 0)
    # This test checks whether ITE's using different graphs are different.
    # This seems to fail?
    # assert np.all((ite0 - ite1)[variables.continuous_idxs] != 0)
    # This test checks whether ITE's using the same graph are the same.
    np.testing.assert_allclose(ite_det0, ite_det1)

    intervention_idxs = torch.tensor([0])
    if variables[0].group_name is not None:
        intervention_values = torch.tensor([10.0 for v in variables if v.group_name == variables[0].group_name])
        reference_values = torch.tensor([1.0 for v in variables if v.group_name == variables[0].group_name])
    else:
        intervention_values = torch.tensor([10.0])
        reference_values = torch.tensor([1.0])

    ite, _ = model.ite(
        base_samples,
        intervention_idxs,
        intervention_values,
        reference_values=reference_values,
        most_likely_graph=True,
        Ngraphs=1,
    )

    if len(intervention_values) > 1:
        intervention_idxs = torch.tensor(
            [i for i, v in enumerate(variables) if v.group_name == variables[0].group_name]
        )
        np.testing.assert_array_almost_equal(
            ite[:, intervention_idxs], (intervention_values - reference_values).expand(len(ite), -1)
        )
    else:
        np.testing.assert_array_almost_equal(
            ite[:, intervention_idxs], (intervention_values - reference_values).expand(len(ite))
        )


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
def test_ite_no_intervention(tmpdir_factory, model_config, variables):
    all_variables = _add_latent_variables(variables, 1)

    model = DDECI.create(
        model_id="model_id",
        variables=all_variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        device="cpu",
        model_config_dict=model_config,
    )

    samples = model.sample(100, most_likely_graph=True, samples_per_graph=100)
    ite, _ = model.ite(samples, most_likely_graph=True, Ngraphs=1)
    assert abs(ite.max()) < 1e-5

    # Should still work, as no intervention is applied
    ite2, _ = model.ite(samples, most_likely_graph=False)
    assert abs(ite2.max()) < 1e-5


@pytest.mark.parametrize(
    "base_offset,deltas",
    [
        (0.0, torch.tensor([0.0, 0.0, 0.0])),
        (1.0, torch.tensor([0.0, 0.0, 0.0])),
        (0.0, torch.tensor([0.1, 0.5, 1.0])),
        (1.0, torch.tensor([1.0, 2.0, 3.0])),
        (1.0, torch.tensor([-1.0, 2.0, -1.0])),
    ],
)
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


def test_run_train(tmpdir_factory, dataset, model_config, train_config):
    """Tests if run_train runs smoothly."""

    all_variables = _add_latent_variables(dataset.variables, 1)

    model = DDECI.create(
        model_id="model_id",
        variables=all_variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        device="cpu",
        model_config_dict=model_config,
    )
    model.run_train(dataset, train_config_dict=train_config)
