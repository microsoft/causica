import logging
import os
from unittest.mock import patch

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from causica.baselines.end2end_causal.true_graph_dowhy import TrueGraphDoWhy
from causica.datasets.dataset import Dataset, SparseDataset
from causica.datasets.intervention_data import InterventionData
from causica.datasets.variables import Variable, Variables
from causica.experiment.steps.eval_step import (
    eval_causal_discovery,
    eval_individual_treatment_effects,
    eval_latent_confounded_causal_discovery,
    eval_treatment_effects,
    evaluate_treatment_effect_estimation,
    run_eval_main,
)
from causica.models.deci.ddeci import ADMGParameterisedDDECI, BowFreeDDECI
from causica.models.deci.deci import DECI
from causica.models.deci.fold_time_deci import FoldTimeDECI
from causica.models.deci.rhino import Rhino
from causica.models.imodel import IModelForCausalInference, IModelForCounterfactuals, IModelForInterventions
from causica.utils.io_utils import read_json_as, save_json

from ..utils.mock_model_for_objective import MockModelForObjective

# pylint: disable=redefined-outer-name


@pytest.fixture(scope="function")
def model(variables, tmpdir_factory):
    return MockModelForObjective("mock_model", variables, tmpdir_factory.mktemp("save_dir"))


@pytest.fixture(scope="function")
def variables():
    return Variables(
        [
            Variable("cts_input", True, "continuous", 0.0, 5.0),
            Variable("binary_input", True, "binary", 0, 1),
            Variable("cts_target", False, "continuous", 0.0, 5.0),
        ]
    )


def test_run_eval_main_user_id_too_high(model, variables):
    impute_config = {"sample_count": 100, "batch_size": 100}
    dataset = Dataset(
        train_data=np.ones((5, 3)),
        train_mask=np.ones((5, 3), dtype=bool),
        val_data=np.ones((5, 3)),
        val_mask=np.ones((5, 3), dtype=bool),
        test_data=np.ones((1, 3)),
        test_mask=np.ones((1, 3), dtype=bool),
        variables=variables,
    )
    with pytest.raises(AssertionError):
        run_eval_main(
            model=model,
            dataset=dataset,
            vamp_prior_data=None,
            impute_config=impute_config,
            extra_eval=False,
            seed=0,
            user_id=1,
            impute_train_data=True,
        )


@pytest.mark.skip("Removed PVAE")
def test_run_eval_main(model, variables):
    impute_config = {"sample_count": 100, "batch_size": 100}
    dataset = Dataset(
        train_data=np.ones((5, 3)),
        train_mask=np.ones((5, 3), dtype=bool),
        val_data=np.ones((5, 3)),
        val_mask=np.ones((5, 3), dtype=bool),
        test_data=np.ones((5, 3)),
        test_mask=np.ones((5, 3), dtype=bool),
        variables=variables,
    )

    vamp_prior_info_gain = [{"0": 0.000621266255620867, "1": 0.0009173454018309712}]
    save_json(vamp_prior_info_gain, os.path.join(model.save_dir, "vamp_prior_info_gain.json"))

    run_eval_main(
        model=model,
        dataset=dataset,
        vamp_prior_data=None,
        impute_config=impute_config,
        extra_eval=True,
        seed=0,
        user_id=1,
        impute_train_data=True,
    )
    results = read_json_as(os.path.join(model.save_dir, "results.json"), dict)
    assert "RMSE" in results["train_data"]["cts_input"]
    assert "RMSE" in results["test_data"]["cts_input"]
    assert "RMSE" in results["val_data"]["cts_input"]
    target_results = read_json_as(os.path.join(model.save_dir, "target_results.json"), dict)
    assert "RMSE" in target_results["train_data"]["cts_target"]
    assert "RMSE" in target_results["test_data"]["cts_target"]
    assert "RMSE" in target_results["val_data"]["cts_target"]
    assert os.path.isfile(os.path.join(model.save_dir, "imputed_values_violin_plot_user1.png"))


@pytest.mark.skip("Removed PVAE")
def test_run_eval_main_no_val_data(model, variables):
    impute_config = {"sample_count": 100, "batch_size": 100}
    dataset = Dataset(
        train_data=np.ones((5, 3)),
        train_mask=np.ones((5, 3), dtype=bool),
        val_data=None,
        val_mask=None,
        test_data=np.ones((5, 3)),
        test_mask=np.ones((5, 3), dtype=bool),
        variables=variables,
    )
    run_eval_main(
        model=model,
        dataset=dataset,
        vamp_prior_data=None,
        impute_config=impute_config,
        extra_eval=False,
        seed=0,
        user_id=1,
        impute_train_data=True,
    )
    results = read_json_as(os.path.join(model.save_dir, "results.json"), dict)
    assert "RMSE" in results["train_data"]["cts_input"]
    assert "RMSE" in results["test_data"]["cts_input"]
    assert results["val_data"] == {}
    target_results = read_json_as(os.path.join(model.save_dir, "target_results.json"), dict)
    assert "RMSE" in target_results["train_data"]["cts_target"]
    assert "RMSE" in target_results["test_data"]["cts_target"]
    assert target_results["val_data"] == {}


@pytest.mark.skip("Removed PVAE")
def test_run_eval_main_impute_train_data_false(model, variables):
    impute_config = {"sample_count": 100, "batch_size": 100}
    dataset = Dataset(
        train_data=np.ones((5, 3)),
        train_mask=np.ones((5, 3), dtype=bool),
        val_data=np.ones((5, 3)),
        val_mask=np.ones((5, 3), dtype=bool),
        test_data=np.ones((5, 3)),
        test_mask=np.ones((5, 3), dtype=bool),
        variables=variables,
    )
    run_eval_main(
        model=model,
        dataset=dataset,
        vamp_prior_data=None,
        impute_config=impute_config,
        extra_eval=False,
        seed=0,
        user_id=1,
        impute_train_data=False,
    )
    results = read_json_as(os.path.join(model.save_dir, "results.json"), dict)
    assert results["train_data"] == {}
    assert "RMSE" in results["test_data"]["cts_input"]
    assert "RMSE" in results["val_data"]["cts_input"]
    target_results = read_json_as(os.path.join(model.save_dir, "target_results.json"), dict)
    assert target_results["train_data"] == {}
    assert "RMSE" in target_results["test_data"]["cts_target"]
    assert "RMSE" in target_results["val_data"]["cts_target"]


@pytest.mark.skip("Removed PVAE")
def test_run_eval_main_no_target_var_idxs(tmpdir_factory):
    impute_config = {"sample_count": 100, "batch_size": 100}
    variables = Variables(
        [
            Variable("cts_input", True, "continuous", 0.0, 5.0),
            Variable("binary_input", True, "binary", 0, 1),
            Variable("cts_target", True, "continuous", 0.0, 5.0),
        ]
    )

    model = MockModelForObjective("mock_model", variables, tmpdir_factory.mktemp("save_dir"))
    dataset = Dataset(
        train_data=np.ones((5, 3)),
        train_mask=np.ones((5, 3), dtype=bool),
        val_data=np.ones((5, 3)),
        val_mask=np.ones((5, 3), dtype=bool),
        test_data=np.ones((5, 3)),
        test_mask=np.ones((5, 3), dtype=bool),
        variables=variables,
    )
    vamp_prior_info_gain = [{"0": 0.000621266255620867, "1": 0.0009173454018309712, "2": 0.0009173454018309712}]
    save_json(vamp_prior_info_gain, os.path.join(model.save_dir, "vamp_prior_info_gain.json"))

    run_eval_main(
        model=model,
        dataset=dataset,
        vamp_prior_data=None,
        impute_config=impute_config,
        extra_eval=False,
        seed=0,
        user_id=1,
        impute_train_data=True,
    )
    assert os.path.isfile(os.path.join(model.save_dir, "difficulty.csv"))
    assert os.path.isfile(os.path.join(model.save_dir, "quality.csv"))


def test_run_eval_main_sparse_dataset(model, variables):
    impute_config = {"sample_count": 100, "batch_size": 100}
    dataset = SparseDataset(
        train_data=csr_matrix(np.ones((5, 3))),
        train_mask=csr_matrix(np.ones((5, 3), dtype=bool)),
        val_data=csr_matrix(np.ones((5, 3))),
        val_mask=csr_matrix(np.ones((5, 3), dtype=bool)),
        test_data=csr_matrix(np.ones((5, 3))),
        test_mask=csr_matrix(np.ones((5, 3), dtype=bool)),
        variables=variables,
    )

    vamp_prior_info_gain = [{"0": 0.000621266255620867, "1": 0.0009173454018309712}]
    save_json(vamp_prior_info_gain, os.path.join(model.save_dir, "vamp_prior_info_gain.json"))

    run_eval_main(
        model=model,
        dataset=dataset,
        vamp_prior_data=None,
        impute_config=impute_config,
        extra_eval=True,
        seed=0,
        user_id=1,
        impute_train_data=True,
    )
    results = read_json_as(os.path.join(model.save_dir, "results.json"), dict)
    assert "RMSE" in results["train_data"]["cts_input"]
    assert "RMSE" in results["test_data"]["cts_input"]
    assert "RMSE" in results["val_data"]["cts_input"]
    target_results = read_json_as(os.path.join(model.save_dir, "target_results.json"), dict)
    assert "RMSE" in target_results["train_data"]["cts_target"]
    assert "RMSE" in target_results["test_data"]["cts_target"]
    assert "RMSE" in target_results["val_data"]["cts_target"]
    assert not os.path.isfile(os.path.join(model.save_dir, "imputed_values_violin_plot_user1.png"))


def test_run_eval_main_sparse_dataset_no_target_variables(tmpdir_factory):
    impute_config = {"sample_count": 100, "batch_size": 100}
    variables = Variables(
        [
            Variable("cts_input", True, "continuous", 0.0, 5.0),
            Variable("binary_input", True, "binary", 0, 1),
            Variable("cts_target", True, "continuous", 0.0, 5.0),
        ]
    )

    model = MockModelForObjective("mock_model", variables, tmpdir_factory.mktemp("save_dir"))

    dataset = SparseDataset(
        train_data=csr_matrix(np.ones((5, 3))),
        train_mask=csr_matrix(np.ones((5, 3), dtype=bool)),
        val_data=csr_matrix(np.ones((5, 3))),
        val_mask=csr_matrix(np.ones((5, 3), dtype=bool)),
        test_data=csr_matrix(np.ones((5, 3))),
        test_mask=csr_matrix(np.ones((5, 3), dtype=bool)),
        variables=variables,
    )
    vamp_prior_info_gain = [{"0": 0.000621266255620867, "1": 0.0009173454018309712, "2": 0.0009173454018309712}]
    save_json(vamp_prior_info_gain, os.path.join(model.save_dir, "vamp_prior_info_gain.json"))

    run_eval_main(
        model=model,
        dataset=dataset,
        vamp_prior_data=None,
        impute_config=impute_config,
        extra_eval=False,
        seed=0,
        user_id=1,
        impute_train_data=True,
    )
    assert not os.path.isfile(os.path.join(model.save_dir, "difficulty.csv"))
    assert not os.path.isfile(os.path.join(model.save_dir, "quality.csv"))


def test_run_eval_main_sparse_dataset_no_targets_elements_split(tmpdir_factory):
    impute_config = {"sample_count": 100, "batch_size": 100}
    variables = Variables(
        [
            Variable("cts_input", True, "continuous", 0.0, 5.0),
            Variable("binary_input", True, "binary", 0, 1),
            Variable("cts_target", True, "continuous", 0.0, 5.0),
        ]
    )
    model = MockModelForObjective("mock_model", variables, tmpdir_factory.mktemp("save_dir"))

    dataset = SparseDataset(
        train_data=csr_matrix(1 - np.eye(5, 3)),
        train_mask=csr_matrix(~np.eye(5, 3, dtype=bool)),
        val_data=None,
        val_mask=None,
        test_data=csr_matrix(np.eye(5, 3)),
        test_mask=csr_matrix(np.eye(5, 3, dtype=bool)),
        variables=variables,
    )
    vamp_prior_info_gain = [{"0": 0.000621266255620867, "1": 0.0009173454018309712, "2": 0.0009173454018309712}]
    save_json(vamp_prior_info_gain, os.path.join(model.save_dir, "vamp_prior_info_gain.json"))

    run_eval_main(
        model=model,
        dataset=dataset,
        vamp_prior_data=None,
        impute_config=impute_config,
        extra_eval=False,
        split_type="elements",
        seed=0,
        user_id=1,
        impute_train_data=True,
    )
    assert not os.path.isfile(os.path.join(model.save_dir, "difficulty.csv"))
    assert not os.path.isfile(os.path.join(model.save_dir, "quality.csv"))


def test_run_eval_causal_discovery(tmpdir_factory):
    variables = Variables(
        [
            Variable("continuous_input_1", True, "continuous", 0, 1),
            Variable("continuous_input_2", True, "continuous", 0, 1),
            Variable("continuous_input_3", True, "continuous", 0, 1),
        ]
    )

    model_config = {
        "tau_gumbel": 0.25,
        "lambda_dag": 100.0,
        "lambda_sparse": 5.0,
        "base_distribution_type": "gaussian",
        "spline_bins": 8,
        "var_dist_A_mode": "enco",
        "mode_adjacency": "learn",
        "random_seed": [0],
    }

    model = DECI.create(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        model_config_dict=model_config,
        device="cpu",
    )

    dataset = Dataset(
        train_data=np.ones((5, 3)),
        train_mask=np.ones((5, 3), dtype=bool),
        val_data=np.ones((5, 3)),
        val_mask=np.ones((5, 3), dtype=bool),
        test_data=np.ones((5, 3)),
        test_mask=np.ones((5, 3), dtype=bool),
        variables=variables,
    )

    adj_matrix = np.zeros((3, 3))
    adj_mask = np.ones((3, 3))
    causal_dataset = dataset.to_causal(adjacency_data=adj_matrix, subgraph_data=adj_mask, intervention_data=None)

    assert isinstance(model, IModelForCausalInference)
    eval_causal_discovery(causal_dataset, model)

    results = read_json_as(os.path.join(model.save_dir, "target_results_causality.json"), dict)

    assert "adjacency_precision" in results["test_data"]
    assert "adjacency_fscore" in results["test_data"]
    assert "orientation_recall" in results["test_data"]
    assert "orientation_precision" in results["test_data"]
    assert "orientation_fscore" in results["test_data"]
    assert "causal_accuracy" in results["test_data"]
    assert "shd" in results["test_data"]
    assert "nnz" in results["test_data"]

    # Test for temporal causal model
    temporal_model_config = {
        "lag": 2,
        "allow_instantaneous": True,
        "treat_continuous": True,
        "tau_gumbel": 0.25,
        "lambda_dag": 100.0,
        "lambda_sparse": 5.0,
        "base_distribution_type": "gaussian",
        "spline_bins": 8,
        "var_dist_A_mode": "enco",
        "mode_adjacency": "learn",
        "random_seed": [0],
    }
    temporal_model = FoldTimeDECI.create(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        device="cpu",
        model_config_dict=temporal_model_config,
    )
    temporal_adj_matrix = np.zeros((5, 3, 3))
    temporal_causal_dataset = dataset.to_temporal(
        adjacency_data=temporal_adj_matrix, intervention_data=None, transition_matrix=None, counterfactual_data=None
    )
    eval_causal_discovery(temporal_causal_dataset, temporal_model)

    # test ar-deci model
    temporal_model_config = {
        "lag": 2,
        "allow_instantaneous": True,
        "tau_gumbel": 0.25,
        "lambda_dag": 100.0,
        "lambda_sparse": 5.0,
        "base_distribution_type": "gaussian",
        "spline_bins": 8,
        "var_dist_A_mode": "temporal_three",
        "random_seed": [0],
    }
    temporal_model = Rhino.create(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        device="cpu",
        model_config_dict=temporal_model_config,
    )
    eval_causal_discovery(temporal_causal_dataset, temporal_model)


def test_run_eval_latent_confounded_causal_discovery(tmpdir_factory):
    variables = Variables(
        [
            Variable("continuous_input_1", True, "continuous", 0, 1),
            Variable("continuous_input_2", True, "continuous", 0, 1),
            Variable("continuous_input_3", True, "continuous", 0, 1),
        ]
    )

    model_config = {
        "tau_gumbel": 0.25,
        "lambda_dag": 100.0,
        "lambda_sparse": 5.0,
        "base_distribution_type": "gaussian",
        "spline_bins": 8,
        "var_dist_A_mode": "enco",
        "mode_adjacency": "learn",
        "random_seed": [0],
    }

    dataset = Dataset(
        train_data=np.ones((5, 3)),
        train_mask=np.ones((5, 3), dtype=bool),
        val_data=np.ones((5, 3)),
        val_mask=np.ones((5, 3), dtype=bool),
        test_data=np.ones((5, 3)),
        test_mask=np.ones((5, 3), dtype=bool),
        variables=variables,
    )

    directed_adj_matrix = np.zeros((3, 3))
    directed_adj_mask = np.ones((3, 3))
    bidirected_adj_matrix = np.zeros((3, 3))
    bidirected_adj_mask = np.ones((3, 3))
    confounded_causal_dataset = dataset.to_latent_confounded_causal(
        directed_adjacency_data=directed_adj_matrix,
        directed_subgraph_data=directed_adj_mask,
        bidirected_adjacency_data=bidirected_adj_matrix,
        bidirected_subgraph_data=bidirected_adj_mask,
        intervention_data=None,
    )

    model = ADMGParameterisedDDECI.create(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        model_config_dict=model_config,
        device="cpu",
    )
    eval_latent_confounded_causal_discovery(confounded_causal_dataset, model)

    results_directed = read_json_as(os.path.join(model.save_dir, "target_results_causality_directed.json"), dict)
    results_bidirected = read_json_as(os.path.join(model.save_dir, "target_results_causality_bidirected.json"), dict)

    assert "adjacency_precision" in results_directed["test_data"]
    assert "adjacency_fscore" in results_directed["test_data"]
    assert "orientation_recall" in results_directed["test_data"]
    assert "orientation_precision" in results_directed["test_data"]
    assert "orientation_fscore" in results_directed["test_data"]
    assert "causal_accuracy" in results_directed["test_data"]
    assert "shd" in results_directed["test_data"]
    assert "nnz" in results_directed["test_data"]
    assert "adjacency_precision" in results_bidirected["test_data"]
    assert "adjacency_fscore" in results_bidirected["test_data"]
    assert "orientation_recall" in results_bidirected["test_data"]
    assert "orientation_precision" in results_bidirected["test_data"]
    assert "orientation_fscore" in results_bidirected["test_data"]
    assert "causal_accuracy" in results_bidirected["test_data"]
    assert "shd" in results_bidirected["test_data"]
    assert "nnz" in results_bidirected["test_data"]

    model = BowFreeDDECI.create(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        model_config_dict=model_config,
        device="cpu",
    )
    eval_latent_confounded_causal_discovery(confounded_causal_dataset, model)

    model = DECI.create(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        model_config_dict=model_config,
        device="cpu",
    )
    eval_latent_confounded_causal_discovery(confounded_causal_dataset, model)


@pytest.fixture
def causal_dataset_conditioning():
    variables = Variables(
        [
            Variable("continuous_input_1", True, "continuous", 0, 1),
            Variable("continuous_input_2", True, "continuous", 0, 1),
            Variable("continuous_input_3", True, "continuous", 0, 1),
            Variable("cat_input_1", True, "categorical", 0, 2),
            Variable("binary_input_1", True, "binary", 0, 1),
        ]
    )

    cts_data = np.random.rand(5, 3)
    cat_data = np.round(2 * np.random.rand(5, 1))
    bin_data = np.round(np.random.rand(5, 1))
    data = np.concatenate([cts_data, cat_data, bin_data], axis=-1)
    dataset = Dataset(
        train_data=data,
        train_mask=np.ones((5, 5), dtype=bool),
        val_data=data,
        val_mask=np.ones((5, 5), dtype=bool),
        test_data=data,
        test_mask=np.ones((5, 5), dtype=bool),
        variables=variables,
    )

    intervention_data = [
        InterventionData(intervention_idxs=np.array([0]), intervention_values=np.array([0.0]), test_data=data),
        InterventionData(intervention_idxs=np.array([3]), intervention_values=np.array([2]), test_data=data),
    ]

    counterfactual_data = [
        InterventionData(
            intervention_idxs=np.array([0]),
            intervention_values=np.array([1]),
            test_data=np.random.rand(2, 5),
            conditioning_idxs=np.array([0, 1, 2, 3, 4]),
            conditioning_values=np.random.rand(2, 5),
            intervention_reference=np.array([-1]),
            reference_data=np.random.rand(2, 5),
        ),
        InterventionData(
            intervention_idxs=np.array([0]),
            intervention_values=np.array([1]),
            test_data=np.random.rand(2, 5),
            conditioning_idxs=np.array([0, 1, 2, 3, 4]),
            conditioning_values=np.random.rand(2, 5),
            intervention_reference=np.array([-1]),
            reference_data=np.random.rand(2, 5),
        ),
    ]

    causal_dataset = dataset.to_causal(
        adjacency_data=None,
        subgraph_data=None,
        intervention_data=intervention_data,
        counterfactual_data=counterfactual_data,
    )
    return causal_dataset


@pytest.fixture
def causal_dataset_no_conditioning():
    variables = Variables(
        [
            Variable("continuous_input_1", True, "continuous", 0, 1),
            Variable("continuous_input_2", True, "continuous", 0, 1),
            Variable("continuous_input_3", True, "continuous", 0, 1),
        ]
    )

    data = np.random.rand(5, 3)
    dataset = Dataset(
        train_data=data,
        train_mask=np.ones((5, 3), dtype=bool),
        val_data=data,
        val_mask=np.ones((5, 3), dtype=bool),
        test_data=data,
        test_mask=np.ones((5, 3), dtype=bool),
        variables=variables,
    )

    intervention_data = [
        InterventionData(intervention_idxs=np.array([0]), intervention_values=np.array([0.0]), test_data=data),
        InterventionData(intervention_idxs=np.array([2]), intervention_values=np.array([2]), test_data=data),
    ]

    counterfactual_data = [
        InterventionData(
            intervention_idxs=np.array([0]),
            intervention_values=np.array([1]),
            test_data=np.random.rand(2, 3),
            intervention_reference=np.array([-1]),
            reference_data=np.random.rand(2, 3),
        ),
        InterventionData(
            intervention_idxs=np.array([0]),
            intervention_values=np.array([1]),
            test_data=np.random.rand(2, 3),
            intervention_reference=np.array([-1]),
            reference_data=np.random.rand(2, 3),
        ),
    ]

    dag = np.zeros((3, 3))
    dag[1, 0] = 1
    dag[2, 0] = 1
    dag[2, 1] = 1

    causal_dataset = dataset.to_causal(
        adjacency_data=dag,
        subgraph_data=None,
        intervention_data=intervention_data,
        counterfactual_data=counterfactual_data,
    )
    return causal_dataset


@pytest.fixture
def deci_model(tmpdir_factory, causal_dataset_conditioning):

    model_config = {
        "tau_gumbel": 0.25,
        "lambda_dag": 100.0,
        "lambda_sparse": 5.0,
        "base_distribution_type": "gaussian",
        "spline_bins": 8,
        "var_dist_A_mode": "enco",
        "mode_adjacency": "learn",
        "random_seed": [0],
    }

    model = DECI.create(
        model_id="model_id",
        variables=causal_dataset_conditioning.variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        model_config_dict=model_config,
        device="cpu",
    )

    return model


def test_evaluate_treatment_effect_estimation(
    tmpdir_factory, deci_model, causal_dataset_no_conditioning, causal_dataset_conditioning
):

    logger = logging.getLogger()

    with patch("causica.experiment.steps.eval_step.eval_treatment_effects") as mock_eval_treatment_effects:
        with patch(
            "causica.experiment.steps.eval_step.eval_individual_treatment_effects"
        ) as mock_eval_individual_treatment_effects:
            evaluate_treatment_effect_estimation(deci_model, causal_dataset_conditioning, logger)
            assert mock_eval_treatment_effects.called
            assert mock_eval_individual_treatment_effects.called

    model = TrueGraphDoWhy(
        model_id="model_id",
        variables=causal_dataset_no_conditioning.variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        device="cpu",
        **{
            "random_seed": 1,
            "adj_matrix": causal_dataset_no_conditioning.get_adjacency_data_matrix(),
            "inference_config": {"linear": True, "polynomial_order": 2, "polynomial_bias": True},
        },
    )

    model.run_train(causal_dataset_no_conditioning)

    with patch("causica.experiment.steps.eval_step.eval_treatment_effects") as mock_eval_treatment_effects:
        with patch(
            "causica.experiment.steps.eval_step.eval_individual_treatment_effects"
        ) as mock_eval_individual_treatment_effects:
            evaluate_treatment_effect_estimation(model, causal_dataset_no_conditioning, logger)
            assert mock_eval_treatment_effects.called
            assert not mock_eval_individual_treatment_effects.called


def test_run_eval_treatment_effects(causal_dataset_conditioning, deci_model):
    logger = logging.getLogger()

    assert isinstance(deci_model, IModelForInterventions)
    eval_treatment_effects(logger, causal_dataset_conditioning, deci_model)

    results = read_json_as(os.path.join(deci_model.save_dir, "results_interventions.json"), dict)

    for i in range(len(causal_dataset_conditioning.get_intervention_data())):
        assert "log prob mean" in results["test_data"][f"Intervention {i}"]
        assert "log prob std" in results["test_data"][f"Intervention {i}"]

        assert "ATE RMSE" in results["test_data"][f"Intervention {i}"]["all columns"]
        assert "Normalised ATE RMSE" in results["test_data"][f"Intervention {i}"]["all columns"]

        for j in range(causal_dataset_conditioning.variables.num_groups):
            assert "ATE RMSE" in results["test_data"][f"Intervention {i}"][f"Column {j}"]
            assert "Normalised ATE RMSE" in results["test_data"][f"Intervention {i}"][f"Column {j}"]

    assert "log prob mean" in results["test_data"]["all interventions"]
    assert "log prob std" in results["test_data"]["all interventions"]

    assert "ATE RMSE" in results["test_data"]["all interventions"]
    assert "Normalised ATE RMSE" in results["test_data"]["all interventions"]

    assert "test log prob mean" in results["test_data"]
    assert "test log prob mean" in results["test_data"]


def test_eval_individual_treatment_effects(causal_dataset_conditioning, deci_model):
    assert isinstance(deci_model, IModelForCounterfactuals)
    eval_individual_treatment_effects(causal_dataset_conditioning, deci_model)

    results = read_json_as(os.path.join(deci_model.save_dir, "results_counterfactual.json"), dict)

    for i in range(len(causal_dataset_conditioning.get_counterfactual_data())):
        assert "Normalised ITE RMSE" in results["test_data"][f"Intervention {i}"]["all columns"]
        assert "ITE RMSE" in results["test_data"][f"Intervention {i}"]["all columns"]

    assert "ITE RMSE" in results["test_data"]["all interventions"]
    assert "Normalised ITE RMSE" in results["test_data"]["all interventions"]
