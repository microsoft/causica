import os
import tempfile

import pytest
import torch

from causica.datasets.causica_dataset_format import CAUSICA_DATASETS_PATH, DataEnum, load_data, save_data, save_dataset
from causica.datasets.tensordict_utils import tensordict_shapes


def _load_and_test_dataset(root_path: str):
    variables_metadata = load_data(root_path, DataEnum.VARIABLES_JSON)
    train_data = load_data(root_path, DataEnum.TRAIN, variables_metadata)
    test_data = load_data(root_path, DataEnum.TEST, variables_metadata)

    assert tensordict_shapes(train_data) == tensordict_shapes(test_data)
    groups = set(train_data.keys())
    interventions = load_data(root_path, DataEnum.INTERVENTIONS, variables_metadata)
    for (intervention_a, intervention_b, _) in interventions:
        int_groups_a = set(intervention_a.intervention_values.keys()) | intervention_a.sampled_nodes
        int_groups_b = set(intervention_b.intervention_values.keys()) | intervention_b.sampled_nodes
        assert groups == int_groups_a
        assert groups == int_groups_b
        assert tensordict_shapes(intervention_a.intervention_data) == tensordict_shapes(
            intervention_b.intervention_data
        )

    adj_mat = load_data(root_path, DataEnum.TRUE_ADJACENCY)
    num_nodes = len(train_data.keys())
    assert adj_mat.shape == (num_nodes, num_nodes)

    return variables_metadata, train_data, test_data, adj_mat, interventions


def load_and_test_counterfactuals(root_path):
    variables_metadata = load_data(root_path, DataEnum.VARIABLES_JSON)
    # not all counterfactuals exist
    counterfactuals = load_data(root_path, DataEnum.COUNTERFACTUALS, variables_metadata)
    for (intervention_a, intervention_b, _) in counterfactuals:
        int_groups_a = set(intervention_a.intervention_values.keys()) | intervention_a.sampled_nodes
        int_groups_b = set(intervention_b.intervention_values.keys()) | intervention_b.sampled_nodes
        assert int_groups_a == int_groups_b

        # all nodes must have the same dimensions in both factual and counterfactual data
        assert tensordict_shapes(intervention_a.counterfactual_data) == tensordict_shapes(
            intervention_b.counterfactual_data
        )
        assert tensordict_shapes(intervention_a.counterfactual_data) == tensordict_shapes(intervention_a.factual_data)
        assert tensordict_shapes(intervention_a.factual_data) == tensordict_shapes(intervention_b.factual_data)

        # there must be the same numbers of factual and counterfactual in each dataset
        assert intervention_a.factual_data.batch_size == intervention_a.counterfactual_data.batch_size
        assert intervention_b.factual_data.batch_size == intervention_b.counterfactual_data.batch_size

    return variables_metadata, counterfactuals


@pytest.mark.parametrize("dataset", ["csuite_weak_arrows", "csuite_linexp_2", "csuite_cts_to_cat"])
def test_load_save_csuite(dataset):
    """Test that we can load a csuite dataset, save it, and load it again"""
    root_path = os.path.join(CAUSICA_DATASETS_PATH, dataset)

    variables_metadata, train_data, test_data, adj_mat, interventions = _load_and_test_dataset(root_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dataset(
            tmpdir, variables_metadata, adj_mat, train_data, test_data, interventions=interventions, overwrite=True
        )
        variables_metadata_2, train_data_2, test_data_2, adj_mat_2, interventions_2 = _load_and_test_dataset(tmpdir)

    assert variables_metadata == variables_metadata_2
    torch.testing.assert_allclose(adj_mat, adj_mat_2)
    torch.testing.assert_allclose(train_data, train_data_2)
    assert tensordict_shapes(train_data) == tensordict_shapes(train_data_2)
    assert tensordict_shapes(test_data) == tensordict_shapes(test_data_2)
    assert len(interventions) == len(interventions_2)
    for intervention_1, intervention_2 in zip(interventions, interventions_2):
        for field in ["intervention_data", "intervention_values", "condition_values"]:
            torch.testing.assert_allclose(getattr(intervention_1[0], field), getattr(intervention_2[0], field))
            torch.testing.assert_allclose(getattr(intervention_1[1], field), getattr(intervention_2[1], field))
        assert intervention_1[2] == intervention_2[2]


def test_load_save_counterfactuals():
    dataset = "csuite_linexp_2"
    root_path = os.path.join(CAUSICA_DATASETS_PATH, dataset)

    variables_metadata, counterfactuals = load_and_test_counterfactuals(root_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_data(tmpdir, variables_metadata, DataEnum.VARIABLES_JSON, variables_metadata)
        save_data(tmpdir, counterfactuals, DataEnum.COUNTERFACTUALS, variables_metadata)
        variables_metadata_2, counterfactuals_2 = load_and_test_counterfactuals(tmpdir)

    assert variables_metadata == variables_metadata_2
    assert len(counterfactuals) == len(counterfactuals_2)
    for cf_1, cf_2 in zip(counterfactuals, counterfactuals_2):
        for field in ["counterfactual_data", "intervention_values", "factual_data"]:
            torch.testing.assert_allclose(getattr(cf_1[0], field), getattr(cf_2[0], field))
            torch.testing.assert_allclose(getattr(cf_1[1], field), getattr(cf_2[1], field))
        assert cf_1[2] == cf_2[2]
