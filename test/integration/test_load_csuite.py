import pytest

from causica.datasets.csuite_data import CSUITE_DATASETS_PATH, DataEnum, load_data
from causica.datasets.tensordict_utils import tensordict_shapes


@pytest.mark.parametrize("dataset", ["csuite_weak_arrows", "csuite_linexp_2"])
def test_load_csuite(dataset):
    """Test that we can load a csuite dataset"""

    variables_metadata = load_data(CSUITE_DATASETS_PATH, dataset, DataEnum.VARIABLES_JSON)
    train_data = load_data(CSUITE_DATASETS_PATH, dataset, DataEnum.TRAIN, variables_metadata)
    test_data = load_data(CSUITE_DATASETS_PATH, dataset, DataEnum.TEST, variables_metadata)
    assert tensordict_shapes(train_data) == tensordict_shapes(test_data)
    groups = set(train_data.keys())
    for (intervention_a, intervention_b, _) in load_data(
        CSUITE_DATASETS_PATH, dataset, DataEnum.INTERVENTIONS, variables_metadata
    ):
        int_groups_a = set(intervention_a.intervention_values.keys()) | intervention_a.sampled_nodes
        int_groups_b = set(intervention_b.intervention_values.keys()) | intervention_b.sampled_nodes
        assert groups == int_groups_a
        assert groups == int_groups_b
        assert tensordict_shapes(intervention_a.intervention_data) == tensordict_shapes(
            intervention_b.intervention_data
        )

    adj_mat = load_data(CSUITE_DATASETS_PATH, dataset, DataEnum.TRUE_ADJACENCY)
    num_nodes = len(train_data.keys())
    assert adj_mat.shape == (num_nodes, num_nodes)


def test_load_counterfactuals():
    dataset = "csuite_linexp_2"
    # not all counterfactuals exist
    data_list = load_data(CSUITE_DATASETS_PATH, dataset, DataEnum.COUNTERFACTUALS)
    for (intervention_a, intervention_b, _) in data_list:
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
