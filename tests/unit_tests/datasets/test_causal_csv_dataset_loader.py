import json
import os

import numpy as np
import pandas as pd
import pytest

from causica.datasets.causal_csv_dataset_loader import CausalCSVDatasetLoader
from causica.datasets.dataset import CausalDataset, InterventionData
from causica.utils.helper_functions import convert_dict_of_ndarray_to_lists


def test_split_data_and_load_dataset(tmpdir_factory):
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    data = np.array(
        [
            [np.nan, 2.1],  # Full data: 10 rows
            [np.nan, 2.1],
            [np.nan, 2.1],
            [np.nan, 2.1],
            [np.nan, 2.1],
            [np.nan, 2.1],
            [np.nan, 2.1],
            [np.nan, 2.1],
            [np.nan, 2.1],
            [np.nan, 2.1],
        ]
    )

    adjacency_matrix = np.array([[0, 1], [0, 0]])

    #    datafile column order:  conditioning_cols   intervention_cols     reference_cols     effect_cols   sample_cols

    conditioning_cols = np.array(
        [
            [34, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
        ]
    )

    intervention_cols = np.array(
        [
            [np.nan, 0.0],
            [0.0, np.nan],
            [0.0, np.nan],
            [0.0, np.nan],
            [0.0, np.nan],
            [2.1, np.nan],
            [3.2, np.nan],
            [3.2, np.nan],
        ]
    )

    reference_cols = np.array(
        [
            [np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [11, np.nan],
            [11, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [0.0, np.nan],
        ]
    )

    effect_cols = np.array(
        [
            [np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, 1],
            [np.nan, np.nan],
            [np.nan, np.nan],
        ]
    )

    test_cols = np.array([[0, 1.3], [0.9, 1.1], [2.1, 2], [-0.9, 1.8], [-0.9, 1.8], [-1.9, 3.5], [-2, 3.1], [-2, 3.1]])

    intervention_data = np.concatenate(
        [conditioning_cols, intervention_cols, reference_cols, effect_cols, test_cols], axis=1
    )

    pd.DataFrame(data).to_csv(os.path.join(dataset_dir, "all.csv"), header=None, index=None)
    pd.DataFrame(adjacency_matrix).to_csv(os.path.join(dataset_dir, "adj_matrix.csv"), header=None, index=None)
    pd.DataFrame(intervention_data).to_csv(os.path.join(dataset_dir, "interventions.csv"), header=None, index=None)

    dataset_loader = CausalCSVDatasetLoader(dataset_dir=dataset_dir)
    dataset = dataset_loader.split_data_and_load_dataset(test_frac=0.5, val_frac=0.2, random_state=0, max_num_rows=None)

    expected_conditioning_idxs = [np.array([0]), None, None, None]
    expected_conditioning_values = [np.array([34.0]), None, None, None]

    expected_intervention_idxs = [np.array([1]), np.array([0]), np.array([0]), np.array([0])]
    expected_intervention_values = [np.array([0.0]), np.array([0.0]), np.array([2.1]), np.array([3.2])]

    # expected_reference_idxs = [None, np.array([1]), None, None]
    expected_reference_values = [None, np.array([11.0]), None, np.array([0.0])]

    expected_effect_idxs = [None, None, np.array([1]), None]

    # Basic type and shape checks
    assert isinstance(dataset, CausalDataset)
    assert np.array_equal(adjacency_matrix, dataset.get_adjacency_data_matrix())
    assert isinstance(dataset.get_intervention_data(), list)
    assert isinstance(dataset.get_intervention_data()[0], InterventionData)
    assert len(dataset.get_intervention_data()) == 4

    # Check conditioning indices
    assert np.array_equal(dataset.get_intervention_data()[0].conditioning_idxs, expected_conditioning_idxs[0])
    assert np.array_equal(dataset.get_intervention_data()[1].conditioning_idxs, expected_conditioning_idxs[1])
    assert np.array_equal(dataset.get_intervention_data()[2].conditioning_idxs, expected_conditioning_idxs[2])
    assert np.array_equal(dataset.get_intervention_data()[3].conditioning_idxs, expected_conditioning_idxs[3])

    # Check conditioning values
    assert np.array_equal(dataset.get_intervention_data()[0].conditioning_values, expected_conditioning_values[0])
    assert np.array_equal(dataset.get_intervention_data()[1].conditioning_values, expected_conditioning_values[1])
    assert np.array_equal(dataset.get_intervention_data()[2].conditioning_values, expected_conditioning_values[2])
    assert np.array_equal(dataset.get_intervention_data()[3].conditioning_values, expected_conditioning_values[3])

    # Check intervention indices
    assert np.array_equal(dataset.get_intervention_data()[0].intervention_idxs, expected_intervention_idxs[0])
    assert np.array_equal(dataset.get_intervention_data()[1].intervention_idxs, expected_intervention_idxs[1])
    assert np.array_equal(dataset.get_intervention_data()[2].intervention_idxs, expected_intervention_idxs[2])
    assert np.array_equal(dataset.get_intervention_data()[3].intervention_idxs, expected_intervention_idxs[3])

    # Check intervention values
    assert np.array_equal(dataset.get_intervention_data()[0].intervention_values, expected_intervention_values[0])
    assert np.array_equal(dataset.get_intervention_data()[1].intervention_values, expected_intervention_values[1])
    assert np.array_equal(dataset.get_intervention_data()[2].intervention_values, expected_intervention_values[2])
    assert np.array_equal(dataset.get_intervention_data()[3].intervention_values, expected_intervention_values[3])

    # Check reference values
    assert np.array_equal(dataset.get_intervention_data()[0].intervention_reference, expected_reference_values[0])
    assert np.array_equal(dataset.get_intervention_data()[1].intervention_reference, expected_reference_values[1])
    assert np.array_equal(dataset.get_intervention_data()[2].effect_idxs, expected_effect_idxs[2])
    assert np.array_equal(dataset.get_intervention_data()[3].effect_idxs, expected_effect_idxs[3])

    # Check effect indices
    assert np.array_equal(dataset.get_intervention_data()[0].effect_idxs, expected_effect_idxs[0])
    assert np.array_equal(dataset.get_intervention_data()[1].effect_idxs, expected_effect_idxs[1])
    assert np.array_equal(dataset.get_intervention_data()[2].effect_idxs, expected_effect_idxs[2])
    assert np.array_equal(dataset.get_intervention_data()[3].effect_idxs, expected_effect_idxs[3])

    # Check intervened data
    assert np.array_equal(dataset.get_intervention_data()[0].test_data, test_cols[:1])
    assert np.array_equal(dataset.get_intervention_data()[1].test_data, test_cols[1:3])
    assert np.array_equal(dataset.get_intervention_data()[2].test_data, test_cols[5:6])
    assert np.array_equal(dataset.get_intervention_data()[3].test_data, test_cols[7:8])

    assert dataset.get_intervention_data()[0].reference_data is None
    assert np.array_equal(dataset.get_intervention_data()[1].reference_data, test_cols[3:5])
    assert dataset.get_intervention_data()[2].reference_data is None
    assert np.array_equal(dataset.get_intervention_data()[3].reference_data, test_cols[6:7])

    # Test case in which there is a single intervention
    intervention_data = np.array(
        [
            [np.nan, np.nan, np.nan, 0.0, np.nan, np.nan, np.nan, np.nan, 0, 1.3],
            [np.nan, np.nan, np.nan, 0.0, np.nan, np.nan, np.nan, np.nan, -0.3, 1.3],
            [np.nan, np.nan, np.nan, 0.0, np.nan, np.nan, np.nan, np.nan, 0, 0.3],
        ]
    )

    pd.DataFrame(intervention_data).to_csv(os.path.join(dataset_dir, "interventions.csv"), header=None, index=None)

    dataset_loader = CausalCSVDatasetLoader(dataset_dir=dataset_dir)
    dataset = dataset_loader.split_data_and_load_dataset(test_frac=0.5, val_frac=0.2, random_state=0, max_num_rows=None)

    expected_intervention_idxs = [np.array([1])]
    expected_intervention_values = [np.array([0.0])]

    assert isinstance(dataset, CausalDataset)
    assert np.array_equal(adjacency_matrix, dataset.get_adjacency_data_matrix())
    assert isinstance(dataset.get_intervention_data(), list)
    assert isinstance(dataset.get_intervention_data()[0], InterventionData)
    assert len(dataset.get_intervention_data()) == 1
    assert np.array_equal(dataset.get_intervention_data()[0].intervention_idxs, expected_intervention_idxs[0])
    assert np.array_equal(dataset.get_intervention_data()[0].intervention_values, expected_intervention_values[0])
    assert np.array_equal(dataset.get_intervention_data()[0].test_data, intervention_data[:, -2:])

    assert dataset.get_intervention_data()[0].reference_data is None

    # Test case with multiple interventions and multiple values with 3 dimensional input
    intervention_data = np.array(
        [
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0.0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                1.0,
                0.2,
                1.3,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0.0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                -0.3,
                1.3,
                0.7,
            ],
            [np.nan, np.nan, np.nan, np.nan, 0.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 2, 0.3, 2.2],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                2.1,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0.03,
                0.72,
                1.5,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                2.1,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                1.2,
                2.2,
                2.3,
            ],
        ]
    )

    pd.DataFrame(intervention_data).to_csv(os.path.join(dataset_dir, "interventions.csv"), header=None, index=None)

    dataset_loader = CausalCSVDatasetLoader(dataset_dir=dataset_dir)
    dataset = dataset_loader.split_data_and_load_dataset(test_frac=0.5, val_frac=0.2, random_state=0, max_num_rows=None)

    expected_intervention_idxs = [np.array([1]), np.array([2])]
    expected_intervention_values = [np.array([0.0]), np.array([2.1])]

    assert isinstance(dataset, CausalDataset)
    assert np.array_equal(adjacency_matrix, dataset.get_adjacency_data_matrix())
    assert isinstance(dataset.get_intervention_data(), list)
    assert isinstance(dataset.get_intervention_data()[0], InterventionData)
    assert len(dataset.get_intervention_data()) == 2
    assert np.array_equal(dataset.get_intervention_data()[0].intervention_idxs, expected_intervention_idxs[0])
    assert np.array_equal(dataset.get_intervention_data()[0].intervention_values, expected_intervention_values[0])
    assert np.array_equal(dataset.get_intervention_data()[0].test_data, intervention_data[:3, -3:])
    assert np.array_equal(dataset.get_intervention_data()[1].intervention_idxs, expected_intervention_idxs[1])
    assert np.array_equal(dataset.get_intervention_data()[1].intervention_values, expected_intervention_values[1])
    assert np.array_equal(dataset.get_intervention_data()[1].test_data, intervention_data[3:, -3:])

    assert dataset.get_intervention_data()[0].reference_data is None
    assert dataset.get_intervention_data()[1].reference_data is None


def test_load_predefined_dataset(tmpdir_factory):
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    train_data = np.array([[np.nan, 2.1], [2.2, np.nan]])
    val_data = np.array([[np.nan, 3.1]])
    test_data = np.array([[4.1, np.nan]])

    adjacency_matrix = np.array([[0, 1], [0, 0]])

    #    datafile column order:  conditioning_cols   intervention_cols     reference_cols     effect_cols   sample_cols

    conditioning_cols = np.array(
        [
            [34, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
        ]
    )

    intervention_cols = np.array(
        [
            [np.nan, 0.0],
            [0.0, np.nan],
            [0.0, np.nan],
            [0.0, np.nan],
            [0.0, np.nan],
            [2.1, np.nan],
            [3.2, np.nan],
            [3.2, np.nan],
        ]
    )

    reference_cols = np.array(
        [
            [np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [11, np.nan],
            [11, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [0.0, np.nan],
        ]
    )

    effect_cols = np.array(
        [
            [np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, 1],
            [np.nan, np.nan],
            [np.nan, np.nan],
        ]
    )

    test_cols = np.array([[0, 1.3], [0.9, 1.1], [2.1, 2], [-0.9, 1.8], [-0.9, 1.8], [-1.9, 3.5], [-2, 3.1], [-2, 3.1]])

    intervention_data = np.concatenate(
        [conditioning_cols, intervention_cols, reference_cols, effect_cols, test_cols], axis=1
    )

    cf_conditioning_cols = np.array(
        [
            [
                0.0,
                1.0,
                2.0,
            ],
            [1.0, 0.0, 2.0],
            [2.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
        ]
    )
    cf_intervention_cols = np.array(
        [[np.nan, 0.0, np.nan], [np.nan, 0.0, np.nan], [0.0, 0.0, np.nan], [0.0, 0.0, np.nan]]
    )
    cf_reference_cols = np.array(
        [[np.nan, np.nan, np.nan], [np.nan, 1.0, np.nan], [np.nan, np.nan, np.nan], [1.0, 1.0, np.nan]]
    )
    cf_effect_cols = np.array(
        [[np.nan, np.nan, 1.0], [np.nan, np.nan, 1.0], [np.nan, np.nan, 1.0], [np.nan, np.nan, 1.0]]
    )
    cf_test_cols = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])

    cf_data = np.concatenate(
        [cf_conditioning_cols, cf_intervention_cols, cf_reference_cols, cf_effect_cols, cf_test_cols], axis=1
    )

    pd.DataFrame(adjacency_matrix).to_csv(os.path.join(dataset_dir, "adj_matrix.csv"), header=None, index=None)
    pd.DataFrame(intervention_data).to_csv(os.path.join(dataset_dir, "interventions.csv"), header=None, index=None)
    pd.DataFrame(cf_data).to_csv(os.path.join(dataset_dir, "counterfactuals.csv"), header=None, index=None)

    pd.DataFrame(train_data).to_csv(os.path.join(dataset_dir, "train.csv"), header=None, index=None)
    pd.DataFrame(val_data).to_csv(os.path.join(dataset_dir, "val.csv"), header=None, index=None)
    pd.DataFrame(test_data).to_csv(os.path.join(dataset_dir, "test.csv"), header=None, index=None)

    expected_conditioning_idxs = [np.array([0]), None, None, None]
    expected_conditioning_values = [np.array([34.0]), None, None, None]

    expected_intervention_idxs = [np.array([1]), np.array([0]), np.array([0]), np.array([0])]
    expected_intervention_values = [np.array([0.0]), np.array([0.0]), np.array([2.1]), np.array([3.2])]

    # expected_reference_idxs = [None, np.array([1]), None, None]
    expected_reference_values = [None, np.array([11.0]), None, np.array([0.0])]

    expected_effect_idxs = [None, None, np.array([1]), None]

    dataset_loader = CausalCSVDatasetLoader(dataset_dir=dataset_dir)
    dataset = dataset_loader.load_predefined_dataset(max_num_rows=None)

    assert isinstance(dataset, CausalDataset)
    assert np.array_equal(adjacency_matrix, dataset.get_adjacency_data_matrix())
    assert isinstance(dataset.get_intervention_data(), list)
    assert isinstance(dataset.get_intervention_data()[0], InterventionData)
    assert len(dataset.get_intervention_data()) == 4

    # Check conditioning indices
    assert np.array_equal(dataset.get_intervention_data()[0].conditioning_idxs, expected_conditioning_idxs[0])
    assert np.array_equal(dataset.get_intervention_data()[1].conditioning_idxs, expected_conditioning_idxs[1])
    assert np.array_equal(dataset.get_intervention_data()[2].conditioning_idxs, expected_conditioning_idxs[2])
    assert np.array_equal(dataset.get_intervention_data()[3].conditioning_idxs, expected_conditioning_idxs[3])

    # Check conditioning values
    assert np.array_equal(dataset.get_intervention_data()[0].conditioning_values, expected_conditioning_values[0])
    assert np.array_equal(dataset.get_intervention_data()[1].conditioning_values, expected_conditioning_values[1])
    assert np.array_equal(dataset.get_intervention_data()[2].conditioning_values, expected_conditioning_values[2])
    assert np.array_equal(dataset.get_intervention_data()[3].conditioning_values, expected_conditioning_values[3])

    # Check intervention indices
    assert np.array_equal(dataset.get_intervention_data()[0].intervention_idxs, expected_intervention_idxs[0])
    assert np.array_equal(dataset.get_intervention_data()[1].intervention_idxs, expected_intervention_idxs[1])
    assert np.array_equal(dataset.get_intervention_data()[2].intervention_idxs, expected_intervention_idxs[2])
    assert np.array_equal(dataset.get_intervention_data()[3].intervention_idxs, expected_intervention_idxs[3])

    # Check intervention values
    assert np.array_equal(dataset.get_intervention_data()[0].intervention_values, expected_intervention_values[0])
    assert np.array_equal(dataset.get_intervention_data()[1].intervention_values, expected_intervention_values[1])
    assert np.array_equal(dataset.get_intervention_data()[2].intervention_values, expected_intervention_values[2])
    assert np.array_equal(dataset.get_intervention_data()[3].intervention_values, expected_intervention_values[3])

    # Check reference values
    assert np.array_equal(dataset.get_intervention_data()[0].intervention_reference, expected_reference_values[0])
    assert np.array_equal(dataset.get_intervention_data()[1].intervention_reference, expected_reference_values[1])
    assert np.array_equal(dataset.get_intervention_data()[2].effect_idxs, expected_effect_idxs[2])
    assert np.array_equal(dataset.get_intervention_data()[3].effect_idxs, expected_effect_idxs[3])

    # Check effect indices
    assert np.array_equal(dataset.get_intervention_data()[0].effect_idxs, expected_effect_idxs[0])
    assert np.array_equal(dataset.get_intervention_data()[1].effect_idxs, expected_effect_idxs[1])
    assert np.array_equal(dataset.get_intervention_data()[2].effect_idxs, expected_effect_idxs[2])
    assert np.array_equal(dataset.get_intervention_data()[3].effect_idxs, expected_effect_idxs[3])

    # Check intervened data
    assert np.array_equal(dataset.get_intervention_data()[0].test_data, test_cols[:1])
    assert np.array_equal(dataset.get_intervention_data()[1].test_data, test_cols[1:3])
    assert np.array_equal(dataset.get_intervention_data()[2].test_data, test_cols[5:6])
    assert np.array_equal(dataset.get_intervention_data()[3].test_data, test_cols[7:8])

    assert dataset.get_intervention_data()[0].reference_data is None
    assert np.array_equal(dataset.get_intervention_data()[1].reference_data, test_cols[3:5])
    assert dataset.get_intervention_data()[2].reference_data is None
    assert np.array_equal(dataset.get_intervention_data()[3].reference_data, test_cols[6:7])

    # Check CF data
    assert len(dataset.get_counterfactual_data()) == 2
    assert len(dataset.get_counterfactual_data()[0].conditioning_idxs) == 2
    assert dataset.get_counterfactual_data()[0].conditioning_values.shape == (2, 3)

    # Test case in which there is a single intervention
    intervention_data = np.array(
        [
            [np.nan, np.nan, np.nan, 0.0, np.nan, np.nan, np.nan, np.nan, 0, 1.3],
            [np.nan, np.nan, np.nan, 0.0, np.nan, np.nan, np.nan, np.nan, -0.3, 1.3],
            [np.nan, np.nan, np.nan, 0.0, np.nan, np.nan, np.nan, np.nan, 0, 0.3],
        ]
    )
    pd.DataFrame(intervention_data).to_csv(os.path.join(dataset_dir, "interventions.csv"), header=None, index=None)

    dataset_loader = CausalCSVDatasetLoader(dataset_dir=dataset_dir)
    dataset = dataset_loader.load_predefined_dataset(max_num_rows=None)

    expected_intervention_idxs = [np.array([1])]
    expected_intervention_values = [np.array([0.0])]

    assert isinstance(dataset, CausalDataset)
    assert np.array_equal(adjacency_matrix, dataset.get_adjacency_data_matrix())
    assert isinstance(dataset.get_intervention_data(), list)
    assert isinstance(dataset.get_intervention_data()[0], InterventionData)
    assert len(dataset.get_intervention_data()) == 1
    assert np.array_equal(dataset.get_intervention_data()[0].intervention_idxs, expected_intervention_idxs[0])
    assert np.array_equal(dataset.get_intervention_data()[0].intervention_values, expected_intervention_values[0])
    assert np.array_equal(dataset.get_intervention_data()[0].test_data, intervention_data[:, -2:])
    assert dataset.get_intervention_data()[0].reference_data is None

    # Test case with multiple interventions and multiple values with 3 dimensional input
    intervention_data = np.array(
        [
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0.0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                1.0,
                0.2,
                1.3,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0.0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                -0.3,
                1.3,
                0.7,
            ],
            [np.nan, np.nan, np.nan, np.nan, 0.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 2, 0.3, 2.2],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                2.1,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0.03,
                0.72,
                1.5,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                2.1,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                1.2,
                2.2,
                2.3,
            ],
        ]
    )

    pd.DataFrame(intervention_data).to_csv(os.path.join(dataset_dir, "interventions.csv"), header=None, index=None)

    dataset_loader = CausalCSVDatasetLoader(dataset_dir=dataset_dir)
    dataset = dataset_loader.load_predefined_dataset(max_num_rows=None)

    expected_intervention_idxs = [np.array([1]), np.array([2])]
    expected_intervention_values = [np.array([0.0]), np.array([2.1])]

    assert isinstance(dataset, CausalDataset)
    assert np.array_equal(adjacency_matrix, dataset.get_adjacency_data_matrix())
    assert isinstance(dataset.get_intervention_data(), list)
    assert isinstance(dataset.get_intervention_data()[0], InterventionData)
    assert len(dataset.get_intervention_data()) == 2
    assert np.array_equal(dataset.get_intervention_data()[0].intervention_idxs, expected_intervention_idxs[0])
    assert np.array_equal(dataset.get_intervention_data()[0].intervention_values, expected_intervention_values[0])
    assert np.array_equal(dataset.get_intervention_data()[0].test_data, intervention_data[:3, -3:])
    assert np.array_equal(dataset.get_intervention_data()[1].intervention_idxs, expected_intervention_idxs[1])
    assert np.array_equal(dataset.get_intervention_data()[1].intervention_values, expected_intervention_values[1])
    assert np.array_equal(dataset.get_intervention_data()[1].test_data, intervention_data[3:, -3:])
    assert dataset.get_intervention_data()[0].reference_data is None
    assert dataset.get_intervention_data()[1].reference_data is None


@pytest.mark.parametrize("data_format", ["npy", "json"])
def test_load_predefined_dataset_npy_json(tmpdir_factory, data_format):
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    train_data = np.array([[np.nan, 2.1], [2.2, np.nan]])
    val_data = np.array([[np.nan, 3.1]])
    test_data = np.array([[4.1, np.nan]])

    adjacency_matrix = np.array([[0, 1], [0, 0]])

    interventions = {
        "metadata": {"columns_to_nodes": [0, 1, 2, 3]},
        "environments": [
            {
                "conditioning_idxs": np.array([0]),
                "conditioning_values": np.array([1.0]),
                "intervention_idxs": np.array([1]),
                "intervention_values": np.array([1.0]),
                "intervention_reference": np.array([0.0]),
                "effect_idxs": np.array([3]),
                "test_data": np.array([[1, 1, 2.1, 3.1], [1, 1, 1.8, 2.9], [1, 1, 2.4, 3.4], [1, 1, 1.7, 3.2]]),
                "reference_data": np.array([[1, 0, 2.1, 1.1], [1, 0, 1.8, 1.9], [1, 0, 2.4, 1.4], [1, 0, 1.7, 1.2]]),
            },
            {
                "conditioning_idxs": np.array([0]),
                "conditioning_values": np.array([5.0]),
                "intervention_idxs": np.array([1]),
                "intervention_values": np.array([1.0]),
                "intervention_reference": np.array([0.0]),
                "effect_idxs": np.array([3]),
                "test_data": np.array([[1, 1, 2.1, 3.1], [1, 1, 1.8, 2.9], [1, 1, 2.4, 3.4], [1, 1, 1.7, 3.2]]),
                "reference_data": np.array([[1, 0, 2.1, 1.1], [1, 0, 1.8, 1.9], [1, 0, 2.4, 1.4], [1, 0, 1.7, 1.2]]),
            },
        ],
    }

    counterfactuals = {
        "metadata": {"columns_to_nodes": [0, 1, 2, 3]},
        "environments": [
            {
                "conditioning_idxs": np.array([0, 1, 2, 3]),
                "conditioning_values": np.array(
                    [[1, 0.5, 2.1, 3.1], [1, 0.0, 1.8, 2.9], [1, 1.2, 2.4, 3.4], [1, -0.5, 1.7, 3.2]]
                ),
                "intervention_idxs": np.array([1]),
                "intervention_values": np.array([1.0]),
                "intervention_reference": np.array([0.0]),
                "effect_idxs": np.array([3]),
                "test_data": np.array([[1, 1, 2.1, 3.1], [1, 1, 1.8, 2.9], [1, 1, 2.4, 3.4], [1, 1, 1.7, 3.2]]),
                "reference_data": np.array([[1, 0, 2.1, 1.1], [1, 0, 1.8, 1.9], [1, 0, 2.4, 1.4], [1, 0, 1.7, 1.2]]),
            },
        ],
    }

    pd.DataFrame(adjacency_matrix).to_csv(os.path.join(dataset_dir, "adj_matrix.csv"), header=None, index=None)

    pd.DataFrame(train_data).to_csv(os.path.join(dataset_dir, "train.csv"), header=None, index=None)
    pd.DataFrame(val_data).to_csv(os.path.join(dataset_dir, "val.csv"), header=None, index=None)
    pd.DataFrame(test_data).to_csv(os.path.join(dataset_dir, "test.csv"), header=None, index=None)

    if data_format == "npy":
        np.save(os.path.join(dataset_dir, "interventions.npy"), interventions)
        np.save(os.path.join(dataset_dir, "counterfactuals.npy"), counterfactuals)

    elif data_format == "json":
        with open(os.path.join(dataset_dir, "interventions.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": interventions["metadata"],
                    "environments": [convert_dict_of_ndarray_to_lists(e) for e in interventions["environments"]],
                },
                f,
            )

        with open(os.path.join(dataset_dir, "counterfactuals.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": counterfactuals["metadata"],
                    "environments": [convert_dict_of_ndarray_to_lists(e) for e in counterfactuals["environments"]],
                },
                f,
            )

    dataset_loader = CausalCSVDatasetLoader(dataset_dir=dataset_dir)
    dataset = dataset_loader.load_predefined_dataset(max_num_rows=None)

    assert isinstance(dataset, CausalDataset)
    assert np.array_equal(adjacency_matrix, dataset.get_adjacency_data_matrix())
    assert isinstance(dataset.get_intervention_data(), list)
    assert isinstance(dataset.get_intervention_data()[0], InterventionData)
    assert len(dataset.get_intervention_data()) == 2

    # Intervention
    for i, intervention_data in enumerate(dataset.get_intervention_data()):
        original_data = interventions["environments"][i]
        for k in original_data.keys():
            assert np.array_equal(getattr(intervention_data, k), original_data[k])

    # Counterfactual
    for i, intervention_data in enumerate(dataset.get_counterfactual_data()):
        original_data = counterfactuals["environments"][i]
        for k in original_data.keys():
            assert np.array_equal(getattr(intervention_data, k), original_data[k])
