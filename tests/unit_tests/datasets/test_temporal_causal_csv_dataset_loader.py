import os

import numpy as np
import pandas as pd
import pytest

from causica.datasets.dataset import TemporalDataset
from causica.datasets.temporal_causal_csv_dataset_loader import TemporalCausalCSVDatasetLoader
from causica.utils.io_utils import save_json


# pylint: disable=protected-access
def check_temporal_order(dataset: TemporalDataset) -> bool:
    """Tests whether the dataset splitting accounted for temporal ordering.

    This means that the timestamp of samples in train < val < test.

    NOTE: This currently assumes that the csv is already temporally ordered.

    Args:
        dataset (TemporalDataset): A temporal dataset to test.

    Returns:
        bool: Whether the temporal order is accounted for.
    """

    assert isinstance(dataset.data_split, dict), "dataset.data_split should be a dict."
    cur_max = max(dataset.data_split["train_idxs"])

    for split in ["val_idxs", "test_idxs"]:
        if dataset.data_split[split]:
            if cur_max > min(dataset.data_split[split]):
                return False
            cur_max = max(dataset.data_split[split])

    return True


@pytest.fixture
def variable_data_example():
    data = np.stack([np.arange(10), np.ones(10), np.arange(10, 20)], 1)
    variable_orig_type = {
        "variables": [
            {"name": "Column", "type": "categorical"},
            {"name": "Column 0", "type": "categorical"},
            {"name": "Column 1", "type": "continuous"},
        ]
    }
    variable_cts = {
        "variables": [
            {"name": "Column", "type": "categorical"},
            {"name": "Column 0", "type": "continuous"},
            {"name": "Column 1", "type": "continuous"},
        ]
    }
    variable_wrong_number = {
        "variables": [{"name": "data column", "type": "categorical"}, {"name": "Column 0", "type": "categorical"}]
    }
    return data, variable_orig_type, variable_cts, variable_wrong_number


def test_split_data_and_load_dataset(tmpdir_factory):
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    # Single series with column index 0
    data_single = np.stack([np.ones(10), np.arange(10)], 1)
    adjacency_matrix = np.array([[[0, 1], [0, 0]], [[0, 1], [0, 0]]])
    pd.DataFrame(data_single).to_csv(os.path.join(dataset_dir, "all.csv"), header=None, index=None)
    np.save(os.path.join(dataset_dir, "adj_matrix.npy"), adjacency_matrix)

    dataset_loader = TemporalCausalCSVDatasetLoader(dataset_dir=dataset_dir)
    dataset = dataset_loader.split_data_and_load_dataset(
        test_frac=0.5, val_frac=0.2, random_state=0, max_num_rows=None, timeseries_column_index=0
    )
    assert np.array_equal(dataset._train_data, np.expand_dims(np.arange(3), axis=1))
    assert np.array_equal(dataset._val_data, np.expand_dims(np.arange(3, 5), axis=1))
    assert np.array_equal(dataset._test_data, np.expand_dims(np.arange(5, 10), axis=1))
    assert check_temporal_order(dataset)

    # Single series with column index 1
    data_single = np.stack([np.arange(10), np.ones(10)], 1)
    adjacency_matrix = np.array([[[0, 1], [0, 0]], [[0, 1], [0, 0]]])
    pd.DataFrame(data_single).to_csv(os.path.join(dataset_dir, "all.csv"), header=None, index=None)
    np.save(os.path.join(dataset_dir, "adj_matrix.npy"), adjacency_matrix)
    dataset_loader = TemporalCausalCSVDatasetLoader(dataset_dir=dataset_dir)
    dataset = dataset_loader.split_data_and_load_dataset(
        test_frac=0.5, val_frac=0.2, random_state=0, max_num_rows=None, timeseries_column_index=1
    )
    assert np.array_equal(dataset._train_data, np.expand_dims(np.arange(3), axis=1))
    assert np.array_equal(dataset._val_data, np.expand_dims(np.arange(3, 5), axis=1))
    assert np.array_equal(dataset._test_data, np.expand_dims(np.arange(5, 10), axis=1))
    assert check_temporal_order(dataset)

    # Multiple series with column index 0
    name_list = [2, 6, 3, 1, 4, 8, 10, 12, 14, 16]  # 10 series
    length_list = [5, 8, 4, 3, 1, 2, 7, 6, 4, 10]
    data_multi = generate_time_series(length_list=length_list, index_name=name_list)

    pd.DataFrame(data_multi).to_csv(os.path.join(dataset_dir, "all.csv"), header=None, index=None)
    np.save(os.path.join(dataset_dir, "adj_matrix.npy"), adjacency_matrix)

    dataset_loader = TemporalCausalCSVDatasetLoader(dataset_dir=dataset_dir)
    dataset = dataset_loader.split_data_and_load_dataset(
        test_frac=0.5, val_frac=0.2, random_state=0, max_num_rows=None, timeseries_column_index=0
    )
    assert len(dataset.train_segmentation) == 3
    assert len(dataset._test_segmentation) == 5
    assert len(dataset._val_segmentation) == 2

    # Multiple series with column index 1 but no test and val data
    data_multi[:, [0, 1]] = data_multi[:, [1, 0]]
    pd.DataFrame(data_multi).to_csv(os.path.join(dataset_dir, "all.csv"), header=None, index=None)
    np.save(os.path.join(dataset_dir, "adj_matrix.npy"), adjacency_matrix)

    dataset_loader = TemporalCausalCSVDatasetLoader(dataset_dir=dataset_dir)
    with pytest.raises(Exception):
        dataset = dataset_loader.split_data_and_load_dataset(
            test_frac=0, val_frac=0, random_state=0, max_num_rows=None, timeseries_column_index=1
        )
    # Multiple series with column index 1 but no val data
    dataset = dataset_loader.split_data_and_load_dataset(
        test_frac=0.49, val_frac=0.01, random_state=0, max_num_rows=None, timeseries_column_index=1
    )
    assert len(dataset.train_segmentation) == 6
    assert len(dataset._test_segmentation) == 4
    assert dataset._val_data is None

    # Multiple series without training data
    with pytest.raises(Exception):
        dataset = dataset_loader.split_data_and_load_dataset(
            test_frac=0.5, val_frac=0.5, random_state=0, max_num_rows=None, timeseries_column_index=1
        )

    # Basic type and shape checks
    assert isinstance(dataset, TemporalDataset)
    assert np.array_equal(adjacency_matrix, dataset.get_adjacency_data_matrix())


def test_load_predefined_dataset(tmpdir_factory):
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")

    data = np.stack([np.arange(10), np.ones(10)], 1)

    train_data = data[:5]
    val_data = data[5:7]
    test_data = data[7:]

    adjacency_matrix = np.array([[[0, 1], [0, 0]], [[0, 1], [0, 0]]])

    np.save(os.path.join(dataset_dir, "adj_matrix.npy"), adjacency_matrix)

    pd.DataFrame(train_data).to_csv(os.path.join(dataset_dir, "train.csv"), header=None, index=None)
    pd.DataFrame(val_data).to_csv(os.path.join(dataset_dir, "val.csv"), header=None, index=None)
    pd.DataFrame(test_data).to_csv(os.path.join(dataset_dir, "test.csv"), header=None, index=None)

    dataset_loader = TemporalCausalCSVDatasetLoader(dataset_dir=dataset_dir)
    dataset = dataset_loader.load_predefined_dataset(max_num_rows=None)

    # Basic type and shape checks
    assert isinstance(dataset, TemporalDataset)
    assert np.array_equal(adjacency_matrix, dataset.get_adjacency_data_matrix())


def generate_time_series(length_list, index_name):
    assert length_list is not None
    assert len(length_list) == len(index_name)
    assert 0 not in length_list
    column_1 = [[index_name[series_id] for _ in range(length_list[series_id])] for series_id in range(len(length_list))]
    column_1 = np.expand_dims(np.concatenate(column_1), axis=1)
    total_length = column_1.shape[0]
    column_2 = np.expand_dims(np.arange(total_length), axis=1)
    column_3 = np.expand_dims(np.arange(total_length), axis=1)
    data = np.concatenate((column_1, column_2, column_3), axis=1)
    return data


@pytest.fixture
def train_test_val_time_series():
    # Series length is 2,1,3,4. Total is 10
    # Series index name is chosen as 5,8,2,3
    train_length_list = [2, 1, 3, 4]
    train_index_name = [5, 8, 2, 3]
    train_data = generate_time_series(train_length_list, train_index_name)

    test_length_list = [1, 1, 1, 3]
    test_index_name = [1, 5, 6, 0]
    test_data = generate_time_series(test_length_list, test_index_name)

    val_length_list = [5]
    val_index_name = [3]
    val_data = generate_time_series(val_length_list, val_index_name)

    return train_data, test_data, val_data


# pylint: disable=redefined-outer-name
def test_process_dataset(tmpdir_factory, variable_data_example):
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    (train_data, variable_orig_type, variable_cts, variable_wrong_number) = variable_data_example

    pd.DataFrame(train_data).to_csv(os.path.join(dataset_dir, "train.csv"), header=None, index=None)

    dataset_loader = TemporalCausalCSVDatasetLoader(dataset_dir=dataset_dir)

    dataset = dataset_loader.load_predefined_dataset(max_num_rows=None, column_index=0)

    # test train segmentation
    assert isinstance(dataset, TemporalDataset)
    assert dataset._train_data.shape[1] == 2
    assert dataset._train_data.shape[0] == 10
    assert dataset.train_segmentation == [(i, i) for i in range(10)]

    # test variable type: cat and cts
    save_json(variable_orig_type, os.path.join(dataset_dir, "variables.json"))
    dataset_loader = TemporalCausalCSVDatasetLoader(dataset_dir=dataset_dir)
    dataset = dataset_loader.load_predefined_dataset(max_num_rows=None, column_index=0)
    assert len(dataset._variables) == 2
    assert dataset._variables[0].type_ == "categorical"
    assert dataset._variables[1].type_ == "continuous"

    # test variable cts
    save_json(variable_cts, os.path.join(dataset_dir, "variables.json"))
    dataset_loader = TemporalCausalCSVDatasetLoader(dataset_dir=dataset_dir)
    dataset = dataset_loader.load_predefined_dataset(max_num_rows=None, column_index=0)
    assert len(dataset._variables) == 2
    assert dataset._variables[0].type_ == "continuous"
    assert dataset._variables[1].type_ == "continuous"
    # variables_dict has wrong number of variables, raise assertion error
    with pytest.raises(AssertionError):
        save_json(variable_wrong_number, os.path.join(dataset_dir, "variables.json"))
        dataset_loader = TemporalCausalCSVDatasetLoader(dataset_dir=dataset_dir)
        dataset_loader.load_predefined_dataset(max_num_rows=None, column_index=0)
