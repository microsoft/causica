import os

import numpy as np
import pandas as pd
import pytest

from causica.datasets.csv_dataset_loader import CSVDatasetLoader
from causica.datasets.dataset import Dataset
from causica.datasets.variables import Variable, Variables
from causica.utils.io_utils import read_json_as, save_json


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

    pd.DataFrame(data).to_csv(os.path.join(dataset_dir, "all.csv"), header=None, index=None)
    dataset_loader = CSVDatasetLoader(dataset_dir=dataset_dir)
    dataset = dataset_loader.split_data_and_load_dataset(test_frac=0.5, val_frac=0.2, random_state=0, max_num_rows=None)

    expected_train_data = np.array([[0.0, 2.1], [0.0, 2.1], [0.0, 2.1]])  # Train data: 3 rows
    expected_train_mask = np.array([[0, 1], [0, 1], [0, 1]])
    expected_val_data = np.array([[0.0, 2.1], [0.0, 2.1]])  # Val data: 2 rows
    expected_val_mask = np.array([[0, 1], [0, 1]])
    expected_test_data = np.array([[0.0, 2.1], [0.0, 2.1], [0.0, 2.1], [0.0, 2.1], [0.0, 2.1]])  # Test data: 5 rows
    expected_test_mask = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])

    assert isinstance(dataset, Dataset)
    assert np.array_equal(expected_train_data, dataset.train_data_and_mask[0])
    assert np.array_equal(expected_train_mask, dataset.train_data_and_mask[1])
    assert np.array_equal(expected_val_data, dataset.val_data_and_mask[0])
    assert np.array_equal(expected_val_mask, dataset.val_data_and_mask[1])
    assert np.array_equal(expected_test_data, dataset.test_data_and_mask[0])
    assert np.array_equal(expected_test_mask, dataset.test_data_and_mask[1])
    assert not np.isnan(dataset.train_data_and_mask[0].astype(float)).any()
    assert not np.isnan(dataset.val_data_and_mask[0].astype(float)).any()
    assert not np.isnan(dataset.test_data_and_mask[0].astype(float)).any()


def test_split_data_and_load_dataset_zero_val_frac(tmpdir_factory):
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

    pd.DataFrame(data).to_csv(os.path.join(dataset_dir, "all.csv"), header=None, index=None)
    dataset_loader = CSVDatasetLoader(dataset_dir=dataset_dir)
    dataset = dataset_loader.split_data_and_load_dataset(test_frac=0.1, val_frac=0.0, random_state=0, max_num_rows=None)

    expected_train_data = np.array(
        [[0.0, 2.1], [0.0, 2.1], [0.0, 2.1], [0.0, 2.1], [0.0, 2.1], [0.0, 2.1], [0.0, 2.1], [0.0, 2.1], [0.0, 2.1]]
    )  # Train data: 9 rows
    expected_train_mask = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
    expected_val_data = None  # Val data: 0 rows -> None
    expected_val_mask = None
    expected_test_data = np.array([[0.0, 2.1]])  # Test data: 1 row
    expected_test_mask = np.array([[0, 1]])

    assert isinstance(dataset, Dataset)
    assert np.array_equal(expected_train_data, dataset.train_data_and_mask[0])
    assert np.array_equal(expected_train_mask, dataset.train_data_and_mask[1])
    assert np.array_equal(expected_val_data, dataset.val_data_and_mask[0])
    assert np.array_equal(expected_val_mask, dataset.val_data_and_mask[1])
    assert np.array_equal(expected_test_data, dataset.test_data_and_mask[0])
    assert np.array_equal(expected_test_mask, dataset.test_data_and_mask[1])
    assert not np.isnan(dataset.train_data_and_mask[0].astype(float)).any()
    assert not np.isnan(dataset.test_data_and_mask[0].astype(float)).any()


def test_split_data_and_load_dataset_max_num_rows_specified(tmpdir_factory):
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    data = np.array(
        [
            [np.nan, 2.1],  # Full data: 11 rows - Last row is ignored
            [np.nan, 2.1],
            [np.nan, 2.1],
            [np.nan, 2.1],
            [np.nan, 2.1],
            [np.nan, 2.1],
            [np.nan, 2.1],
            [np.nan, 2.1],
            [np.nan, 2.1],
            [np.nan, 2.1],
            [np.nan, 2.2],
        ]
    )

    pd.DataFrame(data).to_csv(os.path.join(dataset_dir, "all.csv"), header=None, index=None)
    dataset_loader = CSVDatasetLoader(dataset_dir=dataset_dir)
    dataset = dataset_loader.split_data_and_load_dataset(test_frac=0.5, val_frac=0.2, random_state=0, max_num_rows=10)

    expected_train_data = np.array([[0.0, 2.1], [0.0, 2.1], [0.0, 2.1]])  # Train data: 3 rows
    expected_train_mask = np.array([[0, 1], [0, 1], [0, 1]])
    expected_val_data = np.array([[0.0, 2.1], [0.0, 2.1]])  # Val data: 2 rows
    expected_val_mask = np.array([[0, 1], [0, 1]])
    expected_test_data = np.array([[0.0, 2.1], [0.0, 2.1], [0.0, 2.1], [0.0, 2.1], [0.0, 2.1]])  # Test data: 5 rows
    expected_test_mask = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])

    assert isinstance(dataset, Dataset)
    assert np.array_equal(expected_train_data, dataset.train_data_and_mask[0])
    assert np.array_equal(expected_train_mask, dataset.train_data_and_mask[1])
    assert np.array_equal(expected_val_data, dataset.val_data_and_mask[0])
    assert np.array_equal(expected_val_mask, dataset.val_data_and_mask[1])
    assert np.array_equal(expected_test_data, dataset.test_data_and_mask[0])
    assert np.array_equal(expected_test_mask, dataset.test_data_and_mask[1])
    assert not np.isnan(dataset.train_data_and_mask[0].astype(float)).any()
    assert not np.isnan(dataset.val_data_and_mask[0].astype(float)).any()
    assert not np.isnan(dataset.test_data_and_mask[0].astype(float)).any()


def test_split_data_and_load_dataset_more_rows(tmpdir_factory):
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    data = np.ones((20, 2))  # Full data: 20 rows

    pd.DataFrame(data).to_csv(os.path.join(dataset_dir, "all.csv"), header=None, index=None)
    dataset_loader = CSVDatasetLoader(dataset_dir=dataset_dir)
    dataset = dataset_loader.split_data_and_load_dataset(test_frac=0.5, val_frac=0.2, random_state=0, max_num_rows=None)

    expected_train_data = np.ones((6, 2))  # Train data: 6 rows
    expected_train_mask = np.ones((6, 2))
    expected_val_data = np.ones((4, 2))  # Val data: 4 rows
    expected_val_mask = np.ones((4, 2))
    expected_test_data = np.ones((10, 2))  # Test data: 10 rows
    expected_test_mask = np.ones((10, 2))

    assert isinstance(dataset, Dataset)
    assert np.array_equal(expected_train_data, dataset.train_data_and_mask[0])
    assert np.array_equal(expected_train_mask, dataset.train_data_and_mask[1])
    assert np.array_equal(expected_val_data, dataset.val_data_and_mask[0])
    assert np.array_equal(expected_val_mask, dataset.val_data_and_mask[1])
    assert np.array_equal(expected_test_data, dataset.test_data_and_mask[0])
    assert np.array_equal(expected_test_mask, dataset.test_data_and_mask[1])
    assert not np.isnan(dataset.train_data_and_mask[0].astype(float)).any()
    assert not np.isnan(dataset.val_data_and_mask[0].astype(float)).any()
    assert not np.isnan(dataset.test_data_and_mask[0].astype(float)).any()


def test_split_data_and_load_dataset_save_data_split(tmpdir_factory):
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    data = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]])

    pd.DataFrame(data).to_csv(os.path.join(dataset_dir, "all.csv"), header=None, index=None)
    dataset_loader = CSVDatasetLoader(dataset_dir=dataset_dir)
    dataset = dataset_loader.split_data_and_load_dataset(test_frac=0.3, val_frac=0.2, random_state=0, max_num_rows=None)

    save_dir = tmpdir_factory.mktemp("save_dir")
    dataset.save_data_split(save_dir=str(save_dir))

    saved_train_data = dataset.train_data_and_mask[0]
    saved_val_data = dataset.val_data_and_mask[0]
    saved_test_data = dataset.test_data_and_mask[0]

    data_split = read_json_as(os.path.join(save_dir, "data_split.json"), dict)

    assert data_split["train_idxs"] == list(saved_train_data[:, 0])
    assert data_split["val_idxs"] == list(saved_val_data[:, 0])
    assert data_split["test_idxs"] == list(saved_test_data[:, 0])


@pytest.mark.parametrize(
    "test_frac, val_frac", [(None, 0.25), (0.25, None), (None, None), (1, 0.25), (0.25, 1), (0.5, 0.5)]
)
def test_split_data_and_load_dataset_invalid_test_frac_val_frac_raises_error(tmpdir_factory, test_frac, val_frac):
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    data = np.ones((4, 5))
    pd.DataFrame(data).to_csv(os.path.join(dataset_dir, "all.csv"), header=None, index=None)
    with pytest.raises(AssertionError):
        dataset_loader = CSVDatasetLoader(dataset_dir=dataset_dir)
        _ = dataset_loader.split_data_and_load_dataset(
            test_frac=test_frac, val_frac=val_frac, random_state=0, max_num_rows=None
        )


@pytest.mark.parametrize("random_state", [(0), ((0, 1))])
def test_split_data_and_load_dataset_deterministic(tmpdir_factory, random_state):
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    data = np.array(
        [
            [0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, 1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, 2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, 3, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, 4, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 5, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 6, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 8, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 9],
        ]
    )
    pd.DataFrame(data).to_csv(os.path.join(dataset_dir, "all.csv"), header=None, index=None)
    dataset_loader = CSVDatasetLoader(dataset_dir=dataset_dir)
    dataset1 = dataset_loader.split_data_and_load_dataset(
        test_frac=0.3, val_frac=0.2, random_state=random_state, max_num_rows=None
    )
    dataset2 = dataset_loader.split_data_and_load_dataset(
        test_frac=0.3, val_frac=0.2, random_state=random_state, max_num_rows=None
    )

    assert np.array_equal(dataset1.train_data_and_mask[0], dataset2.train_data_and_mask[0])
    assert np.array_equal(dataset1.train_data_and_mask[1], dataset2.train_data_and_mask[1])
    assert np.array_equal(dataset1.val_data_and_mask[0], dataset2.val_data_and_mask[0])
    assert np.array_equal(dataset1.val_data_and_mask[1], dataset2.val_data_and_mask[1])
    assert np.array_equal(dataset1.test_data_and_mask[0], dataset2.test_data_and_mask[0])
    assert np.array_equal(dataset1.test_data_and_mask[1], dataset2.test_data_and_mask[1])


def test_split_data_and_load_dataset_deterministic_test_set(tmpdir_factory):
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    data = np.array(
        [
            [0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, 1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, 2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, 3, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, 4, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, 5, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 6, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 8, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 9],
        ]
    )
    pd.DataFrame(data).to_csv(os.path.join(dataset_dir, "all.csv"), header=None, index=None)
    dataset_loader = CSVDatasetLoader(dataset_dir=dataset_dir)
    dataset1 = dataset_loader.split_data_and_load_dataset(
        test_frac=0.3, val_frac=0.2, random_state=(0, 1), max_num_rows=None
    )
    dataset2 = dataset_loader.split_data_and_load_dataset(
        test_frac=0.3, val_frac=0.2, random_state=(0, 2), max_num_rows=None
    )

    assert not np.array_equal(dataset1.train_data_and_mask[0], dataset2.train_data_and_mask[0])
    assert not np.array_equal(dataset1.train_data_and_mask[1], dataset2.train_data_and_mask[1])
    assert not np.array_equal(dataset1.val_data_and_mask[0], dataset2.val_data_and_mask[0])
    assert not np.array_equal(dataset1.val_data_and_mask[1], dataset2.val_data_and_mask[1])
    assert np.array_equal(dataset1.test_data_and_mask[0], dataset2.test_data_and_mask[0])
    assert np.array_equal(dataset1.test_data_and_mask[1], dataset2.test_data_and_mask[1])


def test_load_predefined_dataset(tmpdir_factory):
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    train_data = np.array([[np.nan, 2.1], [2.2, np.nan]])
    val_data = np.array([[np.nan, 3.1]])
    test_data = np.array([[4.1, np.nan]])

    pd.DataFrame(train_data).to_csv(os.path.join(dataset_dir, "train.csv"), header=None, index=None)
    pd.DataFrame(val_data).to_csv(os.path.join(dataset_dir, "val.csv"), header=None, index=None)
    pd.DataFrame(test_data).to_csv(os.path.join(dataset_dir, "test.csv"), header=None, index=None)

    dataset_loader = CSVDatasetLoader(dataset_dir=dataset_dir)
    dataset = dataset_loader.load_predefined_dataset(max_num_rows=None)

    expected_train_data = np.array([[0.0, 2.1], [2.2, 0.0]])
    expected_train_mask = np.array([[0, 1], [1, 0]])
    expected_val_data = np.array([[0.0, 3.1]])
    expected_val_mask = np.array([[0, 1]])
    expected_test_data = np.array([[4.1, 0.0]])
    expected_test_mask = np.array([[1, 0]])

    assert isinstance(dataset, Dataset)
    assert np.array_equal(expected_train_data, dataset.train_data_and_mask[0])
    assert np.array_equal(expected_train_mask, dataset.train_data_and_mask[1])
    assert np.array_equal(expected_val_data, dataset.val_data_and_mask[0])
    assert np.array_equal(expected_val_mask, dataset.val_data_and_mask[1])
    assert np.array_equal(expected_test_data, dataset.test_data_and_mask[0])
    assert np.array_equal(expected_test_mask, dataset.test_data_and_mask[1])
    assert not np.isnan(dataset.train_data_and_mask[0].astype(float)).any()
    assert not np.isnan(dataset.val_data_and_mask[0].astype(float)).any()
    assert not np.isnan(dataset.test_data_and_mask[0].astype(float)).any()


def test_load_predefined_dataset_max_num_rows_specified(tmpdir_factory):
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    train_data = np.array([[np.nan, 2.1], [2.2, np.nan]])  # Train data: 2 rows - Second row is ignored
    val_data = np.array([[np.nan, 3.1], [np.nan, 3.2]])  # Val data: 2 rows - Second row is ignored
    test_data = np.array([[4.1, np.nan], [4.2, np.nan]])  # Test data: 2 rows - Second row is ignored

    pd.DataFrame(train_data).to_csv(os.path.join(dataset_dir, "train.csv"), header=None, index=None)
    pd.DataFrame(val_data).to_csv(os.path.join(dataset_dir, "val.csv"), header=None, index=None)
    pd.DataFrame(test_data).to_csv(os.path.join(dataset_dir, "test.csv"), header=None, index=None)

    dataset_loader = CSVDatasetLoader(dataset_dir=dataset_dir)
    dataset = dataset_loader.load_predefined_dataset(max_num_rows=1)

    expected_train_data = np.array([[0.0, 2.1]])
    expected_train_mask = np.array([[0, 1]])
    expected_val_data = np.array([[0.0, 3.1]])
    expected_val_mask = np.array([[0, 1]])
    expected_test_data = np.array([[4.1, 0.0]])
    expected_test_mask = np.array([[1, 0]])

    assert isinstance(dataset, Dataset)
    assert np.array_equal(expected_train_data, dataset.train_data_and_mask[0])
    assert np.array_equal(expected_train_mask, dataset.train_data_and_mask[1])
    assert np.array_equal(expected_val_data, dataset.val_data_and_mask[0])
    assert np.array_equal(expected_val_mask, dataset.val_data_and_mask[1])
    assert np.array_equal(expected_test_data, dataset.test_data_and_mask[0])
    assert np.array_equal(expected_test_mask, dataset.test_data_and_mask[1])
    assert not np.isnan(dataset.train_data_and_mask[0].astype(float)).any()
    assert not np.isnan(dataset.val_data_and_mask[0].astype(float)).any()
    assert not np.isnan(dataset.test_data_and_mask[0].astype(float)).any()


def test_load_predefined_dataset_save_data_split(tmpdir_factory):
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    train_data = np.ones((3, 2))
    val_data = np.ones((3, 2))
    test_data = np.ones((3, 2))

    pd.DataFrame(train_data).to_csv(os.path.join(dataset_dir, "train.csv"), header=None, index=None)
    pd.DataFrame(val_data).to_csv(os.path.join(dataset_dir, "val.csv"), header=None, index=None)
    pd.DataFrame(test_data).to_csv(os.path.join(dataset_dir, "test.csv"), header=None, index=None)

    dataset_loader = CSVDatasetLoader(dataset_dir=dataset_dir)
    dataset = dataset_loader.load_predefined_dataset(max_num_rows=None)

    save_dir = tmpdir_factory.mktemp("save_dir")
    dataset.save_data_split(save_dir=str(save_dir))

    expected_data_split = {"train_idxs": "predefined", "val_idxs": "predefined", "test_idxs": "predefined"}

    assert dataset.data_split == expected_data_split

    data_split = read_json_as(os.path.join(save_dir, "data_split.json"), dict)
    assert data_split == expected_data_split


# pylint: disable=protected-access
def test_load_variables_dict(tmpdir_factory):
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    variables_dict = {"metadata_variables": [], "variables": [{"id": 0}]}
    variables_dict_path = str(os.path.join(dataset_dir, "variables.json"))
    save_json(data=variables_dict, path=variables_dict_path)

    dataset_loader = CSVDatasetLoader(dataset_dir=dataset_dir)
    loaded_variables_dict = dataset_loader._load_variables_dict()
    assert loaded_variables_dict == variables_dict


def test_load_variables_dict_file_doesnt_exist(tmpdir_factory):
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    dataset_loader = CSVDatasetLoader(dataset_dir=dataset_dir)
    loaded_variables_dict = dataset_loader._load_variables_dict()
    assert loaded_variables_dict is None


@pytest.mark.parametrize(
    "max_num_rows, expected_data, expected_mask",
    [
        (None, np.array([[0.0, 1.1, 2.1], [3.1, 0.0, 5.1]]), np.array([[0, 1, 1], [1, 0, 1]])),
        (1, np.array([[0.0, 1.1, 2.1]]), np.array([[0, 1, 1]])),
    ],
)
def test_read_csv_from_file(tmpdir_factory, max_num_rows, expected_data, expected_mask):
    data = np.array([[np.nan, 1.1, 2.1], [3.1, np.nan, 5.1]])

    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    pd.DataFrame(data).to_csv(os.path.join(dataset_dir, "all.csv"), header=None, index=None)

    path = str(os.path.join(dataset_dir, "all.csv"))
    processed_data, processed_mask = CSVDatasetLoader.read_csv_from_file(path, max_num_rows=max_num_rows)

    assert np.array_equal(processed_data, expected_data)
    assert np.array_equal(processed_mask, expected_mask)
    assert not np.isnan(processed_mask).any()


def test_read_csv_from_strings():
    strings = [",1.1,2.1\n", "3.1,,5.1\n"]

    processed_data, processed_mask = CSVDatasetLoader.read_csv_from_strings(strings)

    expected_data = np.array([[0.0, 1.1, 2.1], [3.1, 0.0, 5.1]])
    expected_mask = np.array([[0, 1, 1], [1, 0, 1]])

    assert np.array_equal(processed_data, expected_data)
    assert np.array_equal(processed_mask, expected_mask)
    assert not np.isnan(processed_mask).any()


def test_load_negative_sampling_levels_path_exists(tmpdir_factory):
    # Test scenario where column IDs in data are [10, 11, ..., 19] and we map to variable IDs [0, 1, ..., 9]
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    dataset_loader = CSVDatasetLoader(dataset_dir=dataset_dir)
    df = pd.DataFrame({"col_id": list(range(10, 20)), "level": list(range(10))})
    df.to_csv(os.path.join(dataset_dir, "negative_sampling_levels.csv"), header=None, index=False)

    variables = Variables(
        [Variable(f"var_{i}", False, "binary", 0, 1) for i in range(10)], used_cols=list(range(10, 20))
    )
    negative_sampling_levels_path = os.path.join(dataset_loader.dataset_dir, dataset_loader._negative_sampling_file)
    levels = dataset_loader.load_negative_sampling_levels(negative_sampling_levels_path, variables)
    assert levels == {i: i for i in range(10)}


def test_load_negative_sampling_levels_path_does_not_exist(tmpdir_factory):
    # Test scenario where column IDs in data are [10, 11, ..., 19] and we map to variable IDs [0, 1, ..., 9]
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    dataset_loader = CSVDatasetLoader(dataset_dir=dataset_dir)

    variables = Variables(
        [Variable(f"var_{i}", False, "binary", 0, 1) for i in range(10)], used_cols=list(range(10, 20))
    )
    negative_sampling_levels_path = os.path.join(dataset_loader.dataset_dir, dataset_loader._negative_sampling_file)
    with pytest.raises(FileNotFoundError):
        _ = dataset_loader.load_negative_sampling_levels(negative_sampling_levels_path, variables)


def test_convert_negative_sampling_df_to_dict():
    df = pd.DataFrame({"col_id": list(range(10)), "level": list(range(10))})
    levels = CSVDatasetLoader._convert_negative_sampling_df_to_dict(df)
    assert levels == {i: i for i in range(10)}


def test_negative_sample(tmpdir):
    data = np.eye(5)
    data[0, 1] = 1
    mask = np.eye(5, dtype=bool)
    mask[0, 1] = 1
    mask[1, 2] = 1
    expected_data = data.copy()  # Make copy of original data before we apply negative sampling
    levels = {i: i for i in range(5)}

    dataset_loader = CSVDatasetLoader(dataset_dir=tmpdir)
    data, mask = dataset_loader.negative_sample(data, mask, levels)

    assert data.shape == mask.shape
    assert mask.dtype == bool
    # Shouldn't add any positive samples so data should stay the same
    assert np.all(data == expected_data)

    # Check original observed elements in mask unchanged
    for i in range(5):
        assert mask[i, i] == 1
    assert mask[0, 1] == 1
    assert mask[1, 2] == 1

    # Should sample 2 extra negative samples for row 0, 0 for row 1 (num positive = num negative), 1 for rows 2 + 3 and
    # 0 for row 4 (no candidates).
    assert mask[0].sum() == 4
    assert mask[1].sum() == 2
    assert mask[2].sum() == 2
    assert mask[3].sum() == 2
    assert mask[4].sum() == 1

    # Check no negative samples chosen below the main diagonal (since levels match the column IDs and row i always has
    # a positive element in col i).
    assert np.all(np.tril(mask, -1) == 0)


def test_apply_negative_sampling(tmpdir_factory):
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    dataset_loader = CSVDatasetLoader(dataset_dir=dataset_dir)
    df = pd.DataFrame({"col_id": list(range(10, 20)), "level": list(range(10))})
    df.to_csv(os.path.join(dataset_dir, "negative_sampling_levels.csv"), header=None, index=False)

    variables = Variables(
        [Variable(f"var_{i}", False, "binary", 0, 1) for i in range(10)], used_cols=list(range(10, 20))
    )
    train_data, train_mask = np.eye(10), np.eye(10)
    val_data, val_mask = np.eye(10), np.eye(10)
    test_data, test_mask = np.eye(10), np.eye(10)

    (
        train_data_ns,
        train_mask_ns,
        val_data_ns,
        val_mask_ns,
        test_data_ns,
        test_mask_ns,
    ) = dataset_loader._apply_negative_sampling(
        variables, train_data, train_mask, val_data, val_mask, test_data, test_mask
    )
    assert np.all(train_data_ns == train_data)
    assert train_mask_ns.sum() > train_mask.sum()
    assert np.all(val_data_ns == val_data)
    assert val_mask_ns.sum() > val_mask.sum()
    assert np.all(test_data_ns == test_data)
    assert test_mask_ns.sum() > test_mask.sum()


def test_apply_negative_sampling_no_file(tmpdir_factory):
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    dataset_loader = CSVDatasetLoader(dataset_dir=dataset_dir)

    variables = Variables(
        [Variable(f"var_{i}", False, "binary", 0, 1) for i in range(10)], used_cols=list(range(10, 20))
    )
    train_data, train_mask = np.eye(10), np.eye(10)
    val_data, val_mask = np.eye(10), np.eye(10)
    test_data, test_mask = np.eye(10), np.eye(10)

    with pytest.raises(FileNotFoundError):
        dataset_loader._apply_negative_sampling(
            variables, train_data, train_mask, val_data, val_mask, test_data, test_mask
        )


def test_process_data_numeric():
    data = np.array([[np.nan, 1.1, 2.1], [3.1, np.nan, 5.1]])

    processed_data, processed_mask = CSVDatasetLoader.process_data(data)

    expected_data = np.array([[0.0, 1.1, 2.1], [3.1, 0.0, 5.1]])
    expected_mask = np.array([[0, 1, 1], [1, 0, 1]])

    assert np.array_equal(processed_data, expected_data)
    assert np.array_equal(processed_mask, expected_mask)
    assert not np.isnan(processed_mask).any()


def test_process_data_text():
    data = np.array([["", "foo", "bar"], ["foo?", "", "bar!"], ["foobar", "!", ""]])
    expected_mask = ~np.eye(3, dtype=bool)

    processed_data, processed_mask = CSVDatasetLoader.process_data(data)

    assert np.array_equal(processed_data, data)
    assert np.array_equal(processed_mask, expected_mask)


def test_process_data_mixed_types():
    data = np.array(
        [[np.nan, "I would like to leave on monday .", 1.1, 2.1], [3.1, "", np.nan, 5.1], [3.1, "NaN", np.nan, 5.1]],
        dtype=object,
    )

    processed_data, processed_mask = CSVDatasetLoader.process_data(data)

    expected_data = np.array(
        [[0.0, "I would like to leave on monday .", 1.1, 2.1], [3.1, "", 0.0, 5.1], [3.1, "NaN", 0.0, 5.1]],
        dtype=object,
    )
    expected_mask = np.array([[0, 1, 1, 1], [1, 0, 0, 1], [1, 1, 0, 1]])

    print(processed_data)

    assert np.array_equal(processed_data, expected_data)
    assert np.array_equal(processed_mask, expected_mask)
    assert not np.isnan(processed_mask).any()
