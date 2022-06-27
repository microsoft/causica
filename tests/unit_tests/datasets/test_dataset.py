import os

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.sparse import csr_matrix

from causica.datasets.dataset import Dataset, SparseDataset
from causica.datasets.temporal_causal_csv_dataset_loader import TemporalCausalCSVDatasetLoader
from causica.datasets.temporal_tensor_dataset import TemporalTensorDataset
from causica.datasets.variables import Variables
from causica.preprocessing.data_processor import DataProcessor
from causica.utils.helper_functions import to_tensors
from causica.utils.io_utils import read_json_as


# pylint: disable=protected-access
@pytest.mark.parametrize(
    "train_mask, val_mask, test_mask",
    [
        (np.ones((2, 4)), np.ones((2, 3)), np.ones((2, 3))),  # Train data mask shape mismatch
        (np.ones((2, 3)), np.ones((2, 4)), np.ones((2, 3))),  # Val data mask shape mismatch
        (np.ones((2, 3)), np.ones((2, 3)), np.ones((2, 4))),  # Test data mask shape mismatch
    ],
)
def test_dataset_nonmatching_data_mask_shapes_raise_error(train_mask, val_mask, test_mask):
    train_data = np.ones((2, 3))
    val_data = np.ones((2, 3))
    test_data = np.ones((2, 3))
    variables_dict = {
        "variables": [
            {"id": 0, "query": True, "type": "binary", "name": "Column 0", "lower": 0, "upper": 1},
            {"id": 1, "query": True, "type": "binary", "name": "Column 1", "lower": 0, "upper": 1},
            {"id": 2, "query": True, "type": "binary", "name": "Column 2", "lower": 0, "upper": 1},
        ],
        "metadata_variables": [],
        "used_cols": [0, 1, 2],
    }
    variables = Variables.create_from_dict(variables_dict)
    with pytest.raises(AssertionError):
        pass
        _ = Dataset(train_data, train_mask, val_data, val_mask, test_data, test_mask, variables, data_split=None)


@pytest.mark.parametrize(
    "train_data, train_mask, val_data, val_mask, test_data, test_mask",
    [
        (
            np.ones((0, 3)),
            np.ones((0, 3)),
            np.ones((2, 3)),
            np.ones((2, 3)),
            np.ones((2, 3)),
            np.ones((2, 3)),
        ),  # Zero train rows
        (
            np.ones((2, 3)),
            np.ones((2, 3)),
            np.ones((0, 3)),
            np.ones((0, 3)),
            np.ones((2, 3)),
            np.ones((2, 3)),
        ),  # Zero val rows
        (
            np.ones((2, 3)),
            np.ones((2, 3)),
            np.ones((2, 3)),
            np.ones((2, 3)),
            np.ones((0, 3)),
            np.ones((0, 3)),
        ),  # Zero test rows
    ],
)
def test_dataset_zero_rows_raises_error(train_data, train_mask, val_data, val_mask, test_data, test_mask):
    variables_dict = {
        "variables": [
            {"id": 0, "query": True, "type": "binary", "name": "Column 0", "lower": 0, "upper": 1},
            {"id": 1, "query": True, "type": "binary", "name": "Column 1", "lower": 0, "upper": 1},
            {"id": 2, "query": True, "type": "binary", "name": "Column 2", "lower": 0, "upper": 1},
        ],
        "metadata_variables": [],
        "used_cols": [0, 1, 2],
    }
    variables = Variables.create_from_dict(variables_dict)
    with pytest.raises(AssertionError):
        _ = Dataset(train_data, train_mask, val_data, val_mask, test_data, test_mask, variables, data_split=None)


@pytest.mark.parametrize(
    "train_data, train_mask, val_data, val_mask, test_data, test_mask",
    [
        (
            np.ones((2, 3)),
            np.ones((2, 3)),
            np.ones((2, 4)),
            np.ones((2, 4)),
            np.ones((2, 3)),
            np.ones((2, 3)),
        ),  # Val data different # columns
        (
            np.ones((2, 3)),
            np.ones((2, 3)),
            np.ones((2, 3)),
            np.ones((2, 3)),
            np.ones((2, 4)),
            np.ones((2, 4)),
        ),  # Test data different # columns
    ],
)
def test_dataset_different_num_of_columns_raises_error(
    train_data, train_mask, val_data, val_mask, test_data, test_mask
):
    variables_dict = {
        "variables": [
            {"id": 0, "query": True, "type": "binary", "name": "Column 0", "lower": 0, "upper": 1},
            {"id": 1, "query": True, "type": "binary", "name": "Column 1", "lower": 0, "upper": 1},
            {"id": 2, "query": True, "type": "binary", "name": "Column 2", "lower": 0, "upper": 1},
        ],
        "metadata_variables": [],
        "used_cols": [0, 1, 2],
    }
    variables = Variables.create_from_dict(variables_dict)
    with pytest.raises(AssertionError):
        _ = Dataset(train_data, train_mask, val_data, val_mask, test_data, test_mask, variables, data_split=None)


def test_dataset_val_data_mask_are_none():
    train_data, train_mask = np.ones((2, 3)), np.ones((2, 3))
    test_data, test_mask = np.ones((2, 3)), np.ones((2, 3))
    variables_dict = {
        "variables": [
            {"id": 0, "query": True, "type": "binary", "name": "Column 0", "lower": 0, "upper": 1},
            {"id": 1, "query": True, "type": "binary", "name": "Column 1", "lower": 0, "upper": 1},
            {"id": 2, "query": True, "type": "binary", "name": "Column 2", "lower": 0, "upper": 1},
        ],
        "metadata_variables": [],
        "used_cols": [0, 1, 2],
    }
    variables = Variables.create_from_dict(variables_dict)

    _ = Dataset(
        train_data=train_data,
        train_mask=train_mask,
        val_data=None,
        val_mask=None,
        test_data=test_data,
        test_mask=test_mask,
        variables=variables,
        data_split=None,
    )


def test_dataset_data_is_immutable():
    train_data = np.ones((2, 3))
    train_mask = np.ones((2, 3))
    val_data = np.ones((2, 3))
    val_mask = np.ones((2, 3))
    test_data = np.ones((2, 3))
    test_mask = np.ones((2, 3))
    variables_dict = {
        "variables": [
            {"id": 0, "query": True, "type": "binary", "name": "Column 0", "lower": 0, "upper": 1},
            {"id": 1, "query": True, "type": "binary", "name": "Column 1", "lower": 0, "upper": 1},
            {"id": 2, "query": True, "type": "binary", "name": "Column 2", "lower": 0, "upper": 1},
        ],
        "metadata_variables": [],
        "used_cols": [0, 1, 2],
    }
    variables = Variables.create_from_dict(variables_dict)
    dataset = Dataset(train_data, train_mask, val_data, val_mask, test_data, test_mask, variables, data_split=None)

    saved_train_data, saved_train_mask = dataset.train_data_and_mask
    saved_val_data, saved_val_mask = dataset.val_data_and_mask
    saved_test_data, saved_test_mask = dataset.test_data_and_mask

    with pytest.raises(ValueError):
        saved_train_data[0, 0] = 0
    with pytest.raises(ValueError):
        saved_train_mask[0, 0] = 0
    with pytest.raises(ValueError):
        saved_val_data[0, 0] = 0
    with pytest.raises(ValueError):
        saved_val_mask[0, 0] = 0
    with pytest.raises(ValueError):
        saved_test_data[0, 0] = 0
    with pytest.raises(ValueError):
        saved_test_mask[0, 0] = 0


def test_dataset_data_and_mask():
    train_data = np.ones((3, 3)) * 2
    train_mask = np.ones((3, 3))
    train_mask[0, 0] = 0

    val_data = np.ones((2, 3)) * 3
    val_mask = np.ones((2, 3))
    val_mask[0, 1] = 0

    test_data = np.ones((1, 3)) * 4
    test_mask = np.ones((1, 3))
    test_mask[0, 2] = 0

    variables_dict = {
        "variables": [
            {"id": 0, "query": True, "type": "binary", "name": "Column 0", "lower": 0, "upper": 1},
            {"id": 1, "query": True, "type": "binary", "name": "Column 1", "lower": 0, "upper": 1},
            {"id": 2, "query": True, "type": "binary", "name": "Column 2", "lower": 0, "upper": 1},
        ],
        "metadata_variables": [],
        "used_cols": [0, 1, 2],
    }
    variables = Variables.create_from_dict(variables_dict)
    dataset = Dataset(
        train_data.copy(),
        train_mask.copy(),
        val_data.copy(),
        val_mask.copy(),
        test_data.copy(),
        test_mask.copy(),
        variables,
        data_split=None,
    )

    saved_train_data, saved_train_mask = dataset.train_data_and_mask
    saved_val_data, saved_val_mask = dataset.val_data_and_mask
    saved_test_data, saved_test_mask = dataset.test_data_and_mask

    assert np.array_equal(saved_train_data, train_data)
    assert np.array_equal(saved_train_mask, train_mask)
    assert np.array_equal(saved_val_data, val_data)
    assert np.array_equal(saved_val_mask, val_mask)
    assert np.array_equal(saved_test_data, test_data)
    assert np.array_equal(saved_test_mask, test_mask)


@pytest.mark.parametrize(
    "variables_dict", [(None), ({"metadata_variables": [], "variables": [{"id": 0}, {"id": 1}, {"id": 2}]})]
)
def test_dataset_variables(variables_dict):
    train_data = np.ones((2, 3))
    train_mask = np.ones((2, 3))
    val_data = np.ones((2, 3))
    val_mask = np.ones((2, 3))
    test_data = np.ones((2, 3))
    test_mask = np.ones((2, 3))
    variables_dict = {
        "variables": [
            {"id": 0, "query": True, "type": "binary", "name": "Column 0", "lower": 0, "upper": 1},
            {"id": 1, "query": True, "type": "binary", "name": "Column 1", "lower": 0, "upper": 1},
            {"id": 2, "query": True, "type": "binary", "name": "Column 2", "lower": 0, "upper": 1},
        ],
        "metadata_variables": [],
        "used_cols": [0, 1, 2],
    }
    variables = Variables.create_from_dict(variables_dict)
    dataset = Dataset(train_data, train_mask, val_data, val_mask, test_data, test_mask, variables, data_split=None)
    assert isinstance(dataset.variables, Variables)
    assert dataset.variables.to_dict() == variables.to_dict()


@pytest.mark.parametrize("data_split", [({"train_idxs": [0, 1], "val_idxs": [2, 3], "test_idxs": [4, 5]}), (None)])
def test_dataset_save_data_split(tmpdir_factory, data_split):
    train_data = np.ones((2, 3))
    train_mask = np.ones((2, 3))
    val_data = np.ones((2, 3))
    val_mask = np.ones((2, 3))
    test_data = np.ones((2, 3))
    test_mask = np.ones((2, 3))
    variables_dict = {
        "variables": [
            {"id": 0, "query": True, "type": "binary", "name": "Column 0", "lower": 0, "upper": 1},
            {"id": 1, "query": True, "type": "binary", "name": "Column 1", "lower": 0, "upper": 1},
            {"id": 2, "query": True, "type": "binary", "name": "Column 2", "lower": 0, "upper": 1},
        ],
        "metadata_variables": [],
        "used_cols": [0, 1, 2],
    }
    variables = Variables.create_from_dict(variables_dict)
    dataset = Dataset(
        train_data, train_mask, val_data, val_mask, test_data, test_mask, variables, data_split=data_split
    )
    dataset.save_data_split(save_dir="")

    save_dir = tmpdir_factory.mktemp("save_dir")
    if data_split is None:
        with pytest.warns(UserWarning):
            dataset.save_data_split(save_dir)
    else:
        dataset.save_data_split(save_dir)
        read_data_split = read_json_as(os.path.join(save_dir, "data_split.json"), dict)
        assert data_split == read_data_split


@pytest.mark.parametrize(
    "train_mask, val_mask, test_mask",
    [
        (
            csr_matrix(([2, 2], ([0, 1], [0, 1])), shape=(2, 2)),
            csr_matrix(([2, 2], ([0, 1], [0, 0])), shape=(2, 1)),
            csr_matrix(([2, 2], ([0, 1], [0, 0])), shape=(2, 1)),
        ),  # Train data mask shape mismatch
        (
            csr_matrix(([2, 2], ([0, 1], [0, 0])), shape=(2, 1)),
            csr_matrix(([2, 2], ([0, 1], [0, 1])), shape=(2, 2)),
            csr_matrix(([2, 2], ([0, 1], [0, 0])), shape=(2, 1)),
        ),  # Val data mask shape mismatch
        (
            csr_matrix(([2, 2], ([0, 1], [0, 0])), shape=(2, 1)),
            csr_matrix(([2, 2], ([0, 1], [0, 0])), shape=(2, 1)),
            csr_matrix(([2, 2], ([0, 1], [0, 1])), shape=(2, 2)),
        ),  # Test data mask shape mismatch
    ],
)
def test_sparse_dataset_nonmatching_data_mask_shapes_raise_error(train_mask, val_mask, test_mask):
    train_data = csr_matrix(([2, 2], ([0, 1], [0, 0])), shape=(2, 1))
    val_data = csr_matrix(([2, 2], ([0, 1], [0, 0])), shape=(2, 1))
    test_data = csr_matrix(([2, 2], ([0, 1], [0, 0])), shape=(2, 1))
    variables_dict = {
        "variables": [{"id": 0, "query": True, "type": "binary", "name": "Column 0", "lower": 0, "upper": 1}],
        "metadata_variables": [],
        "used_cols": [0],
    }
    variables = Variables.create_from_dict(variables_dict)
    with pytest.raises(AssertionError):
        _ = SparseDataset(train_data, train_mask, val_data, val_mask, test_data, test_mask, variables, data_split=None)


@pytest.mark.parametrize(
    "train_data, train_mask, val_data, val_mask, test_data, test_mask",
    [
        (
            csr_matrix(([], ([], [])), shape=(0, 1)),
            csr_matrix(([], ([], [])), shape=(0, 1)),
            csr_matrix(([2, 2], ([0, 1], [0, 0])), shape=(2, 1)),
            csr_matrix(([2, 2], ([0, 1], [0, 0])), shape=(2, 1)),
            csr_matrix(([2, 2], ([0, 1], [0, 0])), shape=(2, 1)),
            csr_matrix(([2, 2], ([0, 1], [0, 0])), shape=(2, 1)),
        ),  # Zero train rows
        (
            csr_matrix(([2, 2], ([0, 1], [0, 0])), shape=(2, 1)),
            csr_matrix(([2, 2], ([0, 1], [0, 0])), shape=(2, 1)),
            csr_matrix(([], ([], [])), shape=(0, 1)),
            csr_matrix(([], ([], [])), shape=(0, 1)),
            csr_matrix(([2, 2], ([0, 1], [0, 0])), shape=(2, 1)),
            csr_matrix(([2, 2], ([0, 1], [0, 0])), shape=(2, 1)),
        ),  # Zero val rows
        (
            csr_matrix(([2, 2], ([0, 1], [0, 0])), shape=(2, 1)),
            csr_matrix(([2, 2], ([0, 1], [0, 0])), shape=(2, 1)),
            csr_matrix(([2, 2], ([0, 1], [0, 0])), shape=(2, 1)),
            csr_matrix(([2, 2], ([0, 1], [0, 0])), shape=(2, 1)),
            csr_matrix(([], ([], [])), shape=(0, 1)),
            csr_matrix(([], ([], [])), shape=(0, 1)),
        ),  # Zero test rows
    ],
)
def test_sparse_dataset_zero_rows_raises_error(train_data, train_mask, val_data, val_mask, test_data, test_mask):
    variables_dict = {
        "variables": [{"id": 0, "query": True, "type": "binary", "name": "Column 0", "lower": 0, "upper": 1}],
        "metadata_variables": [],
        "used_cols": [0],
    }
    variables = Variables.create_from_dict(variables_dict)
    with pytest.raises(AssertionError):
        _ = SparseDataset(train_data, train_mask, val_data, val_mask, test_data, test_mask, variables, data_split=None)


@pytest.mark.parametrize(
    "train_data, train_mask, val_data, val_mask, test_data, test_mask",
    [
        (
            csr_matrix(([2, 2], ([0, 1], [0, 0])), shape=(2, 1)),
            csr_matrix(([2, 2], ([0, 1], [0, 0])), shape=(2, 1)),
            csr_matrix(([2, 2], ([0, 1], [0, 1])), shape=(2, 2)),
            csr_matrix(([2, 2], ([0, 1], [0, 1])), shape=(2, 2)),
            csr_matrix(([2, 2], ([0, 1], [0, 0])), shape=(2, 1)),
            csr_matrix(([2, 2], ([0, 1], [0, 0])), shape=(2, 1)),
        ),  # Val data different # columns
        (
            csr_matrix(([2, 2], ([0, 1], [0, 0])), shape=(2, 1)),
            csr_matrix(([2, 2], ([0, 1], [0, 0])), shape=(2, 1)),
            csr_matrix(([2, 2], ([0, 1], [0, 0])), shape=(2, 1)),
            csr_matrix(([2, 2], ([0, 1], [0, 0])), shape=(2, 1)),
            csr_matrix(([2, 2], ([0, 1], [0, 1])), shape=(2, 2)),
            csr_matrix(([2, 2], ([0, 1], [0, 1])), shape=(2, 2)),
        ),  # Test data different # columns
    ],
)
def test_sparse_dataset_different_num_of_columns_raises_error(
    train_data, train_mask, val_data, val_mask, test_data, test_mask
):
    variables_dict = {
        "variables": [{"id": 0, "query": True, "type": "binary", "name": "Column 0", "lower": 0, "upper": 1}],
        "metadata_variables": [],
        "used_cols": [0],
    }
    variables = Variables.create_from_dict(variables_dict)
    with pytest.raises(AssertionError):
        _ = SparseDataset(train_data, train_mask, val_data, val_mask, test_data, test_mask, variables, data_split=None)


def test_sparse_dataset_data_and_mask():
    train_data = csr_matrix(([2, 2, 2, 2, 2], ([0, 1, 2, 3, 4], [0, 0, 0, 0, 0])), shape=(5, 1))
    train_mask = csr_matrix(([1, 1, 1, 1, 1], ([0, 1, 2, 3, 4], [0, 0, 0, 0, 0])), shape=(5, 1))
    val_data = csr_matrix(([3, 3], ([0, 1], [0, 0])), shape=(2, 1))
    val_mask = csr_matrix(([1, 1], ([0, 1], [0, 0])), shape=(2, 1))
    test_data = csr_matrix(([4, 4, 4], ([0, 1, 2], [0, 0, 0])), shape=(3, 1))
    test_mask = csr_matrix(([1, 1, 1], ([0, 1, 2], [0, 0, 0])), shape=(3, 1))
    variables_dict = {
        "variables": [{"id": 0, "query": True, "type": "binary", "name": "Column 0", "lower": 0, "upper": 1}],
        "metadata_variables": [],
        "used_cols": [0],
    }
    variables = Variables.create_from_dict(variables_dict)

    dataset = SparseDataset(
        train_data, train_mask, val_data, val_mask, test_data, test_mask, variables, data_split=None
    )

    saved_train_data, saved_train_mask = dataset.train_data_and_mask
    saved_val_data, saved_val_mask = dataset.val_data_and_mask
    saved_test_data, saved_test_mask = dataset.test_data_and_mask

    expected_train_data = np.ones((5, 1)) * 2
    expected_train_mask = np.ones((5, 1))
    expected_val_data = np.ones((2, 1)) * 3
    expected_val_mask = np.ones((2, 1))
    expected_test_data = np.ones((3, 1)) * 4
    expected_test_mask = np.ones((3, 1))

    assert np.array_equal(saved_train_data.toarray(), expected_train_data)
    assert np.array_equal(saved_train_mask.toarray(), expected_train_mask)
    assert np.array_equal(saved_val_data.toarray(), expected_val_data)
    assert np.array_equal(saved_val_mask.toarray(), expected_val_mask)
    assert np.array_equal(saved_test_data.toarray(), expected_test_data)
    assert np.array_equal(saved_test_mask.toarray(), expected_test_mask)


@pytest.mark.parametrize("data_split", [({"train_idxs": [0, 1], "val_idxs": [2, 3], "test_idxs": [4, 5]}), (None)])
def test_sparse_dataset_save_data_split(tmpdir_factory, data_split):
    train_data = csr_matrix(([1, 1], ([0, 1], [0, 0])), shape=(2, 1))
    train_mask = csr_matrix(([1, 1], ([0, 1], [0, 0])), shape=(2, 1))
    val_data = csr_matrix(([1, 1], ([0, 1], [0, 0])), shape=(2, 1))
    val_mask = csr_matrix(([1, 1], ([0, 1], [0, 0])), shape=(2, 1))
    test_data = csr_matrix(([1, 1], ([0, 1], [0, 0])), shape=(2, 1))
    test_mask = csr_matrix(([1, 1], ([0, 1], [0, 0])), shape=(2, 1))
    variables_dict = {
        "variables": [{"id": 0, "query": True, "type": "binary", "name": "Column 0", "lower": 0, "upper": 1}],
        "metadata_variables": [],
        "used_cols": [0],
    }
    variables = Variables.create_from_dict(variables_dict)
    dataset = SparseDataset(
        train_data, train_mask, val_data, val_mask, test_data, test_mask, variables, data_split=data_split
    )
    dataset.save_data_split(save_dir="")

    save_dir = tmpdir_factory.mktemp("save_dir")
    if data_split is None:
        with pytest.warns(UserWarning):
            dataset.save_data_split(save_dir)
    else:
        dataset.save_data_split(save_dir)
        read_data_split = read_json_as(os.path.join(save_dir, "data_split.json"), dict)
        assert data_split == read_data_split


def test_sparse_dataset_to_dense():
    train_data = csr_matrix(([1, 1], ([0, 1], [0, 0])), shape=(2, 1))
    train_mask = csr_matrix(([1, 1], ([0, 1], [0, 0])), shape=(2, 1))
    val_data = csr_matrix(([2, 2], ([0, 1], [0, 0])), shape=(2, 1))
    val_mask = csr_matrix(([1, 1], ([0, 1], [0, 0])), shape=(2, 1))
    test_data = csr_matrix(([3, 3], ([0, 1], [0, 0])), shape=(2, 1))
    test_mask = csr_matrix(([1, 1], ([0, 1], [0, 0])), shape=(2, 1))
    variables_dict = {
        "variables": [{"id": 0, "query": True, "type": "binary", "name": "Column 0", "lower": 0, "upper": 1}],
        "metadata_variables": [],
        "used_cols": [0],
    }
    variables = Variables.create_from_dict(variables_dict)
    data_split = {"train_idxs": "predefined", "val_idxs": "predefined", "test_idxs": "predefined"}
    sparse_dataset = SparseDataset(
        train_data, train_mask, val_data, val_mask, test_data, test_mask, variables, data_split
    )
    dense_dataset = sparse_dataset.to_dense()

    assert np.array_equal(dense_dataset.train_data_and_mask[0], np.ones((2, 1)))
    assert np.array_equal(dense_dataset.train_data_and_mask[1], np.ones((2, 1)))
    assert np.array_equal(dense_dataset.val_data_and_mask[0], np.ones((2, 1)) * 2)
    assert np.array_equal(dense_dataset.val_data_and_mask[1], np.ones((2, 1)))
    assert np.array_equal(dense_dataset.test_data_and_mask[0], np.ones((2, 1)) * 3)
    assert np.array_equal(dense_dataset.test_data_and_mask[1], np.ones((2, 1)))
    assert dense_dataset.variables.to_dict() == variables.to_dict()
    assert dense_dataset.data_split == data_split


def generate_time_series(length_list, index_name):
    assert length_list is not None
    assert len(length_list) == len(index_name)
    assert 0 not in length_list
    column_1 = [[index_name[series_id] for _ in range(length)] for series_id, length in enumerate(length_list)]
    column_1 = np.concatenate(column_1)
    total_length = column_1.shape[0]
    column_2 = np.arange(total_length) + 0.1
    data = np.stack((column_1, column_2), axis=1)
    return data


@pytest.fixture
def train_test_val_time_series():
    # Series length is 3,3,3,4. Total is 13
    # Series index name is chosen as 5,8,2,3
    train_length_list = [3, 3, 3, 4]
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
def test_TemporalTensorDataset(tmpdir_factory, train_test_val_time_series):
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    train_data, test_data, val_data = train_test_val_time_series

    pd.DataFrame(train_data).to_csv(os.path.join(dataset_dir, "train.csv"), header=None, index=None)
    pd.DataFrame(val_data).to_csv(os.path.join(dataset_dir, "val.csv"), header=None, index=None)
    pd.DataFrame(test_data).to_csv(os.path.join(dataset_dir, "test.csv"), header=None, index=None)

    dataset_loader = TemporalCausalCSVDatasetLoader(dataset_dir=dataset_dir)

    dataset = dataset_loader.load_predefined_dataset(max_num_rows=None)

    data_processor = DataProcessor(dataset._variables, unit_scale_continuous=False)
    proc_dataset = data_processor.process_dataset(dataset)

    # train data
    data, mask = proc_dataset.train_data_and_mask

    # with lag = 2
    tensor_dataset = TemporalTensorDataset(
        *to_tensors(data, mask, device=torch.device("cpu")),
        lag=2,
        is_autoregressive=True,
        index_segmentation=proc_dataset.train_segmentation
    )
    assert len(tensor_dataset) == 5

    assert torch.all(tensor_dataset[0][0] == torch.Tensor([[0.1], [1.1], [2.1]]).to(torch.device("cpu")))
    assert torch.all(tensor_dataset[3][0] == torch.Tensor([[9.1], [10.1], [11.1]]).to(torch.device("cpu")))
    assert torch.all(tensor_dataset[4][0] == torch.Tensor([[10.1], [11.1], [12.1]]).to(torch.device("cpu")))

    # with lag = 3 should raise an assertion error
    with pytest.raises(Exception):
        tensor_dataset = TemporalTensorDataset(
            *to_tensors(data, mask, device=torch.device("cpu")),
            lag=3,
            is_autoregressive=True,
            index_segmentation=proc_dataset.train_segmentation
        )
    # with lag = 0
    tensor_dataset = TemporalTensorDataset(
        *to_tensors(data, mask, device=torch.device("cpu")),
        lag=0,
        is_autoregressive=True,
        index_segmentation=proc_dataset.train_segmentation
    )
    assert len(tensor_dataset) == 13
    assert tensor_dataset[0][0] == torch.Tensor([[0.1]]).to(torch.device("cpu"))
    assert tensor_dataset[12][0] == torch.Tensor([[12.1]]).to(torch.device("cpu"))

    # Fold-time format
    tensor_dataset = TemporalTensorDataset(
        *to_tensors(data, mask, device=torch.device("cpu")),
        lag=2,
        is_autoregressive=False,
        index_segmentation=proc_dataset.train_segmentation
    )
    assert torch.all(tensor_dataset[0][0] == torch.Tensor([0.1, 1.1, 2.1]).to(torch.device("cpu")))
    assert torch.all(tensor_dataset[4][0] == torch.Tensor([10.1, 11.1, 12.1]).to(torch.device("cpu")))
    assert torch.all(tensor_dataset[3][0] == torch.Tensor([9.1, 10.1, 11.1]).to(torch.device("cpu")))
