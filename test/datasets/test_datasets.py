import numpy as np
import pandas as pd
import torch
from tensordict import TensorDict

from causica.datasets.causica_dataset_format import Variable, tensordict_from_variables_metadata
from causica.datasets.tensordict_utils import convert_one_hot, tensordict_from_pandas
from causica.datasets.variable_types import VariableTypeEnum


def test_dataset_without_groups():
    """Test dataset in the case of 1D node values."""
    data = np.random.rand(500, 5).astype(np.float32)
    columns = [f"x_{i}" for i in range(data.shape[1])]
    dataframe = pd.DataFrame(data, columns=columns)
    dataset_from_df = tensordict_from_pandas(df=dataframe)
    dataset_from_np = tensordict_from_variables_metadata(
        data, variables_list=[Variable(group_name=col, name=col) for col in columns]
    )

    assert all(column in dataset_from_df[5].keys() for column in columns)
    for key, val in dataset_from_df.items():
        assert val.shape == dataset_from_np[key].shape
    # test that getting an item is correct
    assert all(torch.allclose(dataset_from_df[5][key], dataset_from_np[5][key]) for key in dataset_from_np.keys())
    assert all(
        torch.allclose(dataset_from_df[5][key], torch.tensor(data[5, i : i + 1]))
        for i, key in enumerate(dataset_from_np.keys())
    )

    key_to_index = {col: i for i, col in enumerate(columns)}
    # test that the underlying datasets are the same as each other
    for key, df_data in dataset_from_df.items():
        np.testing.assert_allclose(df_data, dataset_from_np[key])

        i = key_to_index[key]
        np.testing.assert_allclose(df_data, data[:, i : i + 1])


def test_dataset_with_groups():
    """Test a dataset with "grouped" variables i.e. multidimensional datapoints per node"""
    dims = [5, 3, 1]
    data = np.random.rand(500, sum(dims)).astype(np.float32)
    col_name_list = list((f"x_{i}", f"x_{i}_{j}") for i, val in enumerate(dims) for j in range(val))

    columns = pd.MultiIndex.from_tuples(col_name_list)
    dataframe = pd.DataFrame(data, columns=columns)
    dataset_from_df = tensordict_from_pandas(df=dataframe)

    dataset_from_np = tensordict_from_variables_metadata(
        data,
        variables_list=[Variable(group_name=group_name, name=group_name) for group_name, _ in col_name_list],
    )

    for key, val in dataset_from_df.items():
        assert val.shape == dataset_from_np[key].shape

    # keys in the dataset should match those in the dataframe
    for key, group_name in zip(dataset_from_df.keys(), columns.get_level_values(0).unique()):
        assert key == group_name

    # check that the order of getitem is the same
    assert all(torch.allclose(dataset_from_df[5][key], dataset_from_np[5][key]) for key in dataset_from_np.keys())

    cum_sums = np.cumsum([0] + dims)
    col_name_dict = {f"x_{i}": [cum_sum, cum_sums[i + 1]] for i, cum_sum in enumerate(cum_sums[:-1])}
    # test that the underlying datasets are the same as each other
    for key, df_data in dataset_from_df.items():
        np.testing.assert_allclose(df_data, dataset_from_np[key])
        # test that the underlying dataset is the same as the original data
        start, end = col_name_dict[key]
        np.testing.assert_allclose(df_data, data[:, start:end])


def test_dataset_categorical():
    """Test the processing of categorical data."""
    key = "foo"
    key2 = "bar"
    batch_size = 321
    data = TensorDict(
        {
            key: np.random.binomial(1, 0.5, batch_size)[:, None].astype(np.int32),
            key2: np.random.rand(batch_size)[:, None].astype(np.float32),
        },
        batch_size=batch_size,
    )

    assert data[key].shape == (batch_size, 1)
    assert data[key2].shape == (batch_size, 1)

    # should be one hot encoded
    dataset = convert_one_hot(data, one_hot_sizes={key: 2})
    assert dataset[key].shape == (batch_size, 2)
    assert dataset[key2].shape == (batch_size, 1)

    dataset = convert_one_hot(data, one_hot_sizes={key: 3})
    assert dataset[key].shape == (batch_size, 3)
    assert dataset[key2].shape == (batch_size, 1)
    torch.testing.assert_close(dataset[key][:, -1], torch.zeros_like(dataset[key][:, -1]))
