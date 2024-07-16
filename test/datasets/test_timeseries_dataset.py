from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader

from causica.datasets.causica_dataset_format import Variable, VariablesMetadata
from causica.datasets.timeseries_dataset import (
    IndexedTimeseriesDataset,
    ensure_adjacency_matrix,
    ensure_variables_metadata,
    index_contiguous_chunks,
    preprocess_data,
)
from causica.distributions.adjacency.erdos_renyi import ErdosRenyiDAGDistribution


@pytest.mark.parametrize("data", [torch.randn((10, 3)), np.random.randn(10, 3)])
def test_preprocess_data_with_tensor(data):
    # Check correctness without metadata
    tensordict_data = preprocess_data(data)
    tensor_data = torch.tensor(data, dtype=torch.float32)  # Will be converted when creating a TensorDict with metadata
    for i, (key, values) in enumerate(sorted(tensordict_data.items())):
        assert i == int(key.lstrip("x"))
        torch.testing.assert_close(values.squeeze(1), tensor_data[..., i])

    # Check correctness with metadata
    keys = ["a", "b", "c"]
    variables_metadata = VariablesMetadata([Variable(key, key) for key in keys])
    tensordict_data = preprocess_data(data, variables_metadata)
    for i, key in enumerate(keys):
        torch.testing.assert_close(tensordict_data[key].squeeze(1), tensor_data[..., i])

    # Check that incorrect metadata raises an error
    with pytest.raises(ValueError):
        preprocess_data(data, replace(variables_metadata, variables=variables_metadata.variables[:-1]))


def test_preprocess_data_with_tensordict():
    data = TensorDict({"a": torch.randn(10, 3), "b": torch.randn(10, 3), "c": torch.randn(10, 3)}, batch_size=10)
    tensordict_data = preprocess_data(data)
    assert tensordict_data is data

    selected_keys = list(data.keys())[:-1]
    variables_metadata = VariablesMetadata([Variable(key, key) for key in selected_keys])
    selected_data = preprocess_data(tensordict_data, variables_metadata)
    torch.testing.assert_close(selected_data, data.select(*selected_keys))


@pytest.mark.parametrize("num_dims", [1, 11, 17])
@pytest.mark.parametrize("num_repetitions", [1, 3, 5])
def test_index_contiguous_chunks(num_dims, num_repetitions):
    td = TensorDict({"value": torch.eye(num_dims), "index": torch.arange(num_dims)}, batch_size=num_dims)
    data = torch.cat([td] * num_repetitions)
    data["order"] = torch.arange(data.size(0))  # Set a global order to verify that the relative order is intact

    contiguous_data, slices = index_contiguous_chunks(data, "index")

    for dim in range(num_dims):
        chunk = contiguous_data[slices[dim]]
        expected_value = torch.eye(num_dims)[[dim] * num_repetitions]

        # Verify that the correct value was retrieved
        torch.testing.assert_close(chunk["value"], expected_value)
        # Verify that the relative order is intact
        torch.testing.assert_close(chunk["order"], torch.sort(chunk["order"]).values)


@pytest.mark.parametrize("num_nodes", [2, 3, 10])
def test_ensure_adjacency_matrix(num_nodes: int, tmp_path: Path):
    adjacency_matrix = ErdosRenyiDAGDistribution(num_nodes, probs=torch.tensor(0.9)).sample().to(torch.int64)

    file_path_csv = tmp_path / "adjacency_matrix.csv"
    np.savetxt(str(file_path_csv), adjacency_matrix.numpy(), delimiter=",")
    loaded_adjacency_matrix = ensure_adjacency_matrix(str(file_path_csv))
    torch.testing.assert_close(loaded_adjacency_matrix, adjacency_matrix)

    file_path_npy = tmp_path / "adjacency_matrix.npy"
    np.save(str(file_path_npy), adjacency_matrix.numpy())
    loaded_adjacency_matrix = ensure_adjacency_matrix(str(file_path_npy))
    torch.testing.assert_close(loaded_adjacency_matrix, adjacency_matrix)

    with pytest.raises(ValueError):
        ensure_adjacency_matrix(str(tmp_path / "adjacency_matrix.txt"))

    torch.testing.assert_close(ensure_adjacency_matrix(adjacency_matrix.numpy()), adjacency_matrix)
    torch.testing.assert_close(ensure_adjacency_matrix(adjacency_matrix), adjacency_matrix)


def test_ensure_variables_metadata(tmp_path: Path):
    keys = ["a", "b", "c"]
    variables_metadata = VariablesMetadata([Variable(key, key) for key in keys])
    assert ensure_variables_metadata(variables_metadata) is variables_metadata

    filepath = tmp_path / "variables_metadata.json"
    with filepath.open("w") as f:
        f.write(variables_metadata.to_json())  # type: ignore  # MyPy is unable to find the method from dataclass_json
    assert ensure_variables_metadata(str(filepath)) == variables_metadata


@pytest.mark.parametrize("num_timeseries", [1, 33, 101])
@pytest.mark.parametrize("max_length", [1, 5, 77])
@pytest.mark.parametrize("num_dims", [1, 3, 10])
def test_indexed_timeseries_dataset(num_timeseries: int, max_length: int, num_dims: int):
    # Create dense timeseries with varying length for easier verification
    data = torch.randn(num_timeseries, max_length, num_dims)
    lengths = torch.randint(1, max_length + 1, (num_timeseries,))
    adjacency_matrix = ErdosRenyiDAGDistribution(num_dims, probs=torch.tensor(0.9)).sample()
    timeseries_dataset = IndexedTimeseriesDataset.from_dense(data, lengths, adjacency_matrix=adjacency_matrix)

    # Verify access to each timeseries
    assert len(timeseries_dataset) == num_timeseries
    for i in range(num_timeseries):
        for j in range(num_dims):
            # Take feature j from the dense timeseries i, add a dummy dim to match how the behavior of TensorDict
            # datasets which generally always have a feature axis even for scalar features.
            xj = data[i][: lengths[i], :][..., j, None]
            torch.testing.assert_close(timeseries_dataset[i][f"x{j}"], xj)


@pytest.mark.parametrize("num_timeseries", [33])
@pytest.mark.parametrize("max_length", [11])
@pytest.mark.parametrize("num_dims", [1, 3])
@pytest.mark.parametrize("batch_size", [1, 5])
def test_batching_indexed_timeseries_dataset_fixed_length(
    num_timeseries: int, max_length: int, num_dims: int, batch_size: int
):
    data = torch.randn(num_timeseries, max_length, num_dims)
    lengths = torch.full((num_timeseries,), max_length, dtype=torch.int32)
    adjacency_matrix = ErdosRenyiDAGDistribution(num_dims, probs=torch.tensor(0.9)).sample()
    timeseries_dataset = IndexedTimeseriesDataset.from_dense(data, lengths, adjacency_matrix=adjacency_matrix)

    data_loader = DataLoader(
        timeseries_dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=torch.stack
    )

    count = 0
    for i, batch in enumerate(data_loader):
        if i == len(timeseries_dataset) // batch_size:
            assert batch.batch_size == (num_timeseries % batch_size, max_length)
        else:
            assert batch.batch_size == (batch_size, max_length)
        count += batch.batch_size[0]
    assert count == num_timeseries
