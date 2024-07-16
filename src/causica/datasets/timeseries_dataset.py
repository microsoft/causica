from typing import Type, TypeVar

import fsspec
import numpy as np
import torch
from tensordict import TensorDictBase
from tensordict.utils import DeviceType
from torch.utils.data import Dataset

from causica.datasets.causica_dataset_format.load import (
    Variable,
    VariablesMetadata,
    get_categorical_sizes,
    tensordict_from_variables_metadata,
)
from causica.datasets.tensordict_utils import convert_one_hot


def load_adjacency_matrix(path: str) -> torch.Tensor:
    """Load an adjacency matrix from a file path.

    Args:
        path: Path to a .csv or .npy file

    Returns:
        Loaded adjacency matrix.
    """
    if path.endswith(".csv"):
        with fsspec.open_files(path, "r") as files:
            (file,) = files
            return torch.tensor(np.loadtxt(file, dtype=int, delimiter=","))
    elif path.endswith(".npy"):
        with fsspec.open_files(path, "rb") as files:
            (file,) = files
            return torch.tensor(np.load(file))
    else:
        raise ValueError("Unsupported file format.")


def ensure_adjacency_matrix(
    adjacency_matrix: str | np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Ensure that the adjacency matrix is a tensor.

    Args:
        adjacency_matrix: Adjacency matrix or path to CSV file with adjacency matrix.

    Returns:
        Adjacency matrix as a tensor.
    """
    if isinstance(adjacency_matrix, str):
        adjacency_matrix = load_adjacency_matrix(adjacency_matrix)
    elif isinstance(adjacency_matrix, np.ndarray):
        adjacency_matrix = torch.tensor(adjacency_matrix)
    return adjacency_matrix


def ensure_variables_metadata(
    variables_metadata: str | VariablesMetadata,
) -> VariablesMetadata:
    """Ensure that the variables metadata is loaded.

    Args:
        variables_metadata: Variables metadata or path to JSON file with variables metadata.

    Returns:
        Variables metadata.
    """
    if isinstance(variables_metadata, str):
        with fsspec.open(variables_metadata, "r") as f:
            return VariablesMetadata.from_json(f.read())  # type: ignore
    return variables_metadata


def preprocess_data(
    data: str | np.ndarray | torch.Tensor | TensorDictBase, variables_metadata: VariablesMetadata | None = None
) -> TensorDictBase:
    """Preprocess and if necessary load and format the data.

    Args:
        data: Data or path to a CSV file with data and no header.
        variables_metadata: Variables metadata, if provided will inform the names and preprocessing of variables and
            only select the ones present here. Otherwise all will be assumed continuous variables and named by their
            index.

    Returns:
        Data is formatted as if read by `tensordict_from_variables_metadata`.
    """
    # First, collapse types so that data is torch.Tensor | TensorDictBase
    if isinstance(data, str):
        with fsspec.open(data, "r") as f:
            data = np.loadtxt(f, delimiter=",")
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)

    # Then, ensure data is a TensorDict with variables following the metadata
    if isinstance(data, torch.Tensor):
        if variables_metadata is None:
            variables_metadata = VariablesMetadata([Variable(f"x{i}", f"x{i}") for i in range(data.shape[1])])
        data = tensordict_from_variables_metadata(data, variables_metadata.variables)
    elif variables_metadata is not None:
        data = data.select(*(variable.name for variable in variables_metadata.variables))
    else:  # No metadata and data is already a TensorDict
        return data

    # One-hot encode categorical variables if indicated by variables metadata
    categorical_sizes = get_categorical_sizes(variables_list=variables_metadata.variables)
    return convert_one_hot(data, one_hot_sizes=categorical_sizes)


TD = TypeVar("TD", bound=TensorDictBase)


def index_contiguous_chunks(td: TD, index_key: str) -> tuple[TD, dict[int, slice]]:
    """Index a TensorDict into contiguous chunks maintaing the relative order by one of its values.

    Args:
        td: TensorDict to index.
        index_key: Key to the index value in td. The value must be scalar or 1D apart from its batch shape.

    Example:
        >> td = TensorDict({"index": torch.tensor([0, 0, 1, 1, 0, 2, 2]), "x": torch.tensor([4, 5, 6, 7, 8, 9, 10])})
        >> index_contiguous_chunks(td, "index")
        ({x: torch.tensor([4, 5, 8, 6, 7, 9, 10])}, {0: slice(0, 3), 1: slice(3, 5), 2: slice(5, 7)})

    Returns:
        A contigous copy of the TensorDict without the index value.
        A mapping from index to slice which can be used to directly map to the chunk in the TensorDict.
    """
    # Get the series index and sort to maintain order within series but make sure that all series are contiguous
    td = td.clone(recurse=False)
    index = td.pop(index_key)
    if index.dim() > 2 or (index.dim() == 2 and index.shape[1] > 1):
        raise ValueError("Pivot index must be scalar or only have one dimension across the batch.")
    if index.dim() == 2:
        index = index.squeeze(1)

    # Order by the index and build contiguous copies
    index, order = torch.sort(index, stable=True)
    td = td[order].contiguous()

    # Compute timeseries locations
    series_boundaries = torch.diff(index, dim=0)
    series_locs, *_ = torch.nonzero(series_boundaries, as_tuple=True)
    # Add the first and last index to build the ranges
    series_ranges = torch.cat(
        [
            torch.zeros((1,), dtype=series_locs.dtype),
            series_locs + 1,
            torch.tensor([len(index)], dtype=series_locs.dtype),
        ]
    )

    # Build a mapping to lookup indices quickly
    return td, {
        index: slice(a.item(), b.item()) for index, (a, b) in enumerate(zip(series_ranges[:-1], series_ranges[1:]))
    }


CLS = TypeVar("CLS", bound="IndexedTimeseriesDataset")


class IndexedTimeseriesDataset(Dataset):
    """Represents a timeseries dataset where each series is indexed by a unique identifier and the steps given by order.

    The raw dataset can be provided in the following formats:
    - TensorDict: the expected batch size is the total number of steps from all series, and the key / values refer
        to individual features.
    - Tensor or filepaths: a tensor type or a path to a CSV file with a table. Both with the expected shape
        [`total number of steps from all series`, `num features`].

    The relative order of each series is maintened and the provided dataset must therefore already be in timestep order.
    """

    @torch.no_grad()
    def __init__(
        self,
        series_index_key: str | int,
        data: str | np.ndarray | torch.Tensor | TensorDictBase,
        adjacency_matrix: str | np.ndarray | torch.Tensor,
        variables_metadata: str | VariablesMetadata | None = None,
        device: None | DeviceType = None,
    ) -> None:
        """
        Args:
            series_index_key: Name of the variable that contains the series index or the index to the key / column.
            data: Data or path to the CSV file with data.
            adjacency_matrix: Adjacency matrix or path to CSV file with adjacency matrix.
            variables_metadata: Variables metadata or path to JSON file with variables metadata. If not provided and
                variable names cannot be inferred from the data, the variables will be named x0, x1, ... in order and
                interpreted as continuous variables.
            device: Device to move the data to.
        """
        super().__init__()

        # Load and fix types of inputs
        if variables_metadata is not None:
            variables_metadata = ensure_variables_metadata(variables_metadata)
        adjacency_matrix = ensure_adjacency_matrix(adjacency_matrix)
        data = preprocess_data(data, variables_metadata)
        if isinstance(series_index_key, int):
            series_index_key = list(data.keys())[series_index_key]
            assert isinstance(series_index_key, str)

        data[series_index_key] = data[series_index_key].to(dtype=torch.int64)
        if device is not None:
            data = data.to(device)
            adjacency_matrix = adjacency_matrix.to(device=device)

        self._data, self._series_lookup = index_contiguous_chunks(data, series_index_key)
        self._adjacency_matrix = adjacency_matrix

    @classmethod
    def from_dense(cls: Type[CLS], data: torch.Tensor, lengths: torch.Tensor, **kwargs) -> CLS:
        """Create an indexed timeseries dataset from a dense representation of data with variable lengths.

        With this representation each timeseries i contains the data of data[i, lengths[i]].

        Args:
            data: Dense timeseries data of shape (num_timeseries, max_length, num_dims).
            lengths: Length of each timeseries in the data, shape (num_timeseries, ).
            **kwargs: Forwarded to `__init__`.

        Returns:
            IndexedTimeseriesDataset with the given data.
        """
        if data.dim() != 3:
            raise ValueError(f"The expected shape of data is (num_timeseries, max_length, num_dims), got {data.shape}")
        if lengths.dim() != 1:
            raise ValueError(f"The expected shape of lengths is (num_timeseries, ), got {lengths.shape}")
        if data.shape[0] != lengths.shape[0]:
            raise ValueError(
                f"The number of timeseries in data ({data.shape[0]}) does not match the number of entries "
                f"in lengths ({lengths.shape[0]})"
            )

        num_timeseries, max_length, num_dims = data.shape
        index = torch.arange(num_timeseries)[:, None].expand(num_timeseries, max_length)

        # Create sequence mask of shape (num_timeseries, max_length, num_dims) which select sequences from start to end,
        # and where all entries have the same value along the last axis.
        mask = torch.arange(max_length)[None, :, None].expand(1, max_length, num_dims) < lengths[:, None, None]

        # Use mask to create sparse data
        sparse_data = torch.cat([data[mask].reshape(-1, num_dims), index[mask[:, :, 0]].reshape(-1, 1)], dim=-1)
        return cls(series_index_key=num_dims, data=sparse_data, **kwargs)

    def __getitem__(self, idx) -> TensorDictBase:
        """Get timeseries by their relative index.

        Args:
            idx: Relative index in [0, len(self)]

        Returns:
            The full timeseries TensorDict(..., batch_size=`number of steps`).
        """
        return self._data[self._series_lookup[idx]]

    def __len__(self) -> int:
        """The total number of timeseries in the dataset."""
        return len(self._series_lookup)
