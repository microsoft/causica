import os
from typing import Any, Dict, Optional, Tuple, Union

from .causal_csv_dataset_loader import CausalCSVDatasetLoader
from .csv_dataset_loader import CSVDatasetLoader
from .dataset import CausalDataset, Dataset, SparseDataset, TemporalDataset
from .latent_confounded_causal_csv_dataset_loader import LatentConfoundedCausalCSVDatasetLoader
from .temporal_causal_csv_dataset_loader import TemporalCausalCSVDatasetLoader


def create_dataset_loader(
    data_dir: str, dataset_name: str, dataset_format: str = "csv"
) -> Union[CSVDatasetLoader, CausalCSVDatasetLoader, TemporalCausalCSVDatasetLoader,]:
    """
    Factory method to create an instance of dataset loader using the information about dataset name and dataset format.

    Args:
        data_dir: Directory in which the datasets are saved locally.
        dataset_name: Name of dataset to load. Files will be loaded from the directory [data_dir]/[dataset_name]/.
        dataset_format: Format of dataset, determines which dataset loader will be used. Valid options are
            'csv' and 'sparse_csv'.

    Returns:
        dataset_loader: DatasetLoader object or its subclass
    """
    dataset_dir = os.path.join(data_dir, dataset_name)
    if dataset_format == "csv":
        return CSVDatasetLoader(dataset_dir)

    if dataset_format == "causal_csv":
        return CausalCSVDatasetLoader(dataset_dir)

    if dataset_format == "temporal_causal_csv":
        return TemporalCausalCSVDatasetLoader(dataset_dir)

    if dataset_format == "latent_confounded_causal_csv":
        return LatentConfoundedCausalCSVDatasetLoader(dataset_dir)

    raise NotImplementedError(
        f"Dataset format {dataset_format} not supported. Valid dataset formats are 'csv' and 'sparse_csv'."
    )


def load_dataset_from_config(
    data_dir: str,
    dataset_name: str,
    dataset_config: Dict[str, Any],
    max_num_rows: Optional[int] = None,
    **kwargs,
) -> Union[Dataset, SparseDataset, CausalDataset, TemporalDataset]:
    """
    Factory method to load a dataset using the dataset config dict and the information about dataset name and dataset
    format.
    Args:
        data_dir: Directory in which the datasets are saved locally.
        dataset_name: Name of dataset to load. Files will be loaded from the directory [data_dir]/[dataset_name]/.
        dataset_config: Information about how to load the dataset.
        max_num_rows: Maximum number of rows to load from the dataset.
    """
    dataset_format = dataset_config.get("dataset_format", "csv")
    use_predefined_dataset = dataset_config.get("use_predefined_dataset", False)
    if use_predefined_dataset:
        return load_predefined_dataset(data_dir, dataset_name, dataset_format, max_num_rows=max_num_rows, **kwargs)

    test_frac = dataset_config.get("test_fraction", 0.1)
    val_frac = dataset_config.get("val_fraction", 0.0)

    random_state = dataset_config.get("random_seed", 0)
    if isinstance(random_state, list):
        random_state = random_state[0]

    return split_data_and_load_dataset(
        data_dir, dataset_name, dataset_format, test_frac, val_frac, random_state, max_num_rows=max_num_rows, **kwargs
    )


def split_data_and_load_dataset(
    data_dir: str,
    dataset_name: str,
    dataset_format: str,
    test_frac: float,
    val_frac: float,
    random_state: Union[int, Tuple[int, int]],
    max_num_rows: Optional[int] = None,
    **kwargs,
) -> Union[Dataset, SparseDataset, CausalDataset, TemporalDataset]:
    """
    Factory method to split data and load a dataset using the information about dataset name and dataset format.
    The data is split deterministically given the random state. If the given random state is a pair of integers,
    the first is used to extract test set and the second is used to extract the validation set from the remaining data.
    If only a single integer is given as random state it is used for both.

    Args:
        data_dir: Directory in which the datasets are saved locally.
        dataset_name: Name of dataset to load. Files will be loaded from the directory [data_dir]/[dataset_name]/.
        dataset_format: Format of dataset, determines which dataset loader will be used. Valid options are
            'csv' and 'sparse_csv'.
        test_frac: Fraction of data to put in the test set.
        val_frac: Fraction of data to put in the validation set.
        random_state: An integer or a tuple of integers to be used as the splitting random state.
        max_num_rows: Maximum number of rows to include when reading data files.

    Returns:
        dataset: Dataset, SparseDataset, or CausalDataset object, holding the data and variable metadata.
    """
    dataset_loader = create_dataset_loader(data_dir=data_dir, dataset_name=dataset_name, dataset_format=dataset_format)
    return dataset_loader.split_data_and_load_dataset(
        test_frac, val_frac, random_state=random_state, max_num_rows=max_num_rows, **kwargs
    )


def load_predefined_dataset(
    data_dir: str, dataset_name: str, dataset_format: str, max_num_rows: Optional[int] = None, **kwargs
) -> Union[Dataset, SparseDataset, CausalDataset, TemporalDataset]:
    """
    Factory method to load a predefined dataset using the information about dataset name and dataset format.

    Args:
        data_dir: Directory in which the datasets are saved locally.
        dataset_name: Name of dataset to load. Files will be loaded from the directory [data_dir]/[dataset_name]/.
        dataset_format: Format of dataset, determines which dataset loader will be used. Valid options are
            'csv' and 'sparse_csv'.
        max_num_rows: Maximum number of rows to include when reading data files.

    Returns:
        dataset: Dataset, SparseDataset, or CausalDataset object, holding the data and variable metadata.
    """
    dataset_loader = create_dataset_loader(data_dir=data_dir, dataset_name=dataset_name, dataset_format=dataset_format)
    return dataset_loader.load_predefined_dataset(max_num_rows=max_num_rows, **kwargs)
