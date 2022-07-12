# This is temporary file to keep logic (as functions), which probably
# shouldn't be steps on their own, but rather be part of each step (i.e. dataset loading)
import os
from typing import Any, Callable, Dict, Tuple, Union

from ...datasets.dataset import CausalDataset, Dataset, SparseDataset
from ...datasets.datasets_factory import create_dataset_loader


def load_data(
    dataset_name: str,
    data_dir: str,
    dataset_seed: Union[int, Tuple[int, int]],
    dataset_config: Dict[str, Any],
    model_config: Dict[str, Any],
    tiny: bool,
    download_dataset: Callable[[str, str], None],
):
    use_predefined_dataset = dataset_config.get("use_predefined_dataset", False)
    dataset_test_fraction = dataset_config.get("test_fraction", 0.1) if not use_predefined_dataset else None
    dataset_val_fraction = dataset_config.get("val_fraction", 0.0) if not use_predefined_dataset else None
    split_type = dataset_config.get("split_type", "rows")
    negative_sample = dataset_config.get("negative_sample", False)
    dataset_format = dataset_config.get("dataset_format", "csv")
    if dataset_name == "temporal_causal_csv":
        timeseries_column_index = dataset_config.get("timeseries_column_index", 0)
    else:
        timeseries_column_index = None

    dataset_loader = create_dataset_loader(data_dir=data_dir, dataset_name=dataset_name, dataset_format=dataset_format)
    max_num_rows = 10 if tiny else None

    if not os.path.isdir(dataset_loader.dataset_dir):
        dataset_dir, dataset_name = os.path.split(dataset_loader.dataset_dir)
        download_dataset(dataset_name, dataset_dir)

    if use_predefined_dataset:
        dataset = dataset_loader.load_predefined_dataset(
            max_num_rows=max_num_rows,
            model_config=model_config,
            split_type=split_type,
            negative_sample=negative_sample,
            timeseries_column_index=timeseries_column_index,
        )
    if not use_predefined_dataset:
        dataset = dataset_loader.split_data_and_load_dataset(
            test_frac=dataset_test_fraction,
            val_frac=dataset_val_fraction,
            random_state=dataset_seed,
            max_num_rows=max_num_rows,
            negative_sample=negative_sample,
            model_config=model_config,
            timeseries_column_index=timeseries_column_index,
        )
    return dataset


# Preprocess configs before running individual steps
def preprocess_configs(
    model_config: Dict[str, Any],
    train_hypers: Dict[str, Any],
    model_type: str,
    dataset: Union[Dataset, SparseDataset, CausalDataset],
    data_dir: str,
    tiny: bool,
):
    # Modify/adapt model_config
    if tiny:
        if model_type in ["vaem", "vaem_predictive", "transformer_encoder_vaem"]:
            train_hypers["marginal_epochs"] = 2
            train_hypers["dep_epochs"] = 2
        elif model_type in ["deci", "deci_gaussian", "deci_spline"]:
            train_hypers["max_steps_auglag"] = 2
        else:
            train_hypers["epochs"] = 2

    # TODO 18548 move metadata filepath to dataset config and simplify metadata handling throughout codebase.
    if "metadata_filename" in model_config:
        if model_config["metadata_filename"] is not None:
            model_config["metadata_filepath"] = os.path.join(data_dir, model_config["metadata_filename"])
        else:
            model_config["metadata_filepath"] = None
        del model_config["metadata_filename"]

    if model_type == "bayesian_pvae":
        model_config["dataset_size"] = dataset.train_data_and_mask[0].shape[
            0
        ]  # with BNN the total number of datapoints is needed
