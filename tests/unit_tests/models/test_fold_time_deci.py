import os

import numpy as np
import pandas as pd
import torch

from causica.datasets.temporal_causal_csv_dataset_loader import TemporalCausalCSVDatasetLoader
from causica.models.deci.fold_time_deci import FoldTimeDECI


# pylint: disable=protected-access
def generate_time_series(length_list, index_name):
    assert length_list is not None
    assert len(length_list) == len(index_name)
    assert 0 not in length_list
    column_1 = [[index_name[series_id] for _ in range(length_list[series_id])] for series_id in range(len(length_list))]
    column_1 = np.expand_dims(np.concatenate(column_1), axis=1)
    total_length = column_1.shape[0]
    column_2 = np.expand_dims(np.arange(total_length) + 0.1, axis=1)
    column_3 = np.expand_dims(np.arange(total_length), axis=1)
    return np.concatenate((column_1, column_2, column_3, column_2), axis=1)


def test_init(tmpdir_factory):
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    model_save_dir = tmpdir_factory.mktemp("model_dir")
    name_list = [2, 6, 3, 1]
    length_list = [5, 8, 4, 3]
    data = generate_time_series(length_list=length_list, index_name=name_list)

    pd.DataFrame(data).to_csv(os.path.join(dataset_dir, "train.csv"), header=None, index=None)
    pd.DataFrame(data).to_csv(os.path.join(dataset_dir, "test.csv"), header=None, index=None)

    dataset_loader = TemporalCausalCSVDatasetLoader(dataset_dir=dataset_dir)
    dataset = dataset_loader.load_predefined_dataset(max_num_rows=None, timeseries_column_index=0)

    assert dataset._variables is not None

    # Treat variables as continuous
    fold_time_deci = FoldTimeDECI(
        model_id="test",
        variables=dataset._variables,
        save_dir=model_save_dir,
        device=torch.device("cpu"),
        lag=2,
        allow_instantaneous=True,
        treat_continuous=True,
    )
    assert fold_time_deci.variables is not None
    assert fold_time_deci.variables.num_groups == 9
    assert fold_time_deci.variables.num_processed_non_aux_cols == 9
    assert fold_time_deci.variables.num_unprocessed_non_aux_cols == 9
    assert fold_time_deci.variables_orig.num_groups == 3
    assert fold_time_deci.variables_orig.num_processed_non_aux_cols == 3
    assert fold_time_deci.variables_orig.num_unprocessed_non_aux_cols == 3
    assert "binary" not in fold_time_deci.variables.processed_cols_by_type.keys()
    assert "categorical" not in fold_time_deci.variables.processed_cols_by_type.keys()
    assert "binary" not in fold_time_deci.variables_orig.processed_cols_by_type.keys()
    assert "categorical" not in fold_time_deci.variables_orig.processed_cols_by_type.keys()

    # Treat variables as categorical + continuous
    fold_time_deci = FoldTimeDECI(
        model_id="test",
        variables=dataset._variables,
        save_dir=model_save_dir,
        device=torch.device("cpu"),
        lag=2,
        allow_instantaneous=True,
        treat_continuous=False,
    )
    assert fold_time_deci.variables is not None
    assert fold_time_deci.variables.num_groups == 9
    assert fold_time_deci.variables.num_processed_non_aux_cols == 66
    assert fold_time_deci.variables.num_unprocessed_non_aux_cols == 9
    assert fold_time_deci.variables_orig.num_groups == 3
    assert fold_time_deci.variables_orig.num_processed_non_aux_cols == 22
    assert fold_time_deci.variables_orig.num_unprocessed_non_aux_cols == 3
    assert "binary" not in fold_time_deci.variables.processed_cols_by_type.keys()
    assert "categorical" in fold_time_deci.variables.processed_cols_by_type.keys()
    assert "binary" not in fold_time_deci.variables_orig.processed_cols_by_type.keys()
    assert "categorical" in fold_time_deci.variables_orig.processed_cols_by_type.keys()

    # Check if hard constraint matrix is reasonable
    assert fold_time_deci.hard_constraint.shape == (9, 9)
    assert np.nan_to_num(fold_time_deci.hard_constraint, nan=1.0).sum() == 54
    # No instantaneous effect
    fold_time_deci = FoldTimeDECI(
        model_id="test",
        variables=dataset._variables,
        save_dir=model_save_dir,
        device=torch.device("cpu"),
        lag=2,
        allow_instantaneous=False,
        treat_continuous=False,
    )
    assert fold_time_deci.hard_constraint.shape == (9, 9)
    assert np.nan_to_num(fold_time_deci.hard_constraint, nan=1.0).sum() == 27


def test_dataprocessor(tmpdir_factory):
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    model_save_dir = tmpdir_factory.mktemp("model_dir")
    name_list = [2, 6, 3, 1]
    length_list = [5, 8, 4, 3]
    data = generate_time_series(length_list=length_list, index_name=name_list)
    pd.DataFrame(data).to_csv(os.path.join(dataset_dir, "train.csv"), header=None, index=None)
    pd.DataFrame(data).to_csv(os.path.join(dataset_dir, "test.csv"), header=None, index=None)

    dataset_loader = TemporalCausalCSVDatasetLoader(dataset_dir=dataset_dir)
    dataset = dataset_loader.load_predefined_dataset(max_num_rows=None, timeseries_column_index=0)

    # Treat as continuous
    fold_time_deci = FoldTimeDECI(
        model_id="test",
        variables=dataset._variables,
        save_dir=model_save_dir,
        device=torch.device("cpu"),
        lag=2,
        allow_instantaneous=True,
        treat_continuous=True,
    )
    train_config = {"stardardize_data_mean": False, "stardardize_data_std": False}
    proc_data, _ = fold_time_deci.process_dataset(dataset, train_config)

    assert proc_data.shape == (20, 3)

    # Treat as categorical
    fold_time_deci = FoldTimeDECI(
        model_id="test",
        variables=dataset._variables,
        save_dir=model_save_dir,
        device=torch.device("cpu"),
        lag=2,
        allow_instantaneous=True,
        treat_continuous=False,
    )
    proc_data, _ = fold_time_deci.process_dataset(dataset, train_config)
    # Assert the middle columns is one-hot
    assert (proc_data[:, 1:-1] - np.eye(20)).sum() == 0
