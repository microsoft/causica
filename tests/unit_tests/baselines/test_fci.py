import os

import numpy as np
import pandas as pd
import pytest

from causica.baselines.fci import FCI
from causica.datasets.causal_csv_dataset_loader import CausalCSVDatasetLoader


@pytest.fixture(name="dataset")
def example_dataset(tmpdir_factory):
    # Dataset with latent variables.
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    data = np.random.randn(100, 8)  # [100, 8]
    train_data = data[:60]
    val_data = data[60:90]
    test_data = data[90:]
    pd.DataFrame(train_data).to_csv(os.path.join(dataset_dir, "train.csv"), header=None, index=None)
    pd.DataFrame(val_data).to_csv(os.path.join(dataset_dir, "val.csv"), header=None, index=None)
    pd.DataFrame(test_data).to_csv(os.path.join(dataset_dir, "test.csv"), header=None, index=None)
    dataset_loader = CausalCSVDatasetLoader(dataset_dir=dataset_dir)

    dataset = dataset_loader.load_predefined_dataset(max_num_rows=None, column_index=0)

    return dataset


def test_fci(tmpdir_factory, dataset):
    """Tests that FCI learns a Markov equivalence class of ADMGs."""

    model = FCI.create(
        model_id="model_id",
        variables=dataset.variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        device="cpu",
        model_config_dict={},
    )
    model.run_train(dataset)

    # Check returns adjacency matrices are of the correct shape.
    adjs = model.get_adj_matrix()
    assert len(adjs.shape) == 3
    assert adjs.shape[1] == adjs.shape[2]

    directed_adjs, bidirected_adjs = model.get_admg_matrices()
    assert directed_adjs.shape == bidirected_adjs.shape
    assert len(directed_adjs.shape) == 3
    assert directed_adjs.shape[1] == directed_adjs.shape[2]
