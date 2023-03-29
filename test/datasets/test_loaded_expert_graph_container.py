import numpy as np
import pytest
import torch
import torch.testing

from causica.datasets.loaded_expert_graph_container import LoadedExpertGraphContainer


@pytest.mark.parametrize("file_type", ["npy", "csv"])
def test_loaded_expert_graph_container(tmp_path, file_type: str):
    """Test the loaded expert graph."""
    adj_matrix = torch.triu(torch.ones(10, 10, dtype=torch.bool), diagonal=1)
    # Add some nans
    adj_matrix = adj_matrix.masked_fill(torch.rand(size=adj_matrix.shape) < 0.5, np.nan)

    mask = ~torch.isnan(adj_matrix)

    if file_type == "npy":
        adj_matrix_path = tmp_path / "expert_graph.npy"
        with adj_matrix_path.open("wb") as f:
            np.save(f, adj_matrix.numpy())
    elif file_type == "csv":
        adj_matrix_path = tmp_path / "expert_graph.csv"
        with adj_matrix_path.open("w") as f:
            np.savetxt(f, adj_matrix.numpy(), delimiter=",")
    else:
        raise ValueError(f"Unknown file type {file_type}")

    expert_graph_container = LoadedExpertGraphContainer(str(adj_matrix_path), confidence=0.9, scale=1.0)
    assert torch.equal(torch.tensor(adj_matrix, dtype=torch.int64), expert_graph_container.dag)
    assert torch.equal(mask, expert_graph_container.mask)
