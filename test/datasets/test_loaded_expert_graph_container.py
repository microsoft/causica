import numpy as np
import torch
import torch.testing

from causica.datasets.loaded_expert_graph_container import LoadedExpertGraphContainer


def test_loaded_expert_graph_container(tmp_path):
    """Test the loaded expert graph."""
    adj_matrix = torch.triu(torch.ones(10, 10, dtype=torch.bool), diagonal=1)
    adj_matrix_path = tmp_path / "expert_graph.npy"
    with adj_matrix_path.open("wb") as f:
        np.save(f, adj_matrix)
    expert_graph_container = LoadedExpertGraphContainer(adj_matrix_path, confidence=0.9, scale=1.0)
    assert torch.equal(torch.tensor(adj_matrix, dtype=torch.int64), expert_graph_container.dag)
    assert torch.equal(torch.ones_like(expert_graph_container.mask), expert_graph_container.mask)
