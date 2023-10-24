import pytest
import torch

from causica.nn import DECIEmbedNN

PROCESSED_DIM = 6
NODE_NUM = 4
GROUP_MASK = torch.tensor(
    [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1],
    ],
    dtype=torch.float32,
)
assert GROUP_MASK.shape == (NODE_NUM, PROCESSED_DIM)


GRAPH_SHAPES = [tuple(), (5,), (2, 3)]
SAMPLE_SHAPES = [tuple(), (3,), (1, 2)]


@pytest.mark.parametrize("graph_shape", GRAPH_SHAPES)
@pytest.mark.parametrize("sample_shape", SAMPLE_SHAPES)
def test_fgnni_broadcast(graph_shape, sample_shape):
    graph_tensor = torch.randint(0, 2, (*graph_shape, NODE_NUM, NODE_NUM), dtype=torch.float32)
    sample_tensor = torch.randn((*sample_shape, PROCESSED_DIM))

    fgnni = DECIEmbedNN(group_mask=GROUP_MASK, embedding_size=32, out_dim_g=32, num_layers_g=2, num_layers_zeta=2)
    out = fgnni(sample_tensor, graph_tensor)
    assert out.shape == sample_shape + graph_shape + (PROCESSED_DIM,)
