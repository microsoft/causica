import pytest
import torch

from causica.models.point_net import PointNet, SparsePointNet


@pytest.mark.parametrize(
    "batch_size, input_dim, embedding_dim, set_embedding_dim, multiply_weights",
    [
        (1, 1, 1, 1, True),
        (1, 1, 1, 1, False),
        (10, 20, 5, 10, True),
        (10, 20, 5, 10, False),
    ],
)
def test_set_encoder_shape(batch_size, input_dim, embedding_dim, set_embedding_dim, multiply_weights):
    model = PointNet(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        set_embedding_dim=set_embedding_dim,
        metadata=None,
        device="cpu",
        multiply_weights=multiply_weights,
        encoding_function="sum",
    )

    # Inputs of shape (batch_size, input_dim)
    data = torch.ones(batch_size * input_dim).view(batch_size, input_dim)
    mask = torch.ones(batch_size * input_dim).view(batch_size, input_dim)

    output = model(data, mask)

    assert output.shape == (batch_size, set_embedding_dim)


@pytest.mark.parametrize(
    "batch_size, input_dim, embedding_dim, set_embedding_dim, multiply_weights",
    [
        (1, 1, 1, 1, True),
        (1, 1, 1, 1, False),
        (10, 20, 5, 10, True),
        (10, 20, 5, 10, False),
    ],
)
def test_sparse_set_encoder_shape(batch_size, input_dim, embedding_dim, set_embedding_dim, multiply_weights):

    model = SparsePointNet(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        set_embedding_dim=set_embedding_dim,
        metadata=None,
        device="cpu",
        multiply_weights=multiply_weights,
        encoding_function="sum",
    )

    # Inputs of shape (batch_size, input_dim)
    data = torch.ones(batch_size * input_dim).view(batch_size, input_dim)
    mask = torch.ones(batch_size * input_dim).view(batch_size, input_dim)

    output = model(data, mask)

    assert output.shape == (batch_size, set_embedding_dim)


@pytest.mark.parametrize(
    "batch_size, input_dim, embedding_dim, set_embedding_dim, multiply_weights, encoder_type, expected_value",
    [
        (
            2,
            3,
            4,
            5,
            True,
            PointNet,
            torch.tensor(
                [
                    [210.0, 692.0, 1174.0, 1656.0, 2138.0],
                    [245.0, 804.0, 1363.0, 1922.0, 2481.0],
                ]
            ),
        ),
        (
            2,
            3,
            4,
            5,
            False,
            PointNet,
            torch.tensor(
                [
                    [130.0, 420.0, 710.0, 1000.0, 1290.0],
                    [65.0, 228.0, 391.0, 554.0, 717.0],
                ]
            ),
        ),
        (
            2,
            3,
            4,
            5,
            True,
            SparsePointNet,
            torch.tensor(
                [
                    [210.0, 692.0, 1174.0, 1656.0, 2138.0],
                    [245.0, 804.0, 1363.0, 1922.0, 2481.0],
                ]
            ),
        ),
        (
            2,
            3,
            4,
            5,
            False,
            SparsePointNet,
            torch.tensor(
                [
                    [130.0, 420.0, 710.0, 1000.0, 1290.0],
                    [65.0, 228.0, 391.0, 554.0, 717.0],
                ]
            ),
        ),
        (1, 2, 3, 4, True, PointNet, torch.tensor([[0.0, 1.0, 2.0, 3.0]])),
    ],
)
def test_set_encoder_output(
    batch_size,
    input_dim,
    embedding_dim,
    set_embedding_dim,
    multiply_weights,
    encoder_type,
    expected_value,
):
    model = encoder_type(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        set_embedding_dim=set_embedding_dim,
        metadata=None,
        device="cpu",
        multiply_weights=multiply_weights,
        encoding_function="sum",
    )
    fill_with_arange(model.feature_embedder.embedding_weights)
    fill_with_arange(model.feature_embedder.embedding_bias)
    fill_with_arange(model.forward_sequence[0].weight)
    fill_with_arange(model.forward_sequence[0].bias)

    # Inputs of shape (batch_size, input_dim)
    data = torch.arange(batch_size * input_dim).view(batch_size, input_dim)
    mask = torch.fmod(data, 2) == 0
    data = data.to(torch.float)
    assert mask.shape == data.shape
    output = model(data, mask)
    assert torch.allclose(output, expected_value)


def fill_with_arange(t: torch.nn.Parameter):
    # Fill a parameter with 0,1,2,3,...
    assert isinstance(t, torch.nn.Parameter)
    with torch.no_grad():
        t.data = torch.arange(t.data.numel(), dtype=t.dtype).reshape(t.shape)
