import pytest
import torch

from causica.models.feature_embedder import FeatureEmbedder, SparseFeatureEmbedder


@pytest.mark.parametrize(
    "batch_size, input_dim, embedding_dim, metadata, multiply_weights, output_dim, output_shape",
    [
        (2, 3, 4, torch.ones((3, 5)), True, 11, torch.Size((6, 11))),
        (2, 3, 4, None, True, 6, torch.Size((6, 6))),
        (2, 3, 4, torch.ones((3, 5)), False, 11, torch.Size((6, 11))),
        (2, 3, 4, None, False, 6, torch.Size((6, 6))),
        (1, 1, 1, torch.ones((1, 1)), True, 4, torch.Size((1, 4))),
        (1, 1, 1, None, True, 3, torch.Size((1, 3))),
        (1, 1, 1, torch.ones((1, 1)), False, 4, torch.Size((1, 4))),
        (1, 1, 1, None, False, 3, torch.Size((1, 3))),
    ],
)
def test_feature_embedder_output_shape(
    batch_size,
    input_dim,
    embedding_dim,
    metadata,
    multiply_weights,
    output_dim,
    output_shape,
):
    feature_embedder = FeatureEmbedder(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        metadata=metadata,
        device="cpu",
        multiply_weights=multiply_weights,
    )
    assert feature_embedder.output_dim == output_dim
    x = torch.ones(batch_size, input_dim)
    x_features_embedded = feature_embedder(x)

    assert x_features_embedded.shape == output_shape


@pytest.mark.parametrize(
    "batch_size, input_dim, embedding_dim, metadata, multiply_weights, output_dim, output_shape",
    [
        (2, 3, 4, None, True, 6, torch.Size((6, 6))),
        (2, 3, 4, None, False, 6, torch.Size((6, 6))),
        (1, 1, 1, None, True, 3, torch.Size((1, 3))),
        (1, 1, 1, None, False, 3, torch.Size((1, 3))),
    ],
)
def test_sparse_feature_embedder_output_shape_all_observed(
    batch_size,
    input_dim,
    embedding_dim,
    metadata,
    multiply_weights,
    output_dim,
    output_shape,
):
    feature_embedder = SparseFeatureEmbedder(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        metadata=metadata,
        device="cpu",
        multiply_weights=multiply_weights,
    )
    assert feature_embedder.output_dim == output_dim
    x = torch.ones(batch_size, input_dim)
    mask = torch.ones(batch_size, input_dim)
    x_features_embedded = feature_embedder(x, mask)

    assert x_features_embedded.shape == output_shape


@pytest.mark.parametrize(
    "batch_size, input_dim, embedding_dim, metadata, multiply_weights, output_dim, output_shape",
    [
        (2, 3, 4, None, True, 6, torch.Size((5, 6))),
        (2, 3, 4, None, False, 6, torch.Size((5, 6))),
        (1, 1, 1, None, True, 3, torch.Size((0, 3))),
        (1, 1, 1, None, False, 3, torch.Size((0, 3))),
    ],
)
def test_sparse_feature_embedder_output_shape_one_unobserved(
    batch_size,
    input_dim,
    embedding_dim,
    metadata,
    multiply_weights,
    output_dim,
    output_shape,
):
    feature_embedder = SparseFeatureEmbedder(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        metadata=metadata,
        device="cpu",
        multiply_weights=multiply_weights,
    )
    assert feature_embedder.output_dim == output_dim
    x = torch.ones(batch_size, input_dim)
    mask = torch.ones(batch_size, input_dim)
    mask[0, 0] = 0.0  # First feature in first element is unobserved
    x_features_embedded = feature_embedder(x, mask)

    assert x_features_embedded.shape == output_shape


def test_feature_embedder_wrong_metadata_shape():
    input_dim = 2
    metadata = torch.ones((3, 3))
    # Because metadata.shape[0] != input_dim, an assertion error is raised
    with pytest.raises(AssertionError):
        _ = FeatureEmbedder(
            input_dim=input_dim,
            embedding_dim=1,
            metadata=metadata,
            device="cpu",
            multiply_weights=True,
        )


@pytest.mark.parametrize(
    "batch_size, input_dim, embedding_dim, metadata, multiply_weights, expected_value",
    [
        (
            2,
            3,
            4,
            torch.ones((3, 2)),
            True,
            torch.tensor(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 4.0, 5.0, 6.0, 7.0, 1.0, 1.0, 1.0],
                    [2.0, 16.0, 18.0, 20.0, 22.0, 2.0, 2.0, 2.0],
                    [3.0, 0.0, 3.0, 6.0, 9.0, 3.0, 3.0, 0.0],
                    [4.0, 16.0, 20.0, 24.0, 28.0, 4.0, 4.0, 1.0],
                    [5.0, 40.0, 45.0, 50.0, 55.0, 5.0, 5.0, 2.0],
                ]
            ),
        ),
        (
            2,
            3,
            4,
            None,
            True,
            torch.tensor(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 4.0, 5.0, 6.0, 7.0, 1.0],
                    [2.0, 16.0, 18.0, 20.0, 22.0, 2.0],
                    [3.0, 0.0, 3.0, 6.0, 9.0, 0.0],
                    [4.0, 16.0, 20.0, 24.0, 28.0, 1.0],
                    [5.0, 40.0, 45.0, 50.0, 55.0, 2.0],
                ]
            ),
        ),
        (
            2,
            3,
            4,
            None,
            False,
            torch.tensor(
                [
                    [0.0, 0.0, 1.0, 2.0, 3.0, 0.0],
                    [1.0, 4.0, 5.0, 6.0, 7.0, 1.0],
                    [2.0, 8.0, 9.0, 10.0, 11.0, 2.0],
                    [3.0, 0.0, 1.0, 2.0, 3.0, 0.0],
                    [4.0, 4.0, 5.0, 6.0, 7.0, 1.0],
                    [5.0, 8.0, 9.0, 10.0, 11.0, 2.0],
                ]
            ),
        ),
        (1, 1, 1, None, True, torch.tensor([[0.0, 0.0, 0.0]])),
    ],
)
def test_feature_embedder_output(batch_size, input_dim, embedding_dim, metadata, multiply_weights, expected_value):
    feature_embedder = FeatureEmbedder(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        metadata=metadata,
        device="cpu",
        multiply_weights=multiply_weights,
    )
    fill_with_arange(feature_embedder.embedding_weights)
    fill_with_arange(feature_embedder.embedding_bias)

    # Inputs of shape (batch_size, input_dim)
    data = torch.arange(batch_size * input_dim).view(batch_size, input_dim).to(torch.float)

    output = feature_embedder(data)
    assert torch.allclose(output, expected_value)


@pytest.mark.parametrize(
    "batch_size, input_dim, embedding_dim, metadata, multiply_weights, expected_output",
    [
        (
            2,
            3,
            4,
            None,
            True,
            torch.tensor(
                [
                    [1.0, 4.0, 5.0, 6.0, 7.0, 1.0],
                    [3.0, 0.0, 3.0, 6.0, 9.0, 0.0],
                    [5.0, 40.0, 45.0, 50.0, 55.0, 2.0],
                ]
            ),
        ),
        (
            2,
            3,
            4,
            None,
            False,
            torch.tensor(
                [
                    [1.0, 4.0, 5.0, 6.0, 7.0, 1.0],
                    [3.0, 0.0, 1.0, 2.0, 3.0, 0.0],
                    [5.0, 8.0, 9.0, 10.0, 11.0, 2.0],
                ]
            ),
        ),
        (1, 1, 1, None, True, torch.empty((0, 3))),
    ],
)
def test_sparse_feature_embedder_output(
    batch_size, input_dim, embedding_dim, metadata, multiply_weights, expected_output
):
    feature_embedder = SparseFeatureEmbedder(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        metadata=metadata,
        device="cpu",
        multiply_weights=multiply_weights,
    )
    fill_with_arange(feature_embedder.embedding_weights)
    fill_with_arange(feature_embedder.embedding_bias)

    # Inputs of shape (batch_size, input_dim)
    data = torch.arange(batch_size * input_dim).view(batch_size, input_dim).to(torch.float)
    mask = torch.fmod(data, 2) == 1  # Every odd feature is unobserved
    mask = mask.to(torch.float)

    output = feature_embedder(data, mask)
    torch.allclose(output, expected_output)


def fill_with_arange(t: torch.nn.Parameter):
    # Fill a parameter with 0, 1, 2, 3, ...
    assert isinstance(t, torch.nn.Parameter)
    with torch.no_grad():
        t.data = torch.arange(t.data.numel(), dtype=t.dtype).reshape(t.shape)
