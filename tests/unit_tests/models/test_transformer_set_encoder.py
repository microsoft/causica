import pytest
import torch

from causica.models.transformer_set_encoder import ISAB, MAB, PMA, SAB, SetTransformer, TransformerSetEncoder


# pylint: disable=protected-access
@pytest.mark.parametrize(
    "batch_size, x_set_size, y_set_size, embedding_dim, num_heads",
    [(2, 3, 4, 5, 1), (2, 3, 4, 6, 2), (1, 1, 1, 1, 1)],
)
def test_mab_shape(batch_size, x_set_size, y_set_size, embedding_dim, num_heads):
    mab = MAB(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        multihead_init_type=SetTransformer.MultiheadInitType["xavier"],
        use_layer_norm=True,
        elementwise_transform_type=SetTransformer.ElementwiseTransformType["single"],
    )
    x = torch.ones((batch_size, x_set_size, embedding_dim))
    y = torch.ones((batch_size, y_set_size, embedding_dim))
    y_mask = torch.ones((batch_size, y_set_size))
    output = mab(x, y, y_mask)
    assert output.shape == (batch_size, x_set_size, embedding_dim)


@pytest.mark.parametrize("embedding_dim, num_heads", [(5, 2), (1, 2)])
def test_mab_invalid_num_heads(embedding_dim, num_heads):
    # num_heads does not divide embedding_dim
    with pytest.raises(AssertionError):
        _ = MAB(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            multihead_init_type=SetTransformer.MultiheadInitType["xavier"],
            use_layer_norm=True,
            elementwise_transform_type=SetTransformer.ElementwiseTransformType["single"],
        )


@pytest.mark.parametrize("batch_size, x_set_size, y_set_size, embedding_dim, num_heads", [(2, 3, 4, 5, 1)])
def test_mab_ignores_unobserved_y(batch_size, x_set_size, y_set_size, embedding_dim, num_heads):
    mab = MAB(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        multihead_init_type=SetTransformer.MultiheadInitType["xavier"],
        use_layer_norm=True,
        elementwise_transform_type=SetTransformer.ElementwiseTransformType["single"],
    )

    x = torch.ones((batch_size, x_set_size, embedding_dim))
    y = torch.ones((batch_size, y_set_size, embedding_dim))
    y_mask = torch.arange(batch_size * y_set_size).view(batch_size, y_set_size).to(torch.float)
    y_mask = torch.fmod(y_mask, 2)  # Only odd values in y are observed
    output1 = mab(x, y, y_mask)

    diff = (1 - y_mask) * 100
    y += diff.unsqueeze(2)
    output2 = mab(x, y, y_mask)

    assert torch.allclose(output1, output2)


@pytest.mark.parametrize(
    "batch_size, set_size, embedding_dim, num_heads",
    [(2, 3, 5, 1), (2, 3, 6, 2), (1, 1, 1, 1)],
)
def test_sab_shape(batch_size, set_size, embedding_dim, num_heads):
    sab = SAB(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        multihead_init_type=SetTransformer.MultiheadInitType["xavier"],
        use_layer_norm=True,
        elementwise_transform_type=SetTransformer.ElementwiseTransformType["single"],
    )
    x = torch.ones((batch_size, set_size, embedding_dim))
    mask = torch.ones((batch_size, set_size))
    output = sab(x, mask)

    assert output.shape == (batch_size, set_size, embedding_dim)


@pytest.mark.parametrize(
    "batch_size, set_size, embedding_dim, num_heads",
    [(2, 3, 5, 1), (2, 3, 6, 2), (1, 1, 1, 1)],
)
def test_isab_shape(batch_size, set_size, embedding_dim, num_heads):
    isab = ISAB(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_inducing_points=14,
        multihead_init_type=SetTransformer.MultiheadInitType["xavier"],
        use_layer_norm=True,
        elementwise_transform_type=SetTransformer.ElementwiseTransformType["single"],
    )
    x = torch.ones((batch_size, set_size, embedding_dim))
    mask = torch.ones((batch_size, set_size))
    output = isab(x, mask)

    assert output.shape == (batch_size, set_size, embedding_dim)


@pytest.mark.parametrize(
    "batch_size, set_size, embedding_dim, num_heads, num_seed_vectors",
    [
        (2, 3, 4, 1, 5),
        (2, 3, 4, 2, 5),
        (2, 3, 4, 1, 1),
        (2, 3, 4, 2, 1),
        (1, 1, 1, 1, 1),
    ],
)
def test_pma_shape(batch_size, set_size, embedding_dim, num_heads, num_seed_vectors):
    pma = PMA(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_seed_vectors=num_seed_vectors,
        multihead_init_type=SetTransformer.MultiheadInitType["xavier"],
        use_layer_norm=True,
        elementwise_transform_type=SetTransformer.ElementwiseTransformType["single"],
        use_elementwise_transform_pma=True,
    )
    x = torch.ones((batch_size, set_size, embedding_dim))
    mask = torch.ones((batch_size, set_size))
    output = pma(x, mask)

    assert output.shape == (batch_size, num_seed_vectors, embedding_dim)


@pytest.mark.parametrize(
    "batch_size, set_size, input_embedding_dim, set_embedding_dim, transformer_embedding_dim, num_heads, num_blocks, num_seed_vectors, use_isab, num_inducing_points",
    [
        (2, 3, 4, 5, None, 1, 2, 1, False, None),
        (1, 1, 1, 1, None, 1, 2, 1, False, None),
        (2, 3, 4, 5, None, 1, 3, 1, False, None),
        (2, 3, 4, 5, None, 1, 3, 7, False, None),
        (2, 3, 4, 8, None, 2, 2, 3, False, None),
    ],
)
def test_set_transformer_shape(
    batch_size,
    set_size,
    input_embedding_dim,
    set_embedding_dim,
    transformer_embedding_dim,
    num_heads,
    num_blocks,
    num_seed_vectors,
    use_isab,
    num_inducing_points,
):
    set_transformer = SetTransformer(
        input_embedding_dim=input_embedding_dim,
        transformer_embedding_dim=transformer_embedding_dim,
        set_embedding_dim=set_embedding_dim,
        num_heads=num_heads,
        num_blocks=num_blocks,
        num_seed_vectors=num_seed_vectors,
        use_isab=use_isab,
        num_inducing_points=num_inducing_points,
    )
    x = torch.ones((batch_size, set_size, input_embedding_dim))
    mask = torch.ones((batch_size, set_size))
    output = set_transformer(x, mask)

    assert output.shape == (batch_size, set_embedding_dim)
    assert not set_transformer._transform_input_dimension
    for name, _ in set_transformer.named_parameters():
        if name in ["_input_dimension_transform.weight", "_input_dimension_transform.bias"]:
            assert False


def test_set_transformer_extra_settings_default():
    input_embedding_dim = 3
    set_embedding_dim = 5
    set_transformer = SetTransformer(
        input_embedding_dim=input_embedding_dim,
        transformer_embedding_dim=4,
        set_embedding_dim=set_embedding_dim,
        num_heads=2,
        num_blocks=3,
        num_seed_vectors=4,
        use_isab=True,
        num_inducing_points=4,
        # Extra params
        multihead_init_type="xavier",
        use_layer_norm=True,
        elementwise_transform_type="single",
        use_elementwise_transform_pma=True,
    )
    batch_size = 10
    set_size = 8
    x = torch.ones((batch_size, set_size, input_embedding_dim))
    mask = torch.ones((batch_size, set_size))
    output = set_transformer(x, mask)
    assert output.shape == (batch_size, set_embedding_dim)


def test_set_transformer_extra_settings_nondefault():
    input_embedding_dim = 3
    set_embedding_dim = 5
    set_transformer = SetTransformer(
        input_embedding_dim=input_embedding_dim,
        transformer_embedding_dim=4,
        set_embedding_dim=set_embedding_dim,
        num_heads=2,
        num_blocks=3,
        num_seed_vectors=4,
        use_isab=True,
        num_inducing_points=4,
        # Extra params
        multihead_init_type="kaiming",
        use_layer_norm=False,
        elementwise_transform_type="double",
        use_elementwise_transform_pma=False,
    )
    batch_size = 10
    set_size = 8
    x = torch.ones((batch_size, set_size, input_embedding_dim))
    mask = torch.ones((batch_size, set_size))
    output = set_transformer(x, mask)
    assert output.shape == (batch_size, set_embedding_dim)


@pytest.mark.parametrize(
    "batch_size, set_size, input_embedding_dim, set_embedding_dim, transformer_embedding_dim, num_heads, num_blocks, num_seed_vectors, use_isab, num_inducing_points",
    [
        (2, 3, 4, 5, None, 1, 2, 1, False, None),
    ],
)
def test_set_transformer_shape_no_mask(
    batch_size,
    set_size,
    input_embedding_dim,
    set_embedding_dim,
    transformer_embedding_dim,
    num_heads,
    num_blocks,
    num_seed_vectors,
    use_isab,
    num_inducing_points,
):
    set_transformer = SetTransformer(
        input_embedding_dim=input_embedding_dim,
        transformer_embedding_dim=transformer_embedding_dim,
        set_embedding_dim=set_embedding_dim,
        num_heads=num_heads,
        num_blocks=num_blocks,
        num_seed_vectors=num_seed_vectors,
        use_isab=use_isab,
        num_inducing_points=num_inducing_points,
    )
    x = torch.ones((batch_size, set_size, input_embedding_dim))
    output = set_transformer(x, mask=None)

    assert output.shape == (batch_size, set_embedding_dim)


@pytest.mark.parametrize(
    "batch_size, set_size, input_embedding_dim, set_embedding_dim, transformer_embedding_dim, num_heads, num_blocks, num_seed_vectors, use_isab, num_inducing_points",
    [
        (2, 3, 4, 5, 4, 1, 2, 1, False, None),
        (2, 3, 4, 5, 6, 1, 2, 1, False, None),
    ],
)
def test_set_transformer_shape_transform_input_dimension(
    batch_size,
    set_size,
    input_embedding_dim,
    set_embedding_dim,
    transformer_embedding_dim,
    num_heads,
    num_blocks,
    num_seed_vectors,
    use_isab,
    num_inducing_points,
):
    set_transformer = SetTransformer(
        input_embedding_dim=input_embedding_dim,
        transformer_embedding_dim=transformer_embedding_dim,
        set_embedding_dim=set_embedding_dim,
        num_heads=num_heads,
        num_blocks=num_blocks,
        num_seed_vectors=num_seed_vectors,
        use_isab=use_isab,
        num_inducing_points=num_inducing_points,
    )
    x = torch.ones((batch_size, set_size, input_embedding_dim))
    mask = torch.ones((batch_size, set_size))
    output = set_transformer(x, mask)

    assert output.shape == (batch_size, set_embedding_dim)
    assert set_transformer._transform_input_dimension

    weight_found, bias_found = False, False
    for name, _ in set_transformer.named_parameters():
        if name == "_input_dimension_transform.weight":
            weight_found = True
        if name == "_input_dimension_transform.bias":
            bias_found = True
    assert weight_found and bias_found


@pytest.mark.parametrize(
    "batch_size, set_size, input_embedding_dim, set_embedding_dim, transformer_embedding_dim, num_heads, num_blocks, num_seed_vectors, use_isab, num_inducing_points",
    [(2, 10, 4, 5, None, 1, 2, 1, True, 4)],
)
def test_set_transformer_shape_use_isab(
    batch_size,
    set_size,
    input_embedding_dim,
    set_embedding_dim,
    transformer_embedding_dim,
    num_heads,
    num_blocks,
    num_seed_vectors,
    use_isab,
    num_inducing_points,
):
    set_transformer = SetTransformer(
        input_embedding_dim=input_embedding_dim,
        transformer_embedding_dim=transformer_embedding_dim,
        set_embedding_dim=set_embedding_dim,
        num_heads=num_heads,
        num_blocks=num_blocks,
        num_seed_vectors=num_seed_vectors,
        use_isab=use_isab,
        num_inducing_points=num_inducing_points,
    )
    x = torch.ones((batch_size, set_size, input_embedding_dim))
    mask = torch.ones((batch_size, set_size))
    output = set_transformer(x, mask)

    assert output.shape == (batch_size, set_embedding_dim)


@pytest.mark.parametrize(
    "input_embedding_dim, set_embedding_dim, transformer_embedding_dim, num_heads, num_blocks, num_seed_vectors, use_isab, num_inducing_points",
    [(4, 5, None, 1, 2, 1, True, None)],
)
def test_set_transformer_shape_use_isab_num_inducing_points_missing(
    input_embedding_dim,
    set_embedding_dim,
    transformer_embedding_dim,
    num_heads,
    num_blocks,
    num_seed_vectors,
    use_isab,
    num_inducing_points,
):
    with pytest.raises(ValueError):
        _ = SetTransformer(
            input_embedding_dim=input_embedding_dim,
            transformer_embedding_dim=transformer_embedding_dim,
            set_embedding_dim=set_embedding_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            num_seed_vectors=num_seed_vectors,
            use_isab=use_isab,
            num_inducing_points=num_inducing_points,
        )


@pytest.mark.parametrize("batch_size, set_size, input_embedding_dim, set_embedding_dim", [(2, 3, 4, 5)])
def test_set_transformer_empty_set(batch_size, set_size, input_embedding_dim, set_embedding_dim):
    set_transformer = SetTransformer(
        input_embedding_dim=input_embedding_dim,
        transformer_embedding_dim=input_embedding_dim,
        set_embedding_dim=set_embedding_dim,
    )
    x = torch.ones((batch_size, set_size, input_embedding_dim))
    mask = torch.ones((batch_size, set_size))
    mask[0, :] = 0  # The first feature set in the batch is empty and the second one is observed

    with pytest.raises(AssertionError):
        _ = set_transformer(x, mask)


@pytest.mark.parametrize("set_size, input_embedding_dim, set_embedding_dim", [(3, 4, 5)])
def test_set_transformer_empty_batch(set_size, input_embedding_dim, set_embedding_dim):
    set_transformer = SetTransformer(
        input_embedding_dim=input_embedding_dim,
        transformer_embedding_dim=input_embedding_dim,
        set_embedding_dim=set_embedding_dim,
    )
    batch_size = 0
    x = torch.ones((batch_size, set_size, input_embedding_dim))
    mask = torch.ones((batch_size, set_size))

    with pytest.raises(AssertionError):
        _ = set_transformer(x, mask)


@pytest.mark.parametrize("batch_size, set_size, input_embedding_dim, set_embedding_dim", [(2, 3, 4, 5)])
def test_set_transformer_ignores_unobserved_x(batch_size, set_size, input_embedding_dim, set_embedding_dim):
    set_transformer = SetTransformer(
        input_embedding_dim=input_embedding_dim,
        transformer_embedding_dim=input_embedding_dim,
        set_embedding_dim=set_embedding_dim,
    )

    x = torch.ones(((batch_size, set_size, input_embedding_dim)))
    mask = torch.arange(batch_size * set_size).view(batch_size, set_size).to(torch.float)
    mask = torch.fmod(mask, 2)  # Only odd values in x are observed
    output1 = set_transformer(x, mask)

    diff = (1 - mask) * 100
    x += diff.unsqueeze(2)
    output2 = set_transformer(x, mask)

    assert torch.allclose(output1, output2)


@pytest.mark.parametrize(
    "batch_size, input_dim, embedding_dim, set_embedding_dim, multiply_weights, include_all_vars",
    [(2, 3, 4, 5, True, False), (2, 3, 4, 5, False, True), (1, 1, 1, 1, True, False), (1, 1, 1, 1, False, True)],
)
def test_transformer_set_encoder_shape(
    batch_size, input_dim, embedding_dim, set_embedding_dim, multiply_weights, include_all_vars
):
    model = TransformerSetEncoder(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        set_embedding_dim=set_embedding_dim,
        metadata=None,
        device="cpu",
        multiply_weights=multiply_weights,
        include_all_vars=include_all_vars,
    )
    x = torch.ones((batch_size, input_dim))
    mask = torch.ones((batch_size, input_dim))  # All feature sets are observed
    output = model(x, mask)

    assert output.shape == (batch_size, set_embedding_dim)


@pytest.mark.parametrize(
    "batch_size, input_dim, embedding_dim, set_embedding_dim, multiply_weights, include_all_vars",
    [(2, 3, 4, 5, True, False), (2, 3, 4, 5, False, True)],
)
def test_transformer_set_encoder_ignores_unobserved_x(
    batch_size, input_dim, embedding_dim, set_embedding_dim, multiply_weights, include_all_vars
):
    model = TransformerSetEncoder(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        set_embedding_dim=set_embedding_dim,
        metadata=None,
        device="cpu",
        multiply_weights=multiply_weights,
        include_all_vars=include_all_vars,
    )

    x = torch.ones(((batch_size, input_dim)))
    mask = torch.arange(batch_size * input_dim).view(batch_size, input_dim).to(torch.float)
    mask = torch.fmod(mask, 2)  # Only odd values in x are observed
    output1 = model(x, mask)

    diff = (1 - mask) * 100
    x += diff
    output2 = model(x, mask)

    assert torch.allclose(output1, output2)


@pytest.mark.parametrize("input_dim, embedding_dim, set_embedding_dim", [(3, 4, 5)])
def test_transformer_set_encoder_all_empty_sets(input_dim, embedding_dim, set_embedding_dim):
    batch_size = 2
    model = TransformerSetEncoder(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        set_embedding_dim=set_embedding_dim,
        metadata=None,
        device="cpu",
        multiply_weights=True,
    )
    x = torch.ones((batch_size, input_dim))
    mask = torch.zeros((batch_size, input_dim))  # All feature sets are empty
    output = model(x, mask)

    assert torch.allclose(output[0, :], model._empty_set_embedding) and torch.allclose(
        output[1, :], model._empty_set_embedding
    )


@pytest.mark.parametrize("input_dim, embedding_dim, set_embedding_dim", [(3, 4, 5)])
def test_transformer_set_encoder_one_empty_set(input_dim, embedding_dim, set_embedding_dim):
    batch_size = 2
    model = TransformerSetEncoder(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        set_embedding_dim=set_embedding_dim,
        metadata=None,
        device="cpu",
        multiply_weights=True,
    )
    x = torch.ones((batch_size, input_dim))
    mask = torch.ones((batch_size, input_dim))
    mask[0, :] = 0  # The first feature set in the batch is empty and the second one is observed
    output = model(x, mask)

    assert torch.allclose(output[0, :], model._empty_set_embedding)
