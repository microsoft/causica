"""Module with generic Adjacency Distribution tests."""
from typing import Optional, Type, Union

import numpy as np
import pytest
import torch

from causica.distributions.adjacency import (
    AdjacencyDistribution,
    ConstrainedAdjacencyDistribution,
    ENCOAdjacencyDistribution,
    TemporalConstrainedAdjacencyDistribution,
    ThreeWayAdjacencyDistribution,
)
from causica.distributions.adjacency.temporal_adjacency_distributions import (
    LaggedAdjacencyDistribution,
    RhinoLaggedAdjacencyDistribution,
    TemporalAdjacencyDistribution,
)


def _distribution_factory(
    dist_class: Type[Union[AdjacencyDistribution, TemporalAdjacencyDistribution, LaggedAdjacencyDistribution]],
    num_nodes: int,
    batch_shape: torch.Size,
    context_length: Optional[int] = None,
) -> Union[AdjacencyDistribution, TemporalAdjacencyDistribution, LaggedAdjacencyDistribution]:
    """Create a combined interface for producing Adjacency Distributions (allows us to use `parametrize` over them)"""
    if dist_class is ConstrainedAdjacencyDistribution:
        logits = torch.randn(batch_shape + ((num_nodes * (num_nodes - 1)) // 2, 3))
        inner_dist = ThreeWayAdjacencyDistribution(logits=logits)
        square_ones = torch.ones(num_nodes, num_nodes, dtype=torch.bool)
        positive_constraints = torch.triu(square_ones, diagonal=1)
        negative_constraints = torch.tril(square_ones, diagonal=-1)
        return ConstrainedAdjacencyDistribution(
            inner_dist, positive_constraints=positive_constraints, negative_constraints=negative_constraints
        )
    if dist_class is TemporalConstrainedAdjacencyDistribution:
        assert context_length is not None
        logits_exists = torch.randn(batch_shape + (context_length, num_nodes, num_nodes))
        logits_orient = torch.randn(batch_shape + ((num_nodes * (num_nodes - 1)) // 2,))
        lagged_dist = (
            RhinoLaggedAdjacencyDistribution(logits_edge=logits_exists[..., :-1, :, :], lags=context_length - 1)
            if context_length > 1
            else None
        )
        inst_dist = ENCOAdjacencyDistribution(logits_exist=logits_exists[..., -1, :, :], logits_orient=logits_orient)
        inner_dist_temporal = TemporalAdjacencyDistribution(
            instantaneous_distribution=inst_dist, lagged_distribution=lagged_dist
        )
        square_ones = torch.ones(context_length, num_nodes, num_nodes, dtype=torch.bool)
        positive_constraints = torch.triu(square_ones, diagonal=1)
        negative_constraints = torch.tril(square_ones, diagonal=-1)
        return TemporalConstrainedAdjacencyDistribution(
            inner_dist_temporal, positive_constraints=positive_constraints, negative_constraints=negative_constraints
        )

    if dist_class is ENCOAdjacencyDistribution:
        length = (num_nodes * (num_nodes - 1)) // 2
        return ENCOAdjacencyDistribution(
            logits_exist=torch.randn(batch_shape + (num_nodes, num_nodes)),
            logits_orient=torch.randn(batch_shape + (length,)),
        )
    if dist_class is TemporalAdjacencyDistribution:
        assert context_length is not None
        logits_exists = torch.randn(batch_shape + (context_length, num_nodes, num_nodes))
        logits_orient = torch.randn(batch_shape + ((num_nodes * (num_nodes - 1)) // 2,))
        lagged_dist = (
            RhinoLaggedAdjacencyDistribution(logits_edge=logits_exists[..., :-1, :, :], lags=context_length - 1)
            if context_length > 1
            else None
        )
        inst_dist = ENCOAdjacencyDistribution(logits_exist=logits_exists[..., -1, :, :], logits_orient=logits_orient)
        return TemporalAdjacencyDistribution(instantaneous_distribution=inst_dist, lagged_distribution=lagged_dist)
    if dist_class is ThreeWayAdjacencyDistribution:
        logits = torch.randn(batch_shape + ((num_nodes * (num_nodes - 1)) // 2, 3))
        return ThreeWayAdjacencyDistribution(logits=logits)
    if dist_class is RhinoLaggedAdjacencyDistribution:
        assert context_length is not None
        return RhinoLaggedAdjacencyDistribution(
            logits_edge=torch.randn(batch_shape + (context_length - 1, num_nodes, num_nodes)), lags=context_length - 1
        )
    raise ValueError("Unrecognised Class")


DIST_CLASSES = [
    ConstrainedAdjacencyDistribution,
    ENCOAdjacencyDistribution,
    ThreeWayAdjacencyDistribution,
]
TEMPORAL_DIST_CLASSES = [TemporalAdjacencyDistribution, TemporalConstrainedAdjacencyDistribution]

BATCH_SHAPES = [torch.Size(), (2,)]


# pylint: disable=protected-access
@pytest.mark.parametrize("dist_class", DIST_CLASSES)
@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
def test_support(
    dist_class: Type[AdjacencyDistribution],
    batch_shape: torch.Size,
):
    """Test that the defined support works as expected. This method will be used to test other features."""
    num_nodes = 3

    # static adj distribution
    dist = _distribution_factory(dist_class=dist_class, num_nodes=num_nodes, batch_shape=batch_shape)

    mat = torch.ones((num_nodes, num_nodes))

    # validate sample returns None when there is no error
    dist._validate_sample(mat)

    # validate sample throws when the sample is invalid
    with pytest.raises(ValueError):
        dist._validate_sample(4 * mat)


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
def test_support_lagged(batch_shape: torch.Size):
    """Test that the defined support works as expected for lagged adjacency distribution.

    This method will be used to test other features.
    """
    context_length = 3
    num_nodes = 3
    dist = _distribution_factory(
        dist_class=RhinoLaggedAdjacencyDistribution,
        num_nodes=num_nodes,
        batch_shape=batch_shape,
        context_length=context_length,
    )
    mat = torch.ones((context_length - 1, num_nodes, num_nodes))

    # validate sample returns None when there is no error
    dist._validate_sample(mat)

    # validate sample throws when the sample is invalid
    with pytest.raises(ValueError):
        dist._validate_sample(4 * mat)


@pytest.mark.parametrize("dist_class", TEMPORAL_DIST_CLASSES)
@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
def test_support_temporal(dist_class: Type[TemporalAdjacencyDistribution], batch_shape: torch.Size):
    """Test that the defined support works as expected for temporal adjacency distribution.

    This method will be used to test other features.
    """
    context_length = 3
    num_nodes = 3

    dist = _distribution_factory(
        dist_class=dist_class, num_nodes=num_nodes, batch_shape=batch_shape, context_length=context_length
    )
    mat = torch.ones((context_length, num_nodes, num_nodes))
    # validate sample returns None when there is no error
    dist._validate_sample(mat)

    # validate sample throws when the sample is invalid
    with pytest.raises(ValueError):
        dist._validate_sample(4 * mat)


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("relaxed_sample", [True, False])
@pytest.mark.parametrize(("num_nodes", "sample_shape"), [(3, tuple()), (4, (20,)), (2, (4, 5))])
@pytest.mark.parametrize("dist_class", DIST_CLASSES)
def test_sample_shape(
    dist_class: Type[AdjacencyDistribution],
    num_nodes: int,
    sample_shape: torch.Size,
    relaxed_sample: bool,
    batch_shape: torch.Size,
):
    """Test the sample/rsample method returns binary tensors in the support of the correct shape"""

    dist = _distribution_factory(dist_class=dist_class, num_nodes=num_nodes, batch_shape=batch_shape)
    samples = dist.relaxed_sample(sample_shape, temperature=0.1) if relaxed_sample else dist.sample(sample_shape)
    assert samples.shape == sample_shape + batch_shape + (num_nodes, num_nodes)
    dist._validate_sample(samples)
    assert not np.isnan(samples).any()


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("relaxed_sample", [True, False])
@pytest.mark.parametrize(("num_nodes", "sample_shape"), [(3, tuple()), (4, (20,)), (2, (4, 5))])
@pytest.mark.parametrize("context_length", [3])
def test_sample_shape_lagged(
    num_nodes: int, sample_shape: torch.Size, relaxed_sample: bool, batch_shape: torch.Size, context_length: int
):
    """Test the sample/rsample method returns binary tensors in the support of the correct shape for lagged distribution."""

    dist = _distribution_factory(
        dist_class=RhinoLaggedAdjacencyDistribution,
        num_nodes=num_nodes,
        batch_shape=batch_shape,
        context_length=context_length,
    )
    samples = dist.relaxed_sample(sample_shape, temperature=0.1) if relaxed_sample else dist.sample(sample_shape)
    assert samples.shape == sample_shape + batch_shape + (context_length - 1, num_nodes, num_nodes)

    dist._validate_sample(samples)
    assert not np.isnan(samples).any()


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("relaxed_sample", [True, False])
@pytest.mark.parametrize(("num_nodes", "sample_shape"), [(3, tuple()), (4, (20,)), (2, (4, 5))])
@pytest.mark.parametrize(("context_length"), [1, 3])
@pytest.mark.parametrize("dist_class", TEMPORAL_DIST_CLASSES)
def test_sample_shape_temporal(
    num_nodes: int,
    sample_shape: torch.Size,
    relaxed_sample: bool,
    batch_shape: torch.Size,
    context_length: int,
    dist_class: Type[TemporalAdjacencyDistribution],
):
    """Test the sample/rsample method returns binary tensors in the support of the correct shape for temporal distribution."""

    dist = _distribution_factory(
        dist_class=dist_class, num_nodes=num_nodes, batch_shape=batch_shape, context_length=context_length
    )
    samples = dist.relaxed_sample(sample_shape, temperature=0.1) if relaxed_sample else dist.sample(sample_shape)
    assert samples.shape == sample_shape + batch_shape + (context_length, num_nodes, num_nodes)

    dist._validate_sample(samples)
    assert not np.isnan(samples).any()


@pytest.mark.parametrize("relaxed_sample", [True, False])
@pytest.mark.parametrize("sample_shape", [(2000,), (40, 50)])
@pytest.mark.parametrize("dist_class", DIST_CLASSES)
def test_sample_distinct(
    dist_class: Type[AdjacencyDistribution],
    sample_shape: torch.Size,
    relaxed_sample: bool,
):
    """Test the sample/rsample method returns distinct binary tensors"""
    num_nodes = 4

    dist = _distribution_factory(dist_class=dist_class, num_nodes=num_nodes, batch_shape=torch.Size())
    samples = dist.relaxed_sample(sample_shape, temperature=0.1) if relaxed_sample else dist.sample(sample_shape)
    ref_sample = samples.reshape(-1, num_nodes, num_nodes)
    # Check that the samples are distinct
    assert not np.all(np.isclose(samples, ref_sample[0, ...]))


@pytest.mark.parametrize("relaxed_sample", [True, False])
@pytest.mark.parametrize("sample_shape", [(2000,), (40, 50)])
@pytest.mark.parametrize("context_length", [3])
def test_sample_distinct_lagged(
    sample_shape: torch.Size,
    relaxed_sample: bool,
    context_length: int,
):
    """Test the sample/rsample method returns distinct binary tensors for lagged distribution."""
    num_nodes = 4

    dist = _distribution_factory(
        dist_class=RhinoLaggedAdjacencyDistribution,
        num_nodes=num_nodes,
        batch_shape=torch.Size(),
        context_length=context_length,
    )
    samples = dist.relaxed_sample(sample_shape, temperature=0.1) if relaxed_sample else dist.sample(sample_shape)
    ref_sample = samples.reshape(-1, context_length - 1, num_nodes, num_nodes)

    # Check that the samples are distinct
    assert not np.all(np.isclose(samples, ref_sample[0, ...]))


@pytest.mark.parametrize("relaxed_sample", [True, False])
@pytest.mark.parametrize("sample_shape", [(2000,), (40, 50)])
@pytest.mark.parametrize("dist_class", TEMPORAL_DIST_CLASSES)
@pytest.mark.parametrize("context_length", [1, 3])
def test_sample_distinct_temporal(
    dist_class: Type[TemporalAdjacencyDistribution],
    sample_shape: torch.Size,
    relaxed_sample: bool,
    context_length: int,
):
    """Test the sample/rsample method returns distinct binary tensors for temporal adjacency distribution."""
    num_nodes = 4

    dist = _distribution_factory(
        dist_class=dist_class, num_nodes=num_nodes, batch_shape=torch.Size(), context_length=context_length
    )
    samples = dist.relaxed_sample(sample_shape, temperature=0.1) if relaxed_sample else dist.sample(sample_shape)
    ref_sample = samples.reshape(-1, context_length, num_nodes, num_nodes)

    # Check that the samples are distinct
    assert not np.all(np.isclose(samples, ref_sample[0, ...]))


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("dist_class", DIST_CLASSES)
def test_mean(
    dist_class: Type[AdjacencyDistribution],
    batch_shape: torch.Size,
):
    """Test basic properties of the means of the distributions"""
    num_nodes = 3

    dist = _distribution_factory(dist_class=dist_class, num_nodes=num_nodes, batch_shape=batch_shape)
    mean = dist.mean
    assert mean.shape == batch_shape + (num_nodes, num_nodes)

    assert (mean <= 1.0).all()
    assert (mean >= 0.0).all()


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("context_length", [3])
def test_mean_lagged(
    batch_shape: torch.Size,
    context_length: int,
):
    """Test basic properties of the means of the lagged distributions"""
    num_nodes = 3

    dist = _distribution_factory(
        dist_class=RhinoLaggedAdjacencyDistribution,
        num_nodes=num_nodes,
        batch_shape=batch_shape,
        context_length=context_length,
    )
    mean = dist.mean
    assert mean.shape == batch_shape + (context_length - 1, num_nodes, num_nodes)

    assert (mean <= 1.0).all()
    assert (mean >= 0.0).all()


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("dist_class", TEMPORAL_DIST_CLASSES)
@pytest.mark.parametrize("context_length", [1, 3])
def test_mean_temporal(
    dist_class: Type[TemporalAdjacencyDistribution],
    batch_shape: torch.Size,
    context_length: int,
):
    """Test basic properties of the means of the temporal adjacency distributions"""
    num_nodes = 3
    dist = _distribution_factory(
        dist_class=dist_class, num_nodes=num_nodes, batch_shape=batch_shape, context_length=context_length
    )
    mean = dist.mean
    assert mean.shape == batch_shape + (context_length, num_nodes, num_nodes)
    assert (mean <= 1.0).all()
    assert (mean >= 0.0).all()


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("dist_class", DIST_CLASSES)
def test_mode(
    dist_class: Type[AdjacencyDistribution],
    batch_shape: torch.Size,
):
    """Test basic properties of the modes of the distributions"""
    num_nodes = 3

    dist = _distribution_factory(dist_class=dist_class, num_nodes=num_nodes, batch_shape=batch_shape)
    mode = dist.mode
    assert mode.shape == batch_shape + (num_nodes, num_nodes)


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("context_length", [3])
def test_mode_lagged(
    batch_shape: torch.Size,
    context_length: int,
):
    """Test basic properties of the modes of the lagged distributions"""
    num_nodes = 3

    dist = _distribution_factory(
        dist_class=RhinoLaggedAdjacencyDistribution,
        num_nodes=num_nodes,
        batch_shape=batch_shape,
        context_length=context_length,
    )
    mode = dist.mode
    assert mode.shape == batch_shape + (context_length - 1, num_nodes, num_nodes)


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("dist_class", TEMPORAL_DIST_CLASSES)
@pytest.mark.parametrize("context_length", [1, 3])
def test_mode_temporal(
    dist_class: Type[TemporalAdjacencyDistribution],
    batch_shape: torch.Size,
    context_length: int,
):
    """Test basic properties of the modes of the temporal adjacency distributions"""
    num_nodes = 3

    dist = _distribution_factory(
        dist_class=dist_class, num_nodes=num_nodes, batch_shape=batch_shape, context_length=context_length
    )
    mode = dist.mode
    assert mode.shape == batch_shape + (context_length, num_nodes, num_nodes)


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("sample_shape", [tuple(), (2,), (3,)])
@pytest.mark.parametrize("dist_class", DIST_CLASSES)
def test_log_prob(
    dist_class: Type[AdjacencyDistribution],
    batch_shape: torch.Size,
    sample_shape: torch.Size,
):
    """Test basic properties of the log_prob of the distributions"""
    num_nodes = 4

    dist = _distribution_factory(dist_class=dist_class, num_nodes=num_nodes, batch_shape=batch_shape)
    shapes = sample_shape + batch_shape + (num_nodes, num_nodes)

    values = torch.randint(0, 2, shapes, dtype=torch.float64)
    log_probs = dist.log_prob(values)
    assert log_probs.shape == sample_shape + batch_shape


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("sample_shape", [tuple(), (2,), (3,)])
@pytest.mark.parametrize("context_length", [3])
def test_log_prob_lagged(
    batch_shape: torch.Size,
    sample_shape: torch.Size,
    context_length: int,
):
    """Test basic properties of the log_prob of the lagged distributions"""
    num_nodes = 4

    dist = _distribution_factory(
        dist_class=RhinoLaggedAdjacencyDistribution,
        num_nodes=num_nodes,
        batch_shape=batch_shape,
        context_length=context_length,
    )
    shapes = sample_shape + batch_shape + (context_length - 1, num_nodes, num_nodes)

    values = torch.randint(0, 2, shapes, dtype=torch.float64)
    log_probs = dist.log_prob(values)
    assert log_probs.shape == sample_shape + batch_shape


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("sample_shape", [tuple(), (2,), (3,)])
@pytest.mark.parametrize("dist_class", TEMPORAL_DIST_CLASSES)
@pytest.mark.parametrize("context_length", [1, 3])
def test_log_prob_temporal(
    dist_class: Type[TemporalAdjacencyDistribution],
    batch_shape: torch.Size,
    sample_shape: torch.Size,
    context_length: int,
):
    """Test basic properties of the log_prob of the temporal adjacency distributions"""
    num_nodes = 4

    dist = _distribution_factory(
        dist_class=dist_class, num_nodes=num_nodes, batch_shape=batch_shape, context_length=context_length
    )
    shapes = sample_shape + batch_shape + (context_length, num_nodes, num_nodes)

    values = torch.randint(0, 2, shapes, dtype=torch.float64)
    log_probs = dist.log_prob(values)
    assert log_probs.shape == sample_shape + batch_shape
