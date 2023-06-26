"""Module with generic Adjacency Distribution tests."""
from typing import Type

import numpy as np
import pytest
import torch

from causica.distributions.adjacency import (
    AdjacencyDistribution,
    ConstrainedAdjacencyDistribution,
    ENCOAdjacencyDistribution,
    ThreeWayAdjacencyDistribution,
)


def _distribution_factory(
    dist_class: Type[AdjacencyDistribution], num_nodes: int, batch_shape: torch.Size
) -> AdjacencyDistribution:
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
    if dist_class is ENCOAdjacencyDistribution:
        length = (num_nodes * (num_nodes - 1)) // 2
        return ENCOAdjacencyDistribution(
            logits_exist=torch.randn(batch_shape + (num_nodes, num_nodes)),
            logits_orient=torch.randn(batch_shape + (length,)),
        )
    if dist_class is ThreeWayAdjacencyDistribution:
        logits = torch.randn(batch_shape + ((num_nodes * (num_nodes - 1)) // 2, 3))
        return ThreeWayAdjacencyDistribution(logits=logits)
    raise ValueError("Unrecognised Class")


DIST_CLASSES = [ConstrainedAdjacencyDistribution, ENCOAdjacencyDistribution, ThreeWayAdjacencyDistribution]
BATCH_SHAPES = [torch.Size(), (2,)]

# pylint: disable=protected-access
@pytest.mark.parametrize("dist_class", DIST_CLASSES)
@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
def test_support(dist_class: Type[AdjacencyDistribution], batch_shape: torch.Size):
    """Test that the defined support works as expected. This method will be used to test other features."""
    num_nodes = 3
    dist = _distribution_factory(dist_class=dist_class, num_nodes=num_nodes, batch_shape=batch_shape)

    mat = torch.ones((num_nodes, num_nodes))

    # validate sample returns None when there is no error
    assert dist._validate_sample(mat) is None

    # validate sample throws when the sample is invalid
    with pytest.raises(ValueError):
        dist._validate_sample(4 * mat)


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("dist_class", DIST_CLASSES)
@pytest.mark.parametrize("relaxed_sample", [True, False])
@pytest.mark.parametrize(("num_nodes", "sample_shape"), [(3, tuple()), (4, (20,)), (2, (4, 5))])
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


@pytest.mark.parametrize("dist_class", DIST_CLASSES)
@pytest.mark.parametrize("relaxed_sample", [True, False])
@pytest.mark.parametrize("sample_shape", [(2000,), (40, 50)])
def test_sample_distinct(
    dist_class: Type[AdjacencyDistribution],
    sample_shape: torch.Size,
    relaxed_sample: bool,
):
    """Test the sample/rsample method returns distinct binary tensors"""
    num_nodes = 4
    dist = _distribution_factory(dist_class=dist_class, num_nodes=num_nodes, batch_shape=torch.Size())
    samples = dist.relaxed_sample(sample_shape, temperature=0.1) if relaxed_sample else dist.sample(sample_shape)
    # the samples should be different from each other
    assert not np.all(np.isclose(samples, samples.reshape(-1, num_nodes, num_nodes)[0, ...]))


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("dist_class", DIST_CLASSES)
def test_mean(dist_class: Type[AdjacencyDistribution], batch_shape: torch.Size):
    """Test basic properties of the means of the distributions"""
    num_nodes = 3
    dist = _distribution_factory(dist_class=dist_class, num_nodes=num_nodes, batch_shape=batch_shape)
    mean = dist.mean
    assert mean.shape == batch_shape + (num_nodes, num_nodes)
    assert (mean <= 1.0).all()
    assert (mean >= 0.0).all()


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("dist_class", DIST_CLASSES)
def test_mode(dist_class: Type[AdjacencyDistribution], batch_shape: torch.Size):
    """Test basic properties of the modes of the distributions"""
    num_nodes = 3
    dist = _distribution_factory(dist_class=dist_class, num_nodes=num_nodes, batch_shape=batch_shape)
    mode = dist.mode
    dist._validate_sample(mode)  # mode should be in the support
    assert mode.shape == batch_shape + (num_nodes, num_nodes)


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("sample_shape", [tuple(), (2,), (3,)])
@pytest.mark.parametrize("dist_class", DIST_CLASSES)
def test_log_prob(dist_class: Type[AdjacencyDistribution], batch_shape: torch.Size, sample_shape: torch.Size):
    """Test basic properties of the log_prob of the distributions"""
    num_nodes = 4
    dist = _distribution_factory(dist_class=dist_class, num_nodes=num_nodes, batch_shape=batch_shape)
    values = torch.randint(0, 2, sample_shape + batch_shape + (num_nodes, num_nodes), dtype=torch.float64)
    log_probs = dist.log_prob(values)
    assert log_probs.shape == sample_shape + batch_shape
