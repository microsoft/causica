"""Module with ENCO specific tests."""

import numpy as np
import pytest
import torch
from torch.distributions.utils import probs_to_logits

from causica.distributions.adjacency.enco import ENCOAdjacencyDistribution
from causica.distributions.adjacency.temporal_adjacency_distributions import (
    RhinoLaggedAdjacencyDistribution,
    TemporalAdjacencyDistribution,
)


def test_enco_entropy():
    """Test ENCO entropy is correct for a known distribution."""
    eps = 1e-7

    probs_exist = torch.tensor([[0.0, 1.0 - eps], [1.0 - eps, 0.0]])
    probs_orient = torch.tensor([0.5])
    logits_exist = torch.nn.Parameter(probs_to_logits(probs_exist, is_binary=True), requires_grad=True)
    logits_orient = torch.nn.Parameter(probs_to_logits(probs_orient, is_binary=True), requires_grad=True)
    dist = ENCOAdjacencyDistribution(logits_exist=logits_exist, logits_orient=logits_orient)
    target_entropy = 2 * np.log(2)
    entropy = dist.entropy()

    np.testing.assert_allclose(entropy.detach(), target_entropy)
    entropy.backward()
    assert logits_exist.grad is not None
    assert logits_orient.grad is not None
    np.testing.assert_allclose(logits_exist.grad, np.zeros_like(logits_exist.detach().numpy()), atol=1e-7)
    np.testing.assert_allclose(logits_orient.grad, np.zeros_like(logits_orient.detach().numpy()), atol=1e-7)
    optim = torch.optim.SGD((logits_orient, logits_exist), lr=1e-2, momentum=0.9)
    optim.step()


@pytest.mark.parametrize("context_length", [1, 3])
def test_temporal_enco_entropy(context_length: int):
    """Test temporal ENCO entropy is correct for a known distribution."""
    eps = 1e-7
    probs_exist = torch.cat(
        [0.5 * (torch.ones(context_length - 1, 2, 2) - eps), torch.tensor([[[0.0, 1.0 - eps], [1.0 - eps, 0.0]]])],
        dim=0,
    )
    probs_orient = torch.tensor([0.5])
    logits_exist = torch.nn.Parameter(probs_to_logits(probs_exist, is_binary=True), requires_grad=True)
    logits_orient = torch.nn.Parameter(probs_to_logits(probs_orient, is_binary=True), requires_grad=True)
    inst_dist = ENCOAdjacencyDistribution(logits_exist=logits_exist[..., -1, :, :], logits_orient=logits_orient)
    lagged_dist = (
        RhinoLaggedAdjacencyDistribution(logits_edge=logits_exist[..., :-1, :, :], lags=context_length - 1)
        if context_length > 1
        else None
    )
    dist_temporal = TemporalAdjacencyDistribution(inst_dist, lagged_dist)
    target_entropy = (context_length - 1) * 4 * np.log(2) + 2 * np.log(2)
    entropy = dist_temporal.entropy()

    np.testing.assert_allclose(entropy.detach(), target_entropy)
    entropy.backward()
    assert logits_exist.grad is not None
    assert logits_orient.grad is not None
    np.testing.assert_allclose(logits_exist.grad, np.zeros_like(logits_exist.detach().numpy()), atol=1e-7)
    np.testing.assert_allclose(logits_orient.grad, np.zeros_like(logits_orient.detach().numpy()), atol=1e-7)
    optim = torch.optim.SGD((logits_orient, logits_exist), lr=1e-2, momentum=0.9)
    optim.step()


def test_enco_backprop():
    """Test ENCO backpropagation works."""

    logits_exist = torch.nn.Parameter(torch.zeros((2, 2)), requires_grad=True)
    logits_orient = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
    logits_exist_np = np.array(logits_exist.detach().numpy())
    logits_orient_np = np.array(logits_orient.detach().numpy())
    optim = torch.optim.SGD((logits_orient, logits_exist), lr=1e-2, momentum=0.9)
    dist = ENCOAdjacencyDistribution(logits_exist=logits_exist, logits_orient=logits_orient)

    # maximise the entropy
    loss = -dist.entropy()
    loss_init = loss.detach().numpy()
    loss.backward()
    optim.step()
    loss = -dist.entropy()

    # entropy should increase
    assert (loss.detach().numpy() < loss_init).all()
    logits_exist_after = logits_exist.detach().numpy()
    # diagonal elements (or diagonal in instantaneous adj for temporal adj) should remain the same
    np.testing.assert_allclose(np.diagonal(logits_exist_after), np.diagonal(logits_exist_np))
    # other edges should be more like to exist (their existence logits increase)
    assert logits_exist_after[1, 0] > logits_exist_np[1, 0]
    assert logits_exist_after[0, 1] > logits_exist_np[0, 1]
    # the orientation logit should be heading towards 0 so it should decrease from 1
    assert (logits_orient_np > logits_orient.detach().numpy()).all()
    optim.zero_grad()

    # test we can backpropagate through relaxed_sample
    sample = dist.relaxed_sample(sample_shape=torch.Size((100,)), temperature=0.1)
    loss = torch.sum((torch.eye(2)[None, ...] - sample) ** 2)
    loss.backward()
    # their should be gradients
    assert logits_exist.grad is not None
    assert logits_orient.grad is not None

    assert abs(logits_exist.grad[0, 1]) > 1e-3
    assert abs(logits_orient.grad[-1]) > 1e-3


@pytest.mark.parametrize("context_length", [1, 3])
def test_temporal_enco_backprop(context_length: int):
    """Test temporal ENCO backpropagation works."""

    eps = 1e-1
    logits_exist_prob = torch.cat(
        [eps * torch.ones(context_length - 1, 2, 2), torch.tensor([[[0.0, 0.4], [0, 0.4]]])], dim=0
    )
    logits_exist = torch.nn.Parameter(probs_to_logits(logits_exist_prob), requires_grad=True)
    logits_orient = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
    logits_exist_np = np.array(logits_exist.detach().numpy())
    logits_orient_np = np.array(logits_orient.detach().numpy())
    inst_dist = ENCOAdjacencyDistribution(logits_exist=logits_exist[..., -1, :, :], logits_orient=logits_orient)
    lagged_dist = (
        RhinoLaggedAdjacencyDistribution(logits_edge=logits_exist[..., :-1, :, :], lags=context_length - 1)
        if context_length > 1
        else None
    )
    optim = torch.optim.SGD((logits_orient, logits_exist), lr=1e-2, momentum=0.9)
    dist = TemporalAdjacencyDistribution(inst_dist, lagged_dist)

    # maximise the entropy
    loss = -dist.entropy()
    loss_init = loss.detach().numpy()
    loss.backward()
    optim.step()
    loss = -dist.entropy()

    # entropy should increase
    assert (loss.detach().numpy() < loss_init).all()
    logits_exist_after = logits_exist.detach().numpy()
    # diagonal elements (or diagonal in instantaneous adj for temporal adj) should remain the same
    np.testing.assert_allclose(
        np.diagonal(logits_exist_after[..., -1, :, :], axis1=-1, axis2=-2),
        np.diagonal(logits_exist_np[..., -1, :, :], axis1=-1, axis2=-2),
    )
    # other edges should be more like to exist (their existence logits increase)
    assert np.all(logits_exist_after[:-1, 1, 0] > logits_exist_np[:-1, 1, 0])
    assert np.all(logits_exist_after[:-1, 0, 1] > logits_exist_np[:-1, 0, 1])

    # the orientation logit should be heading towards 0 so it should decrease from 1
    assert (logits_orient_np > logits_orient.detach().numpy()).all()
    optim.zero_grad()

    # test we can backpropagate through relaxed_sample
    sample = dist.relaxed_sample(sample_shape=torch.Size((100,)), temperature=0.1)
    loss = torch.sum((torch.eye(2)[None, ...] - sample) ** 2)
    loss.backward()
    # their should be gradients
    assert logits_exist.grad is not None
    assert logits_orient.grad is not None

    grad_matrix = abs(logits_exist.grad[:-1, 0, 1]) if context_length > 1 else abs(logits_exist.grad[-1, 0, 1])
    assert torch.all(grad_matrix > 1e-3)

    assert abs(logits_orient.grad[-1]) > 1e-3


def test_enco():
    """Test ENCO methods are correct for a known distribution."""
    probs_exist = torch.tensor([[0.0, 0.3], [0.8, 0.0]])
    probs_orient = torch.tensor([0.7])
    logits_exist = probs_to_logits(probs_exist, is_binary=True)
    logits_orient = probs_to_logits(probs_orient, is_binary=True)
    dist = ENCOAdjacencyDistribution(logits_exist=logits_exist, logits_orient=logits_orient)
    np.testing.assert_allclose(torch.tensor([[0.0, 0.09], [0.56, 0.0]]), dist.mean)
    np.testing.assert_allclose(torch.tensor([[0.0, 0.0], [1.0, 0.0]]), dist.mode)
    samples = torch.tensor([[[0.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])
    np.testing.assert_allclose(
        dist.log_prob(samples), np.array([np.log(0.09 * 0.44), np.log(0.91 * 0.56), np.log(0.91 * 0.44)]), atol=1e-7
    )


def test_temporal_enco():
    """Test Temporal ENCO method are correct for a known distribution"""
    context_length = 2
    probs_exist = torch.stack([torch.tensor([[0.3, 0.7], [0.2, 0.6]]), torch.tensor([[0.9, 0.3], [0.8, 0.7]])], dim=0)
    probs_orient = torch.tensor([0.7])
    logits_exist = probs_to_logits(probs_exist, is_binary=True)
    logits_orient = probs_to_logits(probs_orient, is_binary=True)
    inst_dist = ENCOAdjacencyDistribution(logits_exist=logits_exist[..., -1, :, :], logits_orient=logits_orient)
    lagged_dist = (
        RhinoLaggedAdjacencyDistribution(logits_edge=logits_exist[..., :-1, :, :], lags=context_length - 1)
        if context_length > 1
        else None
    )
    dist = TemporalAdjacencyDistribution(instantaneous_distribution=inst_dist, lagged_distribution=lagged_dist)
    np.testing.assert_allclose(torch.tensor([[[0.3, 0.7], [0.2, 0.6]], [[0.0, 0.09], [0.56, 0.0]]]), dist.mean)
    np.testing.assert_allclose(
        torch.tensor(
            [
                [[0.0, 1.0], [0.0, 1.0]],
                [[0.0, 0.0], [1.0, 0.0]],
            ]
        ),
        dist.mode,
    )

    samples = torch.tensor(
        [
            [[[0.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]],
            [[[0.0, 1.0], [1.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]],
            [[[1.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
        ]
    )
    np.testing.assert_allclose(
        dist.log_prob(samples),
        np.array(
            [
                np.log(0.09 * 0.44 * 0.7 * 0.3 * 0.2 * 0.4),
                np.log(0.91 * 0.56 * 0.7 * 0.7 * 0.2 * 0.4),
                np.log(0.91 * 0.44 * 0.3 * 0.3 * 0.2 * 0.4),
            ]
        ),
    )
