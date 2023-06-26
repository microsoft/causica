import pytest
import torch
from tensordict import TensorDict

from causica.distributions import (
    BernoulliNoise,
    CategoricalNoise,
    IndependentNoise,
    JointNoise,
    Noise,
    UnivariateNormalNoise,
)

NOISE_DISTRIBUTIONS = [
    IndependentNoise(BernoulliNoise(torch.randn(3), torch.randn(3)), 1),
    IndependentNoise(UnivariateNormalNoise(torch.randn(5), torch.arange(1, 6, dtype=torch.float)), 1),
    IndependentNoise(UnivariateNormalNoise(torch.randn(1), torch.tensor([4.0])), 1),
    CategoricalNoise(torch.randn(5), torch.randn(5)),
    CategoricalNoise(torch.randn(1), torch.randn(1)),
]
SAMPLE_SHAPES = [torch.Size([]), torch.Size([3]), torch.Size([10, 3]), torch.Size([6, 6, 6])]


@pytest.mark.parametrize("noise", NOISE_DISTRIBUTIONS)
def test_joint_noise_passthrough(noise: Noise):
    """Test that properties from the underlying noise distributions are passed through."""
    joint_noise = JointNoise({"a": noise})

    # Distribution properties
    torch.testing.assert_close(joint_noise.entropy(), noise.entropy())
    torch.testing.assert_close(joint_noise.mode.get("a"), noise.mode)
    torch.testing.assert_close(joint_noise.mean.get("a"), noise.mean)


@pytest.mark.parametrize("noise_a", NOISE_DISTRIBUTIONS)
@pytest.mark.parametrize("noise_b", NOISE_DISTRIBUTIONS)
def test_joint_noise_properties(noise_a: Noise, noise_b: Noise):
    """Test that properties from the underlying noise distributions are passed through for pairs of noise."""
    joint_noise = JointNoise({"a": noise_a, "b": noise_b})

    # Distribution properties
    torch.testing.assert_close(joint_noise.entropy(), noise_a.entropy() + noise_b.entropy())
    torch.testing.assert_close(joint_noise.mode.get("a"), noise_a.mode)
    torch.testing.assert_close(joint_noise.mode.get("b"), noise_b.mode)
    torch.testing.assert_close(joint_noise.mean.get("a"), noise_a.mean)
    torch.testing.assert_close(joint_noise.mean.get("b"), noise_b.mean)


@pytest.mark.parametrize("noise_a", NOISE_DISTRIBUTIONS)
@pytest.mark.parametrize("noise_b", NOISE_DISTRIBUTIONS)
@pytest.mark.parametrize("sample_shape", SAMPLE_SHAPES)
def test_joint_noise_sample_log_prob(noise_a: Noise, noise_b: Noise, sample_shape: torch.Size):
    """Test that sampling and log prob work as expected for pairs of noise."""
    joint_noise = JointNoise({"a": noise_a, "b": noise_b})
    sample_a = noise_a.sample(sample_shape)
    sample_b = noise_b.sample(sample_shape)
    joint_sample = TensorDict({"a": sample_a, "b": sample_b}, batch_size=sample_shape)
    torch.testing.assert_close(
        joint_noise.log_prob(joint_sample), noise_a.log_prob(sample_a) + noise_b.log_prob(sample_b)
    )
    joint_sample = joint_noise.sample()
    sample_a = joint_sample.get("a")
    sample_b = joint_sample.get("b")
    torch.testing.assert_close(
        joint_noise.log_prob(joint_sample), noise_a.log_prob(sample_a) + noise_b.log_prob(sample_b)
    )


@pytest.mark.parametrize("noise_a", NOISE_DISTRIBUTIONS)
@pytest.mark.parametrize("noise_b", NOISE_DISTRIBUTIONS)
def test_joint_noise_empirical(noise_a: Noise, noise_b: Noise):
    """Test that the empirical distributions behave as expected for pairs of noise."""
    joint_noise = JointNoise({"a": noise_a, "b": noise_b})
    sample_shape = torch.Size([1000, 200, 10])

    # Produce samples and log probs in both directions
    samples = TensorDict(
        {"a": noise_a.sample(sample_shape), "b": noise_b.sample(sample_shape)},
        batch_size=sample_shape + noise_a.batch_shape,
    )
    joint_samples = joint_noise.sample(sample_shape)
    log_probs = noise_a.log_prob(joint_samples.get("a")) + noise_b.log_prob(joint_samples.get("b"))
    joint_log_probs = joint_noise.log_prob(samples)
    assert samples.get("a").shape == joint_samples.get("a").shape
    assert samples.get("b").shape == joint_samples.get("b").shape
    assert log_probs.shape == joint_log_probs.shape

    # Similar empirical distribution
    sample_dim = tuple(range(len(sample_shape)))
    mean, std = torch.mean, torch.std
    for key, value in samples.items():
        joint_value = joint_samples.get(key)
        torch.testing.assert_close(mean(value, sample_dim), mean(joint_value, sample_dim), atol=0.01, rtol=0.01)
        torch.testing.assert_close(std(value, sample_dim), std(joint_value, sample_dim), atol=0.01, rtol=0.01)

    # Similar log probs
    torch.testing.assert_close(mean(log_probs), mean(joint_log_probs), atol=0.01, rtol=0.01)
    torch.testing.assert_close(std(log_probs), std(joint_log_probs), atol=0.01, rtol=0.01)
