import torch
import torch.distributions as td


def signed_uniform_1d(low: float, high: float) -> td.Distribution:
    """
    Returns a distribution that is a mixture of two uniform distributions.

    The first uniform distribution is defined on the interval [low, high]
    and the second is defined on the interval [-high, -low].
    The mixture is defined by a Bernoulli distribution with a probability of 0.5.

    Args:
        low: the lower bound of the uniform distribution
        high: the upper bound of the uniform distribution

    Returns:
        a distribution that is a mixture of two uniform distributions
    """

    weight = torch.tensor([0.5, 0.5])
    mix = td.Categorical(probs=weight)
    low_tensor = torch.tensor([low, -high])
    high_tensor = torch.tensor([high, -low])
    comp = td.Uniform(low=low_tensor, high=high_tensor)
    return td.MixtureSameFamily(mix, comp)


class MultivariateSignedUniform(td.Distribution):
    """
    Distribution that is a mixture of two uniform distributions.

    The first uniform distribution is defined on the interval [low, high]
    and the second is defined on the interval [-high, -low].
    The mixture is defined by a Bernoulli distribution with a probability of 0.5.

    Args:
        low: the lower bound of the uniform distribution
        high: the upper bound of the uniform distribution
        size: the size of the distribution
    """

    def __init__(self, low: float, high: float, size: torch.Size):
        self.one_dim_dist = signed_uniform_1d(low, high)
        self.size = size
        super().__init__()

    def sample(self, sample_shape=torch.Size()):
        sample_shape = sample_shape + self.size

        return self.one_dim_dist.sample(sample_shape)
