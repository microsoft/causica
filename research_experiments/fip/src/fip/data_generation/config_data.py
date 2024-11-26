from dataclasses import dataclass, field


@dataclass(frozen=True)
class GaussianConfig:
    """Config for the Gaussian Noise Distribution"""

    low: float = 0.2
    high: float = 2.0


@dataclass(frozen=True)
class LaplaceConfig:
    """Config for the Laplace Noise Distribution"""

    low: float = 0.2
    high: float = 2.0


@dataclass(frozen=True)
class CauchyConfig:
    """Config for the Cauchy Noise Distribution"""

    low: float = 0.2
    high: float = 2.0


@dataclass(frozen=True)
class GRGConfig:
    """
    Config for the GRG Graph Distribution
    """

    radius: list[float] = field(default_factory=lambda: [0.1])


@dataclass(frozen=True)
class WSConfig:
    """
    Config for the WS Graph Distribution
    """

    lattice_dim: list[int] = field(default_factory=lambda: [2, 3])
    rewire_prob: list[float] = field(default_factory=lambda: [0.3])
    neighbors: list[int] = field(default_factory=lambda: [1])


@dataclass(frozen=True)
class SBMConfig:
    """
    Config for the SBM Graph Distribution
    """

    edges_per_node: list[int] = field(default_factory=lambda: [2])
    num_blocks: list[int] = field(default_factory=lambda: [5, 10])
    damping: list[float] = field(default_factory=lambda: [0.1])


@dataclass(frozen=True)
class ERConfig:
    """
    Config for the ER Graph Distribution
    """

    edges_per_node: list[int] = field(default_factory=lambda: [1, 2, 3])


@dataclass(frozen=False)
class SFConfig:
    """
    Config for the SF Graph Distribution
    """

    edges_per_node: list[int] = field(default_factory=lambda: [1, 2, 3])
    attach_power: list[float] = field(default_factory=lambda: [1.0])


@dataclass(frozen=False)
class LinearConfig:
    """
    Config for the Linear Functional Relationship Sampler
    """

    weight_low: float = 1.0
    weight_high: float = 3.0
    bias_low: float = -3.0
    bias_high: float = 3.0


@dataclass(frozen=False)
class RFFConfig:
    """
    Config for the RFF Functional Relationship Sampler
    """

    num_rf: int = 100
    length_low: float = 7.0
    length_high: float = 10.0
    out_low: float = 10.0
    out_high: float = 20.0
    bias_low: float = -3.0
    bias_high: float = 3.0


@dataclass(frozen=False)
class HeteroscedasticRFFConfig:
    """
    Config for the Heteroscedatic RFF Functional Relationship Sampler
    """

    type_noise: str = "gaussian"
    num_rf: int = 100
    length: float = 10.0
    out: float = 2.0
    log_scale: bool = True
