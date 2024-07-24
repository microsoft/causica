import math
from dataclasses import dataclass, field
from itertools import product

import torch
import torch.distributions as td

from causica.data_generation.samplers.functional_relationships_sampler import (
    FunctionalRelationshipsSampler,
    HeteroscedasticRFFFunctionalRelationshipsSampler,
    LinearRelationshipsSampler,
    RFFFunctionalRelationshipsSampler,
)
from causica.data_generation.samplers.noise_dist_sampler import (
    JointNoiseModuleSampler,
    NoiseModuleSampler,
    UnivariateCauchyNoiseModuleSampler,
    UnivariateLaplaceNoiseModuleSampler,
    UnivariateNormalNoiseModuleSampler,
)
from causica.data_generation.samplers.sem_sampler import SEMSampler
from causica.distributions.adjacency import (
    AdjacencyDistribution,
    EdgesPerNodeErdosRenyiDAGDistribution,
    GeometricRandomGraphDAGDistribution,
    ScaleFreeDAGDistribution,
    StochasticBlockModelDAGDistribution,
    WattsStrogatzDAGDistribution,
)
from causica.distributions.signed_uniform import MultivariateSignedUniform


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


@dataclass(frozen=True)
class SFConfig:
    """
    Config for the SF Graph Distribution
    """

    edges_per_node: list[int] = field(default_factory=lambda: [1, 2, 3])
    attach_power: list[float] = field(default_factory=lambda: [1.0])


@dataclass(frozen=True)
class LinearConfig:
    """
    Config for the Linear Functional Relationship Sampler
    """

    weight_low: float = 1.0
    weight_high: float = 3.0
    bias_low: float = -3.0
    bias_high: float = 3.0


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class HeteroscedasticRFFConfig:
    """
    Config for the Heteroscedatic RFF Functional Relationship Sampler
    """

    type_noise: str = "gaussian"
    num_rf: int = 100
    length: float = 10.0
    out: float = 2.0
    log_scale: bool = True


class SemSamplerFactory:
    def __init__(
        self,
        node_nums: list[int] | int,
        noises: list[str],
        graphs: list[str],
        funcs: list[str],
        config_gaussian: GaussianConfig = GaussianConfig(),
        config_laplace: LaplaceConfig = LaplaceConfig(),
        config_cauchy: CauchyConfig = CauchyConfig(),
        config_er: ERConfig = ERConfig(),
        config_sf: SFConfig = SFConfig(),
        config_ws: WSConfig = WSConfig(),
        config_sbm: SBMConfig = SBMConfig(),
        config_grg: GRGConfig = GRGConfig(),
        config_linear: LinearConfig = LinearConfig(),
        config_rff: RFFConfig = RFFConfig(),
        config_heteroscedastic_rff: HeteroscedasticRFFConfig = HeteroscedasticRFFConfig(),
    ) -> None:
        self.node_nums = node_nums
        self.noises = noises
        self.graphs = graphs
        self.funcs = funcs

        self.config_gaussian = config_gaussian
        self.config_laplace = config_laplace
        self.config_cauchy = config_cauchy

        self.config_er = config_er
        self.config_sf = config_sf
        self.config_ws = config_ws
        self.config_sbm = config_sbm
        self.config_grg = config_grg

        self.config_linear = config_linear
        self.config_rff = config_rff
        self.config_heteroscedastic_rff = config_heteroscedastic_rff

    def __call__(self) -> list[SEMSampler]:
        sem_samplers: list[SEMSampler] = []

        if isinstance(self.node_nums, int):
            self.node_nums = [self.node_nums]

        std_dist: td.Distribution | float
        noise_dist_samplers: dict[str, NoiseModuleSampler]
        adjacency_dist: AdjacencyDistribution
        functional_relationships_sampler: FunctionalRelationshipsSampler
        for num_nodes in self.node_nums:
            shapes_dict = {f"x_{i}": torch.Size([1]) for i in range(num_nodes)}
            dim = sum(shape.numel() for shape in shapes_dict.values())
            for type_noise, type_graph, type_func in product(self.noises, self.graphs, self.funcs):
                match type_noise:
                    case "gaussian":
                        if type_func == "linear":
                            # in case of GLM, we fix the variance to be the same for all nodes
                            std_dist = (
                                td.Uniform(low=self.config_gaussian.low, high=self.config_gaussian.high).sample().item()
                            )
                        else:
                            std_dist = td.Uniform(low=self.config_gaussian.low, high=self.config_gaussian.high)
                        noise_dist_samplers = {
                            f"x_{i}": UnivariateNormalNoiseModuleSampler(
                                std_dist=std_dist,
                                dim=1,
                            )
                            for i in range(num_nodes)
                        }
                        log_functional_rescaling_sampler = None
                    case "laplace":
                        noise_dist_samplers = {
                            f"x_{i}": UnivariateLaplaceNoiseModuleSampler(
                                std_dist=td.Uniform(low=self.config_laplace.low, high=self.config_laplace.high), dim=1
                            )
                            for i in range(num_nodes)
                        }
                        log_functional_rescaling_sampler = None
                    case "cauchy":
                        noise_dist_samplers = {
                            f"x_{i}": UnivariateCauchyNoiseModuleSampler(
                                std_dist=td.Uniform(low=self.config_cauchy.low, high=self.config_cauchy.high), dim=1
                            )
                            for i in range(num_nodes)
                        }
                        log_functional_rescaling_sampler = None
                    case "hrff":
                        match self.config_heteroscedastic_rff.type_noise:
                            case "gaussian":
                                noise_dist_samplers = {
                                    f"x_{i}": UnivariateNormalNoiseModuleSampler(std_dist=1.0, dim=1)
                                    for i in range(num_nodes)
                                }
                            case "laplace":
                                noise_dist_samplers = {
                                    f"x_{i}": UnivariateLaplaceNoiseModuleSampler(std_dist=1.0, dim=1)
                                    for i in range(num_nodes)
                                }
                            case "cauchy":
                                noise_dist_samplers = {
                                    f"x_{i}": UnivariateCauchyNoiseModuleSampler(std_dist=1.0, dim=1)
                                    for i in range(num_nodes)
                                }
                        if self.config_heteroscedastic_rff.type_noise == "gaussian" and type_func == "linear":
                            # in case of GLM, we do not apply the data-dependent transformation of the variance
                            log_functional_rescaling_sampler = None
                        else:
                            num_rf = self.config_heteroscedastic_rff.num_rf
                            zeros_vector_rf = torch.zeros((num_rf,), dtype=torch.float32)
                            ones_vector_rf = torch.ones((num_rf,), dtype=torch.float32)
                            zeros_matrix = torch.zeros((num_rf, dim), dtype=torch.float32)
                            ones_vector_dim = torch.ones((dim,), dtype=torch.float32)
                            log_functional_rescaling_sampler = HeteroscedasticRFFFunctionalRelationshipsSampler(
                                rf_dist=td.MultivariateNormal(zeros_matrix, covariance_matrix=torch.eye(dim)),
                                coeff_dist=td.MultivariateNormal(zeros_vector_rf, covariance_matrix=torch.eye(num_rf)),
                                shapes_dict=shapes_dict,
                                length_dist=self.config_heteroscedastic_rff.length,
                                out_dist=self.config_heteroscedastic_rff.out,
                                angle_dist=td.Uniform(low=zeros_vector_rf, high=2 * math.pi * ones_vector_rf),
                                log_scale=self.config_heteroscedastic_rff.log_scale,
                            )
                    case _:
                        raise ValueError(f"Unknown type_dist: {type_noise}")

                joint_noise_module_sampler = JointNoiseModuleSampler(noise_dist_samplers)

                match type_graph:
                    case "er":
                        adjacency_dist = EdgesPerNodeErdosRenyiDAGDistribution(
                            num_nodes=num_nodes,
                            edges_per_node=self.config_er.edges_per_node,
                        )
                    case "sf_in":
                        adjacency_dist = ScaleFreeDAGDistribution(
                            num_nodes=num_nodes,
                            edges_per_node=self.config_sf.edges_per_node,
                            power=self.config_sf.attach_power,
                            in_degree=True,
                        )
                    case "sf_out":
                        adjacency_dist = ScaleFreeDAGDistribution(
                            num_nodes=num_nodes,
                            edges_per_node=self.config_sf.edges_per_node,
                            power=self.config_sf.attach_power,
                            in_degree=False,
                        )
                    case "ws":
                        adjacency_dist = WattsStrogatzDAGDistribution(
                            num_nodes=num_nodes,
                            lattice_dim=self.config_ws.lattice_dim,
                            rewire_prob=self.config_ws.rewire_prob,
                            neighbors=self.config_ws.neighbors,
                        )
                    case "sbm":
                        adjacency_dist = StochasticBlockModelDAGDistribution(
                            num_nodes=num_nodes,
                            edges_per_node=self.config_sbm.edges_per_node,
                            num_blocks=self.config_sbm.num_blocks,
                            damping=self.config_sbm.damping,
                        )
                    case "grg":
                        adjacency_dist = GeometricRandomGraphDAGDistribution(
                            num_nodes=num_nodes, radius=self.config_grg.radius
                        )
                    case _:
                        raise ValueError(f"Unknown type_dist: {type_graph}")

                match type_func:
                    case "linear":
                        one_vector_dim = torch.ones((dim,), dtype=torch.float32)
                        functional_relationships_sampler = LinearRelationshipsSampler(
                            scale_dist=MultivariateSignedUniform(
                                low=self.config_linear.weight_low,
                                high=self.config_linear.weight_high,
                                size=torch.Size([dim, dim]),
                            ),
                            shapes_dict=shapes_dict,
                            bias_dist=td.Uniform(
                                low=self.config_linear.bias_low * one_vector_dim,
                                high=self.config_linear.bias_high * one_vector_dim,
                            ),
                        )
                    case "rff":
                        num_rf = self.config_rff.num_rf
                        zeros_vector_rf = torch.zeros((num_rf,), dtype=torch.float32)
                        ones_vector_rf = torch.ones((num_rf,), dtype=torch.float32)
                        zeros_matrix = torch.zeros((num_rf, dim), dtype=torch.float32)
                        ones_vector_dim = torch.ones((dim,), dtype=torch.float32)
                        functional_relationships_sampler = RFFFunctionalRelationshipsSampler(
                            rf_dist=td.MultivariateNormal(zeros_matrix, covariance_matrix=torch.eye(dim)),
                            coeff_dist=td.MultivariateNormal(zeros_vector_rf, covariance_matrix=torch.eye(num_rf)),
                            shapes_dict=shapes_dict,
                            bias_dist=td.Uniform(
                                low=self.config_rff.bias_low * ones_vector_dim,
                                high=self.config_rff.bias_high * ones_vector_dim,
                            ),
                            length_dist=td.Uniform(
                                low=self.config_rff.length_low * ones_vector_dim,
                                high=self.config_rff.length_high * ones_vector_dim,
                            ),
                            out_dist=td.Uniform(
                                low=self.config_rff.out_low * ones_vector_dim,
                                high=self.config_rff.out_high * ones_vector_dim,
                            ),
                            angle_dist=td.Uniform(low=zeros_vector_rf, high=2 * math.pi * ones_vector_rf),
                        )
                    case _:
                        raise ValueError(f"Unknown type_dist: {type_func}")

                # append the sem sampler in the list
                sem_samplers.append(
                    SEMSampler(
                        adjacency_dist=adjacency_dist,
                        joint_noise_module_sampler=joint_noise_module_sampler,
                        functional_relationships_sampler=functional_relationships_sampler,
                        log_functional_rescaling_sampler=log_functional_rescaling_sampler,
                    )
                )

        return sem_samplers
