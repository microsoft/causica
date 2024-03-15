import torch
from scotch.sdes.sdes_core import SDE


class YeastGlycolysisSDE(SDE):
    """Implementation of yeast glycolysis SDE (Daniels & Nemenman 2015, Bellot et al. 2022); see paper for more details.

    Attributes:
        noise_type: type of SDE noise, required for BaseSDE; always "diagonal" for glycolysis
        sde_type: type of SDE, required for BaseSDE; always "ito" for glycolysis
        noise_scale: diffusion coefficient constant
        others: parameters for yeast glycolysis SDE

    Methods:
        f: drift coefficient for glycolysis system
        g: diffusion coefficient for glycolysis system
    """

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(
        self,
        noise_scale,
        k1=0.52,
        K1=100,
        K2=6,
        K3=16,
        K4=100,
        K5=1.28,
        K6=12,
        K=1.8,
        kappa=13,
        phi=0.1,
        q=4,
        A=4,
        N=1,
        J0=2.5,
    ):
        super().__init__()
        self.noise_scale = noise_scale
        self.k1 = k1
        self.K1 = K1
        self.K2 = K2
        self.K3 = K3
        self.K4 = K4
        self.K5 = K5
        self.K6 = K6
        self.K = K
        self.kappa = kappa
        self.phi = phi
        self.q = q
        self.A = A
        self.N = N
        self.J0 = J0

    @staticmethod
    def graph():
        return torch.tensor(
            [
                [1, 1, 0, 0, 0, 1, 0],
                [0, 1, 1, 0, 1, 0, 0],
                [0, 0, 1, 1, 0, 1, 0],
                [0, 0, 0, 1, 1, 0, 1],
                [0, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 1],
            ],
            dtype=torch.long,
        )

    def f(self, t, y):
        _ = t
        dydt = torch.zeros_like(y)
        dydt[:, 0] = self.J0 - (self.K1 * y[:, 0] * y[:, 5]) / (1 + (y[:, 5] / self.k1) ** self.q)
        dydt[:, 1] = (
            (2 * self.K1 * y[:, 0] * y[:, 5]) / (1 + (y[:, 5] / self.k1) ** self.q)
            - self.K2 * y[:, 1] * (self.N - y[:, 4])
            - self.K6 * y[:, 1] * y[:, 4]
        )
        dydt[:, 2] = self.K2 * y[:, 1] * (self.N - y[:, 4]) - self.K3 * y[:, 2] * (self.A - y[:, 5])
        dydt[:, 3] = (
            self.K3 * y[:, 2] * (self.A - y[:, 5]) - self.K4 * y[:, 3] * y[:, 4] - self.kappa * (y[:, 3] - y[:, 6])
        )
        dydt[:, 4] = self.K2 * y[:, 1] * (self.N - y[:, 4]) - self.K4 * y[:, 3] * y[:, 4] - self.K6 * y[:, 1] * y[:, 4]
        dydt[:, 5] = (
            (-2 * self.K1 * y[:, 0] * y[:, 5]) / (1 + (y[:, 5] / self.k1) ** self.q)
            + 2 * self.K3 * y[:, 2] * (self.A - y[:, 5])
            - self.K5 * y[:, 5]
        )
        dydt[:, 6] = self.phi * self.kappa * (y[:, 3] - y[:, 6]) - self.K * y[:, 6]
        return dydt

    def g(self, t, y):
        _ = t
        return self.noise_scale * torch.ones_like(y)
