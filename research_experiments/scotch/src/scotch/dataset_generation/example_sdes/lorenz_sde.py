import torch
from scotch.sdes.sdes_core import SDE


class LorenzSDE(SDE):
    """Implementation of Lorenz System of SDEs.

    System of SDE consisting of the Lorenz system of ODEs for the drift coefficient and identity diffusion coefficient.

    Attributes:
        noise_type: type of SDE noise, required for BaseSDE; always "diagonal" for Lorenz
        sde_type: type of SDE, required for BaseSDE; always "ito" for Lorenz
        sigma: Lorenz parameter sigma
        rho: Lorenz parameter rho
        beta: Lorenz parameter beta

    Methods:
        f: drift coefficient for Lorenz system
        g: diffusion coefficient for Lorenz system
    """

    noise_type = "diagonal"
    sde_type = "ito"

    @staticmethod
    def graph() -> torch.Tensor:
        return torch.tensor([[1, 1, 1], [1, 1, 1], [0, 1, 1]], dtype=torch.long)

    def __init__(self, sigma: float = 10.0, rho: float = 8.0 / 3, beta: float = 28.0, noise_scale: float = 1.0):
        super().__init__()
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.noise_scale = noise_scale

    def f(self, t: float, y: torch.Tensor) -> torch.Tensor:
        _ = t
        dxdt = self.sigma * (y[:, 1] - y[:, 0])
        dydt = y[:, 0] * (self.rho - y[:, 2]) - y[:, 1]
        dzdt = y[:, 0] * y[:, 1] - self.beta * y[:, 2]
        return torch.stack([dxdt, dydt, dzdt], dim=1)

    def g(self, t: float, y: torch.Tensor) -> torch.Tensor:
        _ = t
        return self.noise_scale * torch.ones_like(y)


class Lorenz96SDE(SDE):
    """Implementation of Lorenz System of SDEs.

    System of SDE consisting of the Lorenz 96 system of ODEs for the drift coefficient and identity diffusion
    coefficient. The Lorenz 96 system of SDEs is given by (for i = 0, ... N-1):

    dx_i = ((x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F) dt + noise_scale * dw_i

    where the indices are taken modulo N.

    Attributes:
        noise_type: type of SDE noise, required for BaseSDE; always "diagonal" for Lorenz
        sde_type: type of SDE, required for BaseSDE; always "ito" for Lorenz
        F: Lorenz 96 parameter F
        noise_scale: diffusion coefficient constant

    Methods:
        f: drift coefficient for Lorenz system
        g: diffusion coefficient for Lorenz system
    """

    noise_type = "diagonal"
    sde_type = "ito"

    @staticmethod
    def graph(state_size: int) -> torch.Tensor:
        """Returns the graph of the Lorenz 96 system of SDEs.

        Args:
            state_size: number of variables in the system
        """
        vec = torch.zeros(state_size, dtype=torch.long)
        vec[1] = 1
        vec[-2] = 1
        vec[-1] = 1
        vec[0] = 1

        vecs = [torch.roll(vec, i, 0) for i in range(state_size)]
        return torch.stack(vecs, dim=1)  # stack along column dimension

    def __init__(self, F: float, noise_scale: float):
        super().__init__()
        self.F = F
        self.noise_scale = noise_scale

    def f(self, t: float, y: torch.Tensor) -> torch.Tensor:
        _ = t
        return (torch.roll(y, -1, 1) - torch.roll(y, 2, 1)) * torch.roll(y, 1, 1) - y + self.F

    def g(self, t: float, y: torch.Tensor) -> torch.Tensor:
        _ = t
        return self.noise_scale * torch.ones_like(y)
