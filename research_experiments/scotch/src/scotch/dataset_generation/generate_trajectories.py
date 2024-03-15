from typing import Tuple

import torch
import torchsde
from scotch.sdes.sdes_core import SDE
from tensordict import TensorDict
from torch import Tensor


def generate_and_return_trajectories(
    sde_class: SDE,
    z0: Tensor,
    num_time_points: int,
    t_max: float,
    device: str = "cuda",
    normalize: bool = False,
    dt: float = 1e-3,
    return_raw: bool = False,
    **kwargs,
) -> Tuple[Tensor, TensorDict, torchsde.BrownianInterval]:
    """Generate synthetic trajectories.
        z0 = torch.full(size=(n, state_size), fill_value=0.0, device=device)
    Args:
        sde_class: SDE class to generate trajectories for.
        z0: Tensor of shape (n, state_size) of initial points.
        num_time_points: Number of time points to generate for each trajectory.
        t_max: Maximum time point to generate for each trajectory.
        return_bm: Whether to return the Brownian motion used to generate the trajectories.
        device: Device to generate trajectories on.
        normalize: Whether to normalize the trajectories per variable.
        dt: Time step to use for SDE integration.
        return_raw: Whether to return the raw trajectories or TensorDict version.
        **kwargs: Any additional arguments to pass to the SDE class.

    Returns:
        ts: Time points of generated trajectories; shape (num_time_points,).
        zs_td: TensorDict of generated trajectories.
        bm: Brownian motion used to generate trajectories; returned if return_bm.
    """
    n, state_size = z0.shape

    ts = torch.linspace(0, t_max, num_time_points)
    bm = torchsde.BrownianInterval(
        t0=0.0, t1=t_max, size=(n, state_size), levy_area_approximation="space-time", device=device
    )
    zs = torchsde.sdeint(sde_class(**kwargs), z0, ts, bm=bm, dt=dt, method="euler")  # (t_size, batch_size, state_size)
    zs = zs.permute(1, 0, 2)  # reshape into format (batch_size, t_size, state_size)
    print(zs.shape)
    zs = (zs - zs.mean(dim=(0, 1))) / zs.std(dim=(0, 1)) if normalize else zs

    zs_td = TensorDict(
        {f"x{i}": zs[:, :, i].unsqueeze(dim=2) for i in range(state_size)},
        batch_size=[n],
    )

    if return_raw:
        return ts, zs, bm
    return ts, zs_td, bm
