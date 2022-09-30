import numpy as np
import torch
import torch.nn.functional as F


def fill_triangular(vec: torch.Tensor, upper: bool = False) -> torch.Tensor:
    """
    Args:
        vec: A tensor of shape (..., n(n-1)/2)
        upper: whether to fill the upper or lower triangle
    Returns:
        An array of shape (..., n, n), where the strictly upper (lower) triangle is filled from vec
        with zeros elsewhere
    """
    num_nodes = num_lower_tri_elements_to_n(vec.shape[-1])
    if upper:
        idxs = torch.triu_indices(num_nodes, num_nodes, offset=1, device=vec.device)
    else:
        idxs = torch.tril_indices(num_nodes, num_nodes, offset=-1, device=vec.device)
    output = torch.zeros(vec.shape[:-1] + (num_nodes, num_nodes), device=vec.device)
    output[..., idxs[0, :], idxs[1, :]] = vec
    return output


def unfill_triangular(mat: torch.Tensor, upper: bool = False) -> torch.Tensor:
    """
    Fill a vector of length n(n-1)/2 with elements from the strictly upper(lower) triangle.

    Args:
        mat: A tensor of shape (..., n, n)
        upper: whether to fill from the upper triangle
    Returns:
        A vector of shape (..., n(n-1)/2), filled from the upper triangle
    """
    num_nodes = mat.shape[-1]
    if upper:
        idxs = torch.triu_indices(num_nodes, num_nodes, offset=1, device=mat.device)
    else:
        idxs = torch.tril_indices(num_nodes, num_nodes, offset=-1, device=mat.device)
    return mat[..., idxs[0, :], idxs[1, :]]


def num_lower_tri_elements_to_n(x: int) -> int:
    """
    Calculate the size of the matrix from the number of strictly lower triangular elements.

    We have x = n(n - 1) / 2 for some n
    n² - n - 2x = 0
    so n = (1 + √(1 + 8x)) / 2
    """
    val = int(np.sqrt(1 + 8 * x) + 1) // 2
    if val * (val - 1) != 2 * x:
        raise ValueError("Invalid number of lower triangular elements")
    return val


def gumbel_softmax_binary(logits: torch.Tensor, tau: float, hard: bool = False) -> torch.Tensor:
    """
    Calculate the gumbel softmax for binary logits.

    Args:
        logits: A tensor of binary logits of shape (...)
        tau: Temperature of the gumbel softmax
        hard: whether samples should be hard {0, 1} or soft (in [0, 1])
    Returns:
        Tensor of shape (...)
    """
    stacked_logits = torch.stack([torch.zeros_like(logits, device=logits.device), logits], dim=-1)  # (..., 2)
    return F.gumbel_softmax(stacked_logits, tau=tau, hard=hard, dim=-1)[..., 1]  # (...) binary
