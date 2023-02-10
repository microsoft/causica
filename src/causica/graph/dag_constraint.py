import torch


def calculate_dagness(A: torch.Tensor) -> torch.Tensor:
    """
    Computes the dag penalty for matrix A as trace(exp(A)) - dim.

    Args:
        A (torch.Tensor): Binary adjacency matrix, size (...., input_dim, input_dim).

    Returns:
        (torch.Tensor): Dagness term.
    """
    return torch.einsum("...ii->...", torch.matrix_exp(A)) - A.shape[-1]
