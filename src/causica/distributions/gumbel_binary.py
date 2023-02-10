import torch
import torch.nn.functional as F


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
