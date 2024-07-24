import torch
from torch import nn


class LearnableGaussianLLH(nn.Module):
    """
    A learnable version of the Gaussian log-likelihood function.

    """

    def __init__(self, max_seq_length: int):
        super().__init__()

        self.log_var = nn.Parameter(torch.ones(max_seq_length))

    def forward(self, x_in: torch.Tensor, y_in: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_in: expected shape (*, max_seq_length, 1)
            y_in: expected shape (*, max_seq_length, 1)

        Returns:
            output: expected shape (1)

        """
        var = torch.exp(self.log_var).repeat(x_in.shape[0], 1).unsqueeze(-1)
        return nn.GaussianNLLLoss(full=True)(x_in, y_in, var)
