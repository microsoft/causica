import math
from typing import Tuple

import torch
from torch import nn


class DotProductCritic(nn.Module):
    """
    A critic π that takes two quantities, π₯ and π¦, as inputs and returns a real number.

    In the context of BED, we want to take as input some observations π¦ and some quantity of interest π₯
    (which we wish to gain information about). These usually live in different spaces, so we first encode
    them (e.g. with two MLPs) so they are in the same space and of the same dimension.

    A dot product critic is defined as the dot product between the encodings:
        π(π₯, π¦) = πΈβ(π₯)α΅πΈβ(π¦)
    Optionally, the dot product can be scaled by sqrt(encoding_dimension), i.e.
         π(π₯, π¦) = πΈβ(π₯)α΅πΈβ(π¦) / sqrt(encoding_dimension)
    """

    def __init__(self, x_encoder: nn.Module, y_encoder: nn.Module, scale: bool = True) -> None:
        super().__init__()
        self.x_encoder = x_encoder
        self.y_encoder = y_encoder
        self.scale = scale

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes π(x_i, y_j) for i, j = 1,..., batch_size
        Args:
            x: first input, expected shape [batch_size, dim(x)])
            y: second input, expected shape [batch_size, dim(y)])
        Returns:
            A tuple of two torch matrices:
                joint scores, containing π(x_i, y_i) on the diagonal.
                product scores, containing π(x_i, y_j) and 0 on the diagonal.
        """
        x_encodings = self.x_encoder(x)  # [batch_size, encoding_dim]
        y_encodings = self.y_encoder(y)  # [batch_size, encoding_dim]
        assert x_encodings.shape == y_encodings.shape, "Encoding and batch dimensions of x and y must be the same."

        factor = 1 / math.sqrt(y_encodings.shape[-1]) if self.scale else 1.0
        score_matrix = torch.matmul(x_encodings, y_encodings.transpose(-1, -2)) * factor
        mask = torch.eye(y_encodings.shape[0], device=y_encodings.device)
        joint_scores = score_matrix * mask
        marginal_product_scores = score_matrix * (1 - mask)

        return joint_scores, marginal_product_scores


class L2Critic(nn.Module):
    """
    A critic π that takes two quantities, π₯ and π¦, as inputs and returns a real number.

    In the context of BED, we want to take as input some observations π¦ and some quantity of interest π₯
    (which we wish to gain information about). These usually live in different spaces, so we first encode
    them (e.g. with two MLPs) so they are in the same space and of the same dimension.

    The critic is defined as L2 norm of the difference between the encodings:
        π(π₯, π¦) = ||πΈβ(π₯) - πΈβ(π¦)||β
    """

    def __init__(self, x_encoder: nn.Module, y_encoder: nn.Module) -> None:
        super().__init__()
        self.x_encoder = x_encoder
        self.y_encoder = y_encoder

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes π(x_i, y_j) for i, j = 1,..., batch_size
        Args:
            x: first input, expected shape [batch_size, dim(x)])
            y: second input, expected shape [batch_size, dim(y)])
        Returns:
            A tuple of two torch matrices:
                joint scores, containing π(x_i, y_i) on the diagonal.
                product scores, containing π(x_i, y_j) and 0 on the diagonal.
        """
        x_encodings = self.x_encoder(x)  # [batch_size, encoding_dim]
        y_encodings = self.x_encoder(y)  # [batch_size, encoding_dim]
        assert x_encodings.shape == y_encodings.shape, "Encoding and batch dimensions of x and y must be the same."

        score_matrix = (x_encodings - y_encodings).norm(p=2, dim=-1)
        mask = torch.eye(y_encodings.shape[0], device=y_encodings.device)
        joint_scores = score_matrix * mask
        marginal_product_scores = score_matrix * (1 - mask)

        return joint_scores, marginal_product_scores
