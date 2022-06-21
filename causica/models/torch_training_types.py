from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch


class LossConfig(NamedTuple):
    # TODO add KL coeff, categorical likelihood coeff, and IWAE vs. ELBO. Remove them from model config.
    max_p_train_dropout: Optional[float] = None
    score_reconstruction: Optional[bool] = None
    score_imputation: Optional[bool] = None


@dataclass(frozen=True)
class LossResults:
    loss: torch.Tensor
    mask_sum: torch.Tensor  # TODO: consider using int as type here
