from enum import Enum

import torch


class VariableTypeEnum(Enum):
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    BINARY = "binary"


# Checkpoints store this enum in their hyperparameters; allowlist it so they load with `torch.load(weights_only=True)`,
# the default since PyTorch 2.6.
torch.serialization.add_safe_globals([VariableTypeEnum])


DTYPE_MAP = {
    VariableTypeEnum.CONTINUOUS: torch.float32,
    VariableTypeEnum.CATEGORICAL: torch.int32,
    VariableTypeEnum.BINARY: torch.int32,
}
