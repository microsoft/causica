from enum import Enum

import torch


class VariableTypeEnum(Enum):
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    BINARY = "binary"


DTYPE_MAP = {
    VariableTypeEnum.CONTINUOUS: torch.float32,
    VariableTypeEnum.CATEGORICAL: torch.int32,
    VariableTypeEnum.BINARY: torch.int32,
}
