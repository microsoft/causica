# This is required in python 3 to allow return types of the same class.
from __future__ import annotations

import os
from typing import Tuple

import numpy as np

from ..datasets.variables import Variables
from ..utils.helper_functions import write_git_info
from .imodel import IModel


class Model(IModel):
    """
    Abstract base model class.
    """

    def __init__(self, model_id: str, variables: Variables, save_dir: str) -> None:
        """
        Args:
            model_id: Unique model ID for referencing this model instance.
            variables: Information about variables/features used by this model.
            save_dir: Location to save any information about this model, including training data.
                It will be created if it doesn't exist.
        """
        super().__init__(model_id, variables, save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        try:
            write_git_info(self.save_dir)
        except (ValueError, FileExistsError):
            # ValueError is likely if not running in a git repo, e.g. in AzureML
            # FileExistsError is likely if we are running inference and git_info.txt was already
            # written during training.
            pass

    @staticmethod
    def _split_inputs_and_targets(data: np.ndarray, target_col: int) -> Tuple[np.ndarray, np.ndarray]:
        inputs = np.concatenate([data[:, :target_col], data[:, target_col + 1 :]], -1)
        targets = data[:, target_col]
        return inputs, targets
