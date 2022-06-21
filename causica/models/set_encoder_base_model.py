from abc import ABC, abstractmethod

import torch

from .torch_model import ONNXNotImplemented


class SetEncoderBaseModel(ABC, torch.nn.Module):
    """
    Abstract model class.

    This ABC should be inherited by classes that transform a set of observed features to a single set embedding vector.

    To instantiate this class, the forward function has to be implemented.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        set_embedding_dim: int,
        device: torch.device,
    ):
        """
        Args:
            input_dim: Dimension of input data to embedding model.
            embedding_dim: Dimension of embedding for each input.
            set_embedding_dim: Dimension of output set embedding.
            device: torch device to use.
        """
        ABC.__init__(self)
        torch.nn.Module.__init__(self)

        self._input_dim = input_dim
        self._embedding_dim = embedding_dim
        self._set_embedding_dim = set_embedding_dim
        self.device = device

    @abstractmethod
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor with shape (batch_size, input_dim).
            mask: Mask indicting observed variables with shape (batch_size, input_dim). 1 is observed, 0 is un-observed.
        Returns:
            set_embedding: Embedded output tensor with shape (batch_size, set_embedding_dim)
        """
        raise NotImplementedError()

    def save_onnx(self, save_dir: str):
        raise ONNXNotImplemented
