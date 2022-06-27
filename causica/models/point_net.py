import os
from typing import Optional

import torch
from torch.nn import Linear, ReLU, Sequential

from .feature_embedder import FeatureEmbedder, SparseFeatureEmbedder
from .set_encoder_base_model import SetEncoderBaseModel
from .torch_model import ONNXNotImplemented


class PointNet(SetEncoderBaseModel):
    """
    Embeds features using a FeatureEmbedder, transforms each feature independently,
    then pools all the features in each set using sum or max.
    """

    feature_embedder_class = FeatureEmbedder

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        set_embedding_dim: int,
        metadata: Optional[torch.Tensor],
        device: torch.device,
        multiply_weights: bool = True,
        encoding_function: str = "sum",
    ):
        """
        Args:
            input_dim: Dimension of input data to embedding model.
            embedding_dim: Dimension of embedding for each input.
            set_embedding_dim: Dimension of output set embedding.
            metadata: Optional torch tensor. Each row represents a feature and each column is a metadata dimension for the feature.
                Shape (input_dim, metadata_dim).
            device: torch device to use.
            multiply_weights: Boolean. Whether or not to take the product of x with embedding weights when feeding
                 through. Defaults to True.
            encoding_function: Function to use to summarise set input. Defaults to "sum".
        """
        super().__init__(input_dim, embedding_dim, set_embedding_dim, device)

        self.feature_embedder = self.feature_embedder_class(
            input_dim, embedding_dim, metadata, device, multiply_weights
        )
        self._set_encoding_func = self._get_function_from_function_name(encoding_function)

        self.forward_sequence = Sequential(
            Linear(self.feature_embedder.output_dim, set_embedding_dim).to(device),
            ReLU(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor with shape (batch_size, input_dim).
            mask: Mask indicating observed variables with shape (batch_size, input_dim). 1 is observed, 0 is unobserved.
        Returns:
            set_embedding: Embedded output tensor with shape (batch_size, set_embedding_dim).
        """
        feature_embedded_x = self.feature_embedder(
            x
        )  # Shape (batch_size * input_dim, self.feature_embedder.output_dim)
        batch_size, _ = x.size()
        embedded = self.forward_sequence(feature_embedded_x)
        embedded = embedded.reshape(
            [batch_size, self._input_dim, self._set_embedding_dim]
        )  # Shape (batch_size, input_dim, set_embedding_dim)

        mask = mask.reshape((batch_size, self._input_dim, 1))
        mask = mask.repeat([1, 1, self._set_embedding_dim])  # Shape (batch_size, input_dim, set_embedding_dim)

        masked_embedding = embedded * mask  # Shape (batch_size, input_dim, set_embedding_dim)
        set_embedding = self._set_encoding_func(masked_embedding, dim=1)  # Shape (batch_size, set_embedding_dim)

        return set_embedding

    @staticmethod
    def max_vals(input_tensor: torch.Tensor, dim: int):
        vals, _ = torch.max(input_tensor, dim=dim)
        return vals

    @staticmethod
    def _get_function_from_function_name(name: str):
        if name == "sum":
            return torch.sum
        elif name == "max":
            return PointNet.max_vals

        raise ValueError(f"Function name should be one of 'sum', 'max'. Was {name}.")

    def save_onnx(self, save_dir: str):
        dummy_input = (
            torch.rand(1, self._input_dim, dtype=torch.float, device=self.device),  # Data
            torch.randint(2, (1, self._input_dim), dtype=torch.float, device=self.device),  # Mask
        )
        # TODO this filename should be defined in base class, not here
        path = os.path.join(save_dir, "set_encoder.onnx")
        torch.onnx.export(self, dummy_input, path)


class SparsePointNet(PointNet):
    """
    Behaves identically to PointNet, but the forward pass filters on observed values in each
    data point before concatenating feature embeddings. This requires a small overhead in order to locate the unmasked
    elements in each datapoint on each forward pass, but can substantially reduce the memory usage when the data is
    large and sparsely-observed.

    This encoder cannot currently be serialised into ONNX format, hopefully this will be updated in future.
    """

    feature_embedder_class = SparseFeatureEmbedder

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor with shape (batch_size, input_dim).
            mask: Mask indicting observed variables with shape (batch_size, input_dim). 1 is observed, 0 is unobserved.
        Returns:
            set_embedding: Embedded output tensor with shape (batch_size, set_embedding_dim).
        """

        # Shape (total_observed_features, 1 + embedding_dim + 1)
        feature_embedded_x = self.feature_embedder(x, mask)

        mask = mask.reshape(-1, 1)
        obs_features = torch.nonzero(mask, as_tuple=False)[:, 0]
        batch_size, _ = x.size()  # Shape (batch_size, input_dim).

        # Shape (total_observed_features, set_embedding_dim)
        feature_embedded_x = self.forward_sequence(feature_embedded_x)

        # Create empty output tensor to copy sparse set embedding outputs into
        embedded = torch.zeros([batch_size * self._input_dim, self._set_embedding_dim], device=self.device)
        embedded[obs_features, :] = feature_embedded_x
        embedded = embedded.reshape([batch_size, self._input_dim, self._set_embedding_dim])
        set_embedding = self._set_encoding_func(embedded, dim=1)  # Shape (batch_size, set_embedding_dim)

        return set_embedding

    def save_onnx(self, save_dir: str):
        # TODO: ONNX serialisation currently not supported for using torch.nonzero or torch.where, which is required in
        # call() function here
        raise ONNXNotImplemented
