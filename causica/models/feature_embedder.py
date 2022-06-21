from typing import Optional

import torch


class FeatureEmbedder(torch.nn.Module):
    """
    Combines each feature value with its feature ID. The embedding of feature IDs is a trainable parameter.

    This is analogous to position encoding in a transformer.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        metadata: Optional[torch.Tensor],
        device: torch.device,
        multiply_weights: bool,
    ):
        """
        Args:
            input_dim (int): Number of features.
            embedding_dim (int): Size of embedding for each feature ID.
            metadata (Optional[torch.Tensor]): Each row represents a feature and each column is a metadata dimension for the feature.
                Shape (input_dim, metadata_dim).
            device (torch.device): Pytorch device to use.
            multiply_weights (bool): Whether or not to take the product of x with embedding weights when feeding through.
        """
        super().__init__()
        self._input_dim = input_dim
        self._embedding_dim = embedding_dim
        if metadata is not None:
            assert metadata.shape[0] == input_dim
        self._metadata = metadata
        self._multiply_weights = multiply_weights

        self.embedding_weights = torch.nn.Parameter(
            torch.zeros(input_dim, embedding_dim, device=device), requires_grad=True
        )
        self.embedding_bias = torch.nn.Parameter(torch.zeros(input_dim, 1, device=device), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.embedding_weights)
        torch.nn.init.xavier_uniform_(self.embedding_bias)

    @property
    def output_dim(self) -> int:
        """
        The final output dimension depends on how features and embeddings are combined in the forward method.

        Returns:
            output_dim (int): The final output dimension of the feature embedder.
        """
        metadata_dim = 0 if self._metadata is None else self._metadata.shape[1]
        output_dim = metadata_dim + self._embedding_dim + 2
        return output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        Map each element of each set to a vector, according to which feature it represents.

        Args:
            x (torch.Tensor): Data to embed of shape (batch_size, input_dim).

        Returns:
            feature_embedded_x (torch.Tensor): of shape (batch_size * input_dim, output_dim)
        """

        batch_size, _ = x.size()  # Shape (batch_size, input_dim).
        x_flat = x.reshape(batch_size * self._input_dim, 1)

        # Repeat weights and bias for each instance of each feature.
        if self._metadata is not None:
            embedding_weights_and_metadata = torch.cat((self.embedding_weights, self._metadata), dim=1)
            repeat_embedding_weights = embedding_weights_and_metadata.repeat([batch_size, 1, 1])
        else:
            repeat_embedding_weights = self.embedding_weights.repeat([batch_size, 1, 1])

        # Shape (batch_size * input_dim, embedding_dim)
        repeat_embedding_weights = repeat_embedding_weights.reshape([batch_size * self._input_dim, -1])

        repeat_embedding_bias = self.embedding_bias.repeat((batch_size, 1, 1))
        repeat_embedding_bias = repeat_embedding_bias.reshape((batch_size * self._input_dim, 1))

        if self._multiply_weights:
            features_to_concatenate = [
                x_flat,
                x_flat * repeat_embedding_weights,
                repeat_embedding_bias,
            ]
        else:
            features_to_concatenate = [
                x_flat,
                repeat_embedding_weights,
                repeat_embedding_bias,
            ]

        # Shape (batch_size*input_dim, output_dim)
        feature_embedded_x = torch.cat(features_to_concatenate, dim=1)
        return feature_embedded_x

    def __repr__(self):
        return f"FeatureEmbedder(input_dim={self._input_dim}, embedding_dim={self._embedding_dim}, multiply_weights={self._multiply_weights}, output_dim={self.output_dim})"


class SparseFeatureEmbedder(FeatureEmbedder):
    """
    Feature embedder to use with SparsePointNet.
    """

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # type: ignore
        """

        Maps each observed element of each set to a vector, according to which feature it represents.

        Args:
            x (torch.Tensor): Data to embed of shape (batch_size, input_dim).
            mask (torch.Tensor): Mask of shape (batch_size, input_dim) indicating observed variables.
                1 is observed, 0 is un-observed.

        Returns:
            feature_embedded_x (torch.Tensor):
                Observed features, embedded with their feature IDs of shape (num_observed_features, output_dim).

        """
        if self._metadata is not None:
            raise NotImplementedError("metadata parameter is not currently supported in SparseFeatureEmbedder.")

        batch_size, _ = x.size()

        # Reshape input so that the first dimension contains all input datapoints stacked on top of one another
        x_flat = x.reshape(batch_size * self._input_dim, 1)
        mask = mask.reshape(batch_size * self._input_dim, 1)

        # Select only the observed features
        obs_features = torch.nonzero(mask, as_tuple=False)[:, 0]
        x_flat = x_flat[
            obs_features,
        ]  # shape (num_observed_features, 1)

        # Repeat weights and bias for each observed instance of each feature.
        repeat_embedding_weights = self.embedding_weights.repeat([batch_size, 1, 1])
        repeat_embedding_weights = repeat_embedding_weights.reshape([batch_size * self._input_dim, self._embedding_dim])

        # shape (num_observed_features, embedding_dim)
        repeat_embedding_weights = repeat_embedding_weights[
            obs_features,
        ]

        repeat_embedding_bias = self.embedding_bias.repeat((batch_size, 1, 1))
        repeat_embedding_bias = repeat_embedding_bias.reshape((batch_size * self._input_dim, 1))

        # shape (num_observed_features, 1)
        repeat_embedding_bias = repeat_embedding_bias[
            obs_features,
        ]

        if self._multiply_weights:
            features_to_concatenate = [
                x_flat,
                x_flat * repeat_embedding_weights,
                repeat_embedding_bias,
            ]
        else:
            features_to_concatenate = [
                x_flat,
                repeat_embedding_weights,
                repeat_embedding_bias,
            ]

        # Shape (num_observed_features, output_dim)
        return torch.cat(features_to_concatenate, dim=1)
