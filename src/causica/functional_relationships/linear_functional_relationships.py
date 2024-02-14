from typing import Optional

import torch
from tensordict import TensorDict

from causica.functional_relationships.functional_relationships import FunctionalRelationships


class LinearFunctionalRelationships(FunctionalRelationships):
    """
    A simple linear functional relationship.
    """

    def __init__(
        self,
        shapes: dict[str, torch.Size],
        initial_linear_coefficient_matrix: torch.Tensor,
        initial_bias: Optional[torch.Tensor] = None,
        trainable: bool = False,
    ) -> None:
        """
        Args:
            shapes: Dict of node shapes (how many dimensions a variable has)
                Order corresponds to the order in graph(s).
            initial_linear_coefficient_matrix: the linear coefficients [output_shape, output_shape]
            initial_bias: Optional, None or a tensor containing the bias [output_shape,]
            trainable: whether the coefficient matrix should be learnable
        """
        super().__init__(shapes=shapes)

        shape = self.tensor_to_td.output_shape
        assert initial_linear_coefficient_matrix.shape == (shape, shape)
        if initial_bias is not None:
            assert initial_bias.shape[0] == shape
        else:
            initial_bias = torch.zeros(shape)

        self.bias = torch.nn.Parameter(initial_bias, requires_grad=trainable)
        self.linear_coefficients = torch.nn.Parameter(initial_linear_coefficient_matrix, requires_grad=trainable)

    def forward(self, samples: TensorDict, graphs: torch.Tensor) -> TensorDict:
        """
        Args:
            samples: Batched inputs, size batch_size_x + (concatenated_shape).
            graphs: Weighted adjacency matrix, size batch_size_g + (n, n)
        Returns:
            A Dict of tensors of shape batch_shape_x + batch_shape_g + (concatenated_shape)
        """
        return self.tensor_to_td(self.linear_map(self.tensor_to_td.inv(samples), graphs))

    def linear_map(self, samples: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        """
        Applies the linear function to a concatenated tensor of samples.

        Args:
            samples: tensor of shape batch_shape_x + batch_shape_f + batch_shape_g + [n_cols]
            graph: tensor of shape batch_shape_g + [n_nodes, n_nodes]
        Returns:
            tensor of shape batch_shape_x + batch_shape_f + batch_shape_g + [n_cols]
        """
        batch_shape_samples = samples.shape[:-1]
        batch_shape_g = graph.shape[:-2]
        batch_shape_x_f = batch_shape_samples[: -len(batch_shape_g)]
        if len(batch_shape_g) > 0 and batch_shape_samples[-len(batch_shape_g) :] != batch_shape_g:
            raise ValueError(
                f"Batch shape of samples and graph must match but got {batch_shape_samples} and {batch_shape_g}"
            )

        masked_graph = torch.einsum("ji,...jk,kl->...il", self.stacked_key_masks, graph, self.stacked_key_masks)

        graph_broad = masked_graph.expand(*(batch_shape_x_f + (-1,) * len(masked_graph.shape)))

        return self.bias + torch.einsum("...i,...ij->...j", samples, graph_broad * self.linear_coefficients)
