import math

import torch
from tensordict import TensorDict

from causica.functional_relationships.functional_relationships import FunctionalRelationships


class RFFFunctionalRelationships(FunctionalRelationships):
    """
    A simple random fourier feature-based functional relationship.
    The formula implemented here is:
    x_i = sqrt(2/M) * sum_{i}^{M} alpha_i sin(<w_i, pa(x_i)>)
    """

    def __init__(
        self,
        shapes: dict[str, torch.Size],
        initial_random_features: torch.Tensor,
        initial_coefficients: torch.Tensor,
        trainable: bool = False,
    ) -> None:
        """
        Args:
            shapes: Dict of node shapes (how many dimensions a variable has)
                Order corresponds to the order in graph(s).
            initial_random_features: a tensor containing the random features [num_rf, output_shape]
            initial_coefficients: a tensor containing the linear outer coefficients [num_rf,]
            trainable: whether the coefficient matrix should be learnable
        """
        super().__init__(shapes=shapes)

        assert initial_random_features.shape[0] == initial_coefficients.shape[0]
        self.num_rf = initial_random_features.shape[0]

        self.shape = self.tensor_to_td.output_shape
        assert initial_random_features.shape[1] == self.shape

        self.linear_coefficients_inner = torch.nn.Parameter(initial_random_features, requires_grad=trainable)
        self.linear_coefficients_outer = torch.nn.Parameter(initial_coefficients, requires_grad=trainable)

    def forward(self, samples: TensorDict, graphs: torch.Tensor) -> TensorDict:
        """
        Args:
            samples: Batched inputs, size batch_size_x + (concatenated_shape).
            graphs: Weighted adjacency matrix, size batch_size_g + (n, n)
        Returns:
            A Dict of tensors of shape batch_shape_x + batch_shape_g + (concatenated_shape)
        """
        return self.tensor_to_td(self.non_linear_map(self.tensor_to_td.inv(samples), graphs))

    def non_linear_map(self, samples: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        """
        Applies the non linear function to a concatenated tensor of samples.

        Args:
            samples: tensor of shape batch_shape_x + [n_cols]
            graph: tensor of shape batch_shape_g + [n_nodes, n_nodes]
        Returns:
            tensor of shape batch_shape_x + batch_shape_g + [n_cols]
        """
        batch_shape_x = samples.shape[:-1]
        batch_shape_g = graph.shape[:-2]

        masked_graph = torch.einsum("ji,...jk,kl->...il", self.stacked_key_masks, graph, self.stacked_key_masks)

        graph_broad = masked_graph.expand(*(batch_shape_x + tuple([-1] * len(graph.shape))))
        target_shape = batch_shape_x + batch_shape_g + samples.shape[-1:]
        view_shape = batch_shape_x + (1,) * len(batch_shape_g) + samples.shape[-1:]
        # Shape batch_shape_x + batch_shape_g + (num_nodes, out_dim_g)
        samples_broad = samples.view(view_shape).expand(target_shape)

        inner_prods = torch.einsum("ij,...j,...jk->...ik", self.linear_coefficients_inner, samples_broad, graph_broad)
        return math.sqrt(2 / self.num_rf) * torch.einsum(
            "i,...ij->...j", self.linear_coefficients_outer, torch.sin(inner_prods)
        )
