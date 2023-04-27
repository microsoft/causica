import torch
from tensordict import TensorDict

from causica.functional_relationships.functional_relationships import (
    FunctionalRelationships,
    sample_dict_to_tensor,
    tensor_to_sample_dict,
)


class LinearFunctionalRelationships(FunctionalRelationships):
    """
    A simple linear functional relationship.
    """

    def __init__(
        self,
        variables: dict[str, torch.Size],
        initial_linear_coefficient_matrix: torch.Tensor,
        trainable: bool = False,
    ) -> None:
        """
        Args:
            variables: Dict of node shapes (how many dimensions a variable has)
                Order corresponds to the order in graph(s).
            initial_linear_coefficient_matrix: the linear coefficients [output_shape, output_shape]
            trainable: whether the coefficient matrix should be learnable
        """
        super().__init__(variables)

        self.stacked_variable_masks = torch.nn.Parameter(
            torch.stack(list(self.variable_masks.values())).float(), requires_grad=False
        )

        assert initial_linear_coefficient_matrix.shape == (self.output_shape, self.output_shape)
        self.linear_coefficients = torch.nn.Parameter(initial_linear_coefficient_matrix, requires_grad=trainable)

    def forward(self, samples: TensorDict, graphs: torch.Tensor) -> TensorDict:
        """
        Args:
            samples: Batched inputs, size batch_size_x + (processed_dim_all).
            graphs: Weighted adjacency matrix, size batch_size_g + (n, n)
        Returns:
            A Dict of tensors of shape batch_shape_x + batch_shape_g + (processed_dim_all)
        """
        return tensor_to_sample_dict(
            self.linear_map(sample_dict_to_tensor(samples, self.variable_masks), graphs), self.variable_masks
        )

    def linear_map(self, samples: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        """
        Applies the linear function to a concatenated tensor of samples.

        Args:
            samples: tensor of shape batch_shape_x + [n_cols]
            graph: tensor of shape batch_shape_g + [n_nodes, n_nodes]
        Returns:
            tensor of shape batch_shape_x + batch_shape_g + [n_cols]
        """
        batch_shape_x = samples.shape[:-1]
        batch_shape_g = graph.shape[:-2]

        masked_graph = torch.einsum(
            "ji,...jk,kl->...il", self.stacked_variable_masks, graph, self.stacked_variable_masks
        )

        graph_broad = masked_graph.expand(*(batch_shape_x + tuple([-1] * len(graph.shape))))
        target_shape = batch_shape_x + batch_shape_g + samples.shape[-1:]
        view_shape = batch_shape_x + (1,) * len(batch_shape_g) + samples.shape[-1:]
        # Shape batch_shape_x + batch_shape_g + (num_nodes, out_dim_g)
        samples_broad = samples.view(view_shape).expand(target_shape)

        return torch.einsum("...i,...ij->...j", samples_broad, graph_broad * self.linear_coefficients)
