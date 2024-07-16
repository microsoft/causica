import math
from typing import Optional

import torch
from tensordict import TensorDict

from causica.functional_relationships.functional_relationships import FunctionalRelationships


class RFFFunctionalRelationships(FunctionalRelationships):
    """
    A simple random fourier feature-based functional relationship.

    The formula implemented here is:
    x_i = sqrt(2/M) * output_scales_i sum_{i}^{M} alpha_i sin( <w_i, pa(x_i)> / length_scale_i )
    """

    def __init__(
        self,
        shapes: dict[str, torch.Size],
        initial_random_features: torch.Tensor,
        initial_coefficients: torch.Tensor,
        initial_bias: Optional[torch.Tensor] = None,
        initial_length_scales: Optional[torch.Tensor] = None,
        initial_output_scales: Optional[torch.Tensor] = None,
        initial_angles: Optional[torch.Tensor] = None,
        trainable: bool = False,
    ) -> None:
        """
        Args:
            shapes: Dict of node shapes (how many dimensions a variable has)
                Order corresponds to the order in graph(s).
            initial_random_features: a tensor containing the random features [num_rf, output_shape]
            initial_coefficients: a tensor containing the linear outer coefficients [num_rf,]
            initial_bias: Optional, None or a tensor containing the bias [output_shape,]
            initial_length_scales: Optional, None or a tensor containing the length scales [output_shape,]
            initial_output_scales: Optional, None or a tensor containing the output scales [output_shape,]
            initial_angles: Optional, None or a tensor containing the angles [num_rf,]
            trainable: whether the coefficient matrix should be learnable
        """
        super().__init__(shapes=shapes)

        assert initial_random_features.shape[0] == initial_coefficients.shape[0]
        self.num_rf = initial_random_features.shape[0]

        self.shape = self.tensor_to_td.output_shape
        assert initial_random_features.shape[1] == self.shape

        if initial_length_scales is not None:
            assert initial_length_scales.shape[0] == self.shape
        else:
            initial_length_scales = torch.ones(self.shape)

        if initial_output_scales is not None:
            assert initial_output_scales.shape[0] == self.shape
        else:
            initial_output_scales = torch.ones(self.shape)

        if initial_bias is not None:
            assert initial_bias.shape[0] == self.shape
        else:
            initial_bias = torch.zeros(self.shape)

        if initial_angles is not None:
            assert initial_angles.shape[0] == self.num_rf
        else:
            initial_angles = (-math.pi / 2) * torch.ones(self.num_rf)

        self.bias = torch.nn.Parameter(initial_bias, requires_grad=trainable)
        self.length_scales = torch.nn.Parameter(initial_length_scales, requires_grad=trainable)
        self.output_scales = torch.nn.Parameter(initial_output_scales, requires_grad=trainable)
        self.angles = torch.nn.Parameter(initial_angles, requires_grad=trainable)
        self.linear_coefficients_inner = torch.nn.Parameter(initial_random_features, requires_grad=trainable)
        self.linear_coefficients_outer = torch.nn.Parameter(initial_coefficients, requires_grad=trainable)

    def forward(self, samples: TensorDict, graphs: torch.Tensor) -> TensorDict:
        """
        Args:
            samples: Batched inputs, size batch_size_x + batch_shape_g + (concatenated_shape).
            graphs: Weighted adjacency matrix, size batch_size_g + (n, n)
        Returns:
            A Dict of tensors of shape batch_shape_x + batch_shape_g + (concatenated_shape)
        """
        return self.tensor_to_td(self.non_linear_map(self.tensor_to_td.inv(samples), graphs))

    def non_linear_map(self, samples: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        """
        Applies the non linear function to a concatenated tensor of samples.

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

        inner_prods = torch.einsum("ij,...j,...jk->...ik", self.linear_coefficients_inner, samples, graph_broad)
        rescaled_inner_prods = inner_prods / self.length_scales
        rescaled_and_translated_inner_prods = rescaled_inner_prods + self.angles[..., None]
        return self.bias + (
            math.sqrt(2 / self.num_rf)
            * self.output_scales
            * torch.einsum(
                "i,...ij->...j", self.linear_coefficients_outer, torch.cos(rescaled_and_translated_inner_prods)
            )
        )
