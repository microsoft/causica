from typing import Optional

import torch

from causica.functional_relationships.rff_functional_relationships import RFFFunctionalRelationships


class HeteroscedasticRFFFunctionalRelationships(RFFFunctionalRelationships):
    """
    A simple random fourier feature-based functional relationship.

    The formula implemented here is:
    x_i = sqrt(2/M) * output_scales_i sum_{i}^{M} alpha_i sin( <w_i, pa(x_i)> / length_scale_i)
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
        log_scale: bool = False,
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
            log_scale: whether to apply a log scale to the output
        """
        super().__init__(
            shapes=shapes,
            initial_random_features=initial_random_features,
            initial_coefficients=initial_coefficients,
            initial_bias=initial_bias,
            initial_length_scales=initial_length_scales,
            initial_output_scales=initial_output_scales,
            initial_angles=initial_angles,
            trainable=trainable,
        )
        self.log_scale = log_scale

    def non_linear_map(self, samples: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        """
        Applies the non linear function to a concatenated tensor of samples.

        Args:
            samples: tensor of shape batch_shape_x + [n_cols]
            graph: tensor of shape batch_shape_g + [n_nodes, n_nodes]
        Returns:
            tensor of shape batch_shape_x + batch_shape_g + [n_cols]
        """
        res = super().non_linear_map(samples, graph)
        res = torch.log(1.0 + torch.exp(res))
        if self.log_scale:
            res = torch.log(torch.log(1.0 + torch.exp(res)))
        return res
