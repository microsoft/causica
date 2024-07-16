import torch
from torch import nn

from causica.nn.deci_embed_nn import _generate_fully_connected


class TemporalEmbedNN(nn.Module):
    """Defines functional relationships for variables in a timeseries.

    Follows the model described in:
        Gong, W., Jennings, J., Zhang, C. and Pawlowski, N., 2022.
        Rhino: Deep causal temporal relationship learning with history-dependent noise.
        ICLR 2023
        arXiv:2210.14706

    The model is a generalization of the DECI model to include temporal dependencies but represents relationships
    between the current timestep and a fixed number of steps backwards.

    For each variable xᵢ we use
        ζᵢ(x) = ζ(eᵢ,  Σ_{k in pa(i)} l(eₖ, xₖ)),

    where eᵢ is a learned embedding for node i and pa(i) includes the variables from any timestep in the context.
    """

    def __init__(
        self,
        group_mask: torch.Tensor,
        embedding_size: int,
        out_dim_l: int,
        num_layers_l: int,
        num_layers_zeta: int,
        context_length: int,
    ):
        """
        Args:
            group_mask: A mask of shape (num_nodes, num_processed_cols) such that group_mask[i, j] = 1. when col j is in
                group i.
            embedding_size: Size of the embeddings used by each node. Also affects the hidden dimensionality of l and ζ,
                which is set to the larger of 4 * concatenated_shape, embedding_size and 64.
            out_dim_l: Output dimension of the "inner" NN, l. If none, default is embedding size.
            num_layers_l: Number of layers in the "inner" NN, l.
            num_layers_zeta: Number of layers in the "outer" NN, ζ.
            context_length: The total number of timesteps in the context of the model, including the current timestep.
        """
        super().__init__()
        self.group_mask = group_mask
        num_nodes, concatenated_shape = group_mask.shape
        self.context_length = context_length

        # Initialize embeddings uₜⱼ and graph weights Wₜᵢⱼ
        # Notice that we store the embeddings and the graph with the time axis in the reversed order from the paper,
        # e.g. embedding[lag - t] are the embeddings for the nodes t steps behind, where t=0 corresponds to the
        # instantaneous effect. lag, in the paper, is equivalent to context_length - 1, and is the number of steps back
        # from the current that's visible to the model.
        self.embeddings = nn.parameter.Parameter(0.01 * torch.randn(context_length, num_nodes, embedding_size))
        self.w = torch.nn.Parameter(torch.zeros((context_length, num_nodes, num_nodes)), requires_grad=True)

        # Set NNs sizes
        a = max(4 * concatenated_shape, embedding_size, 64)
        in_dim_g = embedding_size + concatenated_shape
        in_dim_f = embedding_size + out_dim_l
        self.l = _generate_fully_connected(
            input_dim=in_dim_g,
            output_dim=out_dim_l,
            hidden_dims=[a] * num_layers_l,
        )
        self.zeta = _generate_fully_connected(
            input_dim=in_dim_f,
            output_dim=concatenated_shape,
            hidden_dims=[a] * num_layers_zeta,
        )

    def forward(self, samples: torch.Tensor, graphs: torch.Tensor) -> torch.Tensor:
        """Forward of the module.

        Computes non-linear function hᵢ(X, G) using the given adjacency matrix.

            hᵢ(x, G) =  ζᵢ(Σₜⱼ Wₜⱼᵢ Gₜⱼᵢ lⱼ(xₜⱼ)

        We also use an embedding u so:

            hᵢ(x, G) =  ζ(uₗᵢ, Σₜⱼ Wₜⱼᵢ Gₜⱼᵢ l(xₜⱼ, uₜⱼ))

        l takes inputs of size batch_shape + (context_length, embedding_size + concatenated_shape) and outputs
        batch_shape + (out_dim_g). The input will be appropriately masked to correspond to one variable group.

        ζ takes inputs of size batch_shape + (embedding_size + out_dim_g, ) and outputs batch_shape +
        (concatenated_shape, ) the ouptut is then masked to correspond to one variable.

        Args:
            samples: tensor of shape batch_shape_x + batch_shape_f + batch_shape_g + [context_length, n_cols]
            graph: tensor of shape batch_shape_g + [context_length, n_nodes, n_nodes]

        Returns:
            tensor of shape batch_shape_x + batch_shape_f + batch_shape_g + [n_cols]
        """
        batch_shape_samples = samples.shape[:-2]
        batch_shape_g = graphs.shape[:-3]
        if len(batch_shape_g) > 0 and batch_shape_samples[-len(batch_shape_g) :] != batch_shape_g:
            raise ValueError(
                f"Batch shape of samples and graph must match but got {batch_shape_samples} and {batch_shape_g}"
            )

        # Shape batch_shape_x + batch_shape_f + batch_shape_g + (context_length, num_nodes, concatenated_shape)
        masked_samples = torch.einsum("...i,ji->...ji", samples, self.group_mask)
        # Shape batch_shape_x + batch_shape_f + batch_shape_g + (context_length, num_nodes, embedding_size)
        expanded_embed = self.embeddings.expand(*batch_shape_samples, -1, -1, -1)

        # Shape batch_shape_x + batch_shape_f + batch_shape_g +
        #     (context_length, num_nodes, concatenated_shape + embedding_size)
        x_and_embeddings = torch.cat([masked_samples, expanded_embed], dim=-1)  # (concatenate xⱼ and embeddings uⱼ)
        # l(uⱼ, xⱼ): Shape batch_shape_x + batch_shape_f + batch_shape_g + (context_length, num_nodes, out_dim_g)
        encoded_samples = self.l(x_and_embeddings)

        # Shape batch_shape_samples + batch_shape_g + (num_nodes, out_dim_g)
        aggregated_effects = torch.einsum("...lij,...lio->...jo", self.w * graphs, encoded_samples)

        # ζ(uᵢ, Σⱼ Wᵢⱼ Gⱼᵢ l(uⱼ, xⱼ))
        # Shape batch_shape_x + batch_shape_f + batch_shape_g + (num_nodes, concatenated_shape)
        decoded_samples = self.zeta(torch.cat([aggregated_effects, expanded_embed[..., -1, :, :]], dim=-1))

        # Mask and aggregate Shape batch_shape_x + batch_shape_f + batch_shape_g + (concatenated_shape)
        return torch.einsum("...ij,ij->...j", decoded_samples, self.group_mask)
