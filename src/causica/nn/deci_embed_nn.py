import torch
from torch import nn


class DECIEmbedNN(nn.Module):
    """
    Defines the function f for the SEM. For each variable x_i we use
    f_i(x) = f(e_i, sum_{k in pa(i)} g(e_k, x_k)), where e_i is a learned embedding
    for node i.
    """

    def __init__(
        self,
        group_mask: torch.Tensor,
        embedding_size: int,
        out_dim_g: int,
        num_layers_g: int,
        num_layers_zeta: int,
    ):
        """
        Args:
            group_mask: A mask of shape (num_nodes, num_processed_cols) such that group_mask[i, j] = 1. when col j is in group i.
            embedding_size: Size of the embeddings used by each node. Uses the larger of 4 * concatenated_shape or embedding_size.
            out_dim_g: Output dimension of the "inner" NN, l. If none, default is embedding size.
            num_layers_g: Number of layers in the "inner" NN, l.
            num_layers_zeta: Number of layers in the "outer" NN, ζ.
        """
        super().__init__()
        self.group_mask = group_mask
        num_nodes, concatenated_shape = group_mask.shape
        # Initialize embeddings uⱼ
        self.embeddings = nn.parameter.Parameter(0.01 * torch.randn(num_nodes, embedding_size), requires_grad=True)

        # Set value for out_dim_g
        # Set NNs sizes
        a = max(4 * concatenated_shape, embedding_size, 64)
        in_dim_g = embedding_size + concatenated_shape
        in_dim_f = embedding_size + out_dim_g
        self.l = _generate_fully_connected(
            input_dim=in_dim_g,
            output_dim=out_dim_g,
            hidden_dims=[
                a,
            ]
            * num_layers_g,
        )
        self.zeta = _generate_fully_connected(
            input_dim=in_dim_f,
            output_dim=concatenated_shape,
            hidden_dims=[
                a,
            ]
            * num_layers_zeta,
        )
        self.w = torch.nn.Parameter(torch.zeros((num_nodes, num_nodes)), requires_grad=True)

    def forward(self, samples: torch.Tensor, graphs: torch.Tensor) -> torch.Tensor:
        """
        Computes non-linear function hᵢ(X, G) using the given adjacency matrix.

            hᵢ(x, G) =  ζᵢ(Σⱼ Wᵢⱼ Gⱼᵢ lⱼ(xⱼ)

        We also use an embedding u so:

            hᵢ(x, G) =  ζ(uᵢ, Σⱼ Wᵢⱼ Gⱼᵢ l(uⱼ, xⱼ))

        l takes inputs of size batch_shape + (embedding_size + concatenated_shape) and outputs batch_shape + (out_dim_g)
        the input will be appropriately masked to correspond to one variable group

        ζ takes inputs of size batch_shape + (embedding_size + out_dim_g) and outputs batch_shape + (concatenated_shape)
        the ouptut is then masked to correspond to one variable

        Args:
            samples: tensor of shape batch_shape_x + batch_shape_f + batch_shape_g + [n_cols]
            graph: tensor of shape batch_shape_g + [n_nodes, n_nodes]
        Returns:
            tensor of shape batch_shape_x + batch_shape_f + batch_shape_g + [n_cols]
        """
        batch_shape_samples = samples.shape[:-1]
        batch_shape_g = graphs.shape[:-2]
        if len(batch_shape_g) > 0 and batch_shape_samples[-len(batch_shape_g) :] != batch_shape_g:
            raise ValueError(
                f"Batch shape of samples and graph must match but got {batch_shape_samples} and {batch_shape_g}"
            )

        # Shape batch_shape_x + batch_shape_f + batch_shape_g + (num_nodes, concatenated_shape)
        masked_samples = torch.einsum("...i,ji->...ji", samples, self.group_mask)
        # Shape batch_shape_x + batch_shape_f + batch_shape_g + (num_nodes, embedding_size)
        expanded_embed = self.embeddings.expand(*batch_shape_samples, -1, -1)

        # l(uⱼ, xⱼ) Shape batch_shape_x + batch_shape_f + batch_shape_g + (num_nodes, embedding_size + concatenated_shape)
        encoded_samples = self.l(
            torch.cat([masked_samples, expanded_embed], dim=-1)  # (concatenate xⱼ and embeddings uⱼ)
        )  # Shape batch_shape_x + batch_shape_f + batch_shape_g + (num_nodes, out_dim_g)

        # Aggregate sum and generate input for f (concatenate X_aggr and embeddings)
        # Σⱼ Wᵢⱼ Gⱼᵢ l(uⱼ, xⱼ) Shape batch_shape_x + batch_shape_f + batch_shape_g + (num_nodes, out_dim_g)
        encoded_samples_aggr = torch.einsum("...jk,...jl->...lk", encoded_samples, self.w * graphs)

        # ζ(uᵢ, Σⱼ Wᵢⱼ Gⱼᵢ l(uⱼ, xⱼ)) Shape batch_shape_x + batch_shape_f + batch_shape_g + (num_nodes, concatenated_shape)
        decoded_samples = self.zeta(torch.cat([encoded_samples_aggr, expanded_embed], dim=-1))

        # Mask and aggregate Shape batch_shape_x + batch_shape_f + batch_shape_g + (concatenated_shape)
        return torch.einsum("...ij,ij->...j", decoded_samples, self.group_mask)


def _generate_fully_connected(
    input_dim: int,
    output_dim: int,
    hidden_dims: list[int],
) -> nn.Module:
    """
    Generate a fully connected network.

    Args:
        input_dim: Int. Size of input to network.
        output_dim: Int. Size of output of network.
        hidden_dims: List of int. Sizes of internal hidden layers. i.e. [a, b] is three linear layers with shapes (input_dim, a), (a, b), (b, output_dim)

    Returns:
        Sequential object containing the desired network.
    """
    layers: list[nn.Module] = []

    prev_dim = input_dim
    for idx, hidden_dim in enumerate(hidden_dims):

        block: list[nn.Module] = []

        if idx > 0:
            block.append(nn.LayerNorm(prev_dim))
        block.extend([nn.Linear(prev_dim, hidden_dim), nn.LeakyReLU()])

        seq_block: nn.Module = nn.Sequential(*block)
        if prev_dim == hidden_dim:
            seq_block = _ResBlock(seq_block)
        layers.append(seq_block)

        prev_dim = hidden_dim

    layers.extend([nn.LayerNorm(prev_dim), nn.Linear(prev_dim, output_dim)])

    return nn.Sequential(*layers)


class _ResBlock(nn.Module):
    """
    Wraps an nn.Module, adding a skip connection to it.
    """

    def __init__(self, block: nn.Module):
        """
        Args:
            block: module to which skip connection will be added. The input dimension must match the output dimension.
        """
        super().__init__()
        self.block = block

    def forward(self, x):
        return x + self.block(x)
