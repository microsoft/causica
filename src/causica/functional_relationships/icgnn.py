from typing import Optional, Type

import torch
from tensordict import TensorDict
from torch import nn

from causica.functional_relationships.functional_relationships import (
    FunctionalRelationships,
    sample_dict_to_tensor,
    tensor_to_sample_dict,
)


class ICGNN(FunctionalRelationships):
    """
    This is a `FunctionalRelationsips` that implements the ICGNN.

    Details can be found here: https://openreview.net/forum?id=S2pNPZM-w-f

    This wraps the `FGNNI` in a `TensorDict` interface.
    """

    def __init__(
        self,
        variables: dict[str, torch.Size],
        embedding_size: Optional[int] = None,
        out_dim_g: Optional[int] = None,
        norm_layer: Optional[Type[nn.LayerNorm]] = None,
        res_connection: bool = False,
    ) -> None:
        super().__init__(variables)

        # this needs to be a parameter so it is registered to the module
        self.stacked_variable_masks = torch.nn.Parameter(
            torch.stack(list(self.variable_masks.values())).float(), requires_grad=False
        )

        self.nn = FGNNI(self.stacked_variable_masks, embedding_size, out_dim_g, norm_layer, res_connection)

    def forward(self, samples: TensorDict, graphs: torch.Tensor) -> TensorDict:
        return tensor_to_sample_dict(
            self.nn(sample_dict_to_tensor(samples, self.variable_masks), graphs), self.variable_masks
        )


class FGNNI(nn.Module):
    """
    Defines the function f for the SEM. For each variable x_i we use
    f_i(x) = f(e_i, sum_{k in pa(i)} g(e_k, x_k)), where e_i is a learned embedding
    for node i.
    """

    def __init__(
        self,
        group_mask: torch.Tensor,
        embedding_size: Optional[int] = None,
        out_dim_g: Optional[int] = None,
        norm_layer: Optional[Type[nn.LayerNorm]] = None,
        res_connection: bool = False,
    ):
        """
        Args:
            group_mask: A mask of shape (num_nodes, num_processed_cols) such that group_mask[i, j] = 1. when col j is in group i.
            embedding_size: Size of the embeddings used by each node. If none, default is processed_dim_all.
            out_dim_g: Output dimension of the "inner" NN, g. If none, default is embedding size.
            layers_g: Size of the layers of NN g. Does not include input not output dim. If none, default
                      is [a], with a = max(2 * input_dim, embedding_size, 10).
            layers_f: Size of the layers of NN f. Does not include input nor output dim. If none, default
                      is [a], with a = max(2 * input_dim, embedding_size, 10)
        """
        super().__init__()
        self.group_mask = group_mask
        self.num_nodes, self.processed_dim_all = group_mask.shape
        # Initialize embeddings
        self.embedding_size = self.processed_dim_all if embedding_size is None else embedding_size
        aux = torch.randn(self.num_nodes, self.embedding_size) * 0.01
        self.embeddings = nn.parameter.Parameter(aux, requires_grad=True)  # Shape (input_dim, embedding_size)

        # Set value for out_dim_g
        out_dim_g = self.embedding_size if out_dim_g is None else out_dim_g
        # Set NNs sizes
        a = max(4 * self.processed_dim_all, self.embedding_size, 64)
        in_dim_g = self.embedding_size + self.processed_dim_all
        in_dim_f = self.embedding_size + out_dim_g
        self.g = generate_fully_connected(
            input_dim=in_dim_g,
            output_dim=out_dim_g,
            hidden_dims=[a, a],
            non_linearity=nn.LeakyReLU,
            activation=nn.Identity,
            device=group_mask.device,
            normalization=norm_layer,
            res_connection=res_connection,
        )
        self.f = generate_fully_connected(
            input_dim=in_dim_f,
            output_dim=self.processed_dim_all,
            hidden_dims=[a, a],
            non_linearity=nn.LeakyReLU,
            activation=nn.Identity,
            device=group_mask.device,
            normalization=norm_layer,
            res_connection=res_connection,
        )
        self.w = torch.nn.Parameter(torch.zeros((self.num_nodes, self.num_nodes)), requires_grad=True)

    def forward(self, samples: torch.Tensor, graphs: torch.Tensor) -> torch.Tensor:
        """
        Computes non-linear function h(X, W) using the given weighted adjacency matrix.

        g takes inputs of size batch_shape + (embedding_size + processed_dim_all) and outputs batch_shape + (out_dim_g)
        the input will be appropriately masked to correspond to one variable group

        f takes inputs of size batch_shape + (embedding_size + out_dim_g) and outputs batch_shape + (processed_dim_all)
        the ouptut is then masked to correspond to one variable

        Args:
            samples: Batched inputs, size batch_size_x + (processed_dim_all).
            graphs: Weighted adjacency matrix, size batch_size_g + (n, n)
        Returns:
            A tensor of shape batch_shape_x + batch_shape_g + (processed_dim_all)
        """
        batch_shape_x = samples.shape[:-1]
        batch_shape_g = graphs.shape[:-2]

        # Generate required input for g (concatenate X and embeddings)
        # Pointwise multiply X
        # Shape batch_shape_x + (num_nodes, processed_dim_all)
        masked_samples = torch.einsum("...i,ji->...ji", samples, self.group_mask)
        E = self.embeddings.expand(*batch_shape_x, -1, -1)  # Shape batch_shape_x + (num_nodes, embedding_size)

        # Shape batch_shape_x + (num_nodes, embedding_size + processed_dim_all)
        embedded_samples = self.g(
            torch.cat([masked_samples, E], dim=-1)
        )  # Shape batch_shape_x + (num_nodes, out_dim_g)

        target_shape = batch_shape_x + batch_shape_g + embedded_samples.shape[-2:]
        view_shape = batch_shape_x + (1,) * len(batch_shape_g) + embedded_samples.shape[-2:]
        # Shape batch_shape_x + batch_shape_g + (num_nodes, out_dim_g)
        embedded_samples_broad = embedded_samples.view(view_shape).expand(target_shape)
        # Aggregate sum and generate input for f (concatenate X_aggr and embeddings)
        # Shape batch_shape_x + batch_shape_g + (num_nodes, out_dim_g)
        samples_aggr_sum = torch.einsum("...jk,...jl->...lk", embedded_samples_broad, graphs * self.w)

        # expand dimensions of E batch_shape_x + batch_shape_g + (num_nodes, embedding_size)
        E_broad = E.view(view_shape).expand(target_shape)
        # Run f Shape batch_shape_x + batch_shape_g + (num_nodes, processed_dim_all)
        samples_rec = self.f(torch.cat([samples_aggr_sum, E_broad], dim=-1))
        # Mask and aggregate Shape batch_shape_x + batch_shape_g + (processed_dim_all)
        return torch.einsum("...ij,ij->...j", samples_rec, self.group_mask)


def generate_fully_connected(
    input_dim: int,
    output_dim: int,
    hidden_dims: list[int],
    non_linearity: Optional[Type[nn.Module]],
    activation: Optional[Type[nn.Module]],
    device: torch.device,
    p_dropout: float = 0.0,
    normalization: Optional[Type[nn.LayerNorm]] = None,
    res_connection: bool = False,
) -> nn.Module:
    """
    Generate a fully connected network.

    Args:
        input_dim: Int. Size of input to network.
        output_dim: Int. Size of output of network.
        hidden_dims: List of int. Sizes of internal hidden layers. i.e. [a, b] is three linear layers with shapes (input_dim, a), (a, b), (b, output_dim)
        non_linearity: Non linear activation function used between Linear layers.
        activation: Final layer activation to use.
        device: torch device to load weights to.
        p_dropout: Float. Dropout probability at the hidden layers.
        init_method: initialization method
        normalization: Normalisation layer to use (batchnorm, layer norm, etc). Will be placed before linear layers, excluding the input layer.
        res_connection : Whether to use residual connections where possible (if previous layer width matches next layer width)

    Returns:
        Sequential object containing the desired network.
    """
    layers: list[nn.Module] = []

    prev_dim = input_dim
    for idx, hidden_dim in enumerate(hidden_dims):

        block: list[nn.Module] = []

        if normalization is not None and idx > 0:
            block.append(normalization(prev_dim).to(device))
        block.append(nn.Linear(prev_dim, hidden_dim).to(device))

        if non_linearity is not None:
            block.append(non_linearity())
        if p_dropout != 0:
            block.append(nn.Dropout(p_dropout))

        if res_connection and (prev_dim == hidden_dim):
            layers.append(_ResBlock(nn.Sequential(*block)))
        else:
            layers.append(nn.Sequential(*block))
        prev_dim = hidden_dim

    if normalization is not None:
        layers.append(normalization(prev_dim).to(device))
    layers.append(nn.Linear(prev_dim, output_dim).to(device))

    if activation is not None:
        layers.append(activation())

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
