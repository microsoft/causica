from typing import Optional

import torch

from fip.methods.base_method_module import DifferentialCausalDiscovery
from fip.models.causal_models import CausalTransformer


class SCMLearning(DifferentialCausalDiscovery):
    """Method for learning structural equation with fixed-point solvers"""

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 2,
        dim_key: int = 32,
        d_feedforward: int = 128,
        total_nodes: int = 10,
        total_layers: int = 1,
        dropout_prob: float = 0.1,
        mask_type: str = "diag",
        attn_type: str = "causal",
        cost_type: str = "dot_product",
        special_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            d_model: Embedding dimension used in the transformer model
            num_heads: Total number heads for self attention in the transformer model
            dim_key: Dimension of the key in the transformer model
            d_feedforward: Hidden dimension for feedforward layer in the transformer model
            total_nodes: Total number of nodes in the graph
            total_layers: Total self attention layers for the transformer model
            dropout_prob: Dropout probability for the transformer model
            mask_type: Type of the mask for the cross-attention layers
            attn_type: Type of attention for the cross-attention layers
            cost_type: Type of cost for the cross-attention layers
            special_mask: Add a special mask in the forward pass for the causal transformer model
        """

        super().__init__()

        self.special_mask = special_mask
        self.total_nodes = total_nodes

        self.model = CausalTransformer(
            d_model=d_model,
            num_heads=num_heads,
            dim_key=dim_key,
            num_layers=total_layers,
            d_ff=d_feedforward,
            max_seq_length=total_nodes,
            dropout=dropout_prob,
            attn_type=attn_type,
            cost_type=cost_type,
            mask_type=mask_type,
        )

    def sample_to_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the noise for learning the model under the additive noise assumption."""
        n_hat = x - self.model(x, torch.zeros_like(x), special_mask=self.special_mask)
        return n_hat

    def noise_to_sample(self, n: torch.Tensor) -> torch.Tensor:
        """Computes the fixed-point of the transformer under the additive noise assumption."""

        def func(z):
            return self.model(z, torch.zeros_like(z), special_mask=self.special_mask) + n

        x_hat = torch.ones_like(n)
        while not torch.allclose(func(x_hat), x_hat):
            x_hat = func(x_hat)

        return x_hat

    def forward(self, x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        return self.model(x, torch.zeros_like(x), special_mask=self.special_mask) + n

    def noise_to_sample_ite(
        self,
        n: torch.Tensor,
        mean_data: torch.Tensor,
        std_data: torch.Tensor,
        idx_nodes: list[int] | int,
        val_nodes: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(idx_nodes, int):
            idx_nodes = [idx_nodes]

        assert val_nodes.size(0) == n.size(0), "val_nodes.size(0) must be equal to n.size(0)"
        n[:, idx_nodes] = val_nodes.view(n.size(0), len(idx_nodes))

        special_mask = torch.ones(1, self.total_nodes, self.total_nodes)
        special_mask[:, idx_nodes, :] = 0
        special_mask = special_mask.bool().to(n.device)

        if self.special_mask is not None:
            special_mask = special_mask & self.special_mask

        def func(z):
            z = (z - mean_data) / std_data
            func_res = self.model(z, torch.zeros_like(z), special_mask=special_mask)
            func_res_rescale = std_data * func_res + mean_data
            func_res_rescale[:, idx_nodes] = 0.0
            res = func_res_rescale + n
            return res

        x_hat = torch.ones_like(n)
        while not torch.allclose(func(x_hat), x_hat):
            x_hat = func(x_hat)

        return x_hat

    def ite_prediction(
        self,
        x: torch.Tensor,
        mean_data: torch.Tensor,
        std_data: torch.Tensor,
        idx_nodes: list[int] | int,
        val_nodes: torch.Tensor,
        n_hat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes the ITE prediction for the given nodes and values."""

        x = (x - mean_data) / std_data
        if n_hat is None:
            n_hat = self.sample_to_noise(x)
        n_hat_rescale = n_hat * std_data

        return self.noise_to_sample_ite(n_hat_rescale, mean_data, std_data, idx_nodes, val_nodes)
