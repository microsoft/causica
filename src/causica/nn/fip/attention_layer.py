import math
from typing import Optional

import torch
from torch import nn


def cost_matrix_lp(x_1: torch.Tensor, y_1: torch.Tensor, power: int = 2) -> torch.Tensor:
    """
    Returns the matrix of $|x_1-y_1|^p$.

    Args:
        x_1: expected shape: (*, total_nodes, dim_nodes)
        y_1: expected shape: (*, batch_size, total_nodes, dim_nodes)

    Returns:
        output: expected shape: (*, total_nodes, total_nodes)
    """
    x_col = x_1.unsqueeze(-2)
    y_lin = y_1.unsqueeze(-3)
    return torch.sum((torch.abs(x_col - y_lin)) ** power, -1)


def compute_score_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    d_k: float,
    mask: Optional[torch.Tensor] = None,
    cost_type: str = "dot_product",
) -> torch.Tensor:
    """
    Compute the cost matrix between queries and keys.

    Args:
        queries: expected shape: (*, total_nodes, dim_nodes)
        keys: expected shape: (*, total_nodes, dim_nodes)
        d_k (float): rescaling factor
        mask: expected shape: (*, total_nodes, total_nodes)
        cost_type (str): dot_product or l2

    Returns:
        output: expected shape: (*, total_nodes, total_nodes)
    """

    if cost_type == "l2":
        attn_scores = cost_matrix_lp(queries, keys, power=2) / math.sqrt(d_k)
    elif cost_type == "dot_product":
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(d_k)
    else:
        raise ValueError(f"cost_type {cost_type} not recognized")

    if mask is not None:
        val_masking = -1e4 if attn_scores.dtype == torch.float16 else -1e9
        attn_scores = attn_scores.masked_fill(mask == 0, val_masking)

    return attn_scores


def scaled_dot_product_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    d_k: float,
    mask: Optional[torch.Tensor] = None,
    cost_type: str = "dot_product",
) -> torch.Tensor:

    """
    Compute the standard attention matrix

    Args:
        queries: expected shape: (*, total_nodes, dim_nodes)
        keys: expected shape: (*, total_nodes, dim_nodes)
        d_k (float): rescaling factor
        mask: expected shape: (*, total_nodes, total_nodes)
        cost_type (str): dot_product or l2

    Returns:
        output: expected shape: (*, total_nodes, total_nodes)

    """

    attn_scores = compute_score_attention(queries, keys, d_k, mask=mask, cost_type=cost_type)
    return torch.softmax(attn_scores, dim=-1)


def causal_scaled_dot_product_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    d_k: float,
    mask: Optional[torch.Tensor] = None,
    cost_type: str = "dot_product",
) -> torch.Tensor:
    r"""
    Partial Optimal Transport problem

    Here we are solving: $\min \langle C, P \rangle$ -\espilon H(P)$ s.t. $P \leq 1$
    where $C$ is the cost matrix, $P$ is the transport matrix, and $H$ is the entropy.

    Args:
        queries: expected shape: (*, total_nodes, dim_nodes)
        keys: expected shape: (*, total_nodes, dim_nodes)
        d_k (float): rescaling factor
        mask: expected shape: (*, total_nodes, total_nodes)
        cost_type (str): dot_product or l2

    Returns:
        output: expected shape: (*, total_nodes, total_nodes)

    """
    attn_scores = compute_score_attention(queries, keys, d_k, mask=mask, cost_type=cost_type)

    max_val = torch.max(attn_scores, dim=-1, keepdim=True)[0]
    attn_scores_rescaled = attn_scores - max_val
    attn_probs_rescaled = torch.exp(attn_scores_rescaled)
    den_rescaled = torch.sum(attn_probs_rescaled, dim=-1, keepdim=True)
    den_rescaled = torch.max(den_rescaled, torch.exp(-max_val))

    return attn_probs_rescaled / den_rescaled


class MultiHeadAttention(nn.Module):
    r"""
    Multi-Head Attention module

    Attention type: attn_type = causal, causal_fixed, diagonal, standard, linear

    In attn_type is "causal_fixed", the attention matrix is fixed
    to be the same for all inputs. In that case, the attention matrix is fixed for any inputs.

    If attn_type is "causal", the attention matrix depends on the input.

    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_length: int,
        dim_key: Optional[int] = None,
        attn_type: str = "causal",
        cost_type: str = "dot_product",
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn_type = attn_type
        self.cost_type = cost_type

        self.dim_latent = num_heads * dim_key if dim_key is not None else d_model
        assert self.dim_latent % num_heads == 0, "dim_latent must be divisible by num_heads"
        self.d_k = self.dim_latent // num_heads

        self.w_q = nn.Linear(d_model, self.dim_latent)
        self.w_k = nn.Linear(d_model, self.dim_latent)
        self.w_v = nn.Linear(d_model, self.dim_latent)
        self.w_o = nn.Linear(self.dim_latent, d_model)

        if self.attn_type == "standard":
            self.attn = scaled_dot_product_attention
        elif self.attn_type == "linear":
            self.attn = compute_score_attention
        else:
            self.position: torch.Tensor
            self.register_buffer("position", torch.arange(0, max_seq_length, dtype=torch.long))
            self.emb = nn.Embedding(max_seq_length, d_model)
            self.attn = causal_scaled_dot_product_attention

    def split_heads(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inp: expected shape (*, max_seq_length, d_model)

        Returns:
            out: expected shape: (*, num_heads, max_seq_length, d_model // num_heads)

        """
        all_dims = inp.size()
        inp = inp.view(*all_dims[:-1], self.num_heads, self.d_k)
        return inp.transpose(-3, -2)

    def combine_heads(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inp: expected shape (*, num_heads, max_seq_length, d_model // num_heads)

        Returns:
            out: expected shape: (*, max_seq_length, d_model)

        """
        inp = inp.transpose(-3, -2).contiguous()
        all_dims = inp.size()
        return inp.view(*all_dims[:-2], self.dim_latent)

    def compute_attn(
        self, queries: torch.Tensor, keys: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            queries: expected shape: (batch_size, max_seq_length, d_model)
            keys: expected shape: (batch_size, max_seq_length, d_model)

        Returns:
            output: expected shape: (batch_size, num_heads, max_seq_length, max_seq_length)

        """

        if self.attn_type == "causal_fixed":
            keys = self.emb(self.position).repeat(keys.size(0), 1, 1)
            queries = self.emb(self.position).repeat(queries.size(0), 1, 1)

        queries = self.split_heads(self.w_q(queries))
        keys = self.split_heads(self.w_k(keys))
        return self.attn(queries, keys, self.d_k, mask=mask, cost_type=self.cost_type)

    def compute_cost(
        self, queries: torch.Tensor, keys: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            queries: expected shape: (batch_size, max_seq_length, d_model)
            keys: expected shape: (batch_size, max_seq_length, d_model)

        Returns:
            output: expected shape: (batch_size, num_heads, max_seq_length, max_seq_length)

        """
        if self.attn_type == "causal_fixed":
            keys = self.emb(self.position).repeat(keys.size(0), 1, 1)
            queries = self.emb(self.position).repeat(queries.size(0), 1, 1)

        queries = self.split_heads(self.w_q(queries))
        keys = self.split_heads(self.w_k(keys))
        return compute_score_attention(queries, keys, self.d_k, mask=mask, cost_type=self.cost_type)

    def forward(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            queries: expected shape: (batch_size, max_seq_length, d_model)
            keys: expected shape: (batch_size, max_seq_length, d_model)
            values: expected shape: (batch_size, max_seq_length, d_model)

        Returns:
            output: expected shape: (batch_size, max_seq_length, max_seq_length)

        """
        values = self.split_heads(self.w_v(values))
        attn_probs = self.compute_attn(queries, keys, mask=mask)
        attn_output = attn_probs @ values

        return self.w_o(self.combine_heads(attn_output))
