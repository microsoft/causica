from typing import Optional

import torch
from torch import nn

from causica.nn.fip.attention_layer import MultiHeadAttention
from causica.nn.fip.embeddings import PositionWiseFeedForward


class CausalDecoderLayer(nn.Module):
    """
    Causal Decoder Layer.

    It consists of a multi-head self-attention mechanism, followed by a feed-forward network.
    It also allows the model to use the causal attention mechanism.

    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_key: int,
        d_ff: int,
        dropout: float,
        max_seq_length: int,
        attn_type: str = "standard",
        cost_type: str = "dot_product",
    ):
        super().__init__()

        self.self_attn = MultiHeadAttention(
            d_model, num_heads, max_seq_length, dim_key=dim_key, attn_type=attn_type, cost_type=cost_type
        )

        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.max_seq_length = max_seq_length

    def forward(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:

        """
        Args:
            queries: expected shape: (batch_size, max_seq_length, d_model)
            keys: expected shape: (batch_size, max_seq_length, d_model)
            values: expected shape: (batch_size, max_seq_length, d_model)
            mask: expected shape: (batch_size, max_seq_length, max_seq_length)

        Returns:
            output: expected shape: (batch_size, max_seq_length, d_model)

        """

        attn_output = self.self_attn(queries, keys, values, mask)
        x_trans = self.norm1(queries + self.dropout(attn_output))

        ff_output = self.feed_forward(x_trans)

        return self.norm2(x_trans + self.dropout(ff_output))


class AmortizedDecoderLayer(nn.Module):
    """
    Amortized Causal Decoder Layer.

    This is a re-implementation of the decoder layer used in the AVICI model in PyTorch:

    https://github.com/larslorch/avici

    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_key: int,
        d_ff: int,
        dropout: float,
        max_seq_length: int,
        attn_type: str = "standard",
        cost_type: str = "dot_product",
    ):
        super().__init__()

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_k = nn.LayerNorm(d_model)
        self.norm_v = nn.LayerNorm(d_model)

        self.self_attn = MultiHeadAttention(
            d_model, num_heads, max_seq_length, dim_key=dim_key, attn_type=attn_type, cost_type=cost_type
        )

        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:

        """
        Args:
            queries: expected shape: (batch_size, max_seq_length, d_model)
            keys: expected shape: (batch_size, max_seq_length, d_model)
            values: expected shape: (batch_size, max_seq_length, d_model)

        Returns:
            output: expected shape: (batch_size, max_seq_length, d_model)

        """
        queries_in = self.norm_q(queries)
        keys_in = self.norm_k(keys)
        values_in = self.norm_v(values)
        attn_output = self.self_attn(queries_in, keys_in, values_in, mask)
        x = queries + self.dropout(attn_output)
        x_in = self.norm(x)
        ff_output = self.dropout(self.feed_forward(x_in))
        return x + ff_output
