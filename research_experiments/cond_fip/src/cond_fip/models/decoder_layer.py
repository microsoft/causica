from typing import Optional

import torch
from torch import nn

from causica.nn.fip.attention_layer import MultiHeadAttention
from causica.nn.fip.embeddings import PositionWiseFeedForward


class AmortizedDecoderLayer(nn.Module):
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
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        """
        Args:
            queries: expected shape: (batch_size, max_seq_length, d_model)
            keys: expected shape: (batch_size, max_seq_length, d_model)
            values: expected shape: (batch_size, max_seq_length, d_model)

        Returns:
            output: expected shape: (batch_size, max_seq_length, d_model)

        """
        queries_n = self.norm_q(queries)
        keys_n = self.norm_k(keys)
        values_n = self.norm_v(values)
        attn_output = self.self_attn(queries_n, keys_n, values_n, mask)

        x_trans = queries + self.dropout(attn_output)

        x_trans_n = self.norm(x_trans)
        ff_output = self.dropout(self.feed_forward(x_trans_n))

        res = x_trans + ff_output

        return res


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class AdaptiveAmortizedDecoderLayer(nn.Module):
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

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model, bias=True))

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        condition: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        """
        Args:
            queries: expected shape: (batch_size, max_seq_length, d_model)
            keys: expected shape: (batch_size, max_seq_length, d_model)
            values: expected shape: (batch_size, max_seq_length, d_model)
            condition: expected shape: (**, d_model)

        Returns:
            output: expected shape: (batch_size, max_seq_length, d_model)

        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(condition).chunk(
            6, dim=-1
        )

        queries_n = self.norm_q(queries)
        keys_n = self.norm_k(keys)
        values_n = self.norm_v(values)
        queries_n_modulated = modulate(queries_n, shift_msa, scale_msa)
        keys_n_modulated = modulate(keys_n, shift_msa, scale_msa)
        values_n_modulated = modulate(values_n, shift_msa, scale_msa)

        attn_output = self.self_attn(queries_n_modulated, keys_n_modulated, values_n_modulated, mask)
        x_trans = queries + gate_msa * self.dropout(attn_output)

        x_trans_n = self.norm(x_trans)
        x_trans_n_modulated = modulate(x_trans_n, shift_mlp, scale_mlp)

        ff_output = self.dropout(self.feed_forward(x_trans_n_modulated))
        res = x_trans + gate_mlp * ff_output

        return res
