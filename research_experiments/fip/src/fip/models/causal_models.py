from typing import Optional

import torch
from torch import nn

from causica.nn.fip.decoder_layer import CausalDecoderLayer
from causica.nn.fip.embeddings import Encoding


class CausalTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_key: int,
        num_layers: int,
        d_ff: int,
        max_seq_length: int,
        dropout: float,
        attn_type: str = "causal",
        cost_type: str = "l2",
        mask_type: str = "diag",
    ):
        super().__init__()

        self.dec_mask: torch.Tensor
        if mask_type == "self":
            dec_mask = (torch.eye(max_seq_length).reshape(1, max_seq_length, max_seq_length)).bool()
        elif mask_type == "diag":
            dec_mask = (1 - torch.eye(max_seq_length).reshape(1, max_seq_length, max_seq_length)).bool()
        elif mask_type == "triang":
            dec_mask = (1 - torch.triu(torch.ones(1, max_seq_length, max_seq_length), diagonal=0)).bool()
        else:
            dec_mask = torch.ones(1, max_seq_length, max_seq_length).bool()
        self.register_buffer("dec_mask", dec_mask)

        self.positional_encoding = Encoding(d_model, max_seq_length, encoding_type="position")
        self.dropout = nn.Dropout(dropout)
        self.emb_data = nn.Sequential(
            Encoding(d_model, max_seq_length, encoding_type="node"), self.positional_encoding, self.dropout
        )
        self.emb_noise = nn.Sequential(
            Encoding(d_model, max_seq_length, encoding_type="node"), self.positional_encoding, self.dropout
        )

        self.decoder_layers = nn.ModuleList(
            [
                CausalDecoderLayer(
                    d_model,
                    num_heads,
                    dim_key,
                    d_ff,
                    dropout,
                    max_seq_length,
                    attn_type=attn_type,
                    cost_type=cost_type,
                )
                for _ in range(num_layers)
            ]
        )

        self.positional_encoding_f = Encoding(d_model, max_seq_length, encoding_type="position")
        self.final_emb = nn.Sequential(
            Encoding(d_model, max_seq_length, encoding_type="node"), self.positional_encoding_f, self.dropout
        )

        self.fc = nn.Linear(d_model, 1)

        self.mask_type = mask_type
        self.max_seq_length = max_seq_length

    def special_mask(self, dec_mask: torch.Tensor, special_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if special_mask is not None:
            dec_mask = dec_mask & special_mask

        return dec_mask

    def add_mask(self, dec_mask: torch.Tensor, added_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            added_mask: expected shape (1, max_seq_length, max_seq_length)

        Returns:
            output: expected shape: (1,  max_seq_length, max_seq_length)

        """

        if added_mask is not None:
            dec_mask = dec_mask + added_mask

        return dec_mask

    def forward(
        self,
        tgt: torch.Tensor,
        noise: torch.Tensor,
        special_mask: Optional[torch.Tensor] = None,
        added_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt: expected shape (*, max_seq_length)
            noise: expected shape (*, max_seq_length)
            special_mask: expected shape (1, max_seq_length, max_seq_length)
            added_mask: expected shape (1, max_seq_length, max_seq_length)

        Returns:
            output: expected shape: (*,  max_seq_length)

        """
        dec_mask = self.dec_mask
        dec_mask = self.special_mask(dec_mask=dec_mask, special_mask=special_mask)
        dec_mask = self.add_mask(dec_mask=dec_mask, added_mask=added_mask)

        tgt_embedded = self.emb_data(tgt.unsqueeze(-1))
        noise_embedded = self.emb_noise(noise.unsqueeze(-1))

        dec_output = noise_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, tgt_embedded, tgt_embedded, dec_mask)

        dec_output = self.final_emb(dec_output)

        return self.fc(dec_output).squeeze(-1)
