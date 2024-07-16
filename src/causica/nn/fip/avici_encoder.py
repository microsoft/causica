import torch
from torch import nn

from causica.nn.fip.decoder_layer import AmortizedDecoderLayer


class AmortizedEncoder(nn.Module):
    """
    Causal Dataset Encoder Model.

    Takes a batch of datasets and encodes it using multiple `AmortizedDecoderLayer`s.

    This is a reinterpretation of the AVICI model in PyTorch:

    https://github.com/larslorch/avici

    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_key: int,
        num_layers: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()

        self.emb_data = nn.Linear(1, d_model)
        self.decoder_layers_nodes = nn.ModuleList(
            [
                AmortizedDecoderLayer(
                    d_model,
                    num_heads,
                    dim_key,
                    d_ff,
                    dropout,
                    max_seq_length=2,
                    attn_type="standard",
                    cost_type="dot_product",
                )
                for _ in range(num_layers)
            ]
        )

        self.decoder_layers_samples = nn.ModuleList(
            [
                AmortizedDecoderLayer(
                    d_model,
                    num_heads,
                    dim_key,
                    d_ff,
                    dropout,
                    max_seq_length=2,
                    attn_type="standard",
                    cost_type="dot_product",
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model)
        self.num_layers = num_layers

    def forward(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tgt: expected shape (*, num_samples, max_seq_length) or (*, num_samples, max_seq_length, 2)

        Returns:
            output: expected shape: (*, num_samples, max_seq_length, d_model)

        """
        dec_output = tgt.unsqueeze(-1)

        # embed the data
        dec_output = self.emb_data(dec_output)

        for k in range(self.num_layers):
            dec_output = self.decoder_layers_nodes[k](dec_output, dec_output, dec_output, None)

            dec_output = dec_output.transpose(-2, -3)
            dec_output = self.decoder_layers_samples[k](dec_output, dec_output, dec_output, None)
            dec_output = dec_output.transpose(-2, -3)

        dec_output = self.norm(dec_output)

        return dec_output
