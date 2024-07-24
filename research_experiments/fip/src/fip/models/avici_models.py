import torch
from torch import nn

from causica.nn.fip.avici_encoder import AmortizedEncoder


class AmortizedLeaf(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_key: int,
        num_layers: int,
        d_ff: int,
        dropout: float,
        max_num_leaf: int = 100,
    ):
        super().__init__()

        self.encoder = AmortizedEncoder(
            d_model=d_model,
            num_heads=num_heads,
            dim_key=dim_key,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.num_layers = num_layers

        self.embedding_final = nn.Sequential(nn.LayerNorm(normalized_shape=d_model), nn.Linear(d_model, d_model))

        self.classifier = nn.Linear(d_model, 1)

        self.max_num_leaf = max_num_leaf
        self.d_model = d_model

    def forward(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tgt: expected shape (*, num_samples, max_seq_length)

        Returns:
            output: expected shape: (*, 1, max_seq_length)

        """
        # encode the data
        dec_output = self.encoder(tgt)

        # add a final embedding layer
        dec_output = self.embedding_final(dec_output)

        # reduce the sample dimension by taking the max
        dec_output = torch.max(dec_output, dim=-3)[0]  # shape (B, num_nodes, d_model)

        # normalize the output
        dec_output = dec_output / torch.linalg.norm(dec_output, dim=-1, ord=2, keepdim=True)

        # output the cost
        logits_leaf = self.classifier(dec_output).squeeze(-1)

        return logits_leaf
