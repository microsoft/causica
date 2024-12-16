import torch
from torch import nn

from cond_fip.models.decoder_layer import AdaptiveAmortizedDecoderLayer, AmortizedDecoderLayer


class AmortizedEncoder(nn.Module):
    """Architecture for the dataset embedding network based on self standard and DAG attentions on sample and nodes respectively."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_key: int,
        num_layers: int,
        d_ff: int,
        dropout: float,
    ):
        """
        Args:
            d_model: Embedding Dimension
            num_heads: Total number of attention heads
            dim_key: Dimension of queries, keys and values per head
            num_layers: Total number of attention layers
            d_ff: Hidden dimension for feedforward layer
            dropout: Dropout probability
        """
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
                    max_seq_length=2,  # this parameter has no effect
                    attn_type="causal",
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
                    max_seq_length=2,  # this parameter has no effect
                    attn_type="standard",
                    cost_type="dot_product",
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model)

        self.num_layers = num_layers
        self.dec_mask = None
        self.sample_mask = None

    def forward(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tgt: expected shape (*, num_samples, max_seq_length)

        Returns:
            output: expected shape: (*, num_samples, max_seq_length, d_model)

        """

        dec_output = tgt.unsqueeze(-1)
        dec_output = self.emb_data(dec_output)

        for k in range(self.num_layers):
            dec_output = dec_output.transpose(-2, -3)
            dec_output = self.decoder_layers_samples[k](dec_output, dec_output, dec_output, self.sample_mask)
            dec_output = dec_output.transpose(-2, -3)

            dec_output = self.decoder_layers_nodes[k](dec_output, dec_output, dec_output, self.dec_mask)

        dec_output = self.norm(dec_output)

        return dec_output


class AmortizedNoise(nn.Module):
    """Architecture for the encoder comprising of dataset embedding network and noise prediction layer."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        dropout: float,
        dim_key: int,
        d_hidden_head: int,
    ):
        """
        Args:
            d_model: Embedding Dimension
            num_heads: Total number of attention heads
            num_layers: Total number of attention layers
            d_ff: Hidden dimension for feedforward layer
            dropout: Dropout probability
            dim_key: Dimension of queries, keys and values per head
            d_hidden_head: Hidden dimension of the head MLP
        """
        super().__init__()

        self.encoder = AmortizedEncoder(
            d_model=d_model,
            num_heads=num_heads,
            dim_key=dim_key,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.pred_network = nn.Sequential(
            nn.Linear(d_model, d_hidden_head),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_hidden_head, d_hidden_head),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_hidden_head, 1),
        )

        self.d_model = d_model

    def compute_encoding(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tgt: expected shape (*, batch_size, max_seq_length)

        Returns:
            output: expected shape: (*, batch_size, max_seq_length, d_model)

        """
        return self.encoder(tgt)

    def compute_proj(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tgt: expected shape (*, batch_size, max_seq_length, d_model)

        Returns:
            output: expected shape: (*, batch_size, max_seq_length)

        """
        preds = self.pred_network(tgt)
        return preds.squeeze(-1)

    def forward(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tgt: expected shape (*, batch_size, max_seq_length)

        Returns:
            output: expected shape: (*, batch_size, max_seq_length)

        """
        dec_output = self.encoder(tgt)
        return self.compute_proj(dec_output)


class AmortizedEncoderDecoder(nn.Module):
    """Architecture for the conditional fixed-point (cond-FiP) method to learn causal functional relationships."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_key: int,
        num_layers: int,
        d_ff: int,
        dropout: float,
        d_data_emb: int,
        num_layers_dataset: int,
    ):
        """
        Args:
            d_model: Embedding Dimension
            num_heads: Total number of attention heads
            dim_key: Dimension of queries, keys and values per head
            num_layers: Total number of attention layers
            d_ff: Hidden dimension for feedforward layer
            dropout: Dropout probability
            d_data_emb: Dimension of dataset embedding input
            num_layers_dataset: Number of extra layers to embed the condition
        """
        super().__init__()

        self.proj_dataset = nn.Linear(d_data_emb, d_model)
        self.norm_dataset_1 = nn.LayerNorm(d_model)
        self.norm_dataset_2 = nn.LayerNorm(d_model)

        self.emb_dataset = nn.Linear(d_model, d_model, bias=False)
        self.emb_position = nn.Linear(d_model, d_model, bias=False)
        self.emb_condition = nn.Linear(d_model, d_model, bias=False)

        self.norm_dataset = nn.LayerNorm(d_model)
        self.norm_position = nn.LayerNorm(d_model)
        self.norm_condition = nn.LayerNorm(d_model)

        self.num_layers_dataset = num_layers_dataset
        self.decoder_layers_nodes_dataset = nn.ModuleList(
            [
                AmortizedDecoderLayer(
                    d_model,
                    num_heads,
                    dim_key,
                    d_ff,
                    dropout,
                    max_seq_length=2,  # this parameter has no effect
                    attn_type="causal",
                    cost_type="dot_product",
                )
                for _ in range(num_layers_dataset)
            ]
        )

        self.decoder_layers_samples_dataset = nn.ModuleList(
            [
                AmortizedDecoderLayer(
                    d_model,
                    num_heads,
                    dim_key,
                    d_ff,
                    dropout,
                    max_seq_length=2,  # this parameter has no effect
                    attn_type="standard",
                    cost_type="dot_product",
                )
                for _ in range(num_layers_dataset)
            ]
        )

        self.decoder_layers_nodes = nn.ModuleList(
            [
                AdaptiveAmortizedDecoderLayer(
                    d_model,
                    num_heads,
                    dim_key,
                    d_ff,
                    dropout,
                    max_seq_length=2,
                    attn_type="causal",
                    cost_type="dot_product",
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model)
        self.pred_network = nn.Linear(d_model, 1)

        self.num_layers = num_layers
        self.dec_mask = None

    def embed_dataset(self, dataset_embedded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            dataset_embedded: expected shape (*, num_samples, max_seq_length, d_data_emb)

        Returns:
            output: expected shape: (*, 1, max_seq_length, d_model, d_model)

        """
        # proj dataset_embedded
        dataset_embedded = self.proj_dataset(dataset_embedded)

        # perform attention on dataset_embedded
        for k in range(self.num_layers_dataset):

            dataset_embedded = dataset_embedded.transpose(-2, -3)
            dataset_embedded = self.decoder_layers_samples_dataset[k](
                dataset_embedded, dataset_embedded, dataset_embedded, None
            )
            dataset_embedded = dataset_embedded.transpose(-2, -3)

            dataset_embedded = self.decoder_layers_nodes_dataset[k](
                dataset_embedded, dataset_embedded, dataset_embedded, self.dec_mask
            )

        # normalize the dataset_embedded
        dataset_embedded = self.norm_dataset_1(dataset_embedded)

        # take the max w.r.t the sample dimension
        dataset_embedded = torch.max(dataset_embedded, dim=1).values.unsqueeze(1)

        # normalize the dataset_embedded
        dataset_embedded = self.norm_dataset_2(dataset_embedded)

        # get the conditional embeddings
        vector_embedding = self.emb_dataset(dataset_embedded)
        pos_embedding = self.emb_position(dataset_embedded)
        condition_embedding = self.emb_condition(dataset_embedded)

        # normalize the embeddings
        vector_embedding = self.norm_dataset(vector_embedding)
        pos_embedding = self.norm_position(pos_embedding)
        condition_embedding = self.norm_condition(condition_embedding)

        return vector_embedding, pos_embedding, condition_embedding

    def forward(self, tgt: torch.Tensor, dataset_embedded: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tgt: expected shape (*, num_samples, max_seq_length)
            dataset_embedded: expected shape (*, 1, max_seq_length, d_data_emb)

        Returns:
            output: expected shape: (*, num_samples, max_seq_length, d_model)

        """

        vector_embedding, pos_embedding, condition_embedding = self.embed_dataset(dataset_embedded)

        tgt_embedded = tgt.unsqueeze(-1)
        tgt_embedded = tgt_embedded * vector_embedding + pos_embedding

        for k in range(self.num_layers):
            pos_embedding = self.decoder_layers_nodes[k](
                pos_embedding, tgt_embedded, tgt_embedded, condition_embedding, self.dec_mask
            )

        dec_output = self.norm(pos_embedding)
        preds = self.pred_network(dec_output)

        return preds.squeeze(-1)
