import torch
from torch import nn


class PositionalEncodingFixed(nn.Module):
    """
    Standard positional encoding.

    """

    def __init__(self, d_model: int, max_seq_length: int):
        super().__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.pow(1e5, torch.linspace(0.0, 1.0, d_model // 2 + 1))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe: torch.Tensor
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inp: expected shape: (batch_size, max_seq_length, d_model)

        Returns:
            out: expected shape: (batch_size, max_seq_length, d_model)

        """

        return inp + self.pe[: inp.size(1), :]


class Encoding(nn.Module):
    """Learnable encoding.

    Either for nodes or for positions.
    For the nodes, we multiply (in the hadamard sense) the node value with the positional embedding

    """

    def __init__(self, d_model: int, max_seq_length: int, encoding_type: str = "node"):
        super().__init__()
        self.emb = nn.Embedding(max_seq_length, d_model)

        self.position: torch.Tensor
        self.register_buffer("position", torch.arange(0, max_seq_length, dtype=torch.long))

        self.encoding_type = encoding_type

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inp: expected shape: (batch_size, max_seq_length, 1)

        Returns:
            out: expected shape: (batch_size, max_seq_length, d_model)

        """
        if self.encoding_type == "node":
            return inp * self.emb(self.position)

        if self.encoding_type == "position":
            return inp + self.emb(self.position)

        raise ValueError("Unknown encoding type")


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, add_prod: bool = False):
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.add_prod = add_prod
        if add_prod:
            self.fc_m = nn.Linear(d_model, d_ff)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inp: expected shape: (batch_size, max_seq_length, d_model)

        Returns:
            out: expected shape: (batch_size, max_seq_length, d_model)

        """
        if self.add_prod:
            return self.fc2(self.relu(self.fc1(inp)) * self.fc_m(inp))

        return self.fc2(self.relu(self.fc1(inp)))
