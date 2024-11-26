import abc

import torch
from torch import nn

from fip.methods.jacobian import compute_jacobian


class DifferentialCausalDiscovery(nn.Module, abc.ABC):
    """An Abstract Method Module containing the methods required by jacobian-based causal discovery methods."""

    @abc.abstractmethod
    def sample_to_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the noise for learning the model from input data.

        Args:
            x: Data variables input to the model; shape: (Batch Size, Total Nodes)

        Returns:
            n_hat: noise for learning SCM: (Batch Size, Total Nodes)
        """

    @abc.abstractmethod
    def noise_to_sample(self, n: torch.Tensor) -> torch.Tensor:
        """Transforms the noise into data.

        A typical use case would be computing the fixed point of the model conditioned on noise n using solver.

        Args:
            n: Noise variables input to the model; shape: (Batch Size, Total Nodes)

        Returns:
            x_hat: Sample generated for given noise; shape: (Batch Size, Total Nodes)
        """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        """Compute the forward of the model with noise

        Args:
            x: Data of the SCM in the current batch; shape: (Batch Size, Total Nodes)
            n: Noise of the SCM in the current batch; shape: (Batch Size, Total Nodes)

        Returns:
            x_hat: Prediction of the method; shape: (Batch Size, Total Nodes, 1)
        """

    def get_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """Return the jacobian of the model

        Args:
            x: Data variables input to the model; shape: (Batch Size, Total Nodes, 1)

        Return:
            jacobian: Jacobian of the model; shape: (Batch Size, Total Nodes, Total Nodes)
        """

        x = x.requires_grad_()
        output = self.forward(x, torch.zeros_like(x))
        return compute_jacobian(x, output)

    def get_causal_graphs(self, x: torch.Tensor) -> torch.Tensor:
        """Return the continuous graph of the model

        Args:
            x: Data variables input to the model; shape: (Batch Size, Total Nodes)

        Return:
            pred_graph: Predicted continuous graphs as per the casusal discovery method: Expected Shape: (Total Nodes, Total Nodes)
        """

        # decompose x into smaller part to avoid memory issues
        batch_size = x.shape[0]
        batch_size_decompose = min(100, batch_size)
        num_batches = batch_size // batch_size_decompose

        # compute the jacobian for each batch
        jacobian_list = []
        for i in range(num_batches):
            jacobian_curr = self.get_jacobian(x[i * batch_size_decompose : (i + 1) * batch_size_decompose])
            # remove gradient
            jacobian_curr = jacobian_curr.detach()
            # add to list
            jacobian_list.append(jacobian_curr)

        # concatenate the jacobians
        jacobian = torch.cat(jacobian_list, dim=0)

        return torch.abs(jacobian)

    def get_aggregated_causal_graph(self, x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        """Return the mean causal graph of the model

        Args:
            x: Data variables input to the model; shape: (Batch Size, Total Nodes)
            mode: Mode to aggregate the causal graphs; options: ["mean", "median"]

        Return:
            pred_graph: Predicted aggregated graphs as per the casusal discovery method; shape: (Total Nodes, Total Nodes)
        """

        continuous_graphs = self.get_causal_graphs(x)
        if mode == "mean":
            return torch.mean(continuous_graphs, dim=0)
        if mode == "median":
            return torch.median(continuous_graphs, dim=0).values
        raise ValueError("mode not recognized")

    def get_threshold_causal_graph(self, x: torch.Tensor, threshold: torch.Tensor, mode="mean") -> torch.Tensor:
        """Return the causal graph from the jacobian using the threshold to determine edge existence.

        Args:
            x: Data variables input to the model; shape: (Batch Size, Total Nodes, 1)
            threshold: Tensor of shape (K,) which transforms the jacobian to generate K different graph predictions.
            mode: Mode to aggregate the causal graphs; options: ["mean", "median"]

        Return:
            pred_graph: Predicted graphs as per the casusal discovery method: Expected Shape: (Total Nodes, Total Nodes, K)
        """

        continuous_graph = self.get_aggregated_causal_graph(x, mode=mode)
        return (continuous_graph.unsqueeze(-1) >= threshold).long()
