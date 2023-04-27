import torch
from tensordict import TensorDict

from causica.functional_relationships.functional_relationships import FunctionalRelationships


class DoFunctionalRelationships(FunctionalRelationships):
    """
    A `FunctionalRelationship` that one can "do", i.e. condition nodes and cut the links to their parents.
    """

    def __init__(self, func: FunctionalRelationships, do: TensorDict, submatrix: torch.Tensor) -> None:
        """
        Args:
            func: The unintervened functional relationships
            do: the nodes on which to intervene
            submatrix: the submatrix that the unintervened nodes represent in the larger graph
        """
        assert all(val.ndim == 1 for val in do.values()), "Intervention is only supported for 1 vector per variable"

        new_variables = {key: value for key, value in func.variables.items() if key not in do.keys()}
        super().__init__(new_variables)

        self.func = func
        self.do = do  # dict of key to vectors
        self.submatrix = submatrix

        self.do_nodes_mask = torch.tensor(
            [(name in self.do.keys()) for name in self.func.variables.keys()], dtype=torch.bool
        )

    def pad_intervened_graphs(self, graphs: torch.Tensor) -> torch.Tensor:
        """
        Pad the intervened graph with the unintervened nodes.

        Args:
            graphs: Weighted adjacency matrix, size batch_size_g + (do_func_n, do_func_n)
        Returns:
            A tensor of shape batch_shape_g + (func_n, func_n)
        """
        target_shape = graphs.shape[:-2] + (self.func.num_nodes, self.func.num_nodes)

        output_graphs = torch.zeros(target_shape, dtype=graphs.dtype, device=graphs.device)
        assign_submatrix(output_graphs, graphs, ~self.do_nodes_mask, ~self.do_nodes_mask)
        assign_submatrix(output_graphs, self.submatrix, self.do_nodes_mask, ~self.do_nodes_mask)
        return output_graphs

    def forward(self, samples: TensorDict, graphs: torch.Tensor) -> TensorDict:
        """
        Args:
            samples: Batched inputs, size batch_size_x + (processed_dim_all).
            graphs: Weighted adjacency matrix, size batch_size_g + (n, n)
        Returns:
            A tensor of shape batch_shape_x + batch_shape_g + (processed_dim_all)
        """
        # add the expanded intervention values to the samples
        samples_with_do = samples.update(self.do.expand(*samples.batch_size))

        # create the full graphs
        graphs = self.pad_intervened_graphs(graphs)

        forward = self.func.forward(samples_with_do, graphs)
        return forward.select(*(key for key in forward.keys() if key not in self.do.keys()), inplace=True)


def assign_submatrix(A: torch.Tensor, B: torch.Tensor, x_mask: torch.Tensor, y_mask: torch.Tensor) -> None:
    """
    Assign `B` to a submatrix of `A`. The matrix `A` is changed in place.
    Args:
        A: tensor with >=2 dimensions `[*b, x, y]`
        B: tensor with dimensions `[*b, x', y']` where `x'<=x` and `y'<=y`
        x_mask: boolean tensor of shape `[*b, x]` indicating the rows of A that are to be updated. The sum
            of x_mask should equal `x'`
        y_mask: boolean tensor of shape `[*b, y]` indicating the columns of A that are to be updated. The sum
            of y_mask should equal `y'`
    """
    assign_mask = torch.ones_like(A, dtype=torch.bool)
    assign_mask[..., ~x_mask, :] = 0
    assign_mask[..., :, ~y_mask] = 0
    A[assign_mask] = B.flatten()


def create_do_functional_relationship(
    interventions: TensorDict, func: FunctionalRelationships, graph: torch.Tensor
) -> tuple[DoFunctionalRelationships, torch.Tensor]:
    """
    Given a set of interventions, `FunctionalRelationships` and a graph, create a `DoFunctionalRelationships` and an intervened graph.

    Args:
        interventions: the nodes and their intervention values
        func: the functional relationship of the unintervened SEM
        graph: the unintervened graph shape: [..., num_nodes, num_nodes]
    Return:
        A tuple with the intervened functional relationship and the intervened graph
    """
    node_names = list(func.variables.keys())
    do_nodes_mask = torch.zeros(len(node_names), dtype=torch.bool)
    for i, name in enumerate(node_names):
        if name in interventions.keys():
            do_nodes_mask[i] = 1

    do_graph = graph[..., ~do_nodes_mask, :][..., :, ~do_nodes_mask]
    submatrix = graph[..., do_nodes_mask, :][..., :, ~do_nodes_mask]
    return DoFunctionalRelationships(func, interventions, submatrix), do_graph
