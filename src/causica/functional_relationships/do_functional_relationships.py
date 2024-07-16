import torch
from tensordict import TensorDict

from causica.functional_relationships.functional_relationships import FunctionalRelationships
from causica.functional_relationships.temporal_functional_relationships import TemporalEmbedFunctionalRelationships


class DoFunctionalRelationships(FunctionalRelationships):
    """
    A `FunctionalRelationship` that one can "do", i.e. condition nodes and cut the links to their parents.

    The do intervention can be a single intervention (i.e. empty batch shape) or a batch of interventions. The batch
    shape of the do tensordict batches the interventions and the original functional relationship is broadcast across
    the batch shape of the do tensordict.
    """

    def __init__(self, func: FunctionalRelationships, do: TensorDict, submatrix: torch.Tensor) -> None:
        """
        Args:
            func: The unintervened functional relationships
            do: the nodes on which to intervene. If the do has a batch shape, then the functional relationship will be
                broadcast to that batch shape.
            submatrix: the submatrix that the unintervened nodes represent in the larger graph
        """
        if not all(val.ndim >= 1 for val in do.values()):
            raise ValueError("Intervention is only supported for at least vector valued interventions")
        if len({val.ndim for val in do.values()}) > 1:
            raise ValueError("Intervention must have the same number of dimensions for all variables")
        if isinstance(func, TemporalEmbedFunctionalRelationships):
            raise NotImplementedError("Intervention is not yet implemented for temporal functions.")

        new_shapes = {key: shape for key, shape in func.shapes.items() if key not in do.keys()}
        super().__init__(new_shapes, batch_shape=do.batch_size + func.batch_shape)

        self.func = func
        self.do = do  # dict of key to vectors
        self.submatrix = submatrix

        self.do_nodes_mask = torch.tensor(
            [(name in self.do.keys()) for name in self.func.shapes.keys()], dtype=torch.bool
        )

    def pad_intervened_graphs(self, graphs: torch.Tensor) -> torch.Tensor:
        """
        Pad the intervened graph with the unintervened nodes.

        Args:
            graphs: Weighted adjacency matrix, size batch_size_g + (do_func_n, do_func_n)
        Returns:
            A tensor of shape batch_shape_g + (func_n, func_n)
        """

        num_nodes = self.func.tensor_to_td.num_keys
        target_shape = graphs.shape[:-2] + (num_nodes, num_nodes)

        output_graphs = torch.zeros(target_shape, dtype=graphs.dtype, device=graphs.device)
        assign_submatrix(output_graphs, graphs, ~self.do_nodes_mask, ~self.do_nodes_mask)
        assign_submatrix(output_graphs, self.submatrix, self.do_nodes_mask, ~self.do_nodes_mask)
        return output_graphs

    def forward(self, samples: TensorDict, graphs: torch.Tensor) -> TensorDict:
        """
        Run forward on the underlying functional relationship with the intervened nodes filled in.

        The samples are expected to have a batch shape in order: samples, functions, graphs.

        Args:
            samples: Batched inputs, size batch_size_x + batch_size_f + batch_shape_g + (concatenated_shape).
            graphs: Weighted adjacency matrix, size batch_size_g + (n, n)
        Returns:
            A tensor of shape batch_shape_x + batch_size_f + batch_shape_g + (concatenated_shape)
        """
        # add the expanded intervention values to the samples
        batch_shape_g = graphs.shape[:-2]

        do = self.do
        if do.batch_dims > 0:
            do = do[(...,) + (None,) * len(batch_shape_g)]
        expanded_do = do.expand(*samples.batch_size)

        samples_with_do = samples.clone(False).update(expanded_do)

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
    is_temporal = isinstance(func, TemporalEmbedFunctionalRelationships)
    if is_temporal:
        raise NotImplementedError("Interventions are not yet supported for temporal graphs")

    graph_ndims = 3 if is_temporal else 2
    if func.batch_shape == torch.Size((1,)):
        func.batch_shape = torch.Size()
    if interventions.ndim > 1:
        raise ValueError("Interventions must be at most a single batch of interventions")
    if graph.ndim > graph_ndims + 1:
        raise ValueError("Graph must be at most a single batch of graphs")
    if interventions.batch_dims > 0 and len(func.batch_shape) > 0:
        raise ValueError("Cannot intervene on a batch of interventions and a batch of functional relationships")

    node_names = list(func.shapes.keys())
    do_nodes_mask = torch.zeros(len(node_names), dtype=torch.bool)
    for i, name in enumerate(node_names):
        if name in interventions.keys():
            do_nodes_mask[i] = 1

    do_graph = graph[..., ~do_nodes_mask, :][..., :, ~do_nodes_mask]
    submatrix = graph[..., do_nodes_mask, :][..., :, ~do_nodes_mask]

    # Expanding graph if interventions or functions are batched
    if do_graph.ndim == graph_ndims and interventions.batch_dims > 0 or len(func.batch_shape) > 0:
        do_graph = do_graph.unsqueeze(0)
        submatrix = submatrix.unsqueeze(0)
    # Expanding interventions if graph is batched and functions are not
    if do_graph.ndim == graph_ndims + 1 and interventions.batch_dims == 0 and len(func.batch_shape) == 0:
        interventions = interventions.unsqueeze(0)

    return DoFunctionalRelationships(func, interventions, submatrix), do_graph
