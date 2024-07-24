import torch


def find_leaf_nodes(graph: torch.Tensor) -> torch.Tensor:
    """Finds the leaf nodes of the graph.

    The output is a binary tensor.

    Args:
        graph: Graph of shape (batch_size, num_nodes, num_nodes)
    Returns:
        leaf_nodes_probs: expected shape (batch_size, num_nodes)
    """

    find_leaf_scores = torch.sum(graph, dim=-2).float()
    leaf_nodes_probs = torch.zeros_like(find_leaf_scores)
    leaf_nodes_probs[find_leaf_scores == 0.0] = 1

    return leaf_nodes_probs


def remove_leaf_nodes_in_graph(graph: torch.Tensor, ind_leaves: torch.Tensor) -> torch.Tensor:
    """Removes the leaf nodes from the graphs

    Args:
        graph: Graph of shape (batch_size, num_nodes, num_nodes)
        ind_leaves: Indices of the leaf nodes of shape (batch_size,)
    Returns:
        graph_up: Graph with the leaf nodes removed of shape (batch_size, num_nodes - 1, num_nodes - 1)
    """

    graph_up = torch.zeros(graph.shape[0], graph.shape[1] - 1, graph.shape[2] - 1)

    for k in range(graph.shape[0]):
        graph_trans = torch.cat([graph[k, :, : ind_leaves[k]], graph[k, :, ind_leaves[k] + 1 :]], -1)
        graph_trans = torch.cat([graph_trans[: ind_leaves[k], :], graph_trans[ind_leaves[k] + 1 :, :]], -2)
        graph_up[k] = graph_trans

    return graph_up.to(graph.device)


def remove_leaf_nodes_in_data(train_X, ind_leaves: torch.Tensor) -> torch.Tensor:
    """Removes the leaf nodes from the graphs and the data.

    Args:
        train_X: Data of shape (batch_size, num_samples, num_nodes)
        ind_leaves: Indices of the leaf nodes of shape (batch_size,)
    Returns:
        train_X_up: Data with the leaf nodes removed of shape (batch_size, num_samples, num_nodes - 1)

    """
    if len(train_X.shape) == 3:
        train_X_up = torch.zeros(train_X.shape[0], train_X.shape[1], train_X.shape[2] - 1)
    elif len(train_X.shape) == 4:
        train_X_up = torch.zeros(train_X.shape[0], train_X.shape[1], train_X.shape[2] - 1, train_X.shape[3])
    else:
        raise ValueError(
            "train_X must be of shape (batch_size, num_samples, num_nodes) or (batch_size, num_samples, num_nodes, num_features)"
        )

    for k in range(train_X.shape[0]):
        train_X_trans = torch.cat([train_X[k, :, : ind_leaves[k]], train_X[k, :, ind_leaves[k] + 1 :]], 1)
        train_X_up[k] = train_X_trans

    return train_X_up.to(train_X.device)


def remove_leaf_nodes(graph: torch.Tensor, train_X, ind_leaves: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Removes the leaf nodes from the graphs and the data.

    Args:
        graph: Graph of shape (batch_size, num_nodes, num_nodes)
        train_X: Data of shape (batch_size, num_samples, num_nodes)
        ind_leaves: Indices of the leaf nodes of shape (batch_size,)
    Returns:
        graph_up: Graph with the leaf nodes removed of shape (batch_size, num_nodes - 1, num_nodes - 1)
        train_X_up: Data with the leaf nodes removed of shape (batch_size, num_samples, num_nodes - 1)

    """

    graph_up = remove_leaf_nodes_in_graph(graph, ind_leaves)
    train_X_up = remove_leaf_nodes_in_data(train_X, ind_leaves)
    return graph_up, train_X_up


def decreasing_sig_to_perm(list_decreasing_order: list) -> torch.Tensor:
    """Converts a list of decreasing topological order to a permutation"""

    target_size = len(list_decreasing_order) + 1
    perm = torch.zeros(target_size, dtype=torch.long)
    original_list = list(range(target_size))
    for i, ind in enumerate(list_decreasing_order):
        perm[len(list_decreasing_order) - i] = original_list[ind]
        original_list.pop(ind)

    perm[0] = original_list[0]
    return perm


def vote_leaf_predicition(argmax_ind: list[int]) -> int:
    """Returns the most frequent leaf node prediction"""
    return max(argmax_ind, key=argmax_ind.count)


def graph_to_perm_torch(graph: torch.Tensor) -> torch.Tensor:
    """Returns the permutation from the sequential leaves"""

    num_nodes = graph.shape[-1]

    if len(graph.shape) == 2:
        graph = graph.unsqueeze(0)

    perm_decreasing_order = torch.zeros((graph.shape[0], num_nodes - 1), dtype=torch.long)

    for k in range(num_nodes - 1):
        probs_leaves = find_leaf_nodes(graph)  # (batch_size, num_nodes)
        argmax_ind = torch.argmax(probs_leaves, dim=-1)  # (batch_size)
        perm_decreasing_order[:, k] = argmax_ind
        graph = remove_leaf_nodes_in_graph(graph, argmax_ind)

    perm_final = torch.zeros((graph.shape[0], num_nodes), dtype=torch.long)
    for k in range(graph.shape[0]):
        curr_list_decreasing_order = perm_decreasing_order[k].tolist()
        perm_final[k] = decreasing_sig_to_perm(curr_list_decreasing_order)

    return perm_final
