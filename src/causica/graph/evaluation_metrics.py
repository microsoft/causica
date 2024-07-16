import torch

from causica.triangular_transformations import unfill_triangular


def adjacency_precision_recall(graph1: torch.Tensor, graph2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the precision and recall of edge existence for two adjacency matrices."""
    vec1 = torch.abs(_to_vector(graph1)) > 0
    vec2 = torch.abs(_to_vector(graph2)) > 0

    correspondence = (vec1 & vec2).sum()

    if (vec1_sum := vec1.sum()) != 0:
        recall = correspondence / vec1_sum
    else:
        recall = torch.tensor(0.0, device=graph1.device)

    if (vec2_sum := vec2.sum()) != 0:
        precision = correspondence / vec2_sum
    else:
        precision = torch.tensor(0.0, device=graph1.device)

    return precision, recall


def adjacency_f1(graph1: torch.Tensor, graph2: torch.Tensor) -> torch.Tensor:
    """Evaluate the f1 score of edge existence for two adjacency matrices."""
    return f1_score(*adjacency_precision_recall(graph1, graph2))


def orientation_precision_recall(graph1: torch.Tensor, graph2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the precision and recall of edge orientation for two adjacency matrices."""
    vec1 = _to_vector(graph1)
    vec2 = _to_vector(graph2)
    non_zero_vec1 = vec1 != 0
    non_zero_vec2 = vec2 != 0

    if (non_zero_vec1_sum := non_zero_vec1.sum()) != 0:
        recall = ((vec1 == vec2) & non_zero_vec1).sum() / non_zero_vec1_sum
    else:
        recall = torch.tensor(0.0, device=graph1.device)

    if (non_zero_vec2_sum := non_zero_vec2.sum()) != 0:
        precision = ((vec1 == vec2) & non_zero_vec2).sum() / non_zero_vec2_sum
    else:
        precision = torch.tensor(0.0, device=graph1.device)

    return precision, recall


def orientation_fallout_recall(graph1: torch.Tensor, graph2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the precision and recall of edge orientation for two adjacency matrices."""
    vec1 = _to_vector(graph1)
    vec2 = _to_vector(graph2)
    non_zero_vec1 = vec1 != 0
    zero_vec1 = vec1 == 0
    non_zero_vec2 = vec2 != 0

    if (non_zero_vec1_sum := non_zero_vec1.sum()) != 0:
        recall = ((vec1 == vec2) & non_zero_vec1).sum() / non_zero_vec1_sum
    else:
        recall = torch.tensor(0.0, device=graph1.device)

    if (zero_vec1_sum := zero_vec1.sum()) != 0:
        fallout = (non_zero_vec2 & zero_vec1).sum() / zero_vec1_sum
    else:
        fallout = torch.tensor(0.0, device=graph1.device)

    return fallout, recall


def orientation_f1(graph1: torch.Tensor, graph2: torch.Tensor) -> torch.Tensor:
    """Evaluate the f1 score of edge existence for two adjacency matrices."""
    return f1_score(*orientation_precision_recall(graph1, graph2))


def f1_score(precision: torch.Tensor, recall: torch.Tensor) -> torch.Tensor:
    """Calculate f1 score from precision and recall."""
    if torch.abs(denominator := precision + recall) < 1e-8:
        return torch.tensor(0.0, device=precision.device)
    return 2.0 * precision * recall / denominator


def _to_vector(graph: torch.Tensor) -> torch.Tensor:
    """
    Convert an adjacency matrix to a vector of length n(n-1)/2.

    There is a 0 for no edge, -1 or 1 for a single edge and 2 for both edges between a pair of nodes
    """
    lower_tri = unfill_triangular(graph, upper=False)
    upper_tri = unfill_triangular(graph, upper=True)
    diff = upper_tri - lower_tri
    return diff + (1 - torch.abs(diff)) * (upper_tri + lower_tri)
