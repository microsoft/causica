import torch
from torch import Tensor


def confusion_matrix(true_graph: torch.Tensor, pred_graph: torch.Tensor) -> Tensor:
    """Evaluate metrics for the difference between a true and predicted graph.

    Args:
        true_graph: The true graph; Tensor of shape (state_size, state_size).
        pred_graph: The predicted graph; Tensor of shape (state_size, state_size).

    Returns:
        confusion matrix: Tensor of shape (2, 2) reprsenting confusion matrix:
            Entry (0, 0) is the number of true negatives.
            Entry (0, 1) is the number of false positives.
            Entry (1, 0) is the number of false negatives.
            Entry (1, 1) is the number of true positives.
    """
    vec1 = torch.abs(true_graph) > 0
    vec2 = torch.abs(pred_graph) > 0

    tp = (vec1 & vec2).sum()
    tn = ((~vec1) & (~vec2)).sum()
    fp = ((~vec1) & (vec2)).sum()
    fn = ((vec1) & (~vec2)).sum()

    return torch.stack([tn, fp, fn, tp]).view(2, 2)


def confusion_matrix_batched(true_graph: torch.Tensor, pred_graphs: torch.Tensor) -> Tensor:
    """Evaluate metrics for the difference between a true and a set of predicted graphs.

    Args:
        true_graph: The true graph; Tensor of shape (state_size, state_size).
        pred_graphs: Set of predicted graphs; Tensor of shape (batch_size, state_size, state_size).

    Returns:
        confusion matrix: Tensor of shape (2, 2) reprsenting confusion matrix acrpss all predicted graphs.
    """
    batched_confusion_matrix = torch.vmap(confusion_matrix, in_dims=(None, 0), out_dims=0)(true_graph, pred_graphs)
    return torch.sum(batched_confusion_matrix, dim=0)


def true_positive_rate(tp: int, fn: int) -> float:
    """Compute true positive rate given number of true positives and false negatives.

    Args:
        tp: Number of true positives.
        fn: Number of false negatives.

    Returns:
        True positive rate; -1 if tp + fn == 0.
    """
    return -1 if tp + fn == 0 else tp / (tp + fn)


def false_discovery_rate(tp: int, fp: int) -> float:
    """Compute false discovery rate given number of true positives and false positives.

    Args:
        tp: Number of true positives.
        fp: Number of false positives.

    Returns:
        False discovery rate; -1 if tp + fp == 0.
    """
    return -1 if tp + fp == 0 else fp / (tp + fp)


def f1_score(tp: int, fp: int, fn: int) -> float:
    """Compute F1 score given number of true positives, false positives, and false negatives.

    Args:
        tp: Number of true positives.
        fp: Number of false positives.
        fn: Number of false negatives.

    Returns:
        F1 score; -1 if tp + fp + fn == 0.
    """
    return -1 if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
