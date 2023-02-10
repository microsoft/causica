import pytest
import torch

from causica.graph.evaluation_metrics import (
    adjacency_f1,
    adjacency_precision_recall,
    orientation_f1,
    orientation_precision_recall,
)

MATRICES = [
    torch.tensor([[0, 1, 1], [0, 0, 0], [0, 0, 0]]),
    torch.tensor([[0, 0, 0], [1, 0, 0], [1, 0, 0]]),
    torch.tensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]]),
    torch.tensor([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
]

ADJACENCY_METRICS = [
    {"adjacency_recall": 1.0, "adjacency_precision": 1.0, "adjacency_fscore": 1.0},
    {
        "adjacency_recall": 1.0,
        "adjacency_precision": 1 / 2,
        "adjacency_fscore": 2 / 3,
    },
    {
        "adjacency_recall": 1 / 2,
        "adjacency_precision": 1.0,
        "adjacency_fscore": 2 / 3,
    },
]

ORIENTATION_METRICS = [
    {
        "orientation_recall": 1.0,
        "orientation_precision": 1.0,
        "orientation_fscore": 1.0,
    },
    {
        "orientation_recall": 1.0,
        "orientation_precision": 1 / 2,
        "orientation_fscore": 2 / 3,
    },
    {
        "orientation_recall": 1 / 2,
        "orientation_precision": 1.0,
        "orientation_fscore": 2 / 3,
    },
    {
        "orientation_recall": 0.0,
        "orientation_precision": 0.0,
        "orientation_fscore": 0.0,
    },
]

TEST_COMBINATIONS = [
    (MATRICES[0], MATRICES[0], ADJACENCY_METRICS[0], ORIENTATION_METRICS[0]),
    (MATRICES[1], MATRICES[0], ADJACENCY_METRICS[0], ORIENTATION_METRICS[3]),
    (MATRICES[2], MATRICES[0], ADJACENCY_METRICS[1], ORIENTATION_METRICS[3]),
    (MATRICES[3], MATRICES[0], ADJACENCY_METRICS[1], ORIENTATION_METRICS[1]),
    (MATRICES[0], MATRICES[1], ADJACENCY_METRICS[0], ORIENTATION_METRICS[3]),
    (MATRICES[1], MATRICES[1], ADJACENCY_METRICS[0], ORIENTATION_METRICS[0]),
    (MATRICES[2], MATRICES[1], ADJACENCY_METRICS[1], ORIENTATION_METRICS[1]),
    (MATRICES[3], MATRICES[1], ADJACENCY_METRICS[1], ORIENTATION_METRICS[3]),
    (MATRICES[0], MATRICES[2], ADJACENCY_METRICS[2], ORIENTATION_METRICS[3]),
    (MATRICES[1], MATRICES[2], ADJACENCY_METRICS[2], ORIENTATION_METRICS[2]),
    (MATRICES[2], MATRICES[2], ADJACENCY_METRICS[0], ORIENTATION_METRICS[0]),
    (MATRICES[3], MATRICES[2], ADJACENCY_METRICS[0], ORIENTATION_METRICS[3]),
    (MATRICES[0], MATRICES[3], ADJACENCY_METRICS[2], ORIENTATION_METRICS[2]),
    (MATRICES[1], MATRICES[3], ADJACENCY_METRICS[2], ORIENTATION_METRICS[3]),
    (MATRICES[2], MATRICES[3], ADJACENCY_METRICS[0], ORIENTATION_METRICS[3]),
    (MATRICES[3], MATRICES[3], ADJACENCY_METRICS[0], ORIENTATION_METRICS[0]),
]


@pytest.mark.parametrize("graph1, graph2, adj_metrics, orientation_metrics", TEST_COMBINATIONS)
def test_graph_eval_metrics(graph1, graph2, adj_metrics, orientation_metrics):

    assert orientation_precision_recall(graph1, graph2) == (
        orientation_metrics["orientation_precision"],
        orientation_metrics["orientation_recall"],
    )
    assert orientation_f1(graph1, graph2) == orientation_metrics["orientation_fscore"]
    assert adjacency_precision_recall(graph1, graph2) == (
        adj_metrics["adjacency_precision"],
        adj_metrics["adjacency_recall"],
    )
    assert adjacency_f1(graph1, graph2) == adj_metrics["adjacency_fscore"]
