import numpy as np

from causica.utils.nri_utils import edge_prediction_metrics


def test_recover_exactly():
    A = np.triu(np.ones((3, 3))) - np.eye(3)
    B = np.triu(np.ones((3, 3))) - np.eye(3)
    results = edge_prediction_metrics(A, B)
    assert results["adjacency_recall"] == 1.0
    assert results["adjacency_precision"] == 1.0
    assert results["adjacency_fscore"] == 1.0
    assert results["orientation_recall"] == 1.0
    assert results["orientation_precision"] == 1.0
    assert results["orientation_fscore"] == 1.0
    assert results["causal_accuracy"] == 1.0


def test_recover_none():
    A = np.triu(np.ones((3, 3))) - np.eye(3)
    B = np.zeros((3, 3))
    results = edge_prediction_metrics(A, B)
    assert results["adjacency_recall"] == 0.0
    assert results["adjacency_precision"] == 0.0
    assert results["adjacency_fscore"] == 0.0
    assert results["orientation_recall"] == 0.0
    assert results["orientation_precision"] == 0.0
    assert results["orientation_fscore"] == 0.0
    assert results["causal_accuracy"] == 0.0


def test_recover_adj_not_orient():
    A = np.triu(np.ones((3, 3))) - np.eye(3)
    B = np.tril(np.ones((3, 3))) - np.eye(3)
    results = edge_prediction_metrics(A, B)
    assert results["adjacency_recall"] == 1.0
    assert results["adjacency_precision"] == 1.0
    assert results["adjacency_fscore"] == 1.0
    assert results["orientation_recall"] == 0.0
    assert results["orientation_precision"] == 0.0
    assert results["orientation_fscore"] == 0.0
    assert results["causal_accuracy"] == 0.0


def test_recover_adj_some_orient():
    A = np.triu(np.ones((3, 3))) - np.eye(3)
    B = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    results = edge_prediction_metrics(A, B)
    assert results["adjacency_recall"] == 2.0 / 3.0
    assert results["adjacency_precision"] == 1.0
    assert results["adjacency_fscore"] == 4.0 / 5.0
    assert results["orientation_recall"] == 2.0 / 3.0
    assert results["orientation_precision"] == 1.0
    assert results["orientation_fscore"] == 4.0 / 5.0
    assert results["causal_accuracy"] == 2.0 / 3.0


def test_recover_extra():
    A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    B = np.triu(np.ones((3, 3))) - np.eye(3)
    results = edge_prediction_metrics(A, B)
    assert results["adjacency_recall"] == 1.0
    assert results["adjacency_precision"] == 2.0 / 3.0
    assert results["adjacency_fscore"] == 4.0 / 5.0
    assert results["orientation_recall"] == 1.0
    assert results["orientation_precision"] == 2.0 / 3.0
    assert results["orientation_fscore"] == 4.0 / 5.0
    assert results["causal_accuracy"] == 1.0


def test_shd():
    A = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
    B = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
    assert edge_prediction_metrics(A, B)["shd"] == 0.0

    A = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
    B = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
    assert edge_prediction_metrics(A, B)["shd"] == 1.0

    A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    B = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
    assert edge_prediction_metrics(A, B)["shd"] == 1.0
