import numpy as np


def is_there_adjacency(adj_matrix):
    """
    If input is (n,n), this returns a 1D array of size n*(n-1)/2 indicating whether each edge is present or not (not
    considering orientation).
    """
    mask = np.tri(adj_matrix.shape[0], k=-1, dtype=bool)
    is_there_forward = adj_matrix[mask].astype(bool)
    is_there_backward = (adj_matrix.T)[mask].astype(bool)
    return is_there_backward | is_there_forward


def get_adjacency_type(adj_matrix):
    """
    If input is (n,n), this returns a 1D array of size n*(n-1)/2 indicating the type of each edge (that is, 0 if
    there is no edge, 1 if it is forward, -1 if it is backward and 2 if it is in both directions or undirected).
    """

    def aux(f, b):
        if f and b:
            return 2
        elif f and not b:
            return 1
        elif not f and b:
            return -1
        elif not f and not b:
            return 0

    mask = np.tri(adj_matrix.shape[0], k=-1, dtype=bool)
    is_there_forward = adj_matrix[mask].astype(bool)
    is_there_backward = (adj_matrix.T)[mask].astype(bool)
    out = np.array([aux(f, b) for (f, b) in zip(is_there_forward, is_there_backward)])
    return out


def is_there_edge(adj_matrix):
    """
    If input is (n,n), this returns a 1D array of size n*(n-1) indicating whether each edge is present or not (considering orientation).
    """
    mask = (np.ones_like(adj_matrix) - np.eye(adj_matrix.shape[0])).astype(bool)
    return adj_matrix[mask].astype(bool)


def edge_prediction_metrics(adj_matrix_true, adj_matrix_predicted, adj_matrix_mask=None):
    """
    Computes the edge predicition metrics when the ground truth DAG (or CPDAG) is adj_matrix_true and the predicted one
    is adj_matrix_predicted. Both are numpy arrays.
    adj_matrix_mask is the mask matrix for adj_matrices, that indicates which subgraph is partially known in the ground
    truth. 0 indicates the edge is unknwon, and 1 indicates that the edge is known.
    """
    if adj_matrix_mask is None:
        adj_matrix_mask = np.ones_like(adj_matrix_true)

    assert ((adj_matrix_true == 0) | (adj_matrix_true == 1)).all()
    assert ((adj_matrix_predicted == 0) | (adj_matrix_predicted == 1)).all()
    results = {}

    # Computing adjacency precision/recall
    v_mask = is_there_adjacency(adj_matrix_mask)
    # v_mask is true only if we know about at least one direction of the edge
    v_true = is_there_adjacency(adj_matrix_true) & v_mask
    v_predicted = is_there_adjacency(adj_matrix_predicted) & v_mask
    recall = (v_true & v_predicted).sum() / (v_true.sum())
    precision = (v_true & v_predicted).sum() / (v_predicted.sum()) if v_predicted.sum() != 0 else 0.0
    fscore = 2 * recall * precision / (precision + recall) if (recall + precision) != 0 else 0.0
    results["adjacency_recall"] = recall
    results["adjacency_precision"] = precision
    results["adjacency_fscore"] = fscore

    # Computing orientation precision/recall
    v_mask = is_there_adjacency(adj_matrix_mask)
    v_true = get_adjacency_type(adj_matrix_true) * v_mask
    v_predicted = get_adjacency_type(adj_matrix_predicted) * v_mask
    recall = ((v_true == v_predicted) & (v_true != 0)).sum() / (v_true != 0).sum()
    precision = (
        ((v_true == v_predicted) & (v_predicted != 0)).sum() / (v_predicted != 0).sum()
        if (v_predicted != 0).sum() != 0
        else 0.0
    )
    fscore = 2 * recall * precision / (precision + recall) if (recall + precision) != 0 else 0.0
    results["orientation_recall"] = recall
    results["orientation_precision"] = precision
    results["orientation_fscore"] = fscore

    # Computing causal accuracy (as in https://github.com/TURuibo/Neuropathic-Pain-Diagnosis-Simulator/blob/master/source/CauAcc.py)
    v_mask = is_there_edge(adj_matrix_mask)
    # v_mask is true only if we know about the edge
    v_true = is_there_edge(adj_matrix_true) & v_mask
    v_predicted = is_there_edge(adj_matrix_predicted) & v_mask
    causal_acc = (v_true & v_predicted).sum() / v_true.sum()
    results["causal_accuracy"] = causal_acc

    # Compute SHD and number of nonzero edges
    results["shd"] = _shd(adj_matrix_true, adj_matrix_predicted)
    results["nnz"] = adj_matrix_predicted.sum()
    return results


def _shd(adj_true, adj_pred):
    """
    Computes Structural Hamming Distance as E+M+R, where E is the number of extra edges,
    M the number of missing edges, and R the number os reverse edges.
    """
    E, M, R = 0, 0, 0
    for i in range(adj_true.shape[0]):
        for j in range(adj_true.shape[0]):
            if j <= i:
                continue
            if adj_true[i, j] == 1 and adj_true[j, i] == 0:
                if adj_pred[i, j] == 0 and adj_pred[j, i] == 0:
                    M += 1
                elif adj_pred[i, j] == 0 and adj_pred[j, i] == 1:
                    R += 1
                elif adj_pred[i, j] == 1 and adj_pred[j, i] == 1:
                    E += 1
            if adj_true[i, j] == 0 and adj_true[j, i] == 1:
                if adj_pred[i, j] == 0 and adj_pred[j, i] == 0:
                    M += 1
                elif adj_pred[i, j] == 1 and adj_pred[j, i] == 0:
                    R += 1
                elif adj_pred[i, j] == 1 and adj_pred[j, i] == 1:
                    E += 1
            if adj_true[i, j] == 0 and adj_true[j, i] == 0:
                E += adj_pred[i, j] + adj_pred[j, i]
    return E + M + R


def edge_prediction_metrics_multisample(
    adj_matrix_true, adj_matrices_predicted, adj_matrix_mask=None, compute_mean=True
):
    """
    Computes the edge predicition metrics when the ground truth DAG (or CPDAG) is adj_matrix_true and many predicted
    adjacencies are sampled from the distribution. Both are numpy arrays, adj_matrix_true has shape (n, n) and
    adj_matrices_predicted has shape (M, n, n), where M is the number of matrices sampled.
    """
    results = {}
    for i in range(adj_matrices_predicted.shape[0]):
        adj_matrix_predicted = adj_matrices_predicted[i, :, :]  # (n, n)
        results_local = edge_prediction_metrics(adj_matrix_true, adj_matrix_predicted, adj_matrix_mask=adj_matrix_mask)
        for k, result in results_local.items():
            if k not in results:
                results[k] = []
            results[k].append(result)

    if compute_mean:
        return {key: np.mean(val) for key, val in results.items()}
    return results
