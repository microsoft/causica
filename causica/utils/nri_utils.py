from itertools import product
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch
import torch.distributions as td
from sklearn import metrics

from ..datasets.dataset import CausalDataset


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


def edge_prediction_metrics(
    adj_matrix_true, adj_matrix_predicted, adj_matrix_mask=None, adj_matrix_predicted_prob: Optional[np.ndarray] = None
):
    """
    Computes the edge predicition metrics when the ground truth DAG (or CPDAG) is adj_matrix_true and the predicted one
    is adj_matrix_predicted. Both are numpy arrays.
    adj_matrix_mask is the mask matrix for adj_matrices, that indicates which subgraph is partially known in the ground
    truth. 0 indicates the edge is unknwon, and 1 indicates that the edge is known.
    If adj_matrix_predicted_prob is provided and it is generated from aggregated temporal matrix, then AUROC is computed.
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

    # Compute AUROC
    if adj_matrix_predicted_prob is not None:
        assert adj_matrix_predicted_prob.ndim == 2, "AUROC metric only supports 2D matrix"
        # get rid of the diagonal
        adj_matrix_true_fl = adj_matrix_true.flatten()
        adj_pred_fl = adj_matrix_predicted_prob.flatten()
        auroc = metrics.roc_auc_score(adj_matrix_true_fl, adj_pred_fl)
        results["auroc"] = auroc
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
    adj_matrix_true,
    adj_matrices_predicted,
    adj_matrix_mask=None,
    compute_mean=True,
    adj_pred_prob: Optional[np.ndarray] = None,
):
    """
    Computes the edge predicition metrics when the ground truth DAG (or CPDAG) is adj_matrix_true and many predicted
    adjacencies are sampled from the distribution. Both are numpy arrays, adj_matrix_true has shape (n, n) and
    adj_matrices_predicted has shape (M, n, n), where M is the number of matrices sampled. adj_pred_prob has shape [n,n]
    and contains the bernoulli prob, which is used to compute auroc.
    """
    results: dict = {}
    for i in range(adj_matrices_predicted.shape[0]):
        adj_matrix_predicted = adj_matrices_predicted[i, :, :]  # (n, n)
        results_local = edge_prediction_metrics(
            adj_matrix_true,
            adj_matrix_predicted,
            adj_matrix_mask=adj_matrix_mask,
            adj_matrix_predicted_prob=adj_pred_prob,
        )
        for k, result in results_local.items():
            if k not in results:
                results[k] = []
            results[k].append(result)

    if compute_mean:
        return {key: np.mean(val) for key, val in results.items()}
    return results


def make_temporal_adj_matrix_compatible(
    temporal_adj_matrix: np.ndarray, adj_matrix_2: np.ndarray, is_static: bool, adj_matrix_2_lag: Optional[int] = None
):
    """
    This method will transform the temporal adjacency matrix and adj_matrix_2 to have compatible lags. is_static determines whether
    the adj_matrix_2 has a static format similar to the one inferred by fold_time_deci. When is_static=True, adj_matrix_2_lag must be specified
    so that we can compare the lag of temporal_adj_matrix to adj_matrix_2. Then pad 0 to corresponding matrix with smaller lag.
    If is_static=False, then adj_matrix_2 has temporal format, where its lag can be inferred by the shape info.
    Args:
        temporal_adj_matrix: the temporal adj matrix with shape [lag1+1, from, to] or [N, lag1+1, from, to]
        adj_matrix_2: the adj matrix with shape [lag2+1, from, to] or [N, lag2+1, from, to] if is_static=False or
        [(lag2+1)*from, (lag2+1)*to] or [N, (lag2+1)*from, (lag2+1)*to] if is_static=True.
        is_static: whether the adj_matrix_2 has a static format similar to the one inferred by fold_time_deci.
        adj_matrix_2_lag: the lag of adj_matrix_2. It will be used when is_static=True.

    Returns:
        Tuple(ndarray, ndarray):
            The temporal adj matrix with shape [max(lag1,lag2)+1, from, to] or [N, max(lag1,lag2)+1, from, to]
            The adj matrix with shape [max(lag1,lag2)+1, from, to] or [N, max(lag1,lag2)+1, from, to] if is_static=False or
            [(max(lag1,lag2)+1)*from, (max(lag1,lag2)+1)*to] or [N, (max(lag1,lag2)+1)*from, (max(lag1,lag2)+1)*to] if is_static=True.

    """
    if len(temporal_adj_matrix.shape) == 3:
        temporal_adj_matrix = np.expand_dims(temporal_adj_matrix, axis=0)  # [1, lag1+1, from, to]
    lag_1 = temporal_adj_matrix.shape[1]
    node_1 = temporal_adj_matrix.shape[2]
    if is_static:
        assert adj_matrix_2_lag is not None, "When is_static=True, adj_matrix_2_lag must be specified."
        if len(adj_matrix_2.shape) == 2:
            adj_matrix_2 = np.expand_dims(adj_matrix_2, axis=0)  # [1, (lag2+1)*from, (lag2+1)*to]
        lag_2 = adj_matrix_2_lag + 1
        assert adj_matrix_2.shape[1] % lag_2 == 0, "The adj_matrix_2 has incompatible shapes with adj_matrix_2_lag."
        node_2 = adj_matrix_2.shape[1] // lag_2
        assert node_1 == node_2, "The number of nodes for two input adjacency are not consistent."
        max_lag = max(lag_1, lag_2)
        # pad temporal_adj_matrix
        compatible_temporal_adj_matrix = np.pad(temporal_adj_matrix, [(0, 0), (0, max_lag - lag_1), (0, 0), (0, 0)])
        # pad adj_matrix_2
        max_node = max_lag * node_2
        compatible_adj_2 = np.pad(
            adj_matrix_2, [(0, 0), (max_node - lag_2 * node_2, 0), (max_node - lag_2 * node_2, 0)]
        )  # [N, max_node, max_node]

    else:
        if len(adj_matrix_2.shape) == 3:
            adj_matrix_2 = np.expand_dims(adj_matrix_2, axis=0)  # [1, lag2+1, from, to]
        _, lag_2, node_2, _ = adj_matrix_2.shape
        assert node_1 == node_2, "The number of nodes for two input adjacency are not consistent."
        max_lag = max(lag_1, lag_2)
        # pad temporal_adj_matrix
        compatible_temporal_adj_matrix = np.pad(temporal_adj_matrix, [(0, 0), (0, max_lag - lag_1), (0, 0), (0, 0)])
        # pad adj_matrix_2
        compatible_adj_2 = np.pad(adj_matrix_2, [(0, 0), (0, max_lag - lag_2), (0, 0), (0, 0)])

    return np.squeeze(compatible_temporal_adj_matrix), np.squeeze(compatible_adj_2)


def convert_temporal_to_static_adjacency_matrix(
    adj_matrix: np.ndarray, conversion_type: str, fill_value: Union[float, int] = 0.0
) -> np.ndarray:
    """
    This method convert the temporal adjacency matrix to a specified type of static adjacency.
    It supports two types of conversion: "full_time" and "auto_regressive".
    The conversion type determines the connections between time steps.
    "full_time" will convert the temporal adjacency matrix to a full-time static graph, where the connections between lagged time steps are preserved.
    "auto_regressive" will convert temporal adj to a static graph that only keeps the connections to the current time step.
    E.g. a temporal adj matrix with lag 2 is [A,B,C], where A,B and C are also adj matrices. "full_time" will convert this to
    [[A,B,C],[0,A,B],[0,0,A]]. "auto_regressive" will convert this to [[0,0,C],[0,0,B],[0,0,A]].
    "fill_value" is used to specify the value to fill in the converted static adjacency matrix. Default is 0, but sometimes we may want
    other values. E.g. if we have a temporal soft prior with prior mask, then we may want to fill the converted prior mask with value 1 instead of 0,
    since the converted prior mask should never disable the blocks specifying the "arrow-against-time" in converted soft prior.
    Args:
        adj_matrix: The temporal adjacency matrix with shape [lag+1, from, to] or [N, lag+1, from, to].
        conversion_type: The conversion type. It supports "full_time" and "auto_regressive".
        fill_value: The value used to fill the static adj matrix. The default is 0.

    Returns: static adjacency matrix with shape [(lag+1)*from, (lag+1)*to] or [N, (lag+1)*from, (lag+1)*to].

    """
    assert conversion_type in [
        "full_time",
        "auto_regressive",
    ], f"The conversion_type {conversion_type} is not supported."
    if len(adj_matrix.shape) == 3:
        adj_matrix = adj_matrix[None, ...]  # [1, lag+1, num_node, num_node]
    batch_dim, n_lag, n_nodes, _ = adj_matrix.shape  # n_lag is lag+1
    if conversion_type == "full_time":
        block_fill_value = np.full((n_nodes, n_nodes), fill_value)
    else:
        block_fill_value = np.full((batch_dim, n_lag * n_nodes, (n_lag - 1) * n_nodes), fill_value)

    if conversion_type == "full_time":
        static_adj = np.sum(
            np.stack([np.kron(np.diag(np.ones(n_lag - i), k=i), adj_matrix[:, i, :, :]) for i in range(n_lag)], axis=1),
            axis=1,
        )  # [N, n_lag*from, n_lag*to]
        static_adj += np.kron(
            np.tril(np.ones((batch_dim, n_lag, n_lag)), k=-1), block_fill_value
        )  # [N, n_lag*from, n_lag*to]

    if conversion_type == "auto_regressive":
        # Flip the temporal adj and concatenate to form one block column of the static. The flipping is needed due to the
        # format of converted adjacency matrix. E.g. temporal adj [A,B,C], where A is the instant adj matrix. Then, the converted adj
        # is [[[0,0,C],[0,0,B],[0,0,A]]]. The last column is the concatenation of flipped temporal adj.
        block_column = np.flip(adj_matrix, axis=1).reshape(
            -1, n_lag * n_nodes, n_nodes
        )  # [N, (lag+1)*num_node, num_node]
        # Static graph
        static_adj = np.concatenate((block_fill_value, block_column), axis=2)  # [N, (lag+1)*num_node, (lag+1)*num_node]

    return np.squeeze(static_adj)


def compute_dag_loss(vec, num_nodes):
    """
    vec is a n*(n-1) array with the flattened adjacency matrix (without the diag).
    """
    dev = vec.device
    adj_matrix = torch.zeros(num_nodes, num_nodes, device=dev)
    mask = (torch.ones(num_nodes, num_nodes, device=dev) - torch.eye(num_nodes, device=dev)).to(bool)
    adj_matrix[mask] = vec
    return torch.abs(torch.trace(torch.matrix_exp(adj_matrix * adj_matrix)) - num_nodes)


def get_feature_indices_per_node(variables):
    """
    Returns a list in which the i-th element is a list with the features indices that correspond to the i-th node.
    For each Variable in 'variables' argument, the node is specified through the group_name field.
    """
    nodes = [v.group_name for v in variables]
    nodes_unique = sorted(set(nodes))
    if len(nodes_unique) == len(nodes):
        nodes_unique = nodes
    output = []
    for node in nodes_unique:
        output.append([i for (i, e) in enumerate(nodes) if e == node])
    return output, nodes_unique


def kl_categorical(preds, log_prior, num_atoms, eps=1e-16):
    """
    preds: [num_sims, num_edges, num_edge_types]
    log_prior: [1, 1, num_edge_types]
    """
    kl_div = preds * (torch.log(preds + eps) - log_prior)
    return kl_div.sum() / (num_atoms * preds.size(0))


def piecewise_linear(x, start, width, max_val=1):
    """
    Piecewise linear function whose value is:
        0 if x<=start
        max_val if x>=start+width
        grows linearly from 0 to max_val if start<=x<=(start+width)
    It is used to define the coefficient of the DAG-loss in NRI-MV.
    """
    return max_val * max(min((x - start) / width, 1), 0)


def enum_all_graphs(num_nodes: int, dags_only: Optional[bool] = False):
    """
    Enumerates all graphs of size num_nodes with no self-loops (all diagonals elements are strictly 0 in adj. matrix).
    Useful for computing the full posterior and true posterior.

    Args:
        num_nodes: An int specifying the number of nodes in the graphs (should be less than 6).
        dags_only: Whether to return only DAGs of size num_nodes.

    Returns: Adjacency matrices corresponding to all the graphs
    """

    assert (
        num_nodes < 6
    ), f"Enumeration of DAGs possible when No. of nodes is less than 6, received {num_nodes} instead."
    comb_ = list(product([0, 1], repeat=num_nodes * (num_nodes - 1)))  # Exclude diagonal
    comb = np.array(comb_)
    idxs_upper = np.triu_indices(num_nodes, k=1)
    idxs_lower = np.tril_indices(num_nodes, k=-1)
    output = np.zeros(comb.shape[:-1] + (num_nodes, num_nodes))
    output[..., idxs_upper[0], idxs_upper[1]] = comb[..., : (num_nodes * (num_nodes - 1)) // 2]
    output[..., idxs_lower[0], idxs_lower[1]] = comb[..., (num_nodes * (num_nodes - 1)) // 2 :]
    if dags_only:
        return output[dag_constraint(output) == 0]
    return output


def dag_constraint(A: np.ndarray):
    """
    Computes the DAG constraint based on the matrix exponential.
    Computes tr[e^(A * A)] - num_nodes.

    Args:
        A: Batch of adjacency matrices of size [batch_size, num_nodes, num_nodes]
    Returns: The DAG constraint values for each adj. matrix in the batch.
    """
    assert A.shape[-1] == A.shape[-2]

    num_nodes = A.shape[-1]
    expm_A = torch.linalg.matrix_exp(torch.from_numpy(A * A)).cpu().numpy()
    return np.trace(expm_A, axis1=-1, axis2=-2) - num_nodes


def compute_true_posterior(
    dataset: CausalDataset,
    model: Type,
    train_config_dict: Optional[Dict[str, Any]] = None,
):

    """
    Compute the true posterior for a given dataset by enumerating all the graphs and then taking the MLE estimate of the conditional parameters given the graph.

    Args:
        dataset: The dataset to compute the true posterior for, as defined by the Dataset object.
        model: A class of the posterior (of a DECI like class) which has methods:
                1. run_train which trains the MLE of the parameters
                2. log_prob which calulates the log likelihood of a dataset on the MLE of the parameters and a fixed graph.
                3. log_prior_A which calulates the prior for any given graph
        train_config_dict: Config dict for training the MLE estimators of the functional parameter.


    Returns:
        Posterior distribution over graphs (torch.distributions object)

    """

    train_data, _ = dataset.train_data_and_mask
    all_graphs = enum_all_graphs(num_nodes=train_data.shape[-1], dags_only=False)
    logits = torch.zeros(len(all_graphs))
    for i, graph in enumerate(all_graphs):
        dataset.set_adjacency_data_matrix(A=graph)
        model.run_train(dataset=dataset, train_config_dict=train_config_dict)
        train_data, _ = dataset.train_data_and_mask
        logits[i] = model.log_prob(
            X=train_data, most_likely_graph=True
        ) + model._log_prior_A(  # pylint: disable=protected-access
            torch.from_numpy(graph)
        )

    return td.categorical.Categorical(logits=logits)


def evaluate_true_posterior(
    trained_model: Type,
    saved_true_posterior: Optional[str] = None,
    dataset: Optional[CausalDataset] = None,
    model: Optional[Type] = None,
    train_config_dict: Optional[Dict[str, Any]] = None,
    metric="kl",
):

    """
    Evaluate the trained model with respect to the true posterior for a given dataset

    Args:
        trained_model: The trained DECI model to be evaluated.
        saved_true_posterior: Path to saved true posterior if the true posterior is already trained.
        dataset: The dataset to compute the true posterior for, as defined by the Dataset object. Should be specified if the true posterior is not already trained.
        model: A class of the posterior (of a DECI like class) which is used for calculating the MLE. Should be specified if the true posterior is not already trained.
        train_config_dict: Config dict for training the MLE estimators of the functional parameter.
        metric: Type of divergence to use for comparing the distributions.


    Returns:
        Divergence between the model and the true posterior (float)

    """
    if saved_true_posterior is None:
        if model is None or dataset is None:
            raise Exception
        else:
            true_posterior = compute_true_posterior(dataset=dataset, model=model, train_config_dict=train_config_dict)
    else:
        true_posterior = torch.load(saved_true_posterior)

    all_graphs = enum_all_graphs(num_nodes=trained_model.num_nodes, dags_only=False)
    logits = trained_model.var_dist_A.log_prob(value=torch.from_numpy(all_graphs).to(trained_model.device))

    approximate_posterior = td.categorical.Categorical(logits=logits)

    if metric == "kl":
        return td.kl.kl_divergence(true_posterior, approximate_posterior)
    else:
        raise NotImplementedError


def print_AR_DECI_metrics(adj_metrics: dict, is_aggregated: bool = False) -> None:
    """
    This is to print the metrics for AR-DECI.
    Args:
        adj_metrics: the adj_metrics dict
        is_aggregated: Is this evaluated with aggregated adj matrix
    """
    if is_aggregated:
        adj_fscore = np.array(adj_metrics.get("adjacency_fscore_agg", -1))
        ori_fscore = np.array(adj_metrics.get("orientation_fscore_agg", -1))
        shd = np.array(adj_metrics.get("shd_agg", -1))
        auroc = np.array(adj_metrics.get("auroc_agg", -1))
        val_likelihood = adj_metrics.get("val_likelihood", -1)
        print(
            f"adj_fscore: {adj_fscore} ori_fscore: {ori_fscore} shd: {shd} auroc: {auroc} val_likelihood: {val_likelihood}"
        )
    else:
        ori_fscore_overall = np.array(adj_metrics.get("orientation_fscore_overall_temporal", -1))
        shd_overall = np.array(adj_metrics.get("shd_overall_temporal", -1))
        ori_fscore_lag = np.array(adj_metrics.get("orientation_fscore_lag", -1))
        ori_fscore_inst = np.array(adj_metrics.get("orientation_fscore_inst", -1))
        shd_lag = np.array(adj_metrics.get("shd_lag", -1))
        shd_inst = np.array(adj_metrics.get("shd_inst", -1))
        val_likelihood = adj_metrics.get("val_likelihood", -1)
        print(f"ori_fscore_overall: {ori_fscore_overall} shd_overall: {shd_overall} ori_fscore_lag: {ori_fscore_lag}")
        print(
            f"ori_fscore_inst: {ori_fscore_inst} shd_lag: {shd_lag} shd_inst: {shd_inst} val_likelihood: {val_likelihood}"
        )


def update_AR_DECI_metrics_dict(metrics_dict: dict, adj_metrics: dict, is_aggregated: bool = False):
    if is_aggregated:
        key_list = [
            "adjacency_fscore_agg",
            "orientation_fscore_agg",
            "shd_agg",
            "auroc_agg",
            "val_likelihood",
            "FPR_agg",
            "TPR_agg",
        ]
    else:
        key_list = [
            "adjacency_fscore_overall_temporal",
            "orientation_fscore_overall_temporal",
            "shd_overall_temporal",
            "val_likelihood",
        ]

    for key in key_list:
        if key not in metrics_dict.keys():
            metrics_dict[key] = []
        cur_metric = adj_metrics.get(key, None)
        metrics_dict[key].append(cur_metric)
    return metrics_dict
