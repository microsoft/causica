import numpy as np
import pytest
import torch

from causica.models.deci.variational_distributions import (
    CategoricalAdjacency,
    DeterministicAdjacency,
    TemporalThreeWayGrahpDist,
    ThreeWayGraphDist,
    VarDistA_ENCO,
    VarDistA_ENCO_ADMG,
    VarDistA_Simple,
)

# pylint: disable=redefined-outer-name


@pytest.fixture
def deterministic_a_b():
    graph = DeterministicAdjacency("cpu")
    graph.set_adj_matrix(np.array([[0, 1], [0, 0]]))
    return graph


@pytest.fixture
def simple_indep_a_b():
    return VarDistA_Simple("cpu", 2)


@pytest.fixture
def enco_quarter_a_b():
    # Set so that, p(A->B) = p(B->A) = 1/4
    graph = VarDistA_ENCO("cpu", 2)
    logits = graph.logits_edges.detach().numpy()
    logits[1, ...] = 0.0
    graph.logits_edges = torch.nn.Parameter(torch.from_numpy(logits), requires_grad=True)
    return graph


@pytest.fixture
def enco_half_a_b():
    graph = VarDistA_ENCO("cpu", 2)
    logits = graph.logits_edges.detach().numpy()
    logits[0, ...] = -1e6
    graph.logits_edges = torch.nn.Parameter(torch.from_numpy(logits), requires_grad=True)
    return graph


@pytest.fixture
def three_third_a_b():
    return ThreeWayGraphDist("cpu", 2)


@pytest.fixture
def three_half_a_b():
    graph = ThreeWayGraphDist("cpu", 2)
    logits = graph.logits.detach().numpy()
    logits[2, ...] = -1e6
    graph.logits = torch.nn.Parameter(torch.from_numpy(logits), requires_grad=True)
    return graph


@pytest.fixture
def enco_admg_quarter_a_b():
    # Set so that p(A -> B) = p(B -> A) = 1/4, p(A <-> B) = 1/2.
    graph = VarDistA_ENCO_ADMG("cpu", 2)

    directed_logits = graph.logits_edges.detach().numpy()
    directed_logits[1, ...] = 0.0
    graph.logits_edges = torch.nn.Parameter(torch.from_numpy(directed_logits), requires_grad=True)

    bidirected_logits = graph.params_bidirected.detach().numpy()
    bidirected_logits[...] = 0.0
    graph.params_bidirected = torch.nn.Parameter(torch.from_numpy(bidirected_logits), requires_grad=True)
    return graph


@pytest.fixture
def enco_admg_half_a_b():
    # Set so that p(A -> B) = p(B -> A) = 1/2, p(A <-> B) = 1.
    graph = VarDistA_ENCO_ADMG("cpu", 2)

    directed_logits = graph.logits_edges.detach().numpy()
    directed_logits[0, ...] = -1e6
    graph.logits_edges = torch.nn.Parameter(torch.from_numpy(directed_logits), requires_grad=True)

    bidirected_logits = graph.params_bidirected.detach().numpy()
    bidirected_logits[...] = 1e6
    graph.params_bidirected = torch.nn.Parameter(torch.from_numpy(bidirected_logits), requires_grad=True)
    return graph


@pytest.fixture
def categorical_a_b():
    graph = CategoricalAdjacency("cpu")
    graph.set_adj_matrices(np.array([[[0, 1], [0, 0]], [[0, 0], [1, 0]]]))
    return graph


@pytest.mark.parametrize(
    "dist,expected",
    [
        ["deterministic_a_b", np.array([[0, 1], [0, 0]])],
        ["simple_indep_a_b", np.array([[0.0, 0.5], [0.5, 0.0]])],
        ["enco_quarter_a_b", np.array([[0.0, 0.25], [0.25, 0.0]])],
        ["enco_half_a_b", np.array([[0.0, 0.5], [0.5, 0.0]])],
        ["three_third_a_b", np.array([[0.0, 1 / 3], [1 / 3, 0.0]])],
        ["three_half_a_b", np.array([[0.0, 0.5], [0.5, 0.0]])],
        ["enco_admg_quarter_a_b", np.array([[0.0, 0.25, 0.0], [0.25, 0.0, 0.0], [0.5, 0.5, 0.0]])],
        ["enco_admg_half_a_b", np.array([[0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [1.0, 1.0, 0.0]])],
        ["categorical_a_b", np.array([[0, 1], [0, 0]])],
    ],
)
def test_get_adj_matrix(dist, expected, request):

    dist = request.getfixturevalue(dist)
    adj_matrix = dist.get_adj_matrix(do_round=False).detach().cpu().numpy()
    assert np.allclose(adj_matrix, expected)


@pytest.mark.parametrize(
    "dist,expected",
    [
        ["deterministic_a_b", 0.0],
        ["simple_indep_a_b", 2 * np.log(2)],
        ["enco_quarter_a_b", 2 * (0.25 * np.log(4) + 0.75 * np.log(4 / 3))],
        ["enco_half_a_b", 2 * np.log(2)],
        ["three_third_a_b", np.log(3)],
        ["three_half_a_b", np.log(2)],
        ["enco_admg_quarter_a_b", 2 * (0.25 * np.log(4) + 0.75 * np.log(4 / 3)) + np.log(2)],
        ["enco_admg_half_a_b", 2 * np.log(2)],
        ["categorical_a_b", np.log(2)],
    ],
)
def test_entropy(dist, expected, request):

    dist = request.getfixturevalue(dist)
    entropy = dist.entropy().detach().cpu().numpy()
    assert np.allclose(entropy, expected)


@pytest.mark.parametrize(
    "dist",
    [
        "deterministic_a_b",
        "simple_indep_a_b",
        "enco_quarter_a_b",
        "enco_half_a_b",
        "three_third_a_b",
        "three_half_a_b",
        "categorical_a_b",
    ],
)
def test_sample_empty_diag(dist, request):

    dist = request.getfixturevalue(dist)
    sample = dist.sample_A().detach().cpu().numpy()
    assert (sample.diagonal() == 0.0).all()


@pytest.mark.parametrize(
    "dist",
    [
        "simple_indep_a_b",
        "enco_quarter_a_b",
        "enco_half_a_b",
        "categorical_a_b",
    ],
)
def test_log_prob_A(dist, request):
    dist = request.getfixturevalue(dist)
    sample = dist.sample_A()
    dist.log_prob_A(sample)


@pytest.mark.parametrize(
    "dist",
    [
        "three_third_a_b",
        "three_half_a_b",
    ],
)
def test_three_sample_exclusion(dist, request):

    dist = request.getfixturevalue(dist)
    for _ in range(10):
        sample = dist.sample_A().detach().cpu().numpy()
        # The three distribution cannot sample both A->B and B->A
        # To validate this, we cannot have any entropy of A + A.T equal to 2
        assert (sample + sample.transpose() <= 1.0).all()


def test_TemporalThreeWayGraphDist_init():
    """
    This tests the initialization of the TemporalThreeWayGraphDist object.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lag = 2
    temporal_var_dist = TemporalThreeWayGrahpDist(device=device, input_dim=4, lag=lag)

    assert temporal_var_dist.logits.shape[0] == 3
    assert temporal_var_dist.logits.shape[1] == 6
    assert temporal_var_dist.logits_lag.shape[1] == lag


@pytest.fixture
def generate_logits():
    logits = torch.log(
        torch.tensor([[0.3, 0.5, 0.7, 0.9, 0.1, 0.3], [0.5, 0.3, 0.1, 0.01, 0.6, 0.4], [0.2, 0.2, 0.2, 0.09, 0.3, 0.3]])
    )
    logits_lag_0 = torch.log(
        torch.tensor(
            [
                [[0.9, 0.7, 0.6, 0.9], [0.99, 0.99, 0.3, 0.8], [0.6, 0.4, 0.3, 0.5], [0.9, 0.9, 0.2, 0.999]],
                [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],
            ]
        )
    )  # (2,dim,dim)
    logits_lag_1 = torch.log(
        torch.tensor(
            [
                [[0.1, 0.3, 0.4, 0.1], [0.01, 0.01, 0.7, 0.2], [0.4, 0.6, 0.7, 0.5], [0.1, 0.1, 0.8, 0.001]],
                [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],
            ]
        )
    )
    logits_lag = torch.stack((logits_lag_0, logits_lag_1))  # (2, lag, node, node)
    return logits, logits_lag


@pytest.fixture
def generate_logits_extreme():
    logits = torch.log(
        torch.tensor(
            [
                [1e-10, 1e-10, 1 - 1e-10, 1 - 1e-10, 1e-10, 1e-10],
                [1e-10, 1 - 1e-10, 1e-10, 1e-10, 1e-10, 1e-10],
                [1 - 1e-10, 1e-10, 1e-10, 1e-10, 1 - 1e-10, 1 - 1e-10],
            ]
        )
    )
    logits_lag_0 = torch.log(
        torch.tensor(
            [
                [[1, 1e-10, 1, 1e-10], [1e-10, 1, 1e-10, 1], [1, 1e-10, 1, 1], [1, 1, 1e-10, 1]],
                [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                [[1e-10, 1, 1e-10, 1], [1, 1, 1, 1], [1e-10, 1e-10, 1e-10, 1e-10], [1, 1, 1, 1]],
            ]
        )
    )  # (2,dim,dim)
    logits_lag_1 = torch.log(
        torch.tensor(
            [
                [[1e-10, 1, 1e-10, 1], [1, 1e-10, 1, 1e-10], [1e-10, 1, 1e-10, 1e-10], [1e-10, 1e-10, 1, 1e-10]],
                [
                    [1e-10, 1e-10, 1e-10, 1e-10],
                    [1e-10, 1e-10, 1e-10, 1e-10],
                    [1e-10, 1e-10, 1e-10, 1e-10],
                    [1e-10, 1e-10, 1e-10, 1e-10],
                ],
                [[1, 1e-10, 1, 1e-10], [1e-10, 1e-10, 1e-10, 1e-10], [1, 1, 1, 1], [1e-10, 1e-10, 1e-10, 1e-10]],
            ]
        )
    )
    logits_lag = torch.stack((logits_lag_0, logits_lag_1))  # (2, lag, node, node)
    return logits, logits_lag


@pytest.fixture
def create_temporal_var_dist():
    device = torch.device("cpu")
    lag = 2
    temporal_var_dist = TemporalThreeWayGrahpDist(device=device, input_dim=4, lag=lag)
    return temporal_var_dist, device


def test_TemporalThreeWayGraphDist_get_adj_matrix(generate_logits, create_temporal_var_dist):
    """
    This tests the get_adj_matrix method of the TemporalThreeWayGraphDist. We manually specified the logits, and compare the adj
    with the target.
    """
    temporal_var_dist, device = create_temporal_var_dist
    # Manually specify the logits
    logits, logits_lag = generate_logits
    temporal_var_dist.logits.data = logits.to(device)
    temporal_var_dist.logits_lag.data = logits_lag.to(device)

    output_adj = temporal_var_dist.get_adj_matrix(do_round=False)

    output_target = torch.tensor(
        [
            [[0, 0.5, 0.3, 0.01], [0.3, 0, 0.1, 0.6], [0.5, 0.7, 0, 0.4], [0.9, 0.1, 0.3, 0]],
            [[0.1, 0.3, 0.4, 0.1], [0.01, 0.01, 0.7, 0.2], [0.4, 0.6, 0.7, 0.5], [0.1, 0.1, 0.8, 0.001]],
            [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],
        ]
    ).to(device)

    assert torch.allclose(output_target, output_adj)

    # Dense init with -7 and -0.5
    device = torch.device("cpu")
    lag = 2
    temporal_var_dist = TemporalThreeWayGrahpDist(device=device, input_dim=4, lag=lag, init_logits=(-7, -0.5))
    output_adj = temporal_var_dist.get_adj_matrix(do_round=False)
    mask_excluding_diag = torch.ones(output_adj[0, ...].shape).to(device)
    mask_excluding_diag.fill_diagonal_(0)
    assert torch.all(torch.masked_select(output_adj[0, ...], mask_excluding_diag.bool()) > 0.4)
    assert torch.all(output_adj[1, ...] > 0.5)


def test_TemporalThreeWayGraphDist_entropy(create_temporal_var_dist):
    temporal_var_dist, _ = create_temporal_var_dist
    # Compute ground truth entropy and compare with the computed one.
    entropy = temporal_var_dist.entropy()
    entropy_target = torch.Tensor([-((1 / 3 * np.log(1 / 3)) * 3 * 6 + (0.5 * np.log(0.5)) * 2 * 16 * 2)])
    assert torch.allclose(entropy, entropy_target)


def test_TemporalThreeWayGraphDist_sample_A(generate_logits_extreme):
    """
    This tests whether the sampled adj matches the desired adj matrix (by setting extreme logits).
    """
    device = torch.device("cpu")
    lag = 3
    temporal_var_dist = TemporalThreeWayGrahpDist(device=device, input_dim=4, lag=lag)

    # Set extreme logits
    logits, logits_lag = generate_logits_extreme
    temporal_var_dist.logits.data = logits.to(device)
    temporal_var_dist.logits_lag.data = logits_lag.to(device)
    # Sample adj, and compare with target
    sample_adj = temporal_var_dist.sample_A()
    adj_target = torch.tensor(
        [
            [[0, 0, 1, 0], [0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
            [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[1, 0, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]],
        ],
        dtype=sample_adj.dtype,
    ).to(device)
    assert torch.equal(sample_adj, adj_target)


@pytest.mark.parametrize(
    "dist,expected",
    [
        ["enco_admg_quarter_a_b", np.array([[0.0, 0.5], [0.5, 0.0]])],
        ["enco_admg_half_a_b", np.array([[0.0, 1.0], [1.0, 0.0]])],
    ],
)
def test_get_bidirected_adj_matrix(dist, expected, request):
    dist = request.getfixturevalue(dist)
    bidirected_adj_matrix = dist.get_bidirected_adj_matrix().detach().numpy()
    assert np.allclose(bidirected_adj_matrix, expected)


@pytest.mark.parametrize("dist", ["enco_admg_quarter_a_b", "enco_admg_half_a_b"])
def test_sample_bidirected_adj(dist, request):
    dist = request.getfixturevalue(dist)
    sample = dist.sample_bidirected_adj().detach().cpu().numpy()
    assert (sample.diagonal() == 0.0).all()
    assert np.all(sample == np.transpose(sample))
