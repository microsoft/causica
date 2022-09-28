import pytest
import torch

from causica.models.deci.base_distributions import TemporalConditionalSplineFlow
from causica.models.deci.generation_functions import TemporalContractiveInvertibleGNN, TemporalFGNNI, TemporalHyperNet


@pytest.fixture
def generate_scalar_group_masks():
    """
    This predefines the group masks
    """
    group_mask = torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).float()
    return group_mask


@pytest.fixture
def generate_W_adj():
    W_adj = torch.tensor(
        [
            [[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]],
            [[1, 0, 1, 0], [0, 1, 1, 0], [0, 0, 0, 1], [1, 0, 1, 0]],
            [[0, 0, 1, 0], [1, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 1]],
        ]
    ).float()
    return W_adj


# pylint: disable=redefined-outer-name


def test_initialize_embeddings(generate_scalar_group_masks):
    """
    This tests the size of the embeddings has desired shape, and can be trained.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scalar_group_mask = generate_scalar_group_masks
    lag = 2
    temporal_fgnni = TemporalFGNNI(
        scalar_group_mask,
        device,
        lag=lag,
        embedding_size=10,
    )
    assert len(temporal_fgnni.embeddings.shape) == 3
    assert temporal_fgnni.embeddings.shape[0] == lag + 1
    assert temporal_fgnni.embeddings.shape[1] == scalar_group_mask.shape[0]


def test_feed_forward(generate_scalar_group_masks, generate_W_adj):
    """
    This tests the feed forward function that gives the desired output.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scalar_group_mask = generate_scalar_group_masks.to(device)
    W_adj = generate_W_adj.to(device)  # shape [lag+1, n_nodes, n_nodes] = [3,4,4]
    lag = 2
    temporal_fgnni = TemporalFGNNI(
        scalar_group_mask,
        device,
        lag=lag,
        embedding_size=3,
    )
    # Generate X
    X = torch.tensor([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]]).float().to(device)  # shape [1, 3, 4]
    # Modify the network parameters and embedding value
    for param in temporal_fgnni.g.parameters():
        param.data = torch.ones_like(param.data)
    for param in temporal_fgnni.f.parameters():
        param.data = torch.ones_like(param.data)
    temporal_fgnni.embeddings.data = torch.zeros_like(temporal_fgnni.embeddings.data)
    X_rec = temporal_fgnni.feed_forward(X, W_adj)
    # Note that the following is not exactly the target output. I found that when the number is large,
    # Pytorch does not store the exact value you put in. It gets rounded. So, the following is the target value after pytorch rounded.
    X_target = torch.tensor([[354721824.0, 506515488.0, 961896512.0, 860434496.0]]).to(device)
    assert torch.equal(X_rec, X_target)
    # test the input should respect the batch shapes
    with pytest.raises(AssertionError):
        temporal_fgnni.feed_forward(X.squeeze(0), W_adj)
        temporal_fgnni.feed_forward(torch.randn(10, 3, 4).to(device), torch.randn(5, 3, 4, 4).to(device))
    temporal_fgnni.feed_forward(torch.randn(10, 3, 4).to(device), torch.randn(3, 4, 4).to(device))
    temporal_fgnni.feed_forward(torch.randn(10, 3, 4).to(device), torch.randn(10, 3, 4, 4).to(device))


def test_TemporalContractiveInvertibleGNN(generate_scalar_group_masks):
    """
    This tests the initialization of the TemporalContractiveInvertibleGNN class.
    """

    scalar_group_mask = generate_scalar_group_masks
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lag = 2
    temporal_ICGNN = TemporalContractiveInvertibleGNN(scalar_group_mask, lag, device, res_connection=False)
    # Check the shape of the W parameter
    assert len(temporal_ICGNN.W.shape) == 3
    assert temporal_ICGNN.W.shape[1] == 4
    assert temporal_ICGNN.W.shape[0] == lag + 1

    # Check the network embeddings
    assert temporal_ICGNN.f.embeddings.shape[1] == 4
    assert temporal_ICGNN.f.embeddings.shape[0] == lag + 1


@pytest.fixture
def generate_weights_adj():
    weights = torch.tensor(
        [
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
            [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
            [[2, 3, 4, 5], [2, 3, 4, 5], [2, 3, 4, 5], [2, 3, 4, 5]],
        ]
    ).float()
    return weights


def test_get_weighted_adjacency(generate_scalar_group_masks, generate_weights_adj):
    """
    This method tests if we disable the diagonal elements for instantaneous adj, and allow full matrix for lagged one.
    """
    scalar_group_mask = generate_scalar_group_masks
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lag = 2
    temporal_ICGNN = TemporalContractiveInvertibleGNN(scalar_group_mask, lag, device, res_connection=False)
    # Modify the weights
    W = generate_weights_adj.to(device)
    temporal_ICGNN.W.data = W
    # Get the weights and compare with target
    W_out = temporal_ICGNN.get_weighted_adjacency()
    W_target = (
        torch.tensor(
            [
                [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]],
                [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
                [[2, 3, 4, 5], [2, 3, 4, 5], [2, 3, 4, 5], [2, 3, 4, 5]],
            ]
        )
        .float()
        .to(device)
    )
    assert torch.equal(W_out, W_target)
    # Check the inplace operation in get_weighted_adjacency does not alter the original value.
    assert torch.equal(temporal_ICGNN.W, W)


def test_simulate_SEM(generate_scalar_group_masks, generate_W_adj):
    """
    This method tests the simulation of the SEM in TemporalContractiveInvertibleGNN. By specifying the weights of network,
    adjacency matrix, and data, we can compare the output with the desired values.
    """
    scalar_group_mask = generate_scalar_group_masks
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lag = 2
    temporal_ICGNN = TemporalContractiveInvertibleGNN(scalar_group_mask, lag, device, res_connection=False)
    W_adj = generate_W_adj.to(device)  # [lag+1, num_nodes, num_nodes] = [3, 4, 4]
    time_span = 2
    Z = torch.zeros(1, time_span, 4).to(device)  # [bathc_size, time_span, proc_dim] = [1, 2, 4]
    X_history = torch.tensor([[[4, 4, 4, 4], [3, 3, 3, 3], [2, 2, 2, 2], [1, 1, 1, 1]]]).to(
        device
    )  # [batch_size, history_length, proc_dim] = [1, 4, 4]

    # Modify the network parameters and embedding value
    for param in temporal_ICGNN.f.g.parameters():
        param.data = torch.ones_like(param.data)
    for param in temporal_ICGNN.f.f.parameters():
        param.data = torch.ones_like(param.data)
    temporal_ICGNN.f.embeddings.data = torch.zeros_like(temporal_ICGNN.f.embeddings.data)

    # Simulate the SEM
    _, X_simulate = temporal_ICGNN.simulate_SEM(Z, W_adj, X_history)
    assert X_simulate[0, 0, 0].item() == 540069952
    assert X_simulate[0, 0, 1].item() == 675352640
    assert X_simulate[0, 0, 2] == 45322149550985280
    assert X_simulate.shape[1] == time_span

    # Simulate with intervention mask and value
    intervention_mask = torch.tensor([[True, False, False, True], [False, False, True, False]], dtype=torch.bool).to(
        device
    )
    intervention_value = torch.tensor([0.1, 0.01, 0.2], dtype=torch.float).to(device)
    _, X_simulate = temporal_ICGNN.simulate_SEM(
        Z, W_adj, X_history, intervention_mask=intervention_mask, intervention_values=intervention_value
    )
    assert X_simulate[0, 0, 0] == 0.1
    assert X_simulate[0, 0, 3] == 0.01
    assert X_simulate[0, 1, 2] == 0.2

    # Simulate with wrong intervention mask length, raise assertion error
    intervention_mask = torch.tensor(
        [[True, False, False, True], [False, False, True, False], [False, False, False, False]], dtype=torch.bool
    ).to(device)
    intervention_value = torch.tensor([0.1, 0.01, 0.2], dtype=torch.float).to(device)

    with pytest.raises(AssertionError):
        temporal_ICGNN.simulate_SEM(
            Z, W_adj, X_history, intervention_mask=intervention_mask, intervention_values=intervention_value
        )


@pytest.fixture
def generate_conditional_dist(generate_scalar_group_masks):
    group_mask = generate_scalar_group_masks
    cts_node = [0, 1, 3]
    flow_dist = TemporalConditionalSplineFlow(
        cts_node=cts_node, group_mask=group_mask, device=torch.device("cpu"), lag=2
    )
    return flow_dist


def test_simulate_SEM_conditional(generate_scalar_group_masks, generate_W_adj, generate_conditional_dist):
    flow_dist = generate_conditional_dist
    scalar_group_mask = generate_scalar_group_masks
    lag = 2
    device = torch.device("cpu")
    temporal_ICGNN = TemporalContractiveInvertibleGNN(scalar_group_mask, lag, device)

    W_adj = generate_W_adj.to(device)  # [lag+1, num_nodes, num_nodes] = [3, 4, 4]
    time_span = 2
    Z = torch.zeros(1, time_span, 4).to(device)  # [bathc_size, time_span, proc_dim] = [1, 2, 4]
    X_history = torch.tensor([[[4, 4, 4, 4], [3, 3, 3, 3], [2, 2, 2, 2], [1, 1, 1, 1]]]).to(
        device
    )  # [batch_size, history_length, proc_dim] = [1, 4, 4]
    # without intervention masks
    _, X_simulate = temporal_ICGNN.simulate_SEM_conditional(
        conditional_dist=flow_dist,
        Z=Z,
        W_adj=W_adj,
        X_history=X_history,
        gt_zero_region=[[2]],
    )
    assert X_simulate.shape == torch.Size([1, 2, 4])
    # with intervention masks
    intervention_mask = torch.tensor([[True, False, False, True], [False, False, True, False]], dtype=torch.bool).to(
        device
    )
    intervention_value = torch.tensor([0.1, 0.01, 0.2], dtype=torch.float).to(device)
    _, X_simulate = temporal_ICGNN.simulate_SEM_conditional(
        conditional_dist=flow_dist,
        Z=Z,
        W_adj=W_adj,
        X_history=X_history,
        intervention_mask=intervention_mask,
        intervention_values=intervention_value,
    )
    assert X_simulate[0, 0, 0] == 0.1
    assert X_simulate[0, 0, 3] == 0.01
    assert X_simulate[0, 1, 2] == 0.2


@pytest.fixture
def get_hypernet_setup():
    lag = 2
    num_bins = 8
    param_dim = [num_bins, num_bins, (num_bins - 1)]
    embedding_size = 16
    output_dim_g = 64
    layer_g = [13, 13]
    layer_f = [14, 14]

    return lag, param_dim, embedding_size, output_dim_g, layer_g, layer_f


@pytest.fixture
def get_hypernet(generate_scalar_group_masks, get_hypernet_setup):
    group_mask = generate_scalar_group_masks
    lag, param_dim, embedding_size, output_dim_g, layer_g, layer_f = get_hypernet_setup

    hypernet = TemporalHyperNet(
        cts_node=[0, 1, 2, 3],
        group_mask=group_mask,
        device=torch.device("cpu"),
        lag=lag,
        param_dim=param_dim,
        embedding_size=embedding_size,
        out_dim_g=output_dim_g,
        layers_g=layer_g,
        layers_f=layer_f,
    )
    return hypernet


def test_init_TemporalHyperNet(get_hypernet, get_hypernet_setup, generate_scalar_group_masks):
    """
    This test the init of Temporal hypernet. It will test the shape of embedding and f,g, layer sizes.
    """
    group_mask = generate_scalar_group_masks
    lag, _, embedding_size, output_dim_g, layer_g, layer_f = get_hypernet_setup
    hypernet = get_hypernet
    assert hypernet.embeddings.shape == torch.Size([lag + 1, group_mask.shape[0], embedding_size])
    # test the shape of first layer of g
    assert list(hypernet.g.modules())[0][0][0].in_features == group_mask.shape[1] + embedding_size
    assert list(hypernet.g.modules())[0][0][0].out_features == layer_g[0]
    # test the shape of first layer of f
    assert list(hypernet.f.modules())[0][0][0].in_features == output_dim_g + embedding_size
    assert list(hypernet.f.modules())[0][0][0].out_features == layer_f[0]


@pytest.fixture
def simple_hypernet(generate_scalar_group_masks):
    group_mask = generate_scalar_group_masks
    hypernet = TemporalHyperNet(
        cts_node=[0, 2, 3],
        group_mask=group_mask,
        device=torch.device("cpu"),
        lag=1,
        param_dim=[3, 4, 5],
        embedding_size=2,
        out_dim_g=2,
        layers_g=[2],
        layers_f=[2],
    )
    # fix the weight
    for param in hypernet.g.parameters():
        param.data = torch.ones_like(param)
    for param in hypernet.f.parameters():
        param.data = torch.ones_like(param)

    hypernet.embeddings.data = torch.ones_like(hypernet.embeddings)
    return hypernet


def test_TemporalHyperNet_forward(simple_hypernet):
    """
    This will test the temporal hypernet forward method.
    """
    hypernet = simple_hypernet
    X_hist = torch.tensor([[[1, 2, 3, 4]], [[0.1, 0.2, 0.3, 0.4]]])
    W = torch.tensor(
        [
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
            [[0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 0]],
        ]
    ).float()
    X = {}
    X["X"], X["W"] = X_hist, W

    w, h, d = hypernet(X)
    # assert shape
    assert w.shape == torch.Size([2, 9])
    assert h.shape == torch.Size([2, 12])
    assert d.shape == torch.Size([2, 15])
    # assert value, since the hyper network has fixed weights
    assert all(torch.isclose(w[0], torch.tensor([0.067, 0.067, 0.067, 0.0950, 0.0950, 0.0950, 0.0590, 0.0590, 0.0590])))
    assert all(
        torch.isclose(w[1], torch.tensor([0.0382, 0.0382, 0.0382, 0.0662, 0.0662, 0.0662, 0.0374, 0.0374, 0.0374]))
    )
