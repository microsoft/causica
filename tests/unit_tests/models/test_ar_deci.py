import copy
import os

import numpy as np
import pandas as pd
import pytest
import torch

from causica.datasets.temporal_causal_csv_dataset_loader import TemporalCausalCSVDatasetLoader
from causica.datasets.variables import Variable, Variables
from causica.models.deci.generation_functions import TemporalContractiveInvertibleGNN
from causica.models.deci.rhino import Rhino
from causica.models.deci.variational_distributions import TemporalThreeWayGrahpDist


@pytest.fixture
def generate_variables():
    cts_variables = Variables(
        [
            Variable("continuous_input_1", True, "continuous", 0, 1),
            Variable("continuous_input_2", True, "continuous", 0, 1),
            Variable("continuous_input_3", True, "continuous", 0, 1),
            Variable("continuous_input_4", True, "continuous", 0, 1),
            Variable("continuous_input_5", True, "continuous", 0, 1),
        ]
    )
    cat_variables = Variables(
        [
            Variable("continuous_input_1", True, "continuous", 0, 1),
            Variable("categorical_input_2", True, "categorical", 0, 5),
            Variable("continuous_input_3", True, "continuous", 0, 1),
            Variable("binary_input_4", True, "binary", 0, 1),
            Variable("continuous_input_5", True, "continuous", 0, 1),
        ]
    )
    return cts_variables, cat_variables


@pytest.fixture
def generate_constraint_matrix():

    return np.array(
        [
            [
                [0, np.nan, 1, 0, np.nan],
                [np.nan, 0, np.nan, 1, np.nan],
                [np.nan, 0, 0, np.nan, np.nan],
                [np.nan, np.nan, 1, 0, np.nan],
                [np.nan, np.nan, np.nan, np.nan, 0],
            ],
            [[1, 0, 1, 1, 1], [0, 0, 1, 1, 1], [1, 0, 0, 0, 1], [0, 1, 0, 0, 1], [1, 0, 1, 1, 0]],
            [
                [np.nan, 0, 0, 0, 0],
                [1, np.nan, 0, 0, 0],
                [1, 1, np.nan, 0, 0],
                [1, 1, 1, np.nan, 0],
                [1, 1, 1, 1, np.nan],
            ],
        ]
    )


# pylint: disable=redefined-outer-name
@pytest.fixture
def ar_deci_example(tmpdir_factory, generate_variables):
    # Create categorical variables
    _, cat_variables = generate_variables
    # Create model with categorical variables#
    device = torch.device("cpu")
    model = Rhino(
        model_id="test_model",
        variables=cat_variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        device=device,
        lag=2,
        allow_instantaneous=True,
    )
    return model, device, cat_variables


@pytest.fixture
def ar_deci_conditional_spline_example(tmpdir_factory, generate_variables):
    # Create categorical variables
    cts_variables, _ = generate_variables
    # Create AR-DECI based on that
    device = torch.device("cpu")
    model = Rhino(
        model_id="test_model",
        variables=cts_variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        device=device,
        lag=2,
        allow_instantaneous=True,
        base_distribution_type="conditional_spline",
        ICGNN_embedding_size=16,
        conditional_embedding_size=8,
        conditional_spline_order="quadratic",
        conditional_decoder_layer_sizes=[7, 7],
        conditional_encoder_layer_sizes=[6, 6],
        additional_spline_flow=3,
    )
    return model, device, cts_variables


def test_ar_deci_init(tmpdir_factory, generate_constraint_matrix, ar_deci_example, ar_deci_conditional_spline_example):
    """
    This tests the initialization of the Rhino class. It includes
    (1) hard constraints matrix (None, and manually specified with/out instantaneous effect),
    (2) correctly specified neg_constraint_matrix and pos_constraint_matrix,
    (3) Correctly specified ICGNN
    (4) Correctly specified variational distribution
    (5) assert mode_adjacency is "learn"
    (6) test default prior_A
    """
    model, device, cat_variables = ar_deci_example
    # Assert not all variables are continuous
    assert not all(variable.type_ == "continuous" for variable in model.variables)
    # Assert the neg_constraint_matrix and pos_constraint_matrix
    neg_target = torch.ones(5, 5)
    neg_target.fill_diagonal_(0)
    neg_target = torch.cat((neg_target.unsqueeze(0), torch.ones(2, 5, 5)), dim=0).to(device)
    pos_target = torch.zeros(3, 5, 5).to(device)
    assert torch.equal(model.neg_constraint_matrix, neg_target)
    assert torch.equal(model.pos_constraint_matrix, pos_target)
    # Manually specified constraint matrix
    hard_constraint = generate_constraint_matrix
    model = Rhino(
        model_id="test_model",
        variables=cat_variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        device=device,
        lag=2,
        allow_instantaneous=True,
        graph_constraint_matrix=hard_constraint,
    )
    # Assert the neg_constraint_matrix and pos_constraint_matrix
    neg_target = (
        torch.tensor(
            [
                [[0, 1, 1, 0, 1], [1, 0, 1, 1, 1], [1, 0, 0, 1, 1], [1, 1, 1, 0, 1], [1, 1, 1, 1, 0]],
                [[1, 0, 1, 1, 1], [0, 0, 1, 1, 1], [1, 0, 0, 0, 1], [0, 1, 0, 0, 1], [1, 0, 1, 1, 0]],
                [[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1]],
            ]
        )
        .float()
        .to(device)
    )
    pos_target = (
        torch.tensor(
            [
                [[0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]],
                [[1, 0, 1, 1, 1], [0, 0, 1, 1, 1], [1, 0, 0, 0, 1], [0, 1, 0, 0, 1], [1, 0, 1, 1, 0]],
                [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0]],
            ]
        )
        .float()
        .to(device)
    )
    assert torch.equal(model.neg_constraint_matrix, neg_target)
    assert torch.equal(model.pos_constraint_matrix, pos_target)

    # Check ICGNN type
    assert isinstance(model.ICGNN, TemporalContractiveInvertibleGNN)
    # Check variational dist type
    assert isinstance(model.var_dist_A, TemporalThreeWayGrahpDist)
    # Check mode_adjacency
    assert model.mode_adjacency == "learn"
    # Check prior_A, prior_mask
    assert not model.exist_prior
    assert torch.equal(model.prior_A, torch.zeros(3, 5, 5, device=model.device))
    assert torch.equal(model.prior_mask, torch.zeros(3, 5, 5, device=model.device))
    # Check num_nodes, lag
    assert model.num_nodes == 5
    assert model.lag == 2
    # Conditional spline init
    model, _, _ = ar_deci_conditional_spline_example
    # check ICGNN embedding size
    assert model.ICGNN.f.embeddings.shape[-1] == 16
    # Check conditional spline embedding size
    assert model.likelihoods["continuous"].hypernet.embeddings.shape[-1] == 8
    # Check decoder and encoder sizes
    assert list(model.likelihoods["continuous"].hypernet.g.modules())[0][0][0].out_features == 6
    assert list(model.likelihoods["continuous"].hypernet.f.modules())[0][0][0].out_features == 7
    # Check num of additional flow
    assert len(model.likelihoods["continuous"].transform) == 9


@pytest.fixture
def generate_prior_A():
    prior_lag = np.ones((2, 5, 5))
    prior_inst = np.array([[[0, 1, 0, 1, 0], [1, 0, 1, 1, 0], [0, 0, 0, 0, 1], [0, 0, 1, 0, 1], [0, 1, 1, 0, 0]]])
    prior_A = np.concatenate((prior_inst, prior_lag))
    prior_mask = np.stack((np.ones((5, 5)), np.zeros((5, 5)), np.ones((5, 5))), axis=0)
    prior_A_invalid = np.concatenate((np.ones((1, 5, 5)), prior_lag))
    return prior_A, prior_mask, prior_A_invalid


def test_set_prior_A(generate_prior_A, ar_deci_example):
    # Create categorical variables
    model, _, _ = ar_deci_example
    # Get and update the prior
    prior_A, prior_mask, prior_A_invalid = generate_prior_A
    model.set_prior_A(prior_A, prior_mask)
    assert model.exist_prior
    assert torch.equal(model.prior_A, torch.tensor(prior_A, device=model.device, dtype=torch.float32))
    assert torch.equal(model.prior_mask, torch.tensor(prior_mask, device=model.device, dtype=torch.float32))

    with pytest.raises(AssertionError):
        model.set_prior_A(prior_A_invalid, prior_mask)


def test_create_var_dist_A_for_deci(tmpdir_factory, generate_variables):
    """
    This checks the type of var_dist_A.
    """
    # Create categorical variables
    _, cat_variables = generate_variables
    # Create model with categorical variables#
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = Rhino(
        model_id="test_model",
        variables=cat_variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        device=device,
        lag=2,
        allow_instantaneous=True,
        var_dist_A_mode="temporal_three",
    )
    assert isinstance(model.var_dist_A, TemporalThreeWayGrahpDist)
    with pytest.raises(AssertionError):
        model = Rhino(
            model_id="test_model",
            variables=cat_variables,
            save_dir=tmpdir_factory.mktemp("save_dir"),
            device=device,
            lag=2,
            allow_instantaneous=True,
            var_dist_A_mode="weird_dist",
        )


def test_create_ICGNN_for_deci(ar_deci_example):
    """
    This checks the type of ICGNN.
    """
    # Create categorical variables
    model, _, _ = ar_deci_example
    assert isinstance(model.ICGNN, TemporalContractiveInvertibleGNN)


def test_get_adj_matrix(tmpdir_factory, generate_variables):
    # Create categorical variables
    _, cat_variables = generate_variables
    # Create model with categorical variables#
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = Rhino(
        model_id="test_model",
        variables=cat_variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        device=device,
        lag=2,
        allow_instantaneous=True,
        var_dist_A_mode="temporal_three",
    )
    adj = model.get_adj_matrix(do_round=True, samples=1, most_likely_graph=True, squeeze=True)
    assert isinstance(adj, np.ndarray)
    assert len(adj.shape) == 3
    assert adj.shape[0] == 3
    assert adj.shape[1] == 5
    assert all(adj[0, ...].diagonal() == 0)
    # Check when samples>1, wait for variational PR merge.


def test_get_adj_matrix_tensor(tmpdir_factory, ar_deci_example):
    """
    This tests the returned adj matrix (torch.Tensor). It checks the shape and type.
    """

    model, device, cat_variables = ar_deci_example
    probable_adj = model.get_adj_matrix_tensor(do_round=True, samples=1, most_likely_graph=True)
    assert isinstance(probable_adj, torch.Tensor)
    assert len(probable_adj.shape) == 4
    assert probable_adj.shape[0] == 1
    assert probable_adj.shape[1] == 3
    assert probable_adj.shape[2] == 5
    assert all(probable_adj[0, 0, ...].diagonal() == 0)
    # allow_instantaneous = False,
    model = Rhino(
        model_id="test_model",
        variables=cat_variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        device=device,
        lag=2,
        allow_instantaneous=False,
        var_dist_A_mode="temporal_three",
    )
    # change var_dist logits s.t. instant matrix is not 0

    logits = model.var_dist_A.logits.detach().numpy()
    logits[0, ...] = 3.0
    model.var_dist_A.logits = torch.nn.Parameter(torch.from_numpy(logits), requires_grad=True)
    probable_adj = model.get_adj_matrix_tensor(do_round=True, samples=1, most_likely_graph=True)
    assert torch.all(probable_adj[0, ...] == 0)
    # Check when samples > 1, wait for variational Imp PR merge.


def test_networkx_graph(tmpdir_factory, generate_variables):
    """
    This checks the networkx conversion of most_probable_graph, including num_nodes, DAGness.
    """
    # Create categorical variables
    _, cat_variables = generate_variables
    # Create model with categorical variables#
    device = torch.device("cpu")
    model = Rhino(
        model_id="test_model",
        variables=cat_variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        device=device,
        lag=2,
        allow_instantaneous=True,
        var_dist_A_mode="temporal_three",
        init_logits=[-7, -0.5],
    )
    nx_graph = model.networkx_graph()
    assert len(nx_graph.nodes) == 15
    assert nx_graph.size() == 5 * 5 * 3


def test_get_weighted_adj_matrix(tmpdir_factory, generate_variables):
    """
    This tests the weighted temporal adj matrix. It should disable the instant disgonal.
    """

    # Create categorical variables
    _, cat_variables = generate_variables
    # Create model with categorical variables#
    device = torch.device("cpu")
    model = Rhino(
        model_id="test_model",
        variables=cat_variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        device=device,
        lag=2,
        allow_instantaneous=True,
        var_dist_A_mode="temporal_three",
        init_logits=[-7, -0.5],
    )
    weighted_adj = model.get_weighted_adj_matrix(do_round=True, samples=1, most_likely_graph=True, squeeze=True)
    assert torch.all(weighted_adj == 0)
    # modify the weights of temporal ICGNN
    model.ICGNN.W.data = torch.ones(model.ICGNN.W.shape).to(model.device)
    weighted_adj = model.get_weighted_adj_matrix(do_round=True, samples=1, most_likely_graph=True, squeeze=True)
    assert all(weighted_adj[0, ...].diagonal() == 0)


def test_dagness_factor(ar_deci_example):
    """
    This checks the dagness factor: the return dagness matches the target value
    """
    # Create categorical variables
    model, _, _ = ar_deci_example
    A = torch.ones(3, 5, 5)
    A[0, ...] = torch.tril(A[0, ...], diagonal=-1)
    dag_A = model.dagness_factor(A)
    assert dag_A == 0
    with pytest.raises(AssertionError):
        dag_A = model.dagness_factor(torch.ones(1, 3, 5, 5))


def test_sample_graph_posterior(tmpdir_factory, generate_variables):
    """
    This checks the sample graph posterior. We draw repeated graph samples, and assert it can remove the duplicates.
    """
    # Create categorical variables
    _, cat_variables = generate_variables
    # Create model with categorical variables#
    device = torch.device("cpu")
    model = Rhino(
        model_id="test_model",
        variables=cat_variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        device=device,
        lag=2,
        allow_instantaneous=False,
        var_dist_A_mode="temporal_three",
        init_logits=[-7, -1e10],
    )
    networkx_posterior, _ = model.sample_graph_posterior(do_round=True, samples=100)
    assert len(networkx_posterior) == 1


@pytest.fixture
def ar_deci_example_cts(tmpdir_factory, generate_variables):
    cts_variables, _ = generate_variables
    return Rhino(
        model_id="test_model",
        variables=cts_variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        device=torch.device("cpu"),
        lag=3,
        allow_instantaneous=True,
        var_dist_A_mode="temporal_three",
        init_logits=[-7, -0.5],
    )


@pytest.fixture
def ar_deci_example_cat(tmpdir_factory, generate_variables):
    _, cat_variables = generate_variables
    return Rhino(
        model_id="test_model",
        variables=cat_variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        device=torch.device("cpu"),
        lag=3,
        allow_instantaneous=True,
        var_dist_A_mode="temporal_three",
        init_logits=[-7, -0.5],
    )


@pytest.fixture
def ar_deci_example_cond_spline_cts(tmpdir_factory, generate_variables):
    # Create categorical variables
    cts_variables, _ = generate_variables
    # Create AR-DECI based on that
    device = torch.device("cpu")
    model = Rhino(
        model_id="test_model",
        variables=cts_variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        device=device,
        lag=2,
        allow_instantaneous=True,
        base_distribution_type="conditional_spline",
        ICGNN_embedding_size=16,
        conditional_embedding_size=8,
        conditional_spline_order="quadratic",
        conditional_decoder_layer_sizes=[7, 7],
        conditional_encoder_layer_sizes=[6, 6],
        additional_spline_flow=3,
    )
    return model


@pytest.fixture
def ar_deci_example_cond_spline_cat(tmpdir_factory, generate_variables):
    # Create categorical variables
    _, cat_variables = generate_variables
    # Create AR-DECI based on that
    device = torch.device("cpu")
    model = Rhino(
        model_id="test_model",
        variables=cat_variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        device=device,
        lag=2,
        allow_instantaneous=True,
        base_distribution_type="conditional_spline",
        ICGNN_embedding_size=16,
        conditional_embedding_size=8,
        conditional_spline_order="quadratic",
        conditional_decoder_layer_sizes=[7, 7],
        conditional_encoder_layer_sizes=[6, 6],
        additional_spline_flow=3,
    )
    return model


@pytest.mark.parametrize(
    "ar_deci_model, X_batch, X_single",
    [
        ("ar_deci_example_cts", torch.randn(25, 4, 5), torch.randn(4, 5)),
        ("ar_deci_example_cat", torch.randn(25, 4, 10), torch.randn(4, 10)),
        ("ar_deci_example_cond_spline_cts", torch.randn(25, 3, 5), torch.randn(3, 5)),
    ],
)
def test__log_prob(ar_deci_model, X_batch, X_single, request):
    """
    This tests the log prob computation. We only check the output shape.
    """
    # pylint: disable=protected-access
    ar_deci_model = request.getfixturevalue(ar_deci_model)

    W_adj = ar_deci_model.get_adj_matrix_tensor(do_round=False, samples=1, most_likely_graph=True).squeeze(
        0
    )  # [lag+1, node, nodes]
    W_adj_batch = ar_deci_model.get_adj_matrix_tensor(
        do_round=False, samples=25, most_likely_graph=False
    )  # [lag+1, node, nodes]
    pred_batch = ar_deci_model.ICGNN.predict(X_batch, W_adj_batch)  # [batch, proc_dim]
    pred_single = ar_deci_model.ICGNN.predict(X_single, W_adj)  # [proc_dim]

    log_batch = ar_deci_model._log_prob(X_batch, pred_batch, W=W_adj)
    log_single = ar_deci_model._log_prob(X_single, pred_single, W=W_adj)

    assert log_batch.shape[0] == 25
    assert log_single.shape[0] == 1


@pytest.mark.parametrize("ar_deci_model, target_shape", [("ar_deci_example_cts", 5), ("ar_deci_example_cat", 10)])
def test__sample_base(ar_deci_model, target_shape, request):
    """
    This tests the sampling from noise distribution with different types of variables. It checks the shape info.
    """
    # pylint: disable=protected-access
    ar_deci_model = request.getfixturevalue(ar_deci_model)
    Nsample = 100
    time_span = 7
    Z = ar_deci_model._sample_base(Nsample, time_span)
    assert len(Z.shape) == 3
    assert Z.shape[0] == Nsample
    assert Z.shape[1] == time_span
    assert Z.shape[2] == target_shape


@pytest.fixture
def example_binary_interventions():
    return np.array([[0, 2], [0, 1], [3, 3], [4, 0]]), np.array([1.7, 1.8, 1, -1.7])


@pytest.fixture
def example_cts_interventions():
    return np.array([[1, 0], [3, 3], [2, 1], [0, 4]]), np.array([1.1, 1.2, 1.3, 1.4])


@pytest.mark.parametrize(
    "ar_deci_model, target_shape, interventions",
    [
        ("ar_deci_example_cts", 5, "example_cts_interventions"),
        ("ar_deci_example_cat", 10, "example_binary_interventions"),
        ("ar_deci_example_cond_spline_cts", 5, "example_cts_interventions"),
        ("ar_deci_example_cond_spline_cat", 10, "example_binary_interventions"),
    ],
)
def test_sample(ar_deci_model, target_shape, interventions, request):
    """
    This tests the generated observations from ar-deci, It checks the shape info.
    """
    ar_deci_model = request.getfixturevalue(ar_deci_model)
    intervention_idxs, intervention_values = request.getfixturevalue(interventions)
    Nsamples = 50
    N_batch = 70
    hist_len = 30
    time_span = 13
    samples_per_graph = 10
    X_history = torch.randn(N_batch, hist_len, target_shape, device=torch.device("cpu"))
    samples = ar_deci_model.sample(
        Nsamples=Nsamples,
        most_likely_graph=False,
        X_history=X_history,
        time_span=time_span,
        samples_per_graph=samples_per_graph,
        intervention_idxs=intervention_idxs,
        intervention_values=intervention_values,
    )
    assert len(samples.shape) == 4
    assert samples.shape[0] == N_batch
    assert samples.shape[1] == Nsamples
    assert samples.shape[2] == time_span
    assert samples.shape[3] == target_shape
    # Assert for interventions (note: this test does not support intervention for categorical variables,
    # but binary is fine)
    for idx, int_variable in enumerate(intervention_idxs):
        cur_time = int_variable[1]
        cur_int_idx = int_variable[0]
        assert torch.all(
            samples[:, :, cur_time, ar_deci_model.variables.group_mask[cur_int_idx]] == intervention_values[idx]
        )


def test_log_prob(tmpdir_factory, generate_variables):
    """
    This tests the log_prob computation. It checks the stochastic runs (different graphs -> different log_prob),
    and the deterministic run (same graphs -> same log_prob)
    """
    # Create categorical variables
    cts_variables, _ = generate_variables
    # Create model with continuous variables#
    device = torch.device("cpu")
    model = Rhino(
        model_id="test_model",
        variables=cts_variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        device=device,
        lag=2,
        allow_instantaneous=True,
        var_dist_A_mode="temporal_three",
        init_logits=[-7, -0.5],
    )
    X = torch.randn(30, 3, 5)
    model.ICGNN.W.data = torch.ones_like(model.ICGNN.W.data)
    log_prob_1 = model.log_prob(X, Nsamples_per_graph=100, most_likely_graph=False)
    log_prob_2 = model.log_prob(X, Nsamples_per_graph=1, most_likely_graph=True)
    assert not np.allclose(log_prob_1, log_prob_2)
    # Deterministic runs. This is achieved by setting the Bernoulli logits of the edge non-existence in lag adj matrix to be
    # small (-1e10), so that the generated elements in lag adj are all 1. Therefore, no matter how many graphs samples we use,
    # they are the same graph, and the log prob should be the same.
    model = Rhino(
        model_id="test_model",
        variables=cts_variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        device=device,
        lag=2,
        allow_instantaneous=False,
        var_dist_A_mode="temporal_three",
        init_logits=[-7, -1e10],
    )
    model.ICGNN.W.data = torch.ones_like(model.ICGNN.W.data)
    log_prob_1 = model.log_prob(X, Nsamples_per_graph=100, most_likely_graph=False)
    log_prob_2 = model.log_prob(X, Nsamples_per_graph=1, most_likely_graph=True)
    assert np.allclose(log_prob_1, log_prob_2)


def test_process_dataset(tmpdir_factory):
    """
    This tests the process_dataset method. It should raise assertion error with missing values for V0.
    """
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    dataset_dir_m = tmpdir_factory.mktemp("dataset_dir_m")
    data = np.concatenate([np.ones((100, 1)), np.random.randn(100, 7)], axis=-1)  # [100, 8]
    train_data = data[:60]
    val_data = data[60:90]
    test_data = data[90:]

    pd.DataFrame(train_data).to_csv(os.path.join(dataset_dir, "train.csv"), header=None, index=None)
    pd.DataFrame(val_data).to_csv(os.path.join(dataset_dir, "val.csv"), header=None, index=None)
    pd.DataFrame(test_data).to_csv(os.path.join(dataset_dir, "test.csv"), header=None, index=None)

    # with missing values
    data_m = copy.deepcopy(data)
    data_m[0, 1] = np.nan
    train_data_m = data_m[:60]
    val_data_m = data_m[60:90]
    test_data_m = data_m[90:]

    pd.DataFrame(train_data_m).to_csv(os.path.join(dataset_dir_m, "train.csv"), header=None, index=None)
    pd.DataFrame(val_data_m).to_csv(os.path.join(dataset_dir_m, "val.csv"), header=None, index=None)
    pd.DataFrame(test_data_m).to_csv(os.path.join(dataset_dir_m, "test.csv"), header=None, index=None)

    dataset_loader = TemporalCausalCSVDatasetLoader(dataset_dir=dataset_dir)

    dataset = dataset_loader.load_predefined_dataset(max_num_rows=None, column_index=0)

    device = torch.device("cpu")
    model = Rhino(
        model_id="test_model",
        variables=dataset.variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        device=device,
        lag=2,
        allow_instantaneous=True,
        var_dist_A_mode="temporal_three",
        init_logits=[-7, -0.5],
    )
    model.process_dataset(dataset)

    # with missing value, should raise assertion error
    dataset_loader = TemporalCausalCSVDatasetLoader(dataset_dir=dataset_dir_m)

    dataset = dataset_loader.load_predefined_dataset(max_num_rows=None, column_index=0)

    device = torch.device("cpu")
    model = Rhino(
        model_id="test_model",
        variables=dataset.variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        device=device,
        lag=2,
        allow_instantaneous=True,
        var_dist_A_mode="temporal_three",
        init_logits=[-7, -0.5],
    )
    with pytest.raises(AssertionError):
        model.process_dataset(dataset)


@pytest.fixture
def example_dataset(tmpdir_factory):
    # Single time-series
    dataset_dir = tmpdir_factory.mktemp("dataset_dir")
    data = np.concatenate([np.ones((100, 1)), np.random.randn(100, 7)], axis=-1)  # [100, 8]
    train_data = data[:60]
    val_data = data[60:90]
    test_data = data[90:]
    pd.DataFrame(train_data).to_csv(os.path.join(dataset_dir, "train.csv"), header=None, index=None)
    pd.DataFrame(val_data).to_csv(os.path.join(dataset_dir, "val.csv"), header=None, index=None)
    pd.DataFrame(test_data).to_csv(os.path.join(dataset_dir, "test.csv"), header=None, index=None)
    dataset_loader = TemporalCausalCSVDatasetLoader(dataset_dir=dataset_dir)

    dataset_single = dataset_loader.load_predefined_dataset(max_num_rows=None, column_index=0)

    # Multiple time-series
    column_idx = np.concatenate((np.ones((30, 1)), np.ones((30, 1)) * 2, np.ones((40, 1)) * 3), axis=0)
    train_data = np.concatenate([column_idx, np.random.randn(100, 7)], axis=-1)  # [100, 8]
    val_data = np.concatenate([column_idx, np.random.randn(100, 7)], axis=-1)  # [100, 8]
    test_data = np.concatenate([column_idx, np.random.randn(100, 7)], axis=-1)  # [100, 8]
    pd.DataFrame(train_data).to_csv(os.path.join(dataset_dir, "train.csv"), header=None, index=None)
    pd.DataFrame(val_data).to_csv(os.path.join(dataset_dir, "val.csv"), header=None, index=None)
    pd.DataFrame(test_data).to_csv(os.path.join(dataset_dir, "test.csv"), header=None, index=None)
    dataset_loader = TemporalCausalCSVDatasetLoader(dataset_dir=dataset_dir)
    dataset_multi = dataset_loader.load_predefined_dataset(max_num_rows=None, column_index=0)

    return dataset_single, dataset_multi


@pytest.fixture
def example_train_config():
    train_config = {
        "learning_rate": 1e-2,
        "likelihoods_learning_rate": 1e-3,
        "batch_size": 3,
        "stardardize_data_mean": False,
        "stardardize_data_std": False,
        "rho": 1.0,
        "safety_rho": 1e13,
        "alpha": 0.0,
        "safety_alpha": 1e13,
        "tol_dag": 1e-5,
        "progress_rate": 0.65,
        "max_steps_auglag": 1,
        "max_auglag_inner_epochs": 2,
        "max_p_train_dropout": 0,
        "reconstruction_loss_factor": 1.0,
        "anneal_entropy": "noanneal",
    }
    return train_config


def test_create_dataset_for_deci(tmpdir_factory, example_dataset):
    """
    This tests the temporal tensor dataset for training. It should output the AR temporal data format.
    """
    # pylint: disable=protected-access
    dataset, _ = example_dataset

    device = torch.device("cpu")
    model = Rhino(
        model_id="test_model",
        variables=dataset.variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        device=device,
        lag=2,
        allow_instantaneous=True,
        var_dist_A_mode="temporal_three",
        init_logits=[-7, -0.5],
    )
    train_config = {"batch_size": 10}
    dataloader, len_dataset = model._create_dataset_for_deci(dataset, train_config_dict=train_config)
    data_batch = next(iter(dataloader))[0]
    assert len_dataset == 60 - 2
    assert data_batch.shape[0] == 10
    assert data_batch.shape[1] == 3


def test_run_train(tmpdir_factory, example_dataset, example_train_config):
    """
    This will test if run_train can run smoothly.
    """
    _, dataset = example_dataset
    train_config = example_train_config
    device = torch.device("cpu")
    model = Rhino(
        model_id="test_model",
        variables=dataset.variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        device=device,
        lag=2,
        allow_instantaneous=True,
        var_dist_A_mode="temporal_three",
        init_logits=[-7, -0.5],
    )
    model.run_train(
        dataset,
        train_config_dict=train_config,
    )


@pytest.mark.parametrize(
    "ar_deci_model",
    [("ar_deci_example_cat"), ("ar_deci_example_cond_spline_cat")],
)
def test_cate(ar_deci_model, request):
    """
    This tests the cate function for ar-deci. We check the shape, if effect matches the intervention values if intervened variables matches effect, and
    when reference values matches intervention_values, the cate should 0. We use variables containing categorical type.
    """
    ar_deci_model = request.getfixturevalue(ar_deci_model)

    intervention_idxs = np.array([[0, 3], [1, 0], [2, 1], [0, 2]])
    intervention_values = np.array([1.7, 0, 1, 0, 0, 0, 0, 1.8, 1.9])
    conditioning_history = np.random.randn(10, 10)
    # check shape
    model_cate, model_norm_cate = ar_deci_model.cate(
        intervention_idxs=intervention_idxs,
        intervention_values=intervention_values,
        reference_values=None,
        effect_idxs=None,
        conditioning_history=conditioning_history,
        Ngraphs=50,
    )
    assert model_cate.shape == (4, 10)
    assert model_norm_cate.shape == (4, 10)

    # check the intervention_values
    ref_values = np.zeros(9)
    model_cate, _ = ar_deci_model.cate(
        intervention_idxs=intervention_idxs,
        intervention_values=intervention_values,
        reference_values=ref_values,
        effect_idxs=None,
        conditioning_history=conditioning_history,
        Ngraphs=50,
    )
    assert np.isclose(model_cate[3, 0], 1.7)
    assert np.array_equal(model_cate[0, 1:7], np.array([0, 1, 0, 0, 0, 0]))
    assert np.isclose(model_cate[1, 7], 1.8)
    assert np.isclose(model_cate[2, 0], 1.9)

    # check shape with effect_idxs
    effect_idxs = np.array([[1, 0]])
    model_cate, model_norm_cate = ar_deci_model.cate(
        intervention_idxs=intervention_idxs,
        intervention_values=intervention_values,
        reference_values=ref_values,
        effect_idxs=effect_idxs,
        conditioning_history=conditioning_history,
        Ngraphs=50,
    )
    assert model_cate.ndim == 1
    assert np.array_equal(model_cate, np.array([0, 1, 0, 0, 0, 0]))

    # make sure it supports effect_idxs time_span > intervention_idx time span
    effect_idxs = np.array([[1, 10]])
    ar_deci_model.cate(
        intervention_idxs=intervention_idxs,
        intervention_values=intervention_values,
        reference_values=ref_values,
        effect_idxs=effect_idxs,
        conditioning_history=conditioning_history,
        Ngraphs=50,
    )
