import os

import numpy as np
import pytest

from causica.datasets.variables import Variable, Variables
from causica.models.point_net import PointNet, SparsePointNet
from causica.models.transformer_set_encoder import TransformerSetEncoder
from causica.models_factory import create_model, create_set_encoder, set_model_constraint, set_model_prior


@pytest.mark.parametrize(
    "set_encoder_type, kwargs, expected_return_type",
    [
        (
            "default",
            {
                "input_dim": 3,
                "embedding_dim": 2,
                "set_embedding_dim": 19,
                "metadata": None,
                "device": "cpu",
                "multiply_weights": True,
            },
            PointNet,
        ),
        (
            "sparse",
            {
                "input_dim": 3,
                "embedding_dim": 2,
                "set_embedding_dim": 19,
                "metadata": None,
                "device": "cpu",
                "multiply_weights": True,
            },
            SparsePointNet,
        ),
        (
            "transformer",
            {
                "input_dim": 3,
                "embedding_dim": 2,
                "set_embedding_dim": 19,
                "metadata": None,
                "device": "cpu",
                "multiply_weights": True,
                "num_heads": 2,
                "num_blocks": 3,
                "num_seed_vectors": 4,
            },
            TransformerSetEncoder,
        ),
    ],
)
def test_create_set_encoder(set_encoder_type, kwargs, expected_return_type):
    set_encoder = create_set_encoder(set_encoder_type, kwargs)
    assert isinstance(set_encoder, expected_return_type)


def test_create_model(tmpdir):

    variables = Variables([Variable("test", True, "continuous", lower=0, upper=1)])
    model_config_dict = {
        "tau_gumbel": 0.25,
        "lambda_dag": 100.0,
        "lambda_sparse": 5.0,
        "base_distribution_type": "gaussian",
        "spline_bins": 8,
        "var_dist_A_mode": "enco",
        "mode_adjacency": "learn",
        "random_seed": [0],
    }

    deci = create_model(
        model_name="deci", models_dir=tmpdir, variables=variables, device="cpu", model_config_dict=model_config_dict
    )

    assert deci.name() == "deci"


@pytest.fixture(name="random_prior")
def generate_prior():
    prior = np.random.rand(2, 2)
    np.fill_diagonal(prior, 0)
    return prior


@pytest.fixture(name="random_constraint")
def generate_constraint():
    return np.array([[0, np.nan], [1, np.nan]])


@pytest.fixture(name="test_model")
def generate_test_model(tmpdir):
    variables = Variables(
        [
            Variable("test", True, "continuous", lower=0, upper=1),
            Variable("test2", True, "continuous", lower=0, upper=1),
        ]
    )
    model_config_dict = {
        "tau_gumbel": 0.25,
        "lambda_dag": 100.0,
        "lambda_sparse": 5.0,
        "base_distribution_type": "gaussian",
        "spline_bins": 8,
        "var_dist_A_mode": "enco",
        "mode_adjacency": "learn",
        "random_seed": [0],
    }

    return create_model(
        model_name="deci", models_dir=tmpdir, variables=variables, device="cpu", model_config_dict=model_config_dict
    )


def test_set_model_prior_npy(random_prior, tmpdir, test_model):

    prior_dir = tmpdir.mkdir("prior_dir")
    prior_path = os.path.join(prior_dir, "prior.npy")

    # save prior to npy file
    np.save(prior_path, random_prior)

    # set the prior with npy
    set_model_prior(test_model, prior_path)

    # check prior is successfully set
    assert np.array_equal(test_model.prior_A.data.cpu().numpy(), random_prior)


def test_set_model_prior_csv(tmpdir, random_prior, test_model):

    prior_dir = tmpdir.mkdir("prior_dir")
    # set the prior with csv
    prior_path = os.path.join(prior_dir, "prior.csv")
    np.savetxt(prior_path, random_prior, delimiter=",")
    set_model_prior(test_model, prior_path)
    assert np.array_equal(test_model.prior_A.data.cpu().numpy(), random_prior)


def test_set_model_prior_with_wrong_type(tmpdir, random_prior, test_model):

    prior_dir = tmpdir.mkdir("prior_dir")
    with pytest.raises(TypeError):
        prior_path = os.path.join(prior_dir, "prior.txt")
        np.savetxt(prior_path, random_prior)
        set_model_prior(test_model, prior_path)


def test_set_model_constraint(random_constraint, tmpdir, test_model):
    constraint_dir = tmpdir.mkdir("constraint_dir")
    constraint_path = os.path.join(constraint_dir, "constraint.npy")

    # save prior to npy file
    np.save(constraint_path, random_constraint)

    # set the constraint
    set_model_constraint(test_model, constraint_path)

    # check constraint matrix is successfully set
    assert np.array_equal(test_model.neg_constraint_matrix.cpu().data.numpy(), np.array([[0.0, 1.0], [1.0, 0.0]]))
    assert np.array_equal(test_model.pos_constraint_matrix.cpu().data.numpy(), np.array([[0.0, 0.0], [1.0, 0.0]]))


def test_set_model_constraint_with_wrong_type(tmpdir, random_constraint, test_model):

    constraint_dir = tmpdir.mkdir("constraint_dir")
    # set the constraint with wrong file type
    with pytest.raises(TypeError):
        constraint_path = os.path.join(constraint_dir, "constraint.txt")
        np.savetxt(constraint_path, random_constraint)
        set_model_prior(test_model, constraint_path)
