import pytest

from causica.datasets.variables import Variable, Variables
from causica.models.point_net import PointNet, SparsePointNet
from causica.models.transformer_set_encoder import TransformerSetEncoder
from causica.models_factory import create_model, create_set_encoder


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
