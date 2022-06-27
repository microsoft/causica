import numpy as np
import torch
from torch import nn

from causica.utils.torch_utils import LinearModel, MultiROFFeaturiser, create_dataloader, generate_fully_connected


def test_create_dataloader():
    a = np.ones((3, 4))
    b = np.ones((3, 4))

    dataloader = create_dataloader(a, b, batch_size=2)
    iterator = iter(dataloader)

    sample = next(iterator)
    assert len(sample) == 2
    a_sample = sample[0]
    assert a_sample.shape == (2, 4)
    assert a_sample.dtype == torch.float

    # Check final minibatch is of batch_size 1 rather than 2
    sample = next(iterator)
    assert len(sample) == 2
    a_sample = sample[0]
    assert a_sample.shape == (1, 4)
    assert a_sample.dtype == torch.float


def test_generate_fully_connected():
    net = generate_fully_connected(
        input_dim=2,
        output_dim=3,
        hidden_dims=[4, 4, 6],
        non_linearity=nn.ReLU,
        activation=nn.Sigmoid,
        device=torch.device("cpu"),
        p_dropout=0.0,
        init_method="default",
        normalization=None,
        res_connection=True,
    )

    # Check that the input and output dimensions are right
    X = torch.ones(10, 2)
    y = net(X)
    assert y.shape[0] == 10
    assert y.shape[1] == 3
    assert torch.all(y < 1)

    # Architecture check. check that size of linear layers is right
    expected_shapes = [(4, 2), (4, 4), (6, 4), (3, 6)]
    linear_idx = 0
    for _, mod in net.named_modules():
        if isinstance(mod, nn.Linear):
            assert mod.weight.shape == expected_shapes[linear_idx]
            linear_idx += 1


def test_LinearModel_1d_data():
    def gen_toy_data():
        x_train0 = torch.linspace(-1, -0.25, 200)
        x_train1 = torch.linspace(0.25, 1, 200)
        noise = torch.randn(400) * 0.3
        y_train0 = 2 * torch.cos(4 * x_train0)
        y_train1 = torch.sin(15 * x_train1)

        X_train = torch.cat([x_train0, x_train1])
        y_train = torch.cat([y_train0, y_train1]) + noise

        return X_train[:, None].float(), y_train.float()

    X_train, y_train = gen_toy_data()

    lm = LinearModel()
    lm.fit(X_train, y_train, prior_precision=1)
    pred_mu, pred_cov = lm.predict(X_train, compute_covariance=True)
    assert pred_mu.shape[0] == X_train.shape[0]
    assert len(pred_mu.shape) == 1
    assert pred_cov.shape[0] == X_train.shape[0]
    assert pred_cov.shape[1] == X_train.shape[0]
    assert len(pred_cov.shape) == 2

    pred_mu, pred_cov = lm.predict(X_train, compute_covariance=False)
    assert pred_cov is None


def test_LinearModel_5d_data():
    def gen_toy_data():
        x_train0 = torch.linspace(-1, -0.25, 200)
        x_train1 = torch.linspace(0.25, 1, 200)
        noise = torch.randn(400) * 0.3
        y_train0 = 2 * torch.cos(4 * x_train0)
        y_train1 = torch.sin(15 * x_train1)

        X_train = torch.cat([x_train0, x_train1])
        y_train = torch.cat([y_train0, y_train1]) + noise

        return X_train[:, None].float(), y_train.float()

    X_train, y_train = gen_toy_data()

    X_train = torch.cat([X_train] * 5, dim=1)

    lm = LinearModel()
    lm.fit(X_train, y_train, prior_precision=1)
    pred_mu, pred_cov = lm.predict(X_train, compute_covariance=True)
    assert pred_mu.shape[0] == X_train.shape[0]
    assert len(pred_mu.shape) == 1
    assert pred_cov.shape[0] == X_train.shape[0]
    assert pred_cov.shape[1] == X_train.shape[0]
    assert len(pred_cov.shape) == 2

    pred_mu, pred_cov = lm.predict(X_train, compute_covariance=False)
    assert pred_cov is None


def test_MultiROFFeaturiser_1d_data_10_features():
    def gen_toy_data():
        x_train0 = torch.linspace(-1, -0.25, 200)
        x_train1 = torch.linspace(0.25, 1, 200)
        noise = torch.randn(400) * 0.3
        y_train0 = 2 * torch.cos(4 * x_train0)
        y_train1 = torch.sin(15 * x_train1)

        X_train = torch.cat([x_train0, x_train1])
        y_train = torch.cat([y_train0, y_train1]) + noise

        return X_train[:, None].float(), y_train.float()

    X_train, _ = gen_toy_data()

    n_features = 10
    lengthscale = [0.05, 0.5]

    featuriser = MultiROFFeaturiser(n_features, lengthscale)
    featuriser.fit(X_train)
    phi_train = featuriser.transform(X_train)
    assert phi_train.shape[0] == X_train.shape[0]
    assert phi_train.shape[1] == n_features


def test_MultiROFFeaturiser_1d_data_1_features():
    def gen_toy_data():
        x_train0 = torch.linspace(-1, -0.25, 200)
        x_train1 = torch.linspace(0.25, 1, 200)
        noise = torch.randn(400) * 0.3
        y_train0 = 2 * torch.cos(4 * x_train0)
        y_train1 = torch.sin(15 * x_train1)

        X_train = torch.cat([x_train0, x_train1])
        y_train = torch.cat([y_train0, y_train1]) + noise

        return X_train[:, None].float(), y_train.float()

    X_train, _ = gen_toy_data()

    n_features = 1
    lengthscale = [0.05, 0.5]

    featuriser = MultiROFFeaturiser(n_features, lengthscale)
    featuriser.fit(X_train)
    phi_train = featuriser.transform(X_train)
    assert phi_train.shape[0] == X_train.shape[0]
    assert phi_train.shape[1] == n_features


def test_MultiROFFeaturiser_2d_data_10_features():
    def gen_toy_data():
        x_train0 = torch.linspace(-1, -0.25, 200)
        x_train1 = torch.linspace(0.25, 1, 200)
        noise = torch.randn(400) * 0.3
        y_train0 = 2 * torch.cos(4 * x_train0)
        y_train1 = torch.sin(15 * x_train1)

        X_train = torch.cat([x_train0, x_train1])
        y_train = torch.cat([y_train0, y_train1]) + noise

        return X_train[:, None].float(), y_train.float()

    X_train, _ = gen_toy_data()
    X_train = torch.cat([X_train] * 2, dim=1)

    n_features = 10
    lengthscale = [0.05, 0.5]

    featuriser = MultiROFFeaturiser(n_features, lengthscale)
    featuriser.fit(X_train)
    phi_train = featuriser.transform(X_train)
    assert phi_train.shape[0] == X_train.shape[0]
    assert phi_train.shape[1] == n_features


def test_MultiROFFeaturiser_2d_data_1_features():
    def gen_toy_data():
        x_train0 = torch.linspace(-1, -0.25, 200)
        x_train1 = torch.linspace(0.25, 1, 200)
        noise = torch.randn(400) * 0.3
        y_train0 = 2 * torch.cos(4 * x_train0)
        y_train1 = torch.sin(15 * x_train1)

        X_train = torch.cat([x_train0, x_train1])
        y_train = torch.cat([y_train0, y_train1]) + noise

        return X_train[:, None].float(), y_train.float()

    X_train, _ = gen_toy_data()
    X_train = torch.cat([X_train] * 2, dim=1)

    n_features = 1
    lengthscale = [0.05, 0.5]

    featuriser = MultiROFFeaturiser(n_features, lengthscale)
    featuriser.fit(X_train)
    phi_train = featuriser.transform(X_train)
    assert phi_train.shape[0] == X_train.shape[0]
    # note that intended behaviour is returning 2 features even though 1 is specified
    assert phi_train.shape[1] == 2


def test_MultiROFFeaturiser_error_reduction():
    # We check for reduction of training error when fitting non-linear data
    def gen_toy_data():
        x_train0 = torch.linspace(-1, -0.25, 200)
        x_train1 = torch.linspace(0.25, 1, 200)
        noise = torch.randn(400) * 0.3
        y_train0 = 2 * torch.cos(4 * x_train0)
        y_train1 = torch.sin(15 * x_train1)

        X_train = torch.cat([x_train0, x_train1])
        y_train = torch.cat([y_train0, y_train1]) + noise

        return X_train[:, None].float(), y_train.float()

    X_train, y_train = gen_toy_data()

    lm = LinearModel()
    lm.fit(X_train, y_train, prior_precision=1)
    lm.predict(X_train, compute_covariance=False)

    n_features = 5000
    lengthscale = [0.05, 0.5]
    featuriser = MultiROFFeaturiser(n_features, lengthscale)
    featuriser.fit(X_train)
    phi_train = featuriser.transform(X_train)

    lm_feat = LinearModel()
    lm_feat.fit(phi_train, y_train, prior_precision=1)
    lm_feat.predict(phi_train, compute_covariance=False)

    featurised_rmse = (lm_feat.predict(phi_train)[0] - y_train).pow(2).mean()
    base_rmse = (lm.predict(X_train)[0] - y_train).pow(2).mean()

    assert featurised_rmse < base_rmse
