import torch

from causica.training.per_variable_metrics import binary_accuracy, categorical_accuracy, mape, rmse, smape


def test_binary_accuracy():
    logits = torch.tensor([-1.0, 1.0, 1.0, -1.0])

    target = torch.tensor([0.0, 1.0, 0.0, 0.0])

    assert binary_accuracy(logits, target) == 0.75


def test_categorical_accuracy():
    logits = torch.tensor([[0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7]])

    target = torch.tensor([2, 1, 0, 1])

    assert categorical_accuracy(logits, target, False) == 0.25

    target = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0]])

    assert categorical_accuracy(logits, target, True) == 0.25


def test_rmse():
    prediction = torch.tensor([[0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7]])

    target = torch.tensor([[0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7]])

    assert rmse(prediction, target) == 0.0

    prediction = torch.tensor([[0.2, 0.3, 0.8], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7]])

    assert torch.allclose(torch.tensor(0.0075) ** 0.5, rmse(prediction, target))


def test_mape():
    prediction = torch.tensor([[0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7]])

    target = torch.tensor([[0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7]])

    assert mape(prediction, target) == 0.0

    prediction = torch.tensor([[0.2, 0.3, 0.8], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7]])

    true_mape = torch.sum(torch.ones((3,)) * 0.1 / target[0]) / 4

    assert torch.allclose(true_mape, mape(prediction, target))


def test_smape():
    prediction = torch.tensor([[0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7]])

    target = torch.tensor([[0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7]])

    assert smape(prediction, target) == 0.0

    prediction = torch.tensor([[0.2, 0.3, 0.8], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7]])

    true_smape = torch.sum(torch.ones((3,)) * 0.1 / (target[0] + prediction[0])) / 4

    assert torch.allclose(true_smape, smape(prediction, target))
