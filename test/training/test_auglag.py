import torch
from torch.nn import Parameter
from torch.optim import Adam

from causica.training.auglag import AugLagLossCalculator, AugLagLR, AugLagLRConfig


def _get_auglag_config(lr_init_dict: dict[str, float]):
    return AugLagLRConfig(
        lr_update_lag=1,
        lr_update_lag_best=100,
        lr_init_dict=lr_init_dict,
        aggregation_period=1,
        lr_factor=0.5,
        penalty_progress_rate=1.1,
        safety_rho=1e-3,
        safety_alpha=1e-3,
        max_lr_down=10,
        inner_early_stopping_patience=10,
        max_outer_steps=10,
        patience_penalty_reached=10,
        patience_max_rho=10,
        penalty_tolerance=1e-3,
        max_inner_steps=10,
    )


def test_on_train_batch_end():
    group_lr = {"param1": 0.1}
    param = [{"params": torch.randn(3, 2).requires_grad_(), "name": key, "lr": val} for key, val in group_lr.items()]
    auglag_callback = AugLagLR(_get_auglag_config(group_lr))
    optimizer = Adam(param)
    loss = AugLagLossCalculator(init_alpha=0.0, init_rho=1.0)
    random_generator = torch.Generator()
    random_generator.manual_seed(1337)
    for _ in range(10000):
        auglag_callback.step(
            optimizer=optimizer,
            loss=loss,
            loss_value=torch.rand((), generator=random_generator),
            lagrangian_penalty=torch.rand((), generator=random_generator),
        )


def test_on_train_batch_end_list_opt():
    group_lr = {"param1": 0.1, "param2": 0.1}
    group_param = {"param1": torch.randn(3, 2).requires_grad_(), "param2": torch.randn(5, 3, 6).requires_grad_()}
    param = [{"params": group_param[key], "name": key, "lr": val} for key, val in group_lr.items()]
    optimizer_list = [Adam(param[:1]), Adam(param[1:])]
    loss = AugLagLossCalculator(init_alpha=0.0, init_rho=1.0)
    random_generator = torch.Generator()
    random_generator.manual_seed(1337)
    auglag_callback = AugLagLR(_get_auglag_config(group_lr))
    for _ in range(10000):
        auglag_callback.step(
            optimizer=optimizer_list,
            loss=loss,
            loss_value=torch.rand((), generator=random_generator),
            lagrangian_penalty=torch.rand((), generator=random_generator),
        )


def test_solve_auglag():
    """
    Test that `AugLagLR` can solve the constrained optimization problem.

        min x² s.t. x > 3

    The solution being that x is approximately 3
    """
    x = Parameter(torch.zeros((), requires_grad=True))
    group_lr = {"x": 0.1}
    parameter_list = [{"params": x, "name": "x", "lr": group_lr["x"]}]

    optimizer = Adam(parameter_list)
    scheduler = AugLagLR(
        config=AugLagLRConfig(
            lr_update_lag=1,
            lr_update_lag_best=100,
            lr_init_dict=group_lr,
            aggregation_period=1,
            lr_factor=0.5,
            penalty_progress_rate=1.1,
            safety_rho=100,
            safety_alpha=100,
            max_lr_down=10,
            inner_early_stopping_patience=10,
            max_outer_steps=100000,
            patience_penalty_reached=10,
            patience_max_rho=10,
            penalty_tolerance=1e-5,
            max_inner_steps=10,
        )
    )

    auglag_loss = AugLagLossCalculator(init_alpha=0.0, init_rho=1.0)
    step_counter = 0
    max_iter = 10000
    constraint = torch.inf
    for _ in range(max_iter):
        optimizer.zero_grad()
        loss = x**2
        constraint = torch.max(3 - x, torch.zeros(()))
        auglag_loss_tensor = auglag_loss(loss, constraint)
        auglag_loss_tensor.backward()
        optimizer.step()
        converged = scheduler.step(
            optimizer=optimizer, loss=auglag_loss, loss_value=loss, lagrangian_penalty=constraint
        )
        if converged:
            break

        step_counter += 1

    assert constraint < 1e-3
    assert torch.isclose(x, torch.tensor(3.0), atol=0.1)
