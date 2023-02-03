import os
from typing import Callable, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import nn
from numpy.typing import ArrayLike

from ...datasets.intervention_data import InterventionData, InterventionDataContainer, InterventionMetadata
from .pyro_utils import generate_dataset, layer, layerm, plot_conditioning_and_interventions
from .utils import extract_observations, finalise, sample_base_to_variable_dict


def simulate_data(
    n_samples_train: int,
    n_samples_per_test: int,
    foldername: str,
    numpyro_model: Callable,
    adjacency_matrix: np.ndarray,
    intervention_idx: int,
    intervention_value: ArrayLike,
    reference_value: ArrayLike,
    target_idxs: List[int],
    condition_idx: Optional[int] = None,
    condition_value: Optional[ArrayLike] = None,
    counterfactual_intervention_idx: Optional[int] = None,
    counterfactual_reference_value: Optional[ArrayLike] = None,
    counterfactual_intervention_value: Optional[ArrayLike] = None,
    make_plots: bool = True,
    plot_discrete: bool = False,
    override_dtypes: Optional[dict] = None,
):
    """
    Generate data from base distribution, intervened distribution, and optionally conditional distribution.
    These are stored in the standard causal dataset format together with pairplots for the base and intervened distributions
    Args:
        n_samples_train: Number of samples to draw from observation distribution
        n_samples_per_test: Number of samples to draw in each interventional setting
        foldername: folder where samples will be stored as csv files
        numpyro_model: to simulate from
        adjacency_matrix: as numpy array
        intervention_idx: int
        intervention_value
        reference_value
        target_idxs
        condition_idx=None: if None is specified, no conditional data will be generated and dataset will not allow CATE evaluation
        condition_value=None: if None is specified, no conditional data will be generated and dataset will not allow CATE evaluation
        counterfactual_idx=None: if None is specified, no conditional data will be generated and dataset will not allow ITE evaluation
        counterfactual_value=None: if None is specified, no conditional data will be generated and dataset will not allow ITE evaluation
        make_plots: bool
        plot_discrete: bool
        override_dtypes: Optional[dict]: data types are automatically inferred from the numpy types of variables in the Pyro trace.
            However, sometimes the inferred type is wrong. To override a type of a variable, include it in the `override_dtypes` dict.
            The keys can be *either* node names (e.g. "x7") or variable names (e.g. "x7_0"). The latter takes precedence over the former.
            Values in the dict should be chosen from {"continuous", "categorical", "binary", "text"}.
    Returns:
        None
    """
    thinning = 5
    num_warmup = 10000
    labels = [f"x{i}" for i in range(adjacency_matrix.shape[0])]

    os.makedirs(foldername, exist_ok=True)

    intervention_value_arr = np.array(intervention_value)
    reference_value_arr = np.array(reference_value)
    intervention_dict = {f"x{intervention_idx}": intervention_value_arr}
    reference_dict = {f"x{intervention_idx}": reference_value_arr}

    # Make sure that these variables exist to avoid unboundedness errors
    condition_value_arr: Optional[np.ndarray] = None
    counterfactual_intervention_value_arr: Optional[np.ndarray] = None
    counterfactual_reference_value_arr: Optional[np.ndarray] = None

    if condition_idx is not None and condition_value is not None:
        condition_value_arr = np.array(condition_value)
        condition_dict = {f"x{condition_idx}": condition_value_arr}

    else:
        condition_dict = None

    if (
        counterfactual_intervention_idx is not None
        and counterfactual_reference_value is not None
        and counterfactual_intervention_value is not None
    ):
        counterfactual_intervention_value_arr = np.array(counterfactual_intervention_value)
        counterfactual_reference_value_arr = np.array(counterfactual_reference_value)
        counterfactual_intervention_dict = {
            f"x{counterfactual_intervention_idx}": counterfactual_intervention_value_arr
        }
        counterfactual_reference_dict = {f"x{counterfactual_intervention_idx}": counterfactual_reference_value_arr}
        counterfactual_dicts = [
            counterfactual_intervention_dict,
            counterfactual_reference_dict,
        ]

    else:
        counterfactual_dicts = None

    (
        samples_base,
        samples_test,
        samples_val,
        [samples_int, samples_ref],
        [samples_int_cond, samples_ref_cond],
        [counterfactual_int, counterfactual_ref],
    ) = generate_dataset(
        numpyro_model,
        n_samples_train,
        n_samples_per_test,
        thinning,
        num_warmup,
        [intervention_dict, reference_dict],
        condition_dict=condition_dict,
        counterfactual_dicts=counterfactual_dicts,
    )

    if make_plots:
        plot_conditioning_and_interventions(
            samples_base,
            labels,
            samples_int=None,
            samples_ref=None,
            samples_int_cond=None,
            samples_ref_cond=None,
            intervention_dict=None,
            reference_dict=None,
            condition_dict=None,
            savedir=foldername,
            name="base_distribution",
            discrete=plot_discrete,
        )
        if plot_discrete:
            plot_conditioning_and_interventions(
                None,
                labels,
                samples_int=samples_int,
                samples_ref=None,
                samples_int_cond=samples_int_cond,
                samples_ref_cond=None,
                intervention_dict=intervention_dict,
                reference_dict=None,
                condition_dict=condition_dict,
                savedir=foldername,
                name="intervened_distribution",
                discrete=plot_discrete,
            )

            plot_conditioning_and_interventions(
                None,
                labels,
                samples_int=None,
                samples_ref=samples_ref,
                samples_int_cond=None,
                samples_ref_cond=samples_ref_cond,
                intervention_dict=None,
                reference_dict=reference_dict,
                condition_dict=condition_dict,
                savedir=foldername,
                name="reference_distribution",
                discrete=plot_discrete,
            )

        else:
            plot_conditioning_and_interventions(
                samples_base,
                labels,
                samples_int=samples_int,
                samples_ref=samples_ref,
                samples_int_cond=samples_int_cond,
                samples_ref_cond=samples_ref_cond,
                intervention_dict=intervention_dict,
                reference_dict=reference_dict,
                condition_dict=condition_dict,
                savedir=foldername,
                name="intervened_distribution",
                discrete=plot_discrete,
            )

    train_data = extract_observations(samples_base)
    test_data = extract_observations(samples_test)
    val_data = extract_observations(samples_val)
    # Create indices to first component of each variable
    event_size = [np.prod(samples_base[label].shape[1:], dtype=np.int32) for label in labels]
    columns_to_nodes = [node_idx for node_idx, node_dim in enumerate(event_size) for _ in range(node_dim)]
    node_list = list(range(len(event_size)))

    intervention_samples = extract_observations(samples_int)
    reference_samples = extract_observations(samples_ref)
    intervention_envs = [
        InterventionData(
            intervention_idxs=np.array([intervention_idx]),
            intervention_values=intervention_value_arr,
            test_data=intervention_samples,
            intervention_reference=np.array(reference_value),
            reference_data=reference_samples,
            effect_idxs=np.array(target_idxs),
        )
    ]
    if condition_dict is not None:
        assert condition_idx is not None, "condition_idx must be set when condition_dict is set."
        assert samples_int_cond is not None, "samples_int_cond must be set when condition_dict is set."
        assert samples_ref_cond is not None, "samples_ref_cond must be set when condition_dict is set."
        conditional_intervention_samples = extract_observations(samples_int_cond)
        conditional_reference_samples = extract_observations(samples_ref_cond)
        intervention_envs.append(
            InterventionData(
                intervention_idxs=np.array([intervention_idx]),
                intervention_values=intervention_value_arr,
                test_data=conditional_intervention_samples,
                intervention_reference=np.array(reference_value),
                reference_data=np.array(conditional_reference_samples),
                conditioning_idxs=np.array([condition_idx]),
                conditioning_values=condition_value_arr,
                effect_idxs=np.array(target_idxs),
            )
        )

    intervention_data_container = InterventionDataContainer(
        InterventionMetadata(columns_to_nodes),
        intervention_envs,
    )
    intervention_data_container.validate()

    if counterfactual_dicts is not None:
        assert counterfactual_int is not None, "counterfactual_int must be set when counterfactual_dicts is set."
        assert counterfactual_ref is not None, "counterfactual_ref must be set when counterfactual_dicts is set."
        cf_intervention_samples = extract_observations(counterfactual_int)
        cf_reference_samples = extract_observations(counterfactual_ref)

        assert counterfactual_intervention_value_arr is not None
        assert counterfactual_reference_value_arr is not None
        assert counterfactual_intervention_idx is not None

        cf_data_container = InterventionDataContainer(
            InterventionMetadata(columns_to_nodes),
            [
                InterventionData(
                    intervention_idxs=np.array([counterfactual_intervention_idx]),
                    intervention_values=counterfactual_intervention_value_arr,
                    test_data=cf_intervention_samples,
                    intervention_reference=counterfactual_reference_value_arr,
                    reference_data=cf_reference_samples,
                    conditioning_idxs=np.array(node_list),
                    conditioning_values=train_data,
                    effect_idxs=np.array(target_idxs),
                )
            ],
        )
        cf_data_container.validate(counterfactual=True)
    else:
        cf_data_container = None

    finalise(
        foldername,
        train_data,
        test_data,
        val_data,
        adjacency_matrix,
        intervention_data_container,
        cf_data_container,
        sample_base_to_variable_dict(
            {label: samples_base[label] for label in labels},
            override_dtypes=override_dtypes,
        ),
    )


def two_node_lin(n_samples_train, n_samples_per_test, datadir, x0_noise_dist, x1_noise_dist, name):
    """
    Simulate from the graph (x0) -> (x1) with linear relationship.
    Turns into structural equations
    x0 ~ E_0(0, var_0)
    x1 = 0.5 * x_0 + sqrt(3)/2 * E_1(0, var_1)

    where E_0 and E_1 are any distributions corresponding to the noise variables of x0 and x1 with mean 0.
    E_0 and E_1 should have finite variance (possibly unequal).
    This ensures that the marginal distributions of x0 and x1 have mean 0.

    Arguments:
        n_samples_train [int]
        n_samples_per_test [int]
        datadir [str]
        x0_noise_dist: a numpyro distribution corresponding to the noise variable x0. Should have mean 0 and finite variance.
        x1_noise_dist: a numpyro distribution corresponding to the noise variable x1. Should have mean 0 and finite variance.
        name [str]
    """
    adjacency_matrix = np.zeros((2, 2))
    adjacency_matrix[0, 1] = 1

    intervention_value = np.ones(np.prod(x0_noise_dist.shape(), initial=1, dtype=np.int32))
    intervention_idx = 0
    reference_value = -intervention_value
    target_idxs = [1]

    def lin_model():
        x0 = numpyro.sample("x0", x0_noise_dist)

        x1_noise = numpyro.sample("x1_noise", x1_noise_dist)

        numpyro.deterministic("x1", 0.5 * x0 + (np.sqrt(3) / 2) * x1_noise)

    simulate_data(
        n_samples_train,
        n_samples_per_test,
        f"{datadir}/csuite_{name}",
        lin_model,
        adjacency_matrix,
        intervention_idx,
        intervention_value,
        reference_value,
        target_idxs,
        counterfactual_intervention_idx=intervention_idx,
        counterfactual_intervention_value=intervention_value,
        counterfactual_reference_value=reference_value,
    )


def collider_lin(n_samples_train, n_samples_per_test, datadir, noise_dist):
    """
    Simulate from the graph (x0) -> (x1) <- (x2) with linear relationship.
    Ensure x0, x1 and x2 have same standard deviation (1).
    Turns into structural equations
    x0 ~ E(0, 1)
    x2 ~ E(0, 1)
    x1 = sqrt(1/3) * x_0 + sqrt(1/3) * x_2 + sqrt(1/3) * E(0, 1)

    E(0, 1) should be any distribution with mean 0 and variance 1.

    Arguments:
        num_samples_train [int]
        num_samples_per_test [int]
        datadir [str]
        noise_dist: a function from shape to np.array of that shape. Should have mean 0 variance 1
    """
    # Set up the graph (x0) -> (x1) <- (x2)
    adjacency_matrix = np.zeros((3, 3))
    adjacency_matrix[0, 1] = 1
    adjacency_matrix[2, 1] = 1

    intervention_idx = 0
    intervention_value = 1.0
    reference_value = -1.0
    target_idxs = [1, 2]

    def collider_lin_model():
        x0 = numpyro.sample("x0", noise_dist)

        x2 = numpyro.sample("x2", noise_dist)

        x1_noise = numpyro.sample("x1_noise", noise_dist)
        numpyro.deterministic("x1", np.sqrt(1 / 3) * x0 + np.sqrt(1 / 3) * x2 + np.sqrt(1 / 3) * x1_noise)

    simulate_data(
        n_samples_train,
        n_samples_per_test,
        f"{datadir}/csuite_col_lingauss",
        collider_lin_model,
        adjacency_matrix,
        intervention_idx,
        intervention_value,
        reference_value,
        target_idxs,
        counterfactual_intervention_idx=intervention_idx,
        counterfactual_intervention_value=intervention_value,
        counterfactual_reference_value=reference_value,
    )


def chain_lin(n_samples_train, n_samples_per_test, datadir, noise_dist):
    """
    Simulate from the graph (x0) -> (x1) -> (x2) with linear relationship.
    Ensure x0, x1 and x2 have same standard deviation (1).
    Turns into structural equations
    x0 ~ E(0, 1)
    x1 = sqrt(2/3) * x0 + sqrt(1/3) * E(0, 1)
    x2 = sqrt(2/3) * x1 + sqrt(1/3) * E(0, 1)

    E(0, 1) should be any distribution with mean 0 and variance 1.

    Arguments:
        num_samples_train [int]
        num_samples_per_test [int]
        datadir [str]
        noise_dist: a function from shape to np.array of that shape. Should have mean 0 variance 1
    """
    # Set up the graph (x0) -> (x1) -> (x2)
    adjacency_matrix = np.zeros((3, 3))
    adjacency_matrix[0, 1] = 1
    adjacency_matrix[1, 2] = 1

    intervention_idx = 0
    intervention_value = 1.0
    reference_value = -1.0
    target_idxs = [1, 2]

    def chain_lin_model():
        x0 = numpyro.sample("x0", noise_dist)

        x1_noise = numpyro.sample("x1_noise", noise_dist)
        x1 = numpyro.deterministic("x1", np.sqrt(2 / 3) * x0 + np.sqrt(1 / 3) * x1_noise)

        x2_noise = numpyro.sample("x2_noise", noise_dist)
        numpyro.deterministic("x2", np.sqrt(2 / 3) * x1 + np.sqrt(1 / 3) * x2_noise)

    simulate_data(
        n_samples_train,
        n_samples_per_test,
        f"{datadir}/csuite_chain_lingauss",
        chain_lin_model,
        adjacency_matrix,
        intervention_idx,
        intervention_value,
        reference_value,
        target_idxs,
        counterfactual_intervention_idx=intervention_idx,
        counterfactual_intervention_value=intervention_value,
        counterfactual_reference_value=reference_value,
    )


def fork_lin(n_samples_train, n_samples_per_test, datadir, noise_dist):
    """
    Simulate from the graph (x0) <- (x1) -> (x2) with linear relationship.
    Ensure x0, x1 and x2 have same standard deviation (1).
    Turns into structural equations
    x1 ~ E(0, 1)
    x0 = sqrt(2/3) * x1 + sqrt(1/3) * E(0, 1)
    x2 = sqrt(2/3) * x1 + sqrt(1/3) * E(0, 1)

    E(0, 1) should be any distribution with mean 0 and variance 1.

    Arguments:
        num_samples_train [int]
        num_samples_per_test [int]
        datadir [str]
        noise_dist: a function from shape to np.array of that shape. Should have mean 0 variance 1
    """
    # Set up the graph (x0) <- (x1) -> (x2)
    adjacency_matrix = np.zeros((3, 3))
    adjacency_matrix[1, 0] = 1
    adjacency_matrix[1, 2] = 1

    intervention_idx = 0
    intervention_value = 1.0
    reference_value = -1.0
    target_idxs = [1, 2]

    def fork_lin_model():
        x1 = numpyro.sample("x1", noise_dist)

        x0_noise = numpyro.sample("x0_noise", noise_dist)
        numpyro.deterministic("x0", np.sqrt(2 / 3) * x1 + np.sqrt(1 / 3) * x0_noise)

        x2_noise = numpyro.sample("x2_noise", noise_dist)
        numpyro.deterministic("x2", np.sqrt(2 / 3) * x1 + np.sqrt(1 / 3) * x2_noise)

    simulate_data(
        n_samples_train,
        n_samples_per_test,
        f"{datadir}/csuite_fork_lingauss",
        fork_lin_model,
        adjacency_matrix,
        intervention_idx,
        intervention_value,
        reference_value,
        target_idxs,
        counterfactual_intervention_idx=intervention_idx,
        counterfactual_intervention_value=intervention_value,
        counterfactual_reference_value=reference_value,
    )


def two_node_nonlinear_gauss(n_samples_train, n_samples_per_test, datadir):
    """
    Simulate from the graph (x1) -> (x2) with nonlinear relationship.
    Ensure x1 and x2 have same standard deviation of 1, and a linear correlation of 0.
    The structural equation is
        x1 ~ N(0, 1)
        x2 = sqrt(6) * exp(-x_1**2) + magic_number * N(0, 1)
    where
        magic_number = sqrt(1 - 6 * (1/sqrt(5) - 1/3))
    to ensure that x2 has a variance of 1

    Arguments:
        n_samples_train [int]
        n_samples_per_test [int]
        datadir [str]
    """
    magic_number = np.sqrt(1 - 6 * (1 / np.sqrt(5) - 1 / 3))
    adjacency_matrix = np.zeros((2, 2))
    adjacency_matrix[0, 1] = 1

    intervention_idx = 0
    intervention_value = 1.0
    reference_value = 0.0
    target_idxs = [1]

    def nonlingauss_model():
        x0 = numpyro.sample("x0", dist.Normal(0, 1))

        x1_noise = numpyro.sample("x1_noise", dist.Normal(0, 1))
        numpyro.deterministic("x1", np.sqrt(6) * np.exp(-(x0**2)) + magic_number * x1_noise)

    simulate_data(
        n_samples_train,
        n_samples_per_test,
        f"{datadir}/csuite_nonlingauss",
        nonlingauss_model,
        adjacency_matrix,
        intervention_idx,
        intervention_value,
        reference_value,
        target_idxs,
        counterfactual_intervention_idx=intervention_idx,
        counterfactual_intervention_value=intervention_value,
        counterfactual_reference_value=reference_value,
    )


def nonlin_simpson(n_samples, datadir):

    adjacency_matrix = np.zeros((4, 4))
    adjacency_matrix[0, 1] = 1
    adjacency_matrix[0, 2] = 1
    adjacency_matrix[1, 2] = 1
    adjacency_matrix[2, 3] = 1

    intervention_idx = 1
    intervention_value = 2.0
    reference_value = -1.0
    condition_idx = 3
    condition_value = 0.0
    target_idxs = [2]

    def non_lin_simpson_model():

        x0 = numpyro.sample("x0", dist.Normal(0.0, 1.0))

        x1 = numpyro.sample("x1", dist.Normal(nn.softplus(1 - x0) - 1.5, 0.15))

        x2_noise = numpyro.sample("x2_noise", dist.Normal(0.0, 1.0))
        x2 = numpyro.deterministic("x2", jnp.tanh(x1 * 2) + 1.5 * x0 + jnp.tanh(x2_noise) - 1)

        numpyro.sample("x3", dist.Laplace(5 * jnp.tanh((x2 - 4) / 5) + 3, 0.1))

    simulate_data(
        n_samples,
        n_samples,
        f"{datadir}/csuite_nonlin_simpson",
        non_lin_simpson_model,
        adjacency_matrix,
        intervention_idx,
        intervention_value,
        reference_value,
        target_idxs,
        condition_idx=condition_idx,
        condition_value=condition_value,
        counterfactual_intervention_idx=intervention_idx,
        counterfactual_intervention_value=intervention_value,
        counterfactual_reference_value=reference_value,
    )


def symprod_simpson(n_samples, datadir):

    adjacency_matrix = np.zeros((4, 4))
    adjacency_matrix[0, 1] = 1
    adjacency_matrix[1, 2] = 1
    adjacency_matrix[0, 2] = 1
    adjacency_matrix[0, 3] = 1

    intervention_idx = 1
    intervention_value = 2.0
    reference_value = -2.0
    condition_idx = 3
    condition_value = 1.3
    target_idxs = [2]

    def product_simpson_model():
        x0 = numpyro.sample("x0", dist.Normal(0.0, 1.0))
        x1 = numpyro.sample("x1", dist.StudentT(3, 2 * jnp.tanh(x0 * 2), 0.1))
        numpyro.sample("x2", dist.Laplace(0.5 * x0 * x1, 0.5))
        numpyro.sample("x3", dist.Normal(jnp.tanh(1.5 * x0), 0.3))

    simulate_data(
        n_samples,
        n_samples,
        f"{datadir}/csuite_symprod_simpson",
        product_simpson_model,
        adjacency_matrix,
        intervention_idx,
        intervention_value,
        reference_value,
        target_idxs,
        condition_idx=condition_idx,
        condition_value=condition_value,
    )


def large_backdoor(n_samples: int, datadir: str, binary_treatment: bool = False, dim: int = 1):
    """Simulates a graph with a potentially large backdoor set if chosen maximally.

    The graph is pyramidal with arrows x0 -> x1, x0 -> x2, x_i -> x_{i + 2}, (x6, x7) -> x8.
    We treat x7 as the treatment and x8 as the target.
    Variables are tuned to each have approximate variance of 1.
    The functional relationships and noise variables are nonlinear and non-Gaussian.
    We rely on softplus to generate nonlinearities.

    Arguments:
        n_samples: Number of samples in the train and test sets.
        datadir: Base directory to create dataset in.
        binary_t: If True, the treatment x8 is converted to a binary.
        dim: Number of dimensions for each variable (will be the group size).
    """
    adjacency_matrix = np.zeros((9, 9))
    adjacency_matrix[0, 1] = 1
    adjacency_matrix[0, 2] = 1
    adjacency_matrix[1, 3] = 1
    adjacency_matrix[2, 4] = 1
    adjacency_matrix[3, 5] = 1
    adjacency_matrix[4, 6] = 1
    adjacency_matrix[5, 7] = 1
    adjacency_matrix[6, 8] = 1
    adjacency_matrix[7, 8] = 1

    # Set up variable shapes
    shape: Tuple[int, ...] = (dim,)
    if dim == 1:
        shape = ()

    intervention_idx = 7
    if binary_treatment:
        intervention_value = np.ones(shape)
        reference_value = np.zeros(shape)
        condition_idx = None
        condition_value = None
    else:
        intervention_value = np.full(shape, 2.5)
        reference_value = np.full(shape, 0.5)
        if dim == 1:
            condition_idx = 1
            condition_value = np.full(shape, 2.5)
        else:
            condition_idx = None
            condition_value = None
    target_idxs = [8]

    loc = np.zeros(shape)
    scale = np.ones(shape)

    def large_backdoor_model():

        x0_noise = numpyro.sample("x0_noise", dist.Normal(loc, scale).to_event())
        x0 = numpyro.deterministic("x0", nn.softplus(1.8 * x0_noise) - 1)

        d1 = dist.Normal(layer(x0, 0) * 1.5, 0.25 * scale)
        x1 = numpyro.sample("x1", d1.to_event(len(d1.batch_shape[1:])))

        x2_noise = numpyro.sample("x2_noise", dist.Normal(loc, scale).to_event())
        x2 = numpyro.deterministic("x2", layer(x0, x2_noise))

        x3_noise = numpyro.sample("x3_noise", dist.Normal(loc, scale).to_event())
        x3 = numpyro.deterministic("x3", layer(x1, x3_noise))

        x4_noise = numpyro.sample("x4_noise", dist.Normal(loc, scale).to_event())
        x4 = numpyro.deterministic("x4", layer(x2, x4_noise))

        x5_noise = numpyro.sample("x5_noise", dist.Normal(loc, scale).to_event())
        x5 = numpyro.deterministic("x5", layer(x3, x5_noise))

        x6_noise = numpyro.sample("x6_noise", dist.Normal(loc, scale).to_event())
        x6 = numpyro.deterministic("x6", layer(x4, x6_noise))

        if binary_treatment:
            d7_cts = dist.LogNormal(nn.softplus(x5 + 1) / 1.5 - 1, 0.15 * scale)
            x7_cts = numpyro.sample("x7_cts", d7_cts.to_event(len(d7_cts.batch_shape[1:])))
            d7 = dist.Delta(jnp.asarray(x7_cts > 0.5, dtype=x6.dtype))
            x7 = numpyro.sample("x7", d7.to_event(len(d7.batch_shape[1:])))
        else:
            d7 = dist.Normal(nn.softplus(x5 + 1) * 1.5 - 1, 0.3)
            x7 = numpyro.sample("x7", d7.to_event(len(d7.batch_shape[1:])))

        d8 = dist.Laplace(-nn.softplus((-x6 * 1.3 + x7) / 3 + 1) + 2, 0.6)
        numpyro.sample("x8", d8.to_event(len(d8.batch_shape[1:])))

    name = "large_backdoor_binary_t" if binary_treatment else "large_backdoor"
    override_dtypes = {"x7": "binary"} if binary_treatment else None
    if dim != 1:
        name += f"_{dim}"
    simulate_data(
        n_samples,
        n_samples,
        f"{datadir}/csuite_{name}",
        large_backdoor_model,
        adjacency_matrix,
        intervention_idx,
        intervention_value,
        reference_value,
        target_idxs,
        condition_idx=condition_idx,
        condition_value=condition_value,
        make_plots=dim == 1,
        override_dtypes=override_dtypes,
    )


def weak_arrows(n_samples, datadir, binary_treatment=False):
    """
    Simulates a graph with many weak arrows.
    The graph has arrows x0 -> x1, x0 -> x2, x_i -> x_{i + 2}, all x_i -> x8.
    We treat x7 as the treatment and x8 as the target.
    Variables are tuned to each have approximate variance of 1.
    The functional relationships and noise variables are nonlinear and non-Gaussian.
    We rely on softplus to generate nonlinearities.

    Arguments:
        n_samples [int]
        datadir [str]
        binary_t [bool]: If True, the treatment x7 is converted to a binary
    """

    adjacency_matrix = np.zeros((9, 9))
    adjacency_matrix[0, 1] = 1
    adjacency_matrix[0, 2] = 1
    adjacency_matrix[1, 3] = 1
    adjacency_matrix[2, 4] = 1
    adjacency_matrix[3, 5] = 1
    adjacency_matrix[4, 6] = 1
    adjacency_matrix[5, 7] = 1
    for i in range(8):
        adjacency_matrix[i, 8] = 1

    intervention_idx = 7
    if binary_treatment:
        intervention_value = 1.0
        reference_value = 0.0
    else:
        intervention_value = 2.5
        reference_value = 0.5
    target_idxs = [8]

    def weak_arrow_model():
        x0_noise = numpyro.sample("x0_noise", dist.Normal(0.0, 1.0))
        x0 = numpyro.deterministic("x0", nn.softplus(1.8 * x0_noise) - 1)

        x1 = numpyro.sample("x1", dist.Normal(layer(x0, 0) * 0.75, 0.75))

        x2_noise = numpyro.sample("x2_noise", dist.Normal(0.0, 1.0))
        x2 = numpyro.deterministic("x2", layerm(x0, x2_noise))

        x3_noise = numpyro.sample("x3_noise", dist.Normal(0.0, 1.0))
        x3 = numpyro.deterministic("x3", layerm(x1, x3_noise))

        x4_noise = numpyro.sample("x4_noise", dist.Normal(0.0, 1.0))
        x4 = numpyro.deterministic("x4", layer(x2, x4_noise))

        x5_noise = numpyro.sample("x5_noise", dist.Normal(0.0, 1.0))
        x5 = numpyro.deterministic("x5", layer(x3, x5_noise))

        x6_noise = numpyro.sample("x6_noise", dist.Normal(0.0, 1.0))
        x6 = numpyro.deterministic("x6", layer(x4, x6_noise))

        if binary_treatment:
            x7_cts = numpyro.sample("x7_cts", dist.LogNormal(nn.softplus(x5 + 1) / 2 - 1, 0.12))
            x7 = numpyro.sample("x7", dist.Delta(jnp.asarray(x7_cts > 0.6, dtype=x6.dtype)))
        else:
            d7 = dist.Normal(nn.softplus(x5 + 1) * 1.5 - 1, 0.3)
            x7 = numpyro.sample("x7", d7)

        x8_mean = nn.softplus(0.1 * x0 + 0.1 * x1 + 0.1 * x2 + 0.1 * x3 + 0.1 * x4 + 0.1 * x5 + 0.5 * x6 + 0.7 * x7 + 1)
        numpyro.sample(
            "x8",
            dist.Laplace(
                x8_mean - 2,
                0.5,
            ),
        )

    name = "weak_arrows_binary_t" if binary_treatment else "weak_arrows"
    override_dtypes = {"x7": "binary"} if binary_treatment else None
    simulate_data(
        n_samples,
        n_samples,
        f"{datadir}/csuite_{name}",
        weak_arrow_model,
        adjacency_matrix,
        intervention_idx,
        intervention_value,
        reference_value,
        target_idxs,
        override_dtypes=override_dtypes,
    )


def one_hot(cat_samples, n_categories):
    onehot = np.zeros((cat_samples.size, n_categories))
    onehot[np.arange(cat_samples.size), cat_samples] = 1
    return onehot


def cat_collider(num_samples_train, num_samples_per_test, datadir):
    """
    Simulates a graph with a simple collider x0 -> x1 <- x2.
    x0 and x1 are categorical with 3 categories and x2 is binary.

    Arguments:
        num_samples_train [int]
        num_samples_per_test [int]
        datadir [str]
    """

    # Set up the graph x0 -> x1 <- x2
    adjacency_matrix = np.zeros((3, 3))
    adjacency_matrix[0, 1] = 1
    adjacency_matrix[2, 1] = 1

    intervention_idx = 0
    intervention_value = np.array([1])
    reference_value = np.array([0])
    target_idxs = [1]

    def cat_collider_model():
        x0 = numpyro.sample("x0", dist.Categorical(probs=np.array([0.25, 0.25, 0.5])))

        x2 = numpyro.sample("x2", dist.Categorical(probs=np.array([0.5, 0.5])))

        x1_probs = np.array([0.1, 0.1, 0.1]) + one_hot(x0, 3) + np.array([2.0, 1.0, 0]) * x2[..., None]
        x1_probs /= x1_probs.sum(-1, keepdims=True)
        numpyro.sample("x1", dist.Categorical(probs=x1_probs))

    simulate_data(
        num_samples_train,
        num_samples_per_test,
        f"{datadir}/csuite_cat_collider",
        cat_collider_model,
        adjacency_matrix,
        intervention_idx,
        intervention_value,
        reference_value,
        target_idxs,
        plot_discrete=True,
        override_dtypes={"x0": "categorical", "x1": "categorical", "x2": "binary"},
    )


def cat_chain(num_samples_train, num_samples_per_test, datadir):
    """
    Simulates a graph x0 -> x1 -> x2.
    x0 and x1 are categorical with 3 categories and x2 is binary.

    Arguments:
        num_samples_train [int]
        num_samples_per_test [int]
        datadir [str]
    """

    # Set up the graph x0 -> x1 -> x2
    adjacency_matrix = np.zeros((3, 3))
    adjacency_matrix[0, 1] = 1
    adjacency_matrix[1, 2] = 1

    intervention_idx = 1
    intervention_value = np.array([2])
    reference_value = np.array([0])
    target_idxs = [0, 2]

    def cat_chain_model():
        x0 = numpyro.sample("x0", dist.Categorical(probs=np.array([0.25, 0.25, 0.5])))

        x1_probs = np.array([0.2, 0.2, 0.2]) + one_hot(x0, 3)
        x1_probs /= x1_probs.sum(-1, keepdims=True)
        x1 = numpyro.sample("x1", dist.Categorical(probs=x1_probs))

        x2_probs = (
            np.array([0.2, 0.2])
            + np.outer(x1 == 0, np.array([1.0, 0.0]))
            + np.outer(x1 == 1, np.array([1.0, 0.0]))
            + np.outer(x1 == 2, np.array([0.0, 1.0]))
        )
        x2_probs /= x2_probs.sum(-1, keepdims=True)
        numpyro.sample("x2", dist.Categorical(probs=x2_probs))

    simulate_data(
        num_samples_train,
        num_samples_per_test,
        f"{datadir}/csuite_cat_chain",
        cat_chain_model,
        adjacency_matrix,
        intervention_idx,
        intervention_value,
        reference_value,
        target_idxs,
        plot_discrete=True,
        override_dtypes={"x0": "categorical", "x1": "categorical", "x2": "binary"},
    )


def cat_to_cts(num_samples_train, num_samples_per_test, datadir):
    """
    Simulates a graph x0 -> x1.
    x0 is categorical. x1 is continuous with an additive noise model.

    Arguments:
        num_samples_train [int]
        num_samples_per_test [int]
        datadir [str]
    """

    # Set up the graph x0 -> x1
    adjacency_matrix = np.zeros((2, 2))
    adjacency_matrix[0, 1] = 1

    intervention_idx = 0
    intervention_value = np.array([1])
    reference_value = np.array([0])
    target_idxs = [1]

    def cat_to_cts_model():
        x0 = numpyro.sample("x0", dist.Categorical(probs=np.array([0.25, 0.25, 0.5])))
        cond_means = np.array([-0.5, 0.0, 0.86])

        x1_noise = numpyro.sample("x1_noise", dist.Normal(0, 1))
        numpyro.deterministic(
            "x1",
            (one_hot(x0, 3) * cond_means).sum(-1) + 1.6 * (nn.softplus(x1_noise) - 1),
        )

    simulate_data(
        num_samples_train,
        num_samples_per_test,
        f"{datadir}/csuite_cat_to_cts",
        cat_to_cts_model,
        adjacency_matrix,
        intervention_idx,
        intervention_value,
        reference_value,
        target_idxs,
        override_dtypes={"x0": "categorical"},
    )


def cts_to_cat(num_samples_train, num_samples_per_test, datadir):
    """
    Simulates a graph x0 -> x1.
    x0 is continuous centered uniform with variance 1. x1 is categorical.

    Arguments:
        num_samples_train [int]
        num_samples_per_test [int]
        datadir [str]
    """

    # Set up the graph x0 -> x1
    adjacency_matrix = np.zeros((2, 2))
    adjacency_matrix[0, 1] = 1

    # This will give an ATE of 0
    intervention_idx = 1
    intervention_value = 1
    reference_value = 0
    target_idxs = [0]

    def cts_to_cat_model():
        # Chosen to give mean 0, variance 1
        x0 = numpyro.sample("x0", dist.Uniform(-np.sqrt(3), np.sqrt(3)))

        break_0, break_1 = np.sqrt(12) * (1 / 3 - 1 / 2), np.sqrt(12) * (2 / 3 - 1 / 2)
        x1_probs = (
            np.array([0.2, 0.2, 0.2])
            + np.outer(x0 < break_0, np.array([1.0, 0.0, 0.0]))
            + np.outer(x0 < break_1, np.array([0.0, 1.0, 0.0]))
        )
        x1_probs /= x1_probs.sum(-1, keepdims=True)
        numpyro.sample("x1", dist.Categorical(probs=x1_probs))

    simulate_data(
        num_samples_train,
        num_samples_per_test,
        f"{datadir}/csuite_cts_to_cat",
        cts_to_cat_model,
        adjacency_matrix,
        intervention_idx,
        intervention_value,
        reference_value,
        target_idxs,
        plot_discrete=True,
        override_dtypes={"x1": "categorical"},
    )


def mixed_simpson(num_samples_train, num_samples_per_test, datadir):
    """
    Simulated data to produce Simpson's paradox.
    Graph x2 -> x0, x2 -> x1, x0 -> x1, x1 -> x3.
    Intervene on x0 and target x1.
    The aim is to make regressing on the "wrong" predictor set lead to very bad ATE.
    For example, here regressing x1 on x0 gives the wrong sign of effect (Simpson).
    Including x3 as a predictor gives other effects ~0, as x3 is very predictive of x1.
    For mixed type: x2 ~ Unif({0, ..., 5}), x0 is binary, x1 and x3 are continuous.

    Arguments:
        num_samples_train [int]
        num_samples_per_test [int]
        datadir [str]
    """

    # Set up the graph x2 -> x0, x2 -> x1, x0 -> x1, x1 -> x3
    adjacency_matrix = np.zeros((4, 4))
    adjacency_matrix[2, 0] = 1
    adjacency_matrix[2, 1] = 1
    adjacency_matrix[0, 1] = 1
    adjacency_matrix[1, 3] = 1

    intervention_idx = 0
    intervention_value = np.array([1])
    reference_value = np.array([0])
    target_idxs = [1]

    def mixed_simpson_model():
        x2 = numpyro.sample("x2", dist.Categorical(probs=np.ones(6) / 6))

        x0_probs = (
            np.array([0.1, 0.1]) + np.outer(x2 < 3, np.array([0.0, 1.0])) + np.outer(x2 >= 3, np.array([1.0, 0.0]))
        )
        x0_probs /= x0_probs.sum(-1, keepdims=True)
        x0 = numpyro.sample("x0", dist.Categorical(probs=x0_probs))

        x1_noise = numpyro.sample("x1_noise", dist.Normal(0, 1))
        x1 = numpyro.deterministic(
            "x1",
            (x0 - 0.5) * 0.7 + (x2 - 2.5) * 0.7 + nn.softplus(0.5 * x1_noise) - 0.7,
        )

        x3_noise = numpyro.sample("x3_noise", dist.Exponential(1))
        numpyro.deterministic("x3", (10 / 3) * jnp.tanh(x1 / 3) + 0.1 * x3_noise - 0.1)

    simulate_data(
        num_samples_train,
        num_samples_per_test,
        f"{datadir}/csuite_mixed_simpson",
        mixed_simpson_model,
        adjacency_matrix,
        intervention_idx,
        intervention_value,
        reference_value,
        target_idxs,
        override_dtypes={"x2": "categorical", "x0": "binary"},
    )


def mixed_confounding(num_samples_train, num_samples_per_test, datadir):
    """
    Simulated data with various confounders and various non-confounders.
    We have treatment x0 [binary], outcome x1 [continuous].
    Confounders: x2 [categorical], x3 [continuous], x4 [binary].
    Only affect treatment: x5 [categorical], x6 [continuous].
    Only affect outcome: x7 [categorical], x8 [continuous].
    Downstream of target: x9 [continuous].
    Downstream of outcome: x10 [continuous].
    Downstream of both: x11 [continuous].

    Arguments:
        num_samples_train [int]
        num_samples_per_test [int]
        datadir [str]
    """

    # Set up the graph
    adjacency_matrix = np.zeros((12, 12))
    # treatment -> effect
    adjacency_matrix[0, 1] = 1
    # confounders
    for i in [2, 3, 4]:
        adjacency_matrix[i, 0] = 1
        adjacency_matrix[i, 1] = 1
    # causes of treatment
    for i in [5, 6]:
        adjacency_matrix[i, 0] = 1
    # causes of outcomes
    for i in [7, 8]:
        adjacency_matrix[i, 1] = 1
    # downstream variables
    adjacency_matrix[0, 9] = 1
    adjacency_matrix[1, 10] = 1
    adjacency_matrix[0, 11] = 1
    adjacency_matrix[1, 11] = 1

    intervention_idx = 0
    intervention_value = np.array([1])
    reference_value = np.array([0])
    target_idxs = [1]

    def mixed_confounding_model():
        # Root nodes
        x2 = numpyro.sample("x2", dist.Categorical(probs=np.ones(3) / 3))
        x3 = numpyro.sample("x3", dist.Normal(0, 1))
        x4 = numpyro.sample("x4", dist.Categorical(probs=np.ones(2) / 2))
        x5 = numpyro.sample("x5", dist.Categorical(probs=np.ones(3) / 3))
        x6 = numpyro.sample("x6", dist.Normal(0, 1))
        x7 = numpyro.sample("x7", dist.Categorical(probs=np.ones(3) / 3))
        x8 = numpyro.sample("x8", dist.Normal(0, 1))

        probabilities = 0.5 + np.tanh(1.0 * (x2 - 1 + x3 + x4 - 0.5 + x5 - 1 + x6)) / 2
        x0 = numpyro.sample("x0", dist.Bernoulli(probs=probabilities))

        x1_mean = np.tanh(1.0 * (3 * x0 - 1.5 - x2 + 1 - x3 - x4 + 0.5 + x7 - 1 + x8)) - 0.12
        x1 = numpyro.sample("x1", dist.Normal(x1_mean, 0.6))

        # Downstream variables from treatment [tune numbers for mean=0, var=1]
        x9_noise = numpyro.sample("x9_noise", dist.Normal(0, 1))
        numpyro.deterministic("x9", x0 + 1.9 * nn.softplus(x9_noise) - 2.4)

        x10_noise = numpyro.sample("x10_noise", dist.Normal(0, 1))
        numpyro.deterministic("x10", 0.8 * x1**2 + nn.softplus(x10_noise) - 1.6)

        x11_noise = numpyro.sample("x11_noise", dist.Normal(0, 1))
        numpyro.deterministic("x11", x0 - 1.5 * nn.softplus(x1) + nn.softplus(x11_noise) - 0.4)

    override_dtypes = {
        "x0": "binary",
        "x2": "categorical",
        "x4": "binary",
        "x5": "categorical",
        "x7": "categorical",
    }
    simulate_data(
        num_samples_train,
        num_samples_per_test,
        f"{datadir}/csuite_mixed_confounding",
        mixed_confounding_model,
        adjacency_matrix,
        intervention_idx,
        intervention_value,
        reference_value,
        target_idxs,
        override_dtypes=override_dtypes,
    )


def lin_gauss(n_samples, datadir):

    adjacency_matrix = np.zeros((8, 8))
    adjacency_matrix[0, 1] = 1
    adjacency_matrix[0, 2] = 1

    adjacency_matrix[1, 2] = 1
    adjacency_matrix[1, 4] = 1

    adjacency_matrix[2, 3] = 1
    adjacency_matrix[3, 4] = 1

    adjacency_matrix[5, 7] = 1
    adjacency_matrix[6, 7] = 1
    adjacency_matrix[7, 4] = 1

    intervention_idx = 2
    intervention_value = 2.0
    reference_value = 0.0
    condition_idx = None
    condition_value = None
    target_idxs = [4]

    def linGauss_model():
        x0 = numpyro.sample("x0", dist.Normal(0.0, 1.0))

        x1 = numpyro.sample("x1", dist.Normal(0.2 * x0, 0.8))

        x2 = numpyro.sample("x2", dist.Normal(0.4 * x0 + 0.4 * x1, 0.2))

        x3 = numpyro.sample("x3", dist.Normal(0.5 * x2, 0.5))

        x5 = numpyro.sample("x5", dist.Normal(0.0, 1.0))

        x6 = numpyro.sample("x6", dist.Normal(0.0, 1.0))

        x7 = numpyro.sample("x7", dist.Normal(0.1 * x5 + 0.5 * x6, 0.4))

        numpyro.sample("x4", dist.Normal(0.2 * x1 + 0.2 * x3 + 0.3 * x7, 0.3))

    simulate_data(
        n_samples,
        n_samples,
        f"{datadir}/prior_experiment_lin_gauss",
        linGauss_model,
        adjacency_matrix,
        intervention_idx,
        intervention_value,
        reference_value,
        target_idxs,
        condition_idx=condition_idx,
        condition_value=condition_value,
        counterfactual_intervention_idx=intervention_idx,
        counterfactual_intervention_value=intervention_value,
        counterfactual_reference_value=reference_value,
    )


def lin_gaussish(n_samples, datadir):

    adjacency_matrix = np.zeros((8, 8))
    adjacency_matrix[0, 1] = 1
    adjacency_matrix[0, 2] = 1

    adjacency_matrix[1, 2] = 1
    adjacency_matrix[1, 4] = 1

    adjacency_matrix[2, 3] = 1
    adjacency_matrix[3, 4] = 1

    adjacency_matrix[5, 7] = 1
    adjacency_matrix[6, 7] = 1
    adjacency_matrix[7, 4] = 1

    intervention_idx = 2
    intervention_value = 2.0
    reference_value = 0.0
    condition_idx = None
    condition_value = None
    target_idxs = [4]

    def linGaussish_model():
        x0 = numpyro.sample("x0", dist.Laplace(0.0, 1.0))

        x1 = numpyro.sample("x1", dist.Laplace(0.2 * x0, 0.8))

        x2 = numpyro.sample("x2", dist.Normal(0.35 * x0 + 0.35 * jnp.tanh(x1), 0.3))

        x3 = numpyro.sample("x3", dist.Normal(0.5 * x2, 0.5))

        x5 = numpyro.sample("x5", dist.Laplace(0.0, 1.0))

        x6 = numpyro.sample("x6", dist.Laplace(0.0, 1.0))

        x7 = numpyro.sample("x7", dist.Normal(0.1 * jnp.tanh(x5) + 0.5 * jnp.tanh(x6), 0.4))

        numpyro.sample("x4", dist.Normal(0.2 * x1 + 0.2 * x3 + 0.3 * x7, 0.3))

    simulate_data(
        n_samples,
        n_samples,
        f"{datadir}/prior_experiment_lin_gaussish",
        linGaussish_model,
        adjacency_matrix,
        intervention_idx,
        intervention_value,
        reference_value,
        target_idxs,
        condition_idx=condition_idx,
        condition_value=condition_value,
        counterfactual_intervention_idx=intervention_idx,
        counterfactual_intervention_value=intervention_value,
        counterfactual_reference_value=reference_value,
    )


def fork_nonlin(n_samples_train: int, n_samples_per_test: int, datadir: str, noise_dist: dist.Distribution):
    """Simulate fork non-linear causal model.

    Simulate from the graph (x0) <- (x1) -> (x2) with non-linear relationship.
    Ensure x0, x1 and x2 have same standard deviation (1).
    Turns into structural equations
    x1 ~ E(0, 1)
    x0 = sqrt(6) * exp(-x1**2) + magic_number * E(0, 1)
    x2 = softplus(1 - x1) - 1.5 + sqrt(0.15) * E(0, 1)

    E(0, 1) should be any distribution with mean 0 and variance 1.

    Args:
        num_samples_train: Number of training samples.
        num_samples_per_test: Number of test samples.
        datadir: Data directory.
        noise_dist: A function from shape to np.array of that shape. Should have mean 0 variance 1.
    """
    magic_number = np.sqrt(1 - 6 * (1 / np.sqrt(5) - 1 / 3))

    # Set up the graph (x0) <- (x1) -> (x2).
    adjacency_matrix = np.zeros((3, 3))
    adjacency_matrix[1, 0] = 1
    adjacency_matrix[1, 2] = 1

    intervention_idx = 0
    intervention_value = 1.0
    reference_value = -1.0
    target_idxs = [1, 2]

    def fork_nonlin_model():
        x1 = numpyro.sample("x1", noise_dist)

        x0_noise = numpyro.sample("x0_noise", noise_dist)
        numpyro.deterministic("x0", np.sqrt(6) * np.exp(-(x1**2)) + magic_number * x0_noise)

        x2_noise = numpyro.sample("x2_noise", noise_dist)
        numpyro.deterministic("x2", nn.softplus(1 - x1) - 1.5 + np.sqrt(0.15) * x2_noise)

    simulate_data(
        n_samples_train,
        n_samples_per_test,
        f"{datadir}/csuite_fork_nonlin_gauss",
        fork_nonlin_model,
        adjacency_matrix,
        intervention_idx,
        intervention_value,
        reference_value,
        target_idxs,
        counterfactual_intervention_idx=intervention_idx,
        counterfactual_intervention_value=intervention_value,
        counterfactual_reference_value=reference_value,
    )


def fork_lin_nongauss(n_samples_train: int, n_samples_per_test: int, datadir: str, noise_dist: dist.Distribution):
    """Simulate fork linear non-Gaussian model.

    Simulate from the graph (x0) <- (x1) -> (x2) with linear relationship and non-Gaussian noise.
    Ensure x0, x1 and x2 have same standard deviation (1).

    Args:
        num_samples_train: Number of training samples.
        num_samples_per_test: Number of test samples.
        datadir: Data directory.
        noise_dist: A function from shape to np.array of that shape. Should have mean 0 and variance 1.
    """
    # Set up the graph (x0) <- (x1) -> (x2)
    adjacency_matrix = np.zeros((3, 3))
    adjacency_matrix[1, 0] = 1
    adjacency_matrix[1, 2] = 1

    intervention_idx = 0
    intervention_value = 1.0
    reference_value = -1.0
    target_idxs = [1, 2]

    def fork_lin_nongauss_model():
        x1 = numpyro.sample("x1", noise_dist)

        x0_noise = numpyro.sample("x0_noise", noise_dist)
        numpyro.deterministic("x0", np.sqrt(2 / 3) * x1 + np.sqrt(1 / 3) * (nn.softplus(1.8 * x0_noise) - 1))

        x2_noise = numpyro.sample("x2_noise", noise_dist)
        numpyro.deterministic("x2", np.sqrt(2 / 3) * x1 + np.sqrt(1 / 3) * (nn.softplus(1.8 * x2_noise) - 1))

    simulate_data(
        n_samples_train,
        n_samples_per_test,
        f"{datadir}/csuite_fork_lin_nongauss",
        fork_lin_nongauss_model,
        adjacency_matrix,
        intervention_idx,
        intervention_value,
        reference_value,
        target_idxs,
        counterfactual_intervention_idx=intervention_idx,
        counterfactual_intervention_value=intervention_value,
        counterfactual_reference_value=reference_value,
    )


def bow_lin(
    n_samples_train: int,
    n_samples_per_test: int,
    datadir: str,
    noise_dist: dist.Distribution,
    dataset_name: str = "csuite_bow_lin_gauss",
):
    """Simulate bow linear causal model.

    Ensure x0, x1 and x2 have same standard deviation (1).
    Turns into structural equations
    x1 ~ E(0, 1)
    x0 = sqrt(2/3) * x1 + sqrt(1/3) * E(0, 1)
    x2 = sqrt(1/3) * x1 + sqrt(1/3) * x0 + (sqrt(1/3) - sqrt(sqrt(2/3))) * E(0, 1)

    E(0, 1) should be any distribution with mean 0 and variance 1.

    Args:
        num_samples_train: Number of training samples.
        num_samples_per_test: Number of test samples.
        datadir: Data directory.
        noise_dist: A function from shape to np.array of that shape. Should have mean 0 variance 1.
        dataset_name: Name that the simulated dataset will be saved as.
    """
    # Set up the graph.
    adjacency_matrix = np.zeros((3, 3))
    adjacency_matrix[1, 0] = 1
    adjacency_matrix[1, 2] = 1
    adjacency_matrix[0, 2] = 1

    intervention_idx = 0
    intervention_value = 1.0
    reference_value = -1.0
    target_idxs = [1, 2]

    def bow_lin_model():
        x1 = numpyro.sample("x1", noise_dist)

        x0_noise = numpyro.sample("x0_noise", noise_dist)
        x0 = numpyro.deterministic("x0", np.sqrt(2 / 3) * x1 + np.sqrt(1 / 3) * x0_noise)

        x2_noise = numpyro.sample("x2_noise", dist.Normal(0, 1))
        numpyro.deterministic(
            "x2", np.sqrt(1 / 3) * x1 + np.sqrt(1 / 3) * x0 + (np.sqrt(1 / 3) - np.sqrt(np.sqrt(2 / 3))) * x2_noise
        )

    simulate_data(
        n_samples_train,
        n_samples_per_test,
        f"{datadir}/{dataset_name}",
        bow_lin_model,
        adjacency_matrix,
        intervention_idx,
        intervention_value,
        reference_value,
        target_idxs,
        counterfactual_intervention_idx=intervention_idx,
        counterfactual_intervention_value=intervention_value,
        counterfactual_reference_value=reference_value,
    )


def bow_lin_nongauss(noise_dist: dist.Distribution, *args, **kwargs):
    assert not isinstance(noise_dist, dist.Normal)
    bow_lin(*args, **dict(kwargs, noise_dist=noise_dist))


def bow_nonlin(
    n_samples_train: int,
    n_samples_per_test: int,
    datadir: str,
    noise_dist: dist.Distribution,
    dataset_name: str = "csuite_bow_nonlin_gauss",
):
    """Simulate bow linear causal model.

    Ensure x0, x1 and x2 have approximately standard deviation (1).
    Turns into structural equations
    x1 ~ E(0, 1)
    x0 = sqrt(6) * exp(-x1**2) + magic_number * E(0, 1)
    x2 = softplus(1 - x1) + exp(-x0) + 0.7 * E(0, 1)

    E(0, 1) should be any distribution with mean 0 and variance 1.

    Args:
        num_samples_train: Number of training samples.
        num_samples_per_test: Number of test samples.
        datadir: Data directory.
        noise_dist: A function from shape to np.array of that shape. Should have mean 0 variance 1.
        dataset_name: Name that the simulated dataset will be saved as.
    """
    magic_number = np.sqrt(1 - 6 * (1 / np.sqrt(5) - 1 / 3))

    # Set up the graph.
    adjacency_matrix = np.zeros((3, 3))
    adjacency_matrix[1, 0] = 1
    adjacency_matrix[1, 2] = 1
    adjacency_matrix[0, 2] = 1

    intervention_idx = 0
    intervention_value = 1.0
    reference_value = -1.0
    target_idxs = [1, 2]

    def bow_lin_model():
        x1 = numpyro.sample("x1", noise_dist)

        x0_noise = numpyro.sample("x0_noise", noise_dist)
        x0 = numpyro.deterministic("x0", np.sqrt(6) * np.exp(-(x1**2)) + magic_number * x0_noise)

        x2_noise = numpyro.sample("x2_noise", noise_dist)
        numpyro.deterministic("x2", nn.softplus(x1) + np.exp(-x0) + 0.7 * x2_noise)

    simulate_data(
        n_samples_train,
        n_samples_per_test,
        f"{datadir}/{dataset_name}",
        bow_lin_model,
        adjacency_matrix,
        intervention_idx,
        intervention_value,
        reference_value,
        target_idxs,
        counterfactual_intervention_idx=intervention_idx,
        counterfactual_intervention_value=intervention_value,
        counterfactual_reference_value=reference_value,
    )


def fork_and_collider_nonlin(
    n_samples_train: int,
    n_samples_per_test: int,
    datadir: str,
    noise_dist: dist.Distribution,
    dataset_name: str = "csuite_fork_collider_nonlin_gauss",
):
    """Simulate fork and collider non-linear causal model.

    Ensure all variables have same standard deviation (1).
    Turns into structural equations
    x0 ~ E(0, 1)
    x1 = E(0, 1)
    x2 ~ E(0, 1)
    x3 = sqrt(6) * exp(-x0**2) + 0.1 * E(0, 1)
    x4 = np.sqrt(6) * exp(-x0**2) + np.sqrt(6) * exp(-x1**2) 0.1 * E(0, 1)
    x5 = np.sqrt(6) * exp(-x1**3) + np.sqrt(6) * exp(-x2**2) + 0.1 * E(0, 1)
    x6 = sqrt(6) * exp(-x2**2) + 0.1 * E(0, 1)

    E(0, 1) should be any distribution with mean 0 and variance 1.

    Args:
        num_samples_train: Number of training samples.
        num_samples_per_test: Number of test samples.
        datadir: Data directory.
        noise_dist: A function from shape to np.array of that shape. Should have mean 0 variance 1.
        dataset_name: The name the simulated dataset is saved as.
    """
    magic_number_small = 0.1

    # Set up the graph.
    adjacency_matrix = np.zeros((7, 7))
    adjacency_matrix[0, :] = [0, 0, 0, 1, 1, 0, 0]
    adjacency_matrix[1, :] = [0, 0, 0, 0, 1, 1, 0]
    adjacency_matrix[2, :] = [0, 0, 0, 0, 0, 1, 1]

    intervention_idx = 3
    intervention_value = 1.0
    reference_value = -1.0
    target_idxs = [5]

    def fork_and_collider_nonlin_model():
        x0 = numpyro.sample("x0", noise_dist)
        x1 = numpyro.sample("x1", noise_dist)
        x2 = numpyro.sample("x2", noise_dist)

        x3_noise = numpyro.sample("x3_noise", noise_dist)
        numpyro.deterministic("x3", np.sqrt(6) * np.exp(-(x0**2)) + magic_number_small * x3_noise)

        x4_noise = numpyro.sample("x4_noise", noise_dist)
        numpyro.deterministic(
            "x4", np.sqrt(6) * np.exp(-(x0**2)) + np.sqrt(6) * np.exp(-(x1**2)) + magic_number_small * x4_noise
        )

        x5_noise = numpyro.sample("x5_noise", noise_dist)
        numpyro.deterministic(
            "x5", np.sqrt(6) * np.exp(-(x1**2)) + np.sqrt(6) * np.exp(-(x2**2)) + magic_number_small * x5_noise
        )

        x6_noise = numpyro.sample("x6_noise", noise_dist)
        numpyro.deterministic("x6", np.sqrt(6) * np.exp(-(x2**2)) + magic_number_small * x6_noise)

    simulate_data(
        n_samples_train,
        n_samples_per_test,
        f"{datadir}/{dataset_name}",
        fork_and_collider_nonlin_model,
        adjacency_matrix,
        intervention_idx,
        intervention_value,
        reference_value,
        target_idxs,
        counterfactual_intervention_idx=intervention_idx,
        counterfactual_intervention_value=intervention_value,
        counterfactual_reference_value=reference_value,
    )


def main():
    n_observation_samples = 4000
    n_test_samples = 2000
    datadir = "data"

    ###########################################################################################
    # Continuous datasets
    ###########################################################################################

    two_node_lin(
        n_observation_samples,
        n_test_samples,
        datadir,
        dist.Normal(0, 1),
        dist.Normal(0, 1),
        "lingauss",
    )

    two_node_lin(
        n_observation_samples,
        n_test_samples,
        datadir,
        dist.Normal(0, 1),
        dist.Normal(0, 0.5),
        "lingauss_unequal_variance",
    )

    shift_exp_x0 = dist.TransformedDistribution(
        dist.Exponential(1), dist.transforms.AffineTransform(loc=-1.0, scale=1.0)
    )
    shift_exp_x1 = dist.TransformedDistribution(
        dist.Exponential(0.5), dist.transforms.AffineTransform(loc=-1.0, scale=1.0)
    )

    two_node_lin(
        n_observation_samples,
        n_test_samples,
        datadir,
        shift_exp_x0,
        shift_exp_x0,
        "linexp",
    )

    two_node_lin(
        n_observation_samples,
        n_test_samples,
        datadir,
        shift_exp_x0,
        shift_exp_x1,
        "linexp_unequal_variance",
    )

    ndim = 2
    d = dist.TransformedDistribution(
        dist.Exponential(np.ones(ndim)),
        dist.transforms.AffineTransform(loc=-1, scale=1),
    ).to_event()
    two_node_lin(n_observation_samples, n_test_samples, datadir, d, d, f"linexp_{ndim}")

    collider_lin(n_observation_samples, n_test_samples, datadir, dist.Normal(0, 1))

    chain_lin(n_observation_samples, n_test_samples, datadir, dist.Normal(0, 1))

    fork_lin(n_observation_samples, n_test_samples, datadir, dist.Normal(0, 1))

    two_node_nonlinear_gauss(n_observation_samples, n_test_samples, datadir)

    fork_nonlin(n_observation_samples, n_test_samples, datadir, dist.Normal(0, 1))

    fork_lin_nongauss(n_observation_samples, n_test_samples, datadir, dist.Normal(0, 1))

    bow_lin(n_observation_samples, n_test_samples, datadir, dist.Normal(0, 1))

    bow_lin_nongauss(
        n_samples_train=n_observation_samples,
        n_samples_per_test=n_test_samples,
        datadir=datadir,
        noise_dist=dist.Laplace(0, np.sqrt(0.5)),
        dataset_name="csuite_bow_lin_laplace",
    )

    bow_nonlin(n_observation_samples, n_test_samples, datadir, dist.Normal(0, 1))

    fork_and_collider_nonlin(n_observation_samples, n_test_samples, datadir, dist.Normal(0, 1))

    ###########################################################################################
    # Continuous datasets with CATE
    ###########################################################################################

    nonlin_simpson(n_test_samples, datadir)

    symprod_simpson(n_test_samples, datadir)

    large_backdoor(n_test_samples, datadir)

    large_backdoor(n_test_samples, datadir, dim=5)

    weak_arrows(n_test_samples, datadir)

    ###########################################################################################
    # Discrete datasets
    ###########################################################################################

    cat_to_cts(n_observation_samples, n_test_samples, datadir)

    cts_to_cat(n_observation_samples, n_test_samples, datadir)

    cat_collider(n_observation_samples, n_test_samples, datadir)

    cat_chain(n_observation_samples, n_test_samples, datadir)

    mixed_simpson(n_observation_samples, n_test_samples, datadir)

    large_backdoor(n_test_samples, datadir, binary_treatment=True)

    weak_arrows(n_test_samples, datadir, binary_treatment=True)

    mixed_confounding(n_observation_samples, n_test_samples, datadir)

    ###########################################################################################
    # For informed prior experiments
    ###########################################################################################

    # non-identifiable ANM
    lin_gauss(n_test_samples, datadir)

    # Partially identifiable ANM
    lin_gaussish(n_test_samples, datadir)


if __name__ == "__main__":
    main()
