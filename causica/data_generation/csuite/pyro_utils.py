from typing import Callable, List, Optional

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jax import nn, random

# pylint: disable=wrong-import-position
jax.config.update("jax_platform_name", "cpu")
from numpyro import plate
from numpyro.handlers import condition, do, seed, trace
from numpyro.infer import MCMC, NUTS


def layer(parent, x_noise):
    """
    Implements soft truncation for both input and noise variables, Approximately preserves mean=0 and var=1
    """
    return nn.softplus(parent + 1) + nn.softplus(0.5 + x_noise) - 3.0


def layerm(parent, x_noise):
    """
    Implements soft truncation for both input and noise variables, Approximately preserves mean=0 and var=1.
    Reverses sign of input
    """
    return nn.softplus(-parent + 1.5) + nn.softplus(0.5 + x_noise) - 3


def expand_model(model, expansion_factor, plate_name="outer_plate"):
    def expanded_model(*args, **kwargs):
        with plate(plate_name, expansion_factor):
            return model(*args, **kwargs)

    return expanded_model


def generate_dataset(
    base_model: Callable,
    draw_samples_train: int,
    draw_samples_per_test: int,
    thinning: int,
    num_warmup: int,
    intervention_dicts: List[dict],
    condition_dict: Optional[dict] = None,
    counterfactual_dicts: Optional[List[dict]] = None,
    rng_seed: int = 0,
):
    """
    Generate samples form a base distribution specified by a numpyro model, and intervened and conditional versions of the distribution
    Args:
        base_model: numpyro model
        draw_samples_train: how many samples to draw from the observational distribution
        draw_samples_test: how many samples to draw for each interventional distribution
        thinning: HMC chain subsampling factor
        num_warmup: chain warmup steps
        intervention_dicts: list of dictionaries specifying names of variables to be intervened and their values
        condition_dict:  dictionary specifying names of variable to be conditioned on and their values
        counterfactual_dicts: list of dictionaries specifying names of variables to be intervened and their values.
        Performs counterfactual generation if the value passed is not None.
        rng_seed: random seed
    Returns:
        samples_base,
        [samples_int, samples_ref],
        [samples_counterfactual_int, samples_counterfactual_ref],
        samples_cond: [samples_int_cond, samples_ref_cond]: dictionaries with keys are the variable names in the numpyro model and the values
        are an array of samples. In the case that `condition_dict` is not passed then the list [samples_int_cond, samples_ref_cond]
        will be returned as  [None, None], and similarly for `counterfactual_dicts`.

    """
    # Start from this source of randomness. We will split keys for subsequent operations.
    rng_key = random.PRNGKey(rng_seed)
    obs_seed, int_seed, cond_seed = random.split(rng_key, 3)

    # Run base model
    print("Observational")
    seeded_base_model = seed(expand_model(base_model, draw_samples_train, "plate"), obs_seed)
    base_model_trace = trace(seeded_base_model).get_trace()
    samples_base = {k: v["value"] for k, v in base_model_trace.items()}
    samples_base.pop("plate")

    # Run intervention model
    print("Interventional")
    intervention_samples = []
    intervention_rng_keys = random.split(int_seed, len(intervention_dicts))
    for intervention_dict, rng_key_i in zip(intervention_dicts, intervention_rng_keys):
        intervened_model = do(base_model, data=intervention_dict)

        seeded_int_model = seed(expand_model(intervened_model, draw_samples_per_test, "plate"), rng_key_i)
        int_model_trace = trace(seeded_int_model).get_trace()
        samples_int = {k: v["value"] for k, v in int_model_trace.items()}
        samples_int.pop("plate")

        # In numpyro, the do-variables are not actually altered, only subsequent data is changed
        for var in intervention_dict.keys():
            samples_int[var] = np.ones(draw_samples_per_test) * intervention_dict[var]

        intervention_samples.append(samples_int)

    # Counterfactual
    if counterfactual_dicts is not None:
        print("Counterfactual")
        counterfactual_samples: List[Optional[dict]] = []
        for counterfactual_dict in counterfactual_dicts:
            intervened_model = do(base_model, data=counterfactual_dict)
            # Counterfactual generation requires using same seed for each intervention
            seeded_int_model = seed(expand_model(intervened_model, draw_samples_train, "plate"), obs_seed)
            int_model_trace = trace(seeded_int_model).get_trace()
            samples_int = {k: v["value"] for k, v in int_model_trace.items()}
            samples_int.pop("plate")

            for var in counterfactual_dict.keys():
                samples_int[var] = np.ones(draw_samples_train) * counterfactual_dict[var]

            counterfactual_samples.append(samples_int)
    else:
        counterfactual_samples = [None, None]

    # Conditional
    if condition_dict is not None:

        num_samples = draw_samples_per_test * thinning

        # Run intervention condition
        print("Conditional Interventional")

        cond_intervention_samples = []
        intervention_rng_keys = random.split(cond_seed, len(intervention_dicts))
        for intervention_dict, rng_key_i in zip(intervention_dicts, intervention_rng_keys):
            intervened_model = do(base_model, data=intervention_dict)
            conditional_intervened_model = condition(intervened_model, data=condition_dict)

            kernel = NUTS(conditional_intervened_model)
            mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, thinning=thinning)
            mcmc.run(rng_key_i)
            mcmc.print_summary()
            samples_int_cond = mcmc.get_samples()

            for var in intervention_dict.keys():
                samples_int_cond[var] = np.ones(samples_int_cond[var].shape) * intervention_dict[var]

            for var in condition_dict.keys():
                samples_int_cond[var] = np.ones(samples_int[var].shape) * condition_dict[var]

            cond_intervention_samples.append(samples_int_cond)
    else:
        cond_intervention_samples = [None, None]

    return samples_base, intervention_samples, cond_intervention_samples, counterfactual_samples


def plot_conditioning_and_interventions(
    samples_base: Optional[dict],
    labels: list,
    samples_int: Optional[dict] = None,
    samples_ref: Optional[dict] = None,
    samples_int_cond: Optional[dict] = None,
    samples_ref_cond: Optional[dict] = None,
    intervention_dict: Optional[dict] = None,
    reference_dict: Optional[dict] = None,
    condition_dict: Optional[dict] = None,
    savedir: Optional[str] = None,
    name: Optional[str] = None,
    discrete: bool = False,
):
    """
    Plot pairplots for base distribution and interventional distribution
    """
    df_list = []

    if samples_base is not None:
        df1 = pd.DataFrame(data=np.stack([samples_base[a] for a in labels], axis=1), index=None, columns=labels)
        df1["intervention"] = "base dist"
        df_list.append(df1)

    if intervention_dict is not None:
        assert samples_int is not None
        df1_int = pd.DataFrame(data=np.stack([samples_int[a] for a in labels], axis=1), index=None, columns=labels)
        df1_int["intervention"] = f"do({intervention_dict})"
        df_list.append(df1_int)

    if reference_dict is not None:
        assert samples_ref is not None
        df1_ref = pd.DataFrame(data=np.stack([samples_ref[a] for a in labels], axis=1), index=None, columns=labels)
        df1_ref["intervention"] = f"do({reference_dict})"
        df_list.append(df1_ref)

    if condition_dict is not None:
        assert samples_ref_cond is not None
        df1_ref_cond = pd.DataFrame(
            data=np.stack([samples_ref_cond[a] for a in labels], axis=1), index=None, columns=labels
        )
        df1_ref_cond["intervention"] = f"do({reference_dict}), cond {condition_dict}"
        df_list.append(df1_ref_cond)

        assert samples_int_cond is not None
        df1_int_cond = pd.DataFrame(
            data=np.stack([samples_int_cond[a] for a in labels], axis=1), index=None, columns=labels
        )
        df1_int_cond["intervention"] = f"do({intervention_dict}), cond {condition_dict}"
        df_list.append(df1_int_cond)

    sns.set_style("ticks", {"axes.grid": True})

    if discrete:
        sns.pairplot(pd.concat(df_list), hue="intervention", plot_kws={}, grid_kws={}, kind="hist")
    else:
        sns.pairplot(pd.concat(df_list), hue="intervention", plot_kws=dict(alpha=0.05, size=0.7), grid_kws={})

    if name is not None:
        plt.title(name)
        plt.savefig(f"{savedir}/{name}.png")
