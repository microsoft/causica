import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

# pylint: disable=wrong-import-position
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes, SubplotBase
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from tqdm import trange

from ..datasets.variables import Variable, Variables

logger = logging.getLogger()


def violin_plot_imputations(
    imputations: np.ndarray,
    imputations_stats: dict,
    mask: np.ndarray,
    variables: Variables,
    user_id: int,
    title: str = None,
    save_path: str = None,
    plot_name: str = None,
    normalized: bool = True,
    ax: SubplotBase = None,
) -> Optional[Tuple[Dict[str, Any], SubplotBase]]:
    """
    Plot the violin plot for the imputed continuous variables for a given input datapoint.
    Since violin plots only support continuous variables, other variables will be dropped and if there are
    no continuous variables the plot is not generated and None is returned.
    Args:
        imputations (np.array of shape (sample_count, user_count, variable_count): the imputed results from
        model predictions.
        imputations_stats (dictionary): Imputation stats of {variable idx: stats_dictionary},
        where depending on the type of the variable (continuous/binary/categorical) the computed statistics type
        can be different.
        This stats can be obatained by running ImputationStatistics.get_statistics(imputations, variables)
        mask (np.array of shape (user_count, variable_count)): mask of value existence, 0 indicates missing.
        variables (Variables): an instance of Variable class indicating the feature properties,
        should be consistent with imputations_stats.
        user_id (int): the integer indexing the entry in imputations (along user_count axis).
        title (str): Title for plot.
        save_path (path): Path to save generated plot to. Defaults to "plots".
        plot_name (str): File name for generated plot. If set to None, a random one will be generated.
        normalized (bool): Boolean indicator to indicate whether the values are normalised
        ax (SubplotBase): if provided, plot the violin plot in the provided subplot ax object

    Returns:
        user_stats (dictionary): a dictionary containing the statistics used for the violin plot

    """
    # first determine which feature to plot: excluding categorical & binary features
    continuous_list_idx = []
    continuous_variables_idx = []
    continuous_variables_names = []
    target_variables_idx = []
    for var_idx, var in enumerate(variables):
        plot_this_var = (imputations_stats[var_idx]["type"] == "continuous") and (var.type_ == "continuous")
        if plot_this_var:
            # TODO do we still need both variables below?
            continuous_list_idx.append(var_idx)
            continuous_variables_idx.append(var_idx)
            continuous_variables_names.append(var.name)  # check this
            if not var.query:  # target variable
                target_variables_idx.append(var_idx)

    if len(variables) != len(continuous_variables_idx):
        logger.debug(
            f"{len(continuous_variables_idx)}/{len(variables)} variables to use for the violin plot, "
            "ignoring categorical & binary variables (only continuous variables supported)."
        )

    if len(continuous_variables_idx) == 0:
        logger.debug("No continuous variables in the dataset - not generating a violin plot.")
        return None

    # then plot the violin plot for selected user/datapoint/input with user_id
    user_data = imputations[:, user_id, :]  # shape (sample_count, variable_count)
    user_mask = mask[user_id]  # shape (variable_count,)

    # this function collects the stats needed and computes a representation for the extreme values/outliers
    def collect_stats_values(stats, user_id, normalized):
        median = stats["median"][user_id]
        q1 = stats["quartile_1"][user_id]
        q3 = stats["quartile_3"][user_id]
        iqr = q3 - q1
        upper_adjacent_value = np.clip(q3 + iqr * 1.5, q3, stats["max_val"][user_id])
        lower_adjacent_value = np.clip(q1 - iqr * 1.5, stats["min_val"][user_id], q1)
        collected_stats = np.array([median, q1, q3, lower_adjacent_value, upper_adjacent_value])
        if normalized:
            collected_stats = (collected_stats - stats["variable_lower"]) / (
                stats["variable_upper"] - stats["variable_lower"]
            )
        return collected_stats

    if ax is None:
        _, ax = plt.subplots()
        save_figs = True
    else:
        save_figs = False
    missing_idx = list(np.where(user_mask[continuous_list_idx] == 0)[0])
    for i in range(len(continuous_list_idx)):
        user_data_ = user_data[:, continuous_list_idx[i]].astype(float)
        if normalized:
            lower = imputations_stats[continuous_variables_idx[i]]["variable_lower"]
            upper = imputations_stats[continuous_variables_idx[i]]["variable_upper"]
            user_data_ = (user_data_ - lower) / (upper - lower)
        if i in missing_idx:
            # check the variation of the imputations for different Monte Carlo samples
            diff = user_data_.max() - user_data_.min()
            if diff < 5e-3:
                # the difference range is too small, plot mean prediction instead
                ax.scatter([i], user_data_.mean(), marker="_", s=30, zorder=3)
                missing_idx.remove(i)
            else:
                ax.violinplot(
                    dataset=user_data_,
                    positions=[i],
                    showmeans=False,
                    showmedians=False,
                    showextrema=False,
                )
            if continuous_variables_idx[i] in target_variables_idx:
                ax.axvspan(i - 0.5, i + 0.5, alpha=0.1, color="y")
        else:
            # by preserving the observed data, now all the entries in user_data_ should be equal
            ax.scatter([i], user_data_[0], marker="x", color="k", s=30, zorder=3)

    collected_stats = np.asarray(
        [
            collect_stats_values(imputations_stats[continuous_variables_idx[i]], user_id, normalized)
            for i in range(len(continuous_list_idx))
        ],
        dtype="f",
    )
    inds = np.arange(0, len(continuous_list_idx))
    # plot median
    ax.scatter(
        missing_idx,
        collected_stats[:, 0][missing_idx],
        marker="o",
        color="r",
        s=30,
        zorder=3,
    )
    # plot IQR range
    ax.vlines(
        missing_idx,
        collected_stats[:, 1][missing_idx],
        collected_stats[:, 2][missing_idx],
        color="k",
        linestyle="-",
        lw=5,
    )
    # plot extreme values/outlier range
    ax.vlines(
        missing_idx,
        collected_stats[:, 3][missing_idx],
        collected_stats[:, 4][missing_idx],
        color="k",
        linestyle="-",
        lw=1,
    )

    # labeling
    ax.get_xaxis().set_tick_params(direction="out")
    ax.xaxis.set_ticks_position("bottom")
    ax.set_xticks(inds)
    ax.set_xticklabels(continuous_variables_names)
    ax.set_xlim(-0.5, len(continuous_list_idx) + 0.5)
    ax.set_xlabel("variable name")
    if normalized:
        ax.set_ylabel("variable value (normalized)")
        ax.set_ylim(-0.1, 1.1)
    else:
        ax.set_ylabel("variable value")

    if title is None:
        title = "Violin plot for continuous variables"
    ax.set_title(f"{title} (user_id = {user_id})")

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="X",
            color="w",
            markerfacecolor="k",
            label="observed feature",
            markersize=7,
        )
    ]
    ax.legend(handles=legend_elements)

    if save_figs:
        if plot_name is None:
            plot_name = f"violin_plot_user_id{user_id}"
        if save_path is None:
            save_path = "plots"
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(
            os.path.join(save_path, plot_name + ".png"),
            format="png",
            dpi=200,
            bbox_inches="tight",
        )

    # construct return stats
    keys = ["median", "quartile_1", "quartile_3", "lower_fence", "upper_fence"]
    user_stats: Dict[str, Any] = {"variable_names": continuous_variables_names}
    for i in range(5):
        user_stats[keys[i]] = collected_stats[:, i]
    return user_stats, ax


def bar_plot_from_per_var(
    means_per_var,
    errs_per_var,
    labels,
    sort_by=None,
    colours=("b", "r", "g", "y", "c", "k"),
    total_bar_width=0.6,
    title="Bar plot",
    save_path="plots",
    plot_name=None,
):
    """
    Plot multiple bars for different quantities in the same plot, with the x-axis being a set of shared variables for which
    each bar plot has an associated quantity.

    Args:
        means_per_var (dict): Dictionary of the form plot_name : plot_means_per_var where plot_means_per_var is itself a
            dictionary of the form variable_name : bar_value. The x-axis of the generated plot will be all variable_names
            and for each variable_name there will be a bar plotted for each plot_name at height bar_value.
        errs_per_var (dict): Dictionary of the same form as means_per_var but bar_value replaced by error_height. The generated
            plot will add error bars of height error_height.
        labels (dict): Dictionary of the form plot_name : label_string containing y-axis labels for each of the plot_names.
        sort_by (str): Must be equal to one of the plot_name keys found in means_per_var or set to None (default). The x-axis
            will be sorted according to the bar_values (from highest to lowest) of the corresponding plot_name.
        colours (tuple): Tuple of matplotlib colours to use for the bars of the plot. At least as many colours must be passed as
            there are plot_name keys in means_per_var.
        total_bar_width (float): Parameter to adjust total width of the bars for a given variable. Defaults to 0.6.
        title (str): Title for plot.
        save_path (path): Path to save generated plot to. Defaults to "plots".
        plot_name (str): File name for generated plot. If set to None, a random one will be generated.
    """
    lbls = list(set(labels.values()))

    if len(lbls) > 2:
        raise ValueError(
            f"Cannot plot more than 2 different axis. {len(set(labels.values()))} different labels found in labels."
        )

    plots = list(means_per_var.keys())

    plt.clf()
    if sort_by is not None:
        x_labels = sorted(
            means_per_var[sort_by].keys(),
            key=means_per_var[sort_by].__getitem__,
            reverse=True,
        )
    else:
        x_labels = list(means_per_var[plots[0]])
    x_ticks = np.arange(len(x_labels))
    bar_width = total_bar_width / len(plots)

    mean = {}
    err = {}

    for plot in plots:
        mean[plot] = [means_per_var[plot][x_label] for x_label in x_labels]
        err[plot] = [errs_per_var[plot][x_label] for x_label in x_labels]

    ax = {lbls[0]: plt.subplot(111)}
    plt.xticks(x_ticks, x_labels, rotation=45)

    if len(lbls) == 2:
        ax[lbls[1]] = ax[lbls[0]].twinx()

    for label in lbls:
        ax[label].set_ylabel(label)

    displace = -0.3
    x_pos = {sort_by: displace}
    displace += bar_width
    for plot in plots:
        if plot != sort_by:
            x_pos[plot] = displace
            displace += bar_width

    colors = dict(zip(plots, colours[: len(plots)]))

    bars = {}
    for label in lbls:
        for plot in plots:
            if labels[plot] == label:
                bars[plot] = ax[label].bar(
                    x_ticks + x_pos[plot],
                    mean[plot],
                    width=bar_width,
                    align="edge",
                    label=plot,
                    color=colors[plot],
                    yerr=err[plot],
                )

    if len(lbls) == 2:
        align_yaxis_np(ax[lbls[0]], ax[lbls[1]])

    plt.title(title)
    plt.legend([bars[plot] for plot in plots], list(plots))

    if plot_name is None:
        plot_name = str(uuid4())
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(
        os.path.join(save_path, plot_name + ".png"),
        format="png",
        dpi=200,
        bbox_inches="tight",
    )


def align_yaxis_np(ax1, ax2):
    """
    Align zeros of the two axes, zooming them out by same ratio.
    """

    axes = np.array([ax1, ax2])
    extrema = np.array([ax.get_ylim() for ax in axes])
    tops = extrema[:, 1] / (extrema[:, 1] - extrema[:, 0])

    # Ensure that plots (intervals) are ordered bottom to top:
    if tops[0] > tops[1]:
        axes, extrema, tops = [a[::-1] for a in (axes, extrema, tops)]

    # How much would the plot overflow if we kept current zoom levels?
    tot_span = tops[1] + 1 - tops[0]

    extrema[0, 1] = extrema[0, 0] + tot_span * (extrema[0, 1] - extrema[0, 0])
    extrema[1, 0] = extrema[1, 1] + tot_span * (extrema[1, 0] - extrema[1, 1])
    for i in range(2):
        axes[i].set_ylim(*extrema[i])


def plot_rewards_scatter(
    myopic_rewards,
    non_myopic_rewards,
    total_rewards,
    max_steps,
    next_qs_lists,
    feature_count,
):
    # For non myopic policy this is going to plot the rewards for each question
    plt.clf()
    _, axes = plt.subplots(1, max_steps, figsize=(12, 5))

    for step_idx in trange(max_steps):
        axes[step_idx].scatter(
            next_qs_lists[step_idx][0][0],
            total_rewards[step_idx][next_qs_lists[step_idx][0][0]],
            marker="*",
            s=300,
            alpha=0.5,
            zorder=-1,
        )
        axes[step_idx].scatter(
            np.linspace(0, feature_count - 1, feature_count),
            myopic_rewards[step_idx],
            color="blue",
            label="myopic_rewards",
        )
        axes[step_idx].scatter(
            np.linspace(0, feature_count - 1, feature_count),
            non_myopic_rewards[step_idx],
            color="red",
            label="non_myopic",
        )
        axes[step_idx].scatter(
            np.linspace(0, feature_count - 1, feature_count),
            total_rewards[step_idx],
            color="green",
            label="total_rewards",
            marker="x",
            s=200,
        )
        axes[step_idx].legend()
    plt.show()


def plot_rewards_hist(rewards_list, save_dir):
    """
    Plot the histograms of the rewards computed to select next question

    Args:
        rewards_list: list of rewards for each steps. For each step this is a list of rewards for each user.
        save_dir: Directory to save plot to.
    """

    max_num_rows = 5
    max_num_col = 3

    col_count = min(len(rewards_list[0]), max_num_col)
    row_count = min(len(rewards_list), max_num_rows)

    _, axs = plt.subplots(row_count, col_count, figsize=(25, 15))

    for step_idx, reward_per_step in enumerate(rewards_list):
        if step_idx < max_num_rows:

            for user_id, reward_per_user in enumerate(reward_per_step):
                if user_id < max_num_col:

                    if col_count == 1:
                        axs[step_idx].bar(
                            list(reward_per_user.keys()),
                            reward_per_user.values(),
                            color="g",
                            align="center",
                        )
                        axs[step_idx].set_title(f"User {user_id}, Step {step_idx}")
                        axs[step_idx].set_ylabel("Reward")
                    else:
                        axs[step_idx, user_id].bar(
                            list(reward_per_user.keys()),
                            reward_per_user.values(),
                            color="g",
                            align="center",
                        )
                        axs[step_idx, user_id].set_title(f"User {user_id}, Step {step_idx}")
                        axs[step_idx, user_id].set_ylabel("Reward")
                else:
                    break
        else:
            break

    save_path = os.path.join(save_dir, "hist_rewards.png")
    plt.savefig(save_path, format="png", dpi=200, bbox_inches="tight")
    logger.info(f"Saved plot to {save_path}")


def plot_difficulty_curves(strategy, observations, difficulty, save_dir, steps=None):
    """
    Plot difficulty level for picked question over steps

    Args:
        strategy: objective function
        observations: numpy array of shape (user_count, step_count) with observation taken at each step.
        variables: List of variables.
        difficulty: difficulty levels for each feature in the dataset.
        save_dir: Directory to save plots to.
        steps: Number of steps to plot. Defaults to None - all steps are plotted.

    """
    plt.clf()
    plt.figure(figsize=(6.4, 4.8))

    if steps is not None:
        observations = observations[:, :steps]
    else:
        _, steps = observations.shape

    xs = np.arange(1, steps + 1)

    plt.clf()
    plt.xticks(xs)
    plt.xlabel("Steps")
    plt.ylabel(f"{strategy}_Question difficulty")

    average_difficulty = 0.0
    for user in observations:
        difficulty_to_plot = [difficulty[int(i)] for i in user]
        average_difficulty += np.array(difficulty_to_plot)
        # Plot transparent lines - same path with multiple occurrences will appear darker.
        plt.plot(xs, difficulty_to_plot, linestyle="-", marker="o", color="blue", alpha=0.2)

    average_difficulty = average_difficulty / observations.shape[0]

    # Plot average difficulty for each step across students
    plt.plot(
        xs,
        average_difficulty,
        linestyle="-",
        marker="o",
        color="red",
        alpha=1.0,
        zorder=1,
        label="Average difficulty",
    )

    plt.legend()
    save_path = os.path.join(save_dir, "difficulty.png")
    plt.savefig(save_path, format="png", dpi=200, bbox_inches="tight")
    logger.info(f"Saved plot to {save_path}")


def plot_quality_curves(strategy, observations, quality, save_dir, steps=None):
    """
    Plot quality level for picked question over steps

    Args:
        strategy: objective function
        observations: numpy array of shape (user_count, step_count) with observation taken at each step.
        variables: List of variables.
        quality: quality level for each feature in the dataset and for each user.
        save_dir: Directory to save plots to.
        steps: Number of steps to plot. Defaults to None - all steps are plotted.

    """
    plt.clf()
    plt.figure(figsize=(6.4, 4.8))
    if steps is not None:
        observations = observations[:, :steps]
    else:
        _, steps = observations.shape

    xs = np.arange(1, steps + 1)

    plt.clf()
    plt.xticks(xs)
    plt.xlabel("Steps")
    plt.ylabel(str(strategy) + "_" + "Question quality")

    average_quality = 0.0
    for user in observations:

        quality_to_plot = [quality[int(i)] for i in user]
        average_quality += np.array(quality_to_plot)
        # Plot transparent lines - same path with multiple occurrences will appear darker.
        plt.plot(xs, quality_to_plot, linestyle="-", marker="o", color="green", alpha=0.2)

    average_quality = average_quality / observations.shape[0]

    # Plot average quality for each step across students
    plt.plot(
        xs,
        average_quality,
        linestyle="-",
        marker="o",
        color="red",
        alpha=1.0,
        zorder=1,
        label="Average quality",
    )

    plt.legend()
    save_path = os.path.join(save_dir, "quality.png")
    plt.savefig(save_path, format="png", dpi=200, bbox_inches="tight")
    logger.info(f"Saved plot to {save_path}")


def plot_target_curves(imputed_values_per_strategy, save_dir):
    """
    Plot the target value Y per step

    Args:
        imputed_values_per_strategy (dict): {strategy: np array with shape
            (seed_count, user_count, step_count, variable_count)
            or (user_count, step_count, variable_count)}
        save_dir (str): Directory to save plots to.

    """
    plt.clf()
    plt.figure(figsize=(6.4, 4.8))
    save_path = os.path.join(save_dir, "Y.png")

    for strategy, imputed_values in imputed_values_per_strategy.items():

        user_count, step_count, _ = imputed_values.shape

        steps = np.arange(step_count)

        metric_per_step = np.zeros((step_count, user_count))
        stderr_per_step = np.zeros((step_count, user_count))

        for step_idx in steps:
            imputed_for_step = imputed_values[:, step_idx, :]  # Shape (users, features)

            metric_per_step[step_idx] = np.mean(imputed_for_step, axis=1)
            stderr_per_step[step_idx] = np.std(imputed_for_step, axis=1)

        if strategy == "b_ei":
            color = "blue"
            style = "--"
        else:
            style = "-"
            color = "green"

        cmap = plt.get_cmap("jet_r")

        for user_id in np.arange(user_count):
            color = cmap(float(user_id) / user_count)

            if strategy == "b_ei":
                plt.plot(
                    [0, step_count + 1],
                    metric_per_step[:, user_id],
                    c=color,
                    label=strategy + "_student" + str(user_id),
                    linestyle=style,
                )
            else:
                plt.plot(
                    steps,
                    metric_per_step[:, user_id],
                    c=color,
                    label=strategy + "_student" + str(user_id),
                    linestyle=style,
                )

        plt.legend(loc="center left")
        plt.xlabel("Steps")
        plt.ylabel("Avg target")

        plt.savefig(save_path, format="png", dpi=200, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")


def plot_mean_target_curves(imputed_values_per_strategy, max_steps, save_dir):
    """
    Plot the mean target value per step across students

    Args:
        imputed_values_per_strategy (dict): {strategy: np array with shape
            (seed_count, user_count, step_count, variable_count)
            or (user_count, step_count, variable_count)}
        save_dir (str): Directory to save plots to.

    """
    plt.clf()
    plt.figure(figsize=(6.4, 4.8))
    save_path = os.path.join(save_dir, "mean_Y.png")

    for strategy, imputed_values in imputed_values_per_strategy.items():

        user_count, step_count, _ = imputed_values.shape

        if step_count < 3:
            steps = np.arange(step_count)
            steps_to_plot = [0, max_steps]
        else:
            steps = np.arange(step_count)
            steps_to_plot = steps

        metric_per_step = np.zeros((step_count, user_count))
        stderr_per_step = np.zeros((step_count, user_count))

        for step_idx in steps:
            imputed_for_step = imputed_values[:, step_idx, :]  # Shape (users, features)

            metric_per_step[step_idx] = np.mean(imputed_for_step, axis=1)
            stderr_per_step[step_idx] = np.std(imputed_for_step, axis=1)

        if strategy == "ei":
            color = "red"
            label = "EI"
            linestyle = "dashed"
            marker = "D"
        elif strategy == "b_ei":
            color = "blue"
            label = "B_EI"
            linestyle = "dotted"
            marker = "*"
        else:
            color = "pink"
            label = "Other"
            linestyle = "-"
            marker = "o"

        mean_path = np.mean(metric_per_step, axis=1)
        plt.plot(
            steps_to_plot,
            mean_path,
            color=color,
            label=label,
            linestyle=linestyle,
            marker=marker,
        )

        plt.legend()
        plt.xlabel("Steps")
        plt.ylabel("Avg target")

        plt.savefig(save_path, format="png", dpi=200, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")


def plot_time_results_hist(imputed_values_per_strategy, save_dir):
    plt.clf()
    plt.figure(figsize=(6.4, 4.8))
    save_path = os.path.join(save_dir, "results_time_hist.png")

    bp_list = []
    label_list = []

    for strategy, imputed_values in imputed_values_per_strategy.items():
        time = np.loadtxt(save_dir + "/" + strategy + "/time.csv")

        last_imputed_values = imputed_values[:, -1, :]

        y_per_time = np.mean(np.mean(last_imputed_values, axis=1))

        if strategy == "ei":
            color = "red"
            label = "EI"
        elif strategy == "b_ei":
            color = "blue"
            label = "B_EI"
        else:
            color = "pink"

            label = "Other"

        bp = plt.boxplot(
            y_per_time,
            patch_artist=True,
            boxprops=dict(facecolor=color, color=color),
            capprops=dict(color=color),
            whiskerprops=dict(color=color),
            flierprops=dict(color=color, markeredgecolor=color),
            medianprops=dict(color=color),
            positions=[np.around(time, 2)],
        )

        bp_list.append(bp)
        label_list.append(label)

    plt.legend([bp["boxes"][0] for bp in bp_list], label_list, loc="best")
    plt.xlabel("Time (s)")
    plt.ylabel("Avg target")

    plt.savefig(save_path, format="png", dpi=200, bbox_inches="tight")
    logger.info(f"Saved plot to {save_path}")


def plot_training_metrics(
    results_dict: Dict[str, List[float]],
    variable: Variable,
    save_dir: str,
    metric: str = "loss",
):
    """
    Plot training metrics for marginal networks of VAEM
    Args:
        results_dict: Dictionary of training metrics for each marginal network.
        variable: Variable object corresponding to the feature which marginal is being plot.
        save_dir: Directory to save plots to.
        metric: Which metric (loss, kl, nll) . Defaults to'loss'.
                Options: 'loss', 'kl', 'nll', 'all'

    """
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    if metric == "all":
        metrics_to_plot = ["training_loss", "kl", "nll"]
    else:
        metrics_to_plot = [metric if metric in ["kl", "nll"] else "training_loss"]

    for m in metrics_to_plot:
        save_name = os.path.join(save_dir, f"{m}_{variable.name}.png")
        plt.clf()
        plt.xlabel("Epoch")
        plt.ylabel(m)
        plt.plot(results_dict[m])
        plt.title(m.upper() + " " + str(variable.name))
        plt.savefig(save_name, format="png", dpi=200, bbox_inches="tight")


def plot_nonzero_coverage(df: pd.DataFrame) -> Tuple[Figure, Axes]:
    """Plot horizontal bars showing the fraction of nonzero values by column.

    Args:
        df: DataFrame for plot

    Returns:
        Figure: Figure of plot
        Axes: Axes of plot
    """
    coverage = (np.count_nonzero(df, axis=0) - df.isna().sum(0).values) / len(df)
    order = np.argsort(coverage)
    coverage = coverage[order]
    labels = df.columns[order]

    fig, ax = plt.subplots(figsize=[16, 12])
    col_map = plt.get_cmap("tab20_r")
    ax.barh(labels, coverage, color=col_map.colors, edgecolor="maroon")
    ax.set_xlabel("Coverage", fontsize=15)
    ax.set_ylabel("Treatments", fontsize=15)
    return fig, ax
