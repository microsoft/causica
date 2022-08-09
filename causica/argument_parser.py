import argparse
import os
import textwrap

import numpy as np


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a partial VAE model.", formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("dataset_name", help="Name of dataset to use.")
    parser.add_argument(
        "--data_dir",
        "-d",
        type=str,
        default="data",
        help="Directory containing saved datasets. Defaults to ./data",
    )
    parser.add_argument(
        "--model_type",
        "-mt",
        type=str,
        default="deci",
        choices=[
            "visl",
            "deci",
            "deci_gaussian",
            "deci_spline",
            "pc",
            "notears_linear",
            "notears_mlp",
            "notears_sob",
            "grandag",
            "icalingam",
            "dowhy",
            "deci_dowhy",
            "true_graph_dowhy",
            "pc_dowhy",
            "pc_informed_deci",
            "informed_deci",
            "varlingam",
            "fold_time_deci",
            "auto_regressive_deci",
        ],
        help=textwrap.dedent(
            """Type of model to train.
            pvae: Default Partial VAE model from EDDI paper. For the IWAE version, set the use_importance_sampling flag in the model config .
            vaem: Default model from VAEM paper.
            vaem_predictive: The target is treated as a separate node p(y|x,h).
            transformer_pvae: Partial-VAE with a transformer in encoder and decoder.
            transformer_encoder_pvae: Partial-VAE with set transformer in encoder.
            transformer_encoder_vaem: VAEM with set transformer in encoder.
            bayesian_pvae: Weights as stochastic variable. Using inducing point VI to infer.
            mnar_pvae: Default uses identifiable (P)-VAE instead of vanilla VAE for MNAR case.
                With different settings (relative to the default setting for mnar_pvae), we can recover different models:
                     use_importance_sampling = False -> mnar_pvae without importance sampling
                     mask_net_coefficient = 0 -> MISSIWAE
                     mask_net_coefficient = 0, use_importance_sampling = False -> PVAE
                     use_prior_net_to_train = False, latent_connection = False -> Not-MissIWAE
                     use_prior_net_to_train = False -> Not-MIssIWAE with latent connection
                     use_prior_net_to_train = False, latent_connection = False, use_importance_sampling = False -> Not-MissPVAE
                     use_prior_net_to_train = False,  use_importance_sampling = False -> Not-MissPVAE with latent connection
            transformer_imputer: Directly using transformer for imputation. Only works with active learning strategies variance or rand.
            different graph_neural_network models: Graph neural network-based models for missing data imputation. Does not work with active learning.
                Changing the model configuration leads to different GNN recommendation models developed over the past years
                (for detailed model configurations, please refer to the model_config json files in parameters/defaults/ with
                the specified model names):
                    CoRGi:
                        CoRGi is a GNN model that considers the rich data within nodes in the context of their neighbors.
                        This is achieved by endowing CORGI’s message passing with a personalized attention
                        mechanism over the content of each node. This way, CORGI assigns user-item-specific
                        attention scores with respect to the words that appear in items.
                    Graph Convolutional Network (GCN):
                        As a default, "average" is used for the aggregation function
                        and nodes are randomly initialized.
                        We adopt dropout with probability 0.5 for node embedding updates
                        as well as for the prediction MLPs.
                    GRAPE:
                        GRAPE is a GNN model that employs edge embeddings.
                        Also, it adopts edge dropouts that are applied throughout all message-passing layers.
                        Compared to the GRAPE proposed in the oroginal paper, because of the memory issue,
                        we do not initialize nodes with one-hot vectors nor constants (ones).
                    Graph Convolutional Matrix Completion (GC-MC):
                        Compared to GCN, this model has a single message-passing layer.
                        Also, For classification, each label is endowed with a separate message passing channel.
                        Here, we do not implement the weight sharing.
                    GraphSAGE:
                        GraphSAGE extends GCN by allowing the model to be trained on the part of the graph,
                        making the model to be used in inductive settings.
                    Graph Attention Network (GAT):
                        During message aggregation, GAT uses the attention mechanism to allow the target nodes to
                        distinguish the weights of multiple messages from the source nodes for aggregation.

            deep_matrix_factorization: Deep matrix factorization (DMF) for missing data imputation.
                It's a deterministic model that uses an arbitrary value to fill in  the missing entries of the imputation matrix.
                The value that replaces NaN can be assigned in missing_fill_val in the training_hyperparams of the model config file.
                Does not work with active learning.
            visl: Simultaneous missing value imputation and causal discovery using neural relational inference.
            deci: Causal discovery using flow based model while doing approximate inference over the adjacency matrix. Supports estimating treatment effects.
            deci_gaussian: Causal discovery using flow based model while doing approximate inference over the adjacency matrix. Supports estimating treatment effects. Use Gaussian base distribtion
            deci_spline: Causal discovery using flow based model while doing approximate inference over the adjacency matrix. Supports estimating treatment effects. Use Spline base distribution
            min_imputing: Impute missing values using the minimum value for the corresponding variable.
            mean_imputing: Impute missing values using the mean observed value for the corresponding variable.
            zero_imputing: Impute missing values using the value 0.
            majority_vote: Impute missing values using the most common observed value for the feature.
            mice: Impute missing values using the iterative method Multiple Imputation by Chained Equations (MICE).
            missforest: Impute missing values using the iterative random forest method MissForest.
            notears_linear: Linear version of notears algorithm for causal discovery (https://arxiv.org/abs/1803.01422).
            notears_mlp: Nonlinear version (MLP) of notears algorithm for causal discovery (https://arxiv.org/abs/1909.13189).
            notears_sob: Nonlinear version (Sobolev) of notears algorithm for causal discovery (https://arxiv.org/abs/1909.13189).
            grandag: GraNDAG algorithm for causal discovery using MLP as nonlinearities (https://arxiv.org/abs/1906.02226).
            pc: PC algorithm for causal discovery (https://arxiv.org/abs/math/0510436).
            icalingam: ICA based causal discovery (https://dl.acm.org/doi/10.5555/1248547.1248619).
            do_why: Causal inference from predefined causal graph and observational data.
                Note that a causal graph or causal discovery model save will need to be specified in config file. (https://arxiv.org/abs/2011.04216)
            deci_dowhy: Causal discovery using DECI. Causal inference using DoWhy.
            true_graph_dowhy:  Causal inference using DoWhy when the causal graph is set to the ground truth.
            pc_dowhy:  Causal discovery using PC. Causal inference using DoWhy.
            pc_informed_deci: Causal discovery using PC CPDAG as a soft prior for DECI. Causal inference using DECI,
            informed_deci: Causal discovery using the true graph as a soft prior for DECI. Causal inference using DECI,
            varlingam: VARLiNGaM model for causal time series discovery.
            fold_time_deci: The Fold-time DECI model for causal time series discovery.
            auto_regressive_deci: The Auto-regressive DECI model for end-to-end causal inference of time series.
            """
        ),
    )
    parser.add_argument("--model_dir", "-md", default=None, help="Directory containing the model.")
    parser.add_argument("--model_config", "-m", type=str, help="Path to JSON containing model configuration.")
    parser.add_argument("--dataset_config", "-dc", type=str, help="Path to JSON containing dataset configuration.")
    parser.add_argument("--impute_config", "-ic", type=str, help="Path to JSON containing impute configuration.")
    parser.add_argument("--objective_config", "-oc", type=str, help="Path to JSON containing objective configuration.")

    # Whether or not to run inference and active learning
    parser.add_argument("--run_inference", "-i", action="store_true", help="Run inference after training.")
    parser.add_argument("--extra_eval", "-x", action="store_true", help="Run extra eval tests that take longer.")
    parser.add_argument(
        "--max_steps", "-ms", type=int, default=np.inf, help="Maximum number of active learning steps to take."
    )
    parser.add_argument(
        "--max_al_rows",
        "-mar",
        type=int,
        default=np.inf,
        help="Maximum number of rows on which to perform active learning.",
    )
    parser.add_argument(
        "--active-learning",
        "-a",
        nargs="+",
        choices=[
            "eddi",
            "eddi_mc",
            "eddi_rowwise",
            "rand",
            "cond_sing",
            "sing",
            "ei",
            "b_ei",
            "variance",
            "all",
        ],
        help="""Run active learning after train and test.
                                eddi = personalized information acquisition from EDDI paper
                                eddi_mc = personalized information acquisition using Bayesian EDDI with stocastic weights
                                eddi_rowwise = same as eddi but with row-wise parallelization for information gain computation
                                rand = random strategy for information acquisition
                                cond_sing = conditional single order strategy across the whole test dataset where the next best step condition on existing observation
                                sing = single order strategy determinated by first step information gain
                                ei = expected improvement,
                                b_ei = batch ei""",
    )
    parser.add_argument(
        "--users_to_plot", "-up", default=[0], nargs="+", help="Indices of users to plot info gain bar charts for."
    )
    # Whether or not to evaluate causal discovery (only visl at the moment)
    parser.add_argument(
        "--causal_discovery",
        "-c",
        action="store_true",
        help="Whether to evaluate causal discovery against a ground truth during evaluation.",
    )
    parser.add_argument(
        "--treatment_effects",
        "-te",
        action="store_true",
        help="Whether to evaluate treatment effects against a ground truth.",
    )
    # Other options for saving output.

    parser.add_argument("--output_dir", "-o", type=str, default="runs", help="Output path. Defaults to ./runs/.")
    parser.add_argument("--name", "-n", type=str, help="Tag for this run. Output dir will start with this tag.")
    parser.add_argument(
        "--device", "-dv", default="cpu", help="Name (e.g. 'cpu', 'gpu') or ID (e.g. 0 or 1) of device to use."
    )
    parser.add_argument("--tiny", action="store_true", help="Use this flag to do a tiny run for debugging")
    parser.add_argument(
        "--random_seed",
        type=int,
        help="Random seed for training. If not provided, a random seed will be taken from the model config JSON",
    )
    parser.add_argument(
        "--default_configs_dir",
        "-dcd",
        type=str,
        default="configs",
        help="Directory containing configs. Defaults to ./configs",
    )
    # Control the logger level
    parser.add_argument(
        "--logger_level",
        "-ll",
        type=str.upper,
        default="INFO",
        choices=["CRITICAL", "ERROR", "INFO", "DEBUG", "WARNING"],
        help="Control the logger level. Default: %(default)s .",
    )
    # Control whether evaluating the log-likelihood for Causal models
    parser.add_argument(
        "--eval_likelihood",
        "-el",
        action="store_false",
        help="Disable the likelihood computation for causal models during treatment effect estimation.",
    )
    parser.add_argument(
        "--conversion_type",
        "-ct",
        type=str.lower,
        default="full_time",
        choices=["ful_time", "auto_regressive"],
        help="The type of conversion used for converting the temporal adjacency matrix to a static adjacency matrix during causal discovery evaluation",
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate command-line arguments.

    """
    assert os.path.isdir(args.data_dir), f"{args.data_dir} is not a directory."

    # Config files
    for config in (args.model_config, args.dataset_config, args.impute_config, args.objective_config):
        if config is not None and not os.path.isfile(config):
            raise ValueError(f"Config file {config} does not exist.")
