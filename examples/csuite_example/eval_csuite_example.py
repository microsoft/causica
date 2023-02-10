import argparse
import dataclasses
import math
from typing import Dict, Optional

import mlflow
import pytorch_lightning as pl
import torch
from tensordict import TensorDict

from causica.datasets.csuite_data import (
    CSUITE_DATASETS_PATH,
    CounterfactualsWithEffects,
    DataEnum,
    InterventionsWithEffects,
    get_categorical_sizes,
    load_data,
)
from causica.datasets.interventional_data import CounterfactualData, InterventionData
from causica.datasets.tensordict_utils import convert_one_hot, tensordict_shapes
from causica.functional_relationships import ICGNN
from causica.graph.evaluation_metrics import adjacency_f1, orientation_f1
from causica.sem.distribution_parameters_sem import DistributionParametersSEM
from causica.training.evaluation import (
    eval_ate_rmse,
    eval_intervention_likelihoods,
    eval_ite_rmse,
    list_logsumexp,
    list_mean,
)
from causica.training.trainable_container import TrainableContainer, create_noise_dists

NUM_GRAPH_SAMPLES = 100
NUM_ATE_ITE_SEMS = 10
LAST_MODEL = "last_model.pt"


@dataclasses.dataclass
class EvaluationMetrics:
    adjacency_f1: float
    orientation_f1: float
    test_log_likelihood: float
    interventional_log_likelihood: float
    ate_rmse: float
    ite_rmse: Optional[float] = None


def eval_csuite(container: TrainableContainer, ite: bool = False) -> EvaluationMetrics:
    device = "cpu"
    container.to(device)

    true_adj_matrix = torch.tensor(
        load_data(CSUITE_DATASETS_PATH, container.dataset_name, DataEnum.TRUE_ADJACENCY), device=device
    )

    icgnn: ICGNN = container.icgnn
    node_names = list(icgnn.variables.keys())

    vardist = container.vardist.forward()
    graph_samples = vardist.sample(torch.Size([NUM_GRAPH_SAMPLES]))

    adj_f1 = list_mean([adjacency_f1(true_adj_matrix, graph) for graph in graph_samples]).item()
    orient_f1 = list_mean([orientation_f1(true_adj_matrix, graph) for graph in graph_samples]).item()

    variables_metadata = load_data(CSUITE_DATASETS_PATH, container.dataset_name, DataEnum.VARIABLES_JSON)

    categorical_sizes = get_categorical_sizes(variables_list=variables_metadata["variables"])

    def one_hotify(data: TensorDict) -> TensorDict:
        return convert_one_hot(data, categorical_sizes)

    dataset_test: TensorDict = one_hotify(
        load_data(CSUITE_DATASETS_PATH, container.dataset_name, DataEnum.TEST, variables_metadata)
    )
    dataset_test = dataset_test.apply(lambda t: t.to(dtype=torch.float32, device=device))
    variables = variables_metadata["variables"]
    types_dict = {var["group_name"]: var["type"] for var in variables}
    noise_dist_funcs, _ = create_noise_dists(
        shapes=tensordict_shapes(dataset_test),
        types_dict=types_dict,
        noise_dist=container.noise_dist_type,
        noise_dist_params=container.noise_dist,
    )

    # use an iterator to minimize the memory impact
    sems = (
        DistributionParametersSEM(graph=graph, node_names=node_names, noise_dist=noise_dist_funcs, func=icgnn)
        for graph in graph_samples
    )

    # estimate total log prob using graph samples and report the mean over graphs
    log_prob = torch.mean(
        list_logsumexp([sem.log_prob(dataset_test) for sem in sems]) - math.log(NUM_GRAPH_SAMPLES)
    ).item()

    interventions: InterventionsWithEffects = load_data(
        CSUITE_DATASETS_PATH, container.dataset_name, DataEnum.INTERVENTIONS, variables_metadata
    )

    def intersect(dict_1: Dict, dict_2: Dict):
        return {key: dict_1[key] for key in dict_1.keys() & dict_2.keys()}

    def one_hot_intervention(intervention: InterventionData) -> InterventionData:
        return InterventionData(
            convert_one_hot(
                intervention.intervention_data, intersect(categorical_sizes, intervention.intervention_data)
            ),
            convert_one_hot(
                intervention.intervention_values, intersect(categorical_sizes, intervention.intervention_values)
            ),
            convert_one_hot(intervention.condition_values, intersect(categorical_sizes, intervention.condition_values)),
        )

    interventions = [
        (one_hot_intervention(int_a), one_hot_intervention(int_b), effects) for int_a, int_b, effects in interventions
    ]

    graph_samples = vardist.sample(torch.Size([NUM_ATE_ITE_SEMS]))
    sems_list = [
        DistributionParametersSEM(graph=graph, node_names=node_names, noise_dist=noise_dist_funcs, func=icgnn)
        for graph in graph_samples
    ]

    interventional_log_prob = eval_intervention_likelihoods(sems_list, interventions).item()

    mean_ate_rmse = eval_ate_rmse(sems_list, interventions).item()

    if ite:
        counterfactuals: CounterfactualsWithEffects = load_data(
            CSUITE_DATASETS_PATH, container.dataset_name, DataEnum.COUNTERFACTUALS, variables_metadata
        )

        def one_hot_counterfactual(counterfactual: CounterfactualData) -> CounterfactualData:
            return CounterfactualData(
                convert_one_hot(
                    counterfactual.counterfactual_data, intersect(categorical_sizes, counterfactual.counterfactual_data)
                ),
                convert_one_hot(
                    counterfactual.intervention_values, intersect(categorical_sizes, counterfactual.intervention_values)
                ),
                convert_one_hot(counterfactual.factual_data, intersect(categorical_sizes, counterfactual.factual_data)),
            )

        counterfactuals = [
            (one_hot_counterfactual(c_a), one_hot_counterfactual(c_b), effects) for c_a, c_b, effects in counterfactuals
        ]
        mean_ite_rmse = eval_ite_rmse(sems_list, counterfactuals).item()
    else:
        mean_ite_rmse = None

    return EvaluationMetrics(
        adjacency_f1=adj_f1,
        orientation_f1=orient_f1,
        test_log_likelihood=log_prob,
        interventional_log_likelihood=interventional_log_prob,
        ate_rmse=mean_ate_rmse,
        ite_rmse=mean_ite_rmse,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-save-file", "-s", default=LAST_MODEL)
    parser.add_argument("--ite", default=False, action="store_true")

    args = parser.parse_args()

    pl.seed_everything(123)

    train_container: TrainableContainer = torch.load(args.model_save_file)["container"]
    eval_metrics = eval_csuite(train_container, args.ite)

    print(eval_metrics)
    mlflow.log_metric("eval/adjacency.f1", eval_metrics.adjacency_f1)
    mlflow.log_metric("eval/orientation.f1", eval_metrics.orientation_f1)
    mlflow.log_metric("eval/test_LL", eval_metrics.test_log_likelihood)
    mlflow.log_metric("eval/Interventional_LL", eval_metrics.interventional_log_likelihood)
    mlflow.log_metric("eval/ATE_RMSE", eval_metrics.ate_rmse)
    if eval_metrics.ite_rmse is not None:
        mlflow.log_metric("eval/ITE_RMSE", eval_metrics.ite_rmse)
