import logging
import os
from collections import defaultdict
from operator import itemgetter

import fsspec
import networkx as nx
import numpy as np
import pandas as pd
import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader

from causica.datasets.causica_dataset_format import (
    CounterfactualWithEffects,
    InterventionWithEffects,
    VariablesMetadata,
)
from causica.datasets.causica_dataset_format.load import (
    convert_one_hot,
    get_categorical_sizes,
    tensordict_from_variables_metadata,
)
from causica.datasets.interventional_data import CounterfactualData, InterventionData
from causica.datasets.tensordict_utils import identity, tensordict_shapes
from causica.datasets.variable_types import VariableTypeEnum
from causica.distributions.transforms import JointTransformModule
from causica.lightning.data_modules.variable_spec_data import VariableSpecDataModule

logger = logging.getLogger(__name__)


def get_intervention_and_cfs(df: pd.DataFrame) -> tuple[list[CounterfactualWithEffects], list[InterventionWithEffects]]:
    outcome = "Revenue"
    treatment_columns = ["Tech Support", "Discount", "New Engagement Strategy"]

    # Generate CounterfactualData and InterventionData objects for the test set
    cf_data: list[CounterfactualWithEffects] = []
    intervention_data: list[InterventionWithEffects] = []
    observations = df.loc[:, "Global Flag":"Revenue"]  # type: ignore
    for treatment in treatment_columns:
        ite_values = df.loc[:, f"Total Treatment Effect: {treatment}"].values
        interventions_with_effect: list[InterventionData | set[str] | None] = []
        for treatment_value in [0, 1]:
            treatment_mask = df[treatment] == 1 - treatment_value
            factual_td = TensorDict(
                {key: observations[key].values[treatment_mask][..., None] for key in observations.columns},
                batch_size=(treatment_mask.sum(),),
            )
            cf_td = factual_td.clone()
            cf_td[treatment] = 1 - cf_td[treatment]
            # Subtracting ITE from the factual outcome for negative treatments and adding it for positive treatments
            masked_ite_values = ite_values[treatment_mask]
            offset = (cf_td[treatment] * 2 - 1) * masked_ite_values
            cf_td[outcome] = factual_td[outcome] + offset

            treatment_td = TensorDict({treatment: torch.tensor([treatment_value])}, batch_size=[])

            cf_data.append(
                (
                    CounterfactualData(
                        factual_data=factual_td, counterfactual_data=cf_td, intervention_values=treatment_td
                    ),
                    None,
                    {outcome},
                )
            )
            interventions_with_effect.append(
                InterventionData(
                    intervention_data=cf_td,
                    intervention_values=treatment_td,
                    condition_values=TensorDict({}, batch_size=[]),
                )
            )
        interventions_with_effect.append({outcome})
        intervention_data.append(tuple(interventions_with_effect))  # type: ignore

    return cf_data, intervention_data


def get_constraint_prior(node_name_to_idx: dict[str, int]) -> np.ndarray:
    num_nodes = len(node_name_to_idx)
    constraint_matrix = np.full((num_nodes, num_nodes), np.nan, dtype=np.float32)

    revenue_idx = node_name_to_idx["Revenue"]
    planning_summit_idx = node_name_to_idx["Planning Summit"]
    constraint_matrix[revenue_idx, :] = 0.0
    constraint_matrix[revenue_idx, planning_summit_idx] = np.nan

    non_child_nodes = [
        "Commercial Flag",
        "Major Flag",
        "SMC Flag",
        "PC Count",
        "Employee Count",
        "Global Flag",
        "Size",
    ]
    non_child_idxs = itemgetter(*non_child_nodes)(node_name_to_idx)
    constraint_matrix[:, non_child_idxs] = 0.0

    engagement_nodes = ["Tech Support", "Discount", "New Engagement Strategy"]
    engagement_idxs = itemgetter(*engagement_nodes)(node_name_to_idx)
    for i in engagement_idxs:
        constraint_matrix[engagement_idxs, i] = 0.0

    return constraint_matrix


def _get_tensordict_from_df(df: pd.DataFrame, variables_metadata: VariablesMetadata, categorical_sizes) -> TensorDict:
    return convert_one_hot(
        tensordict_from_variables_metadata(df.to_numpy(), variables_metadata.variables),
        one_hot_sizes=categorical_sizes,
    )


class ExampleDataModule(VariableSpecDataModule):
    """Example of a lightning data module.

    This data module loads the multi-attribution dataset and prepares it for training. It uses the custom functions
    `get_intervention_and_cfs` and `get_constraint_prior` to generate CounterfactualData and InterventionData objects
    from the full dataframe. The true graph is also loaded from a file, but this is not available in most cases.

    The `prepare_data` handles all the data loading and preprocessing. In practice, this can be customised to any
    scenario however appropriate.
    """

    def prepare_data(self):
        # Load metadata telling us the data type of each column
        variables_path = os.path.join(self.root_path, "multi_attribution_data_20220819_data_types.json")
        with fsspec.open(variables_path, mode="r", encoding="utf-8") as f:
            self.variables_metadata = VariablesMetadata.from_json(f.read())
        self.categorical_sizes = get_categorical_sizes(variables_list=self.variables_metadata.variables)
        continuous_variables = [
            spec.name for spec in self.variables_metadata.variables if spec.type == VariableTypeEnum.CONTINUOUS
        ]

        # Load the data as a DataFrame
        df = pd.read_csv(os.path.join(self.root_path, "multi_attribution_data_20220819.csv"))
        df[continuous_variables] = df[continuous_variables].astype(float)

        # Load the true graph. In most cases, this will not be available.
        adjacency_path = os.path.join(self.root_path, "true_graph_gml_string.txt")
        with fsspec.open(adjacency_path, mode="r", encoding="utf-8") as f:
            self.true_adj = torch.tensor(nx.to_numpy_array(nx.parse_gml(f.read())))

        # Split into train, validation, and test sets
        shuffled_df = df.sample(frac=1, random_state=1337)
        train_df, valid_df, test_df = np.split(shuffled_df, [int(0.7 * len(df)), int(0.8 * len(df))])

        train_df = train_df.loc[:, "Global Flag":"Revenue"]
        valid_df = valid_df.loc[:, "Global Flag":"Revenue"]

        # Convert the data to TensorDicts
        self._dataset_train = _get_tensordict_from_df(train_df, self.variables_metadata, self.categorical_sizes)
        self._dataset_valid = _get_tensordict_from_df(valid_df, self.variables_metadata, self.categorical_sizes)
        self._dataset_test = _get_tensordict_from_df(
            test_df.loc[:, "Global Flag":"Revenue"], self.variables_metadata, self.categorical_sizes
        )

        # Generate CounterfactualData and InterventionData objects for the test set
        # In most cases this won't be available.
        self.counterfactuals, self.interventions = get_intervention_and_cfs(test_df)

        # Generate the constraint / prior
        node_name_to_idx = {key: i for i, key in enumerate(train_df.columns)}
        self.constraint_prior = get_constraint_prior(node_name_to_idx)

        # Set up utility variables
        self._variable_shapes = tensordict_shapes(self._dataset_train)
        self._variable_types = {var.group_name: var.type for var in self.variables_metadata.variables}
        self._column_names = defaultdict(list)
        for variable in self.variables_metadata.variables:
            self._column_names[variable.group_name].append(variable.name)

        # Normalize the data
        if self.use_normalizer:
            # Only applied to continuous variables
            normalization_variables = {k for k, v in self._variable_types.items() if v == VariableTypeEnum.CONTINUOUS}
            self.normalizer = self.create_normalizer(normalization_variables)(
                self._dataset_train.select(*normalization_variables)
            )
            self.normalize_data()
        else:
            self.normalizer = JointTransformModule({})

    def test_dataloader(self):
        return DataLoader(dataset=self.dataset_test, collate_fn=identity, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(dataset=self.dataset_valid, collate_fn=identity, batch_size=self.batch_size)
