# This is required in python 3 to allow return types of the same class.
from __future__ import annotations

import logging
import warnings
from collections import defaultdict
from distutils.util import strtobool
from typing import Any, DefaultDict, Dict, Iterator, List, Optional, Tuple, Union, cast

import numpy as np
import torch
from scipy.sparse import issparse

from ..utils.io_utils import read_json_as, save_json


class Variables:
    """
    This class represents any variables present in a model.
    """

    def __init__(
        self,
        variables: List[Variable],
        auxiliary_variables: Optional[List[Variable]] = None,
        used_cols: Optional[List[int]] = None,
    ) -> None:
        """
        Args:
            variables: A list Variable objects.
            auxiliary_variables: A list of Variable objects only used for input into VAE,
                not produced in output.
                These are assumed to be appended onto the end of the variables in the data.
                Defaults to None - no aux variables present.
            used_cols: A list of column ids that were used when processing the original data.
        """
        if not auxiliary_variables:
            auxiliary_variables = []
        self.auxiliary_variables = auxiliary_variables
        self._variables = variables

        self._deduplicate_names()

        # Dictionary mapping from variable name to variable index.
        self.name_to_idx = {var.name: idx for idx, var in enumerate(self._variables)}

        # Lists containing query and target variable indices
        self.target_var_idxs = []
        self.not_target_var_idxs = []
        self.query_var_idxs = []
        self.not_query_var_idxs = []
        for idx, var in enumerate(self._variables):
            if var.query:
                self.query_var_idxs.append(idx)
            else:
                self.not_query_var_idxs.append(idx)
            if var.target:
                self.target_var_idxs.append(idx)
            else:
                self.not_target_var_idxs.append(idx)

        if len(self.target_var_idxs) > 0 and all(idx in self.query_var_idxs for idx in self.target_var_idxs):
            warnings.warn(
                "All target variables are marked as queriable, it is likely that active learning will always "
                "select these variables first."
            )

        # Lists containing continuous (including text) and binary/categorical variable indices
        self.var_idxs_by_type: DefaultDict[str, List[int]] = defaultdict(list)
        for idx, var in enumerate(self._variables + self.auxiliary_variables):
            self.var_idxs_by_type[var.type_].append(idx)

        # List of lists, where self.unprocessed_cols[i] gives the columns occupied by the ith variable in the unprocessed
        # data.
        self.unprocessed_cols = []
        start_col = 0
        for var in self._all_variables:
            end_col = start_col + var.unprocessed_dim
            self.unprocessed_cols.append(list(range(start_col, end_col)))
            start_col = end_col

        # List of lists, where self.unprocessed_non_aux_cols[i] gives the columns occupied by the ith variable in the unprocessed
        # data (non-auxiliary).
        self.unprocessed_non_aux_cols = []
        start_col = 0
        for var in self._variables:
            end_col = start_col + var.unprocessed_dim
            self.unprocessed_non_aux_cols.append(list(range(start_col, end_col)))
            start_col = end_col

        # List of lists, where self.processed_cols[i] gives the columns occupied by the ith variable in the processed
        # data.
        self.processed_cols = []
        start_col = 0
        for var in self._all_variables:
            end_col = start_col + var.processed_dim
            self.processed_cols.append(list(range(start_col, end_col)))
            start_col = end_col

        # List of lists, where self.processed_non_aux_cols[i] gives the columns occupied by the ith variable in the processed
        # data (non-auxiliary).
        self.processed_non_aux_cols = []
        start_col = 0
        for var in self._variables:
            end_col = start_col + var.processed_dim
            self.processed_non_aux_cols.append(list(range(start_col, end_col)))
            start_col = end_col

        # Set of all query group names, maintaining order in which they are first encountered when iterating through
        # the variables list. This is the simplest way to do this since dictionaries are guaranteed to be
        # insertion-ordered since Python 3.7
        self.group_names = list(dict.fromkeys([var.group_name for var in self._variables]))

        # List containing indices for each query group, where the query group names are assumed to be in the same order
        # as self.group_names
        self.group_idxs = [
            [idx for idx, var in enumerate(self._variables) if var.group_name == group_name]
            for group_name in self.group_names
        ]

        # Remove groups containing no queriable variables from self.group_names and self.group_idxs, as
        # we can guarantee that we will never query these groups.
        is_group_queriable = [any(self._variables[idx].query for idx in idxs) for idxs in self.group_idxs]

        self.group_names = [name for group_idx, name in enumerate(self.group_names) if is_group_queriable[group_idx]]
        self.group_idxs = [idxs for group_idx, idxs in enumerate(self.group_idxs) if is_group_queriable[group_idx]]

        # Save the list of observed column ids
        default_used_cols = list(range(len(self._variables) + len(auxiliary_variables)))  # All columns observed
        self.used_cols = used_cols if used_cols is not None else default_used_cols
        assert len(self.used_cols) == len(self._variables) + len(self.auxiliary_variables)

        self.col_id_to_var_index = {old: new for new, old in enumerate(self.used_cols)}

    def __repr__(self):
        return str(self._variables)

    def __iter__(self) -> Iterator[Variable]:
        """
        Iterate through the variables within the container.
        Note - Now it iterate through all the variables within the container
        (including auxiliary variables, if they are present)
        """
        for var in self._all_variables:
            yield var

    def __getitem__(self, idx):
        return (self._all_variables)[idx]

    def __len__(self) -> int:
        return len(self._variables) + len(self.auxiliary_variables)

    @classmethod
    def create_from_json(cls, path: str) -> Variables:
        return cls.create_from_dict(read_json_as(path, dict))

    @classmethod
    def create_from_dict(cls, variables_dict: Dict[str, List[Any]]) -> Variables:
        """
        Create variables object from a dictionary
        """
        variables = variables_dict["variables"]
        for var in variables:
            # remove deprecated "id" key if present
            var.pop("id", None)
        var_obj_list = [Variable(**var) for var in variables]

        auxiliary_vars = variables_dict.get("auxiliary_variables", [])
        if len(auxiliary_vars) == 0:
            auxiliary_vars_obj = None
        else:
            for var in auxiliary_vars:
                # remove deprecated "id" key if present
                var.pop("id", None)

            auxiliary_vars_obj = [Variable(**var) for var in auxiliary_vars]

        used_cols = variables_dict.get("used_cols", None)

        return cls(var_obj_list, auxiliary_vars_obj, used_cols)

    @classmethod
    def create_from_data_and_dict(
        cls, data: np.ndarray, mask: np.ndarray, variables_dict: Optional[Dict[str, Any]] = None
    ) -> Variables:
        """
        Create variables object from an input dictionary, inferring missing fields using `data` and `mask`.
        """
        # Infer missing fields in variables_dict
        variables_dict = cls.infer_from_data(data, mask, variables_dict, True)
        variables = cls.create_from_dict(variables_dict)
        return variables

    @staticmethod
    def _metadata_from_dict(
        data, mask, variables_dict, variables_type="variables"
    ) -> Tuple[List[Any], Union[List[Any], None]]:
        """
        Infer variables_metadata from input data

        Args:
            data: NumPy array containing data
            mask: NumPy array containing 1 for observed data values, 0 for unobserved data values.
            variables_dict: Dictionary containing metadata for each variable (column) in the input data. Missing variables,
                or missing fields for a particular variable, will attempt to be inferred from the input data.
            variables_type: is it aux variables, or normal variables
        Returns:
            varaibles_metadata: inferred metadata from input data
            A list of column ids that were used when processing the original data.
        """

        variables_metadata = []
        # Use None rather than {} as default since mutable default args are dangerous in Python.
        used_cols = variables_dict.get("used_cols", None)
        if used_cols:
            used_cols = cast(List[int], used_cols)
            assert len(used_cols) == data.shape[1]

        for idx, variable_metadata in enumerate(variables_dict[variables_type]):
            if not all(
                k in variable_metadata for k in ["name", "type", "lower", "upper", "query", "target", "always_observed"]
            ):
                # If variable metadata fully specified, do not try to infer, as doing column indexing can be expensive
                # for CSR sparse matrices.
                var_data = data[:, idx]
                var_mask = mask[:, idx]
                if issparse(var_data):
                    var_data = var_data.toarray()
                    var_mask = var_mask.toarray()

                if "name" not in variable_metadata:
                    if used_cols:
                        variable_metadata["name"] = str(used_cols[idx])
                    else:
                        variable_metadata["name"] = f"Column {idx}"

                # If data type/min max/num categories specified explicitly, overwrite variables file
                if "type" not in variable_metadata:
                    # Test if all unmasked elements are integers

                    if np.all((var_data * var_mask) // 1 == var_data * var_mask):
                        if (var_data * var_mask).max() <= 1:
                            print(
                                f'Type of variable {variable_metadata["name"]} inferred as binary. This can be '
                                "changed manually in the dataset's variables.json file"
                            )
                            variable_metadata["type"] = "binary"
                        else:
                            # Note that we always infer integer values with a max value > 1 as categorical. This may want to be
                            # reconsidered if support for ordinal variables is introduced at a later date.
                            print(
                                f'Type of variable {variable_metadata["name"]} inferred as categorical. This can be'
                                " changed manually in the dataset's variables.json file"
                            )
                            variable_metadata["type"] = "categorical"
                    else:
                        variable_metadata["type"] = "continuous"

                if "lower" not in variable_metadata:
                    if variable_metadata["type"] == "binary":
                        inferred_lower = 0
                    else:
                        inferred_lower = min(var_data[np.where(var_mask == 1)]).item()
                    variable_metadata["lower"] = inferred_lower
                    print(
                        f'Minimum value of variable {variable_metadata["name"]} inferred as {inferred_lower}. This'
                        " can be changed manually in the dataset's variables.json file"
                    )

                if "upper" not in variable_metadata:
                    if variable_metadata["type"] == "binary":
                        inferred_upper = 1
                    else:
                        inferred_upper = max(var_data[np.where(var_mask == 1)]).item()
                    variable_metadata["upper"] = inferred_upper
                    print(
                        f'Max value of variable {variable_metadata["name"]} inferred as {inferred_upper}. This can '
                        "be changed manually in the dataset's variables.json file"
                    )

                if "query" not in variable_metadata:
                    # By default, assume all variables can be queried unless specified otherwise.
                    if variables_type == "auxiliary_variables":
                        variable_metadata["query"] = False
                        print(
                            f'Variable {variable_metadata["name"]} inferred to be a non-queriable variable. '
                            'This can be changed manually in the dataset\'s variables.json file by updating the "query" field.'
                        )
                    else:
                        variable_metadata["query"] = True
                        print(
                            f'Variable {variable_metadata["name"]} inferred to be a queriable variable. '
                            'This can be changed manually in the dataset\'s variables.json file by updating the "query" field.'
                        )

                if "target" not in variable_metadata:
                    # By default, assume variable is a target if and only if it is not queriable.
                    variable_metadata["target"] = not variable_metadata["query"]
                    fill_string = "not " if not variable_metadata["target"] else ""
                    print(
                        f'Variable {variable_metadata["name"]} inferred as {fill_string}an active learning target variable. '
                        'This can be changed manually in the dataset\'s variables.json file by updating the "target" field.'
                    )

                if "always_observed" not in variable_metadata:
                    # By default, assume variable is always observed if there is no missing in the mask.
                    if np.sum((var_mask - 1) ** 2) == 0:
                        variable_metadata["always_observed"] = True
                    else:
                        variable_metadata["always_observed"] = False
                    fill_string = "not " if not variable_metadata["always_observed"] else ""
                    print(
                        f'Variable {variable_metadata["name"]} inferred as {fill_string}an always observed target variable. '
                        'This can be changed manually in the dataset\'s variables.json file by updating the "always_observed" field.'
                    )

            variables_metadata.append(variable_metadata)

        return variables_metadata, used_cols

    @staticmethod
    def infer_from_data(data, mask, variables_dict=None, infer_aux_variables=False) -> Dict[str, List[Any]]:
        """
        Infer missing values in an input variables dictionary, using the input data.

        Args:
            data: NumPy array containing data
            mask: NumPy array containing 1 for observed data values, 0 for unobserved data values.
            variables_dict: Dictionary containing metadata for each variable (column) in the input data. Missing variables,
                or missing fields for a particular variable, will attempt to be inferred from the input data.
            infer_aux_variables: infer auxiliary variables for GINA or not.
        Returns:
            variables_dict: Updated version of the input variables_dict, with missing variables and fields inferred from the
                data.
        """

        if variables_dict is None:
            variables_dict = {}

        # NOTE this assumes all variables have only one column in unprocessed data, which should always be the case when
        # inferring from a dataset.
        if "auxiliary_variables" not in variables_dict:
            variables_dict["auxiliary_variables"] = []

        if "variables" not in variables_dict or variables_dict["variables"] == []:
            num_var_cols = data.shape[1] - len(variables_dict["auxiliary_variables"])
            variables_dict["variables"] = [{} for _ in range(num_var_cols)]

        variables_metadata, used_cols = Variables._metadata_from_dict(
            data, mask, variables_dict, variables_type="variables"
        )
        variables_dict = {
            "variables": variables_metadata,
            "auxiliary_variables": variables_dict["auxiliary_variables"],
            "used_cols": used_cols,
        }
        if infer_aux_variables:
            aux_variables_metadata, used_cols = Variables._metadata_from_dict(
                data, mask, variables_dict, variables_type="auxiliary_variables"
            )
            variables_dict = {
                "variables": variables_metadata,
                "auxiliary_variables": aux_variables_metadata,
                "used_cols": used_cols,
            }

        return variables_dict

    @property
    def _all_variables(self):
        return self._variables + self.auxiliary_variables

    @property
    def has_auxiliary(self) -> bool:
        """
        True if there are aux variables present.
        """
        return len(self.auxiliary_variables) > 0

    @property
    def binary_idxs(self) -> List[int]:
        """
        Return a list of the indices of all binary variables.
        """
        return self.var_idxs_by_type["binary"]

    @property
    def categorical_idxs(self) -> List[int]:
        """
        Return a list of the indices of all categorical variables.
        """
        return self.var_idxs_by_type["categorical"]

    @property
    def discrete_idxs(self) -> List[int]:
        """
        Return a list of the indices of all discrete (i.e. binary or categorical) variables. We sort to ensure that the
        combined list is in ascending order.
        """
        return sorted(self.var_idxs_by_type["categorical"] + self.var_idxs_by_type["binary"])

    @property
    def continuous_idxs(self) -> List[int]:
        """
        Return a list of the indices of all continuous variables.
        """
        return self.var_idxs_by_type["continuous"]

    @property
    def text_idxs(self) -> List[int]:
        """
        Return a list of the indices of all text variables.
        """
        return self.var_idxs_by_type["text"]

    @property
    def non_text_idxs(self) -> List[bool]:
        """Helper method. Returns list of booleans, where an element
        at index i indicates whether a variable at index i is non-text or not
        e.g. For Variables object of [..."continous"..., ..."text"..., "continuous"],
        the result would be [True, False, True]
        """
        unproc_cols_by_type = self.unprocessed_cols_by_type
        if "text" not in unproc_cols_by_type:
            return [True for _ in range(len(self))]
        return (~np.in1d(range(len(self)), unproc_cols_by_type["text"])).tolist()

    @property
    def num_unprocessed_cols(self) -> int:
        """
        Return number of columns in the unprocessed data represented by all variables
        """
        return sum(len(idxs) for idxs in self.unprocessed_cols)

    @property
    def num_unprocessed_non_aux_cols(self) -> int:
        """
        Return number of columns in the unprocessed data represented by non auxiliary variables
        """
        return sum(len(idxs) for idxs in self.unprocessed_non_aux_cols)

    @property
    def num_processed_cols(self) -> int:
        """
        Return number of columns in the processed data represented by all variables
        """
        return sum(len(idxs) for idxs in self.processed_cols)

    @property
    def num_processed_non_aux_cols(self) -> int:
        """
        Return number of columns in the processed data represented by non auxiliary variables
        """
        return sum(len(idxs) for idxs in self.processed_non_aux_cols)

    @property
    def num_groups(self) -> int:
        """
        Return the number of unique query groups in the variables object.
        """
        return len(self.group_names)

    @property
    def group_mask(self) -> np.ndarray:
        """
        Return a mask of shape (num_groups, num_processed_cols) indicating which column
        corresponds to which group.
        """
        mask = np.zeros((self.num_groups, self.num_processed_cols), dtype=bool)
        for group_idx, group in enumerate(self.group_idxs):
            for var in group:
                for proc_col in self.processed_cols[var]:
                    mask[group_idx, proc_col] = 1
        return mask

    @property
    def proc_always_observed_list(self) -> List[Optional[bool]]:
        """
        The mask that indicates if the variable is always observed (for processed data)
        """
        return sum(([var.always_observed] * var.processed_dim for var in self._all_variables), [])

    @property
    def processed_cols_by_type(self) -> Dict[str, List[List[int]]]:
        """
        Return a dictionary mapping each type of data (e.g. continuous, binary, ...) to a list of lists, where each
        sublist represents indices in the processed (i.e. one-hot) data associated with each variable of that type.

        E.g. for a two categorical variables each taking 3 values, followed by a binary variable, return
        {'categorical': [[0,1,2], [3,4,5]], 'binary': [[6]]}.
        """
        grouped_vars: DefaultDict[str, List[List[int]]] = defaultdict(list)
        for var, cols in zip(self._all_variables, self.processed_cols):
            grouped_vars[var.type_].append(cols)
        return grouped_vars

    @property
    def processed_non_aux_cols_by_type(self) -> Dict[str, List[List[int]]]:
        """
        Return a dictionary mapping each type of data (e.g. continuous, binary, ...) to a list of lists, where each
        sublist represents indices in the processed (i.e. one-hot) data (w/o aux variables) associated with each
        variable of that type.
        E.g. for a two categorical variables each taking 3 values, followed by a binary variable, return
        {'categorical': [[0,1,2], [3,4,5]], 'binary': [[6]]}.
        """
        grouped_vars: DefaultDict[str, List[List[int]]] = defaultdict(list)
        for var, cols in zip(self._variables, self.processed_cols):
            grouped_vars[var.type_].append(cols)
        return grouped_vars

    @property
    def unprocessed_cols_by_type(self) -> DefaultDict[str, List[int]]:
        """
        Return a dictionary mapping each type of data (e.g. continuous, binary, ...) to a list containing the column
        indices in the unprocessed data for all variables of that type.

        E.g. for a two categorical variables each taking 3 values, followed by a binary variable, return
        {'categorical': [0, 1], 'binary': [2]}.
        """
        grouped_vars: DefaultDict[str, List[int]] = defaultdict(list)
        i = 0
        for var, cols in zip(self._all_variables, self.unprocessed_cols):
            grouped_vars[var.type_] += cols
            i += var.unprocessed_dim
        return grouped_vars

    @property
    def unprocessed_non_aux_cols_by_type(self) -> DefaultDict[str, List[int]]:
        """
        Return a dictionary mapping each type of data (e.g. continuous, binary, ...) to a list containing the column
        indices in the unprocessed data for all variables of that type.

        E.g. for a two categorical variables each taking 3 values, followed by a binary variable, return
        {'categorical': [0, 1], 'binary': [2]}.
        """
        grouped_vars: DefaultDict[str, List[int]] = defaultdict(list)
        i = 0
        for var, cols in zip(self._variables, self.unprocessed_cols):
            grouped_vars[var.type_] += cols
            i += var.unprocessed_dim
        return grouped_vars

    def subset(self, idxs: List[int], auxiliary_idxs: Optional[List[int]] = None) -> Variables:
        """
        Returns a new Variables object containing only the Variable objects whose indices are given in `idxs`.
        Note that this currently ignores metadata variables.
        """
        if auxiliary_idxs is None:
            auxiliary_idxs = []

        variables_list = [self._variables[idx] for idx in idxs]
        auxiliary_variables_list = [self.auxiliary_variables[idx] for idx in auxiliary_idxs]
        return Variables(variables_list, auxiliary_variables_list)

    def to_dict(self) -> Dict[str, Any]:
        variables_list = [var.to_json() for var in self._variables]
        if self.auxiliary_variables is None:
            auxiliary_vars_list = []
        else:
            auxiliary_vars_list = [var.to_json() for var in self.auxiliary_variables]

        variables_json = {
            "variables": variables_list,
            "auxiliary_variables": auxiliary_vars_list,
            "used_cols": [int(col) for col in self.used_cols],
        }
        return variables_json

    def save(self, path: str) -> None:
        variables_json = self.to_dict()
        save_json(variables_json, path)

    def as_list(self) -> List[Variable]:
        return self._variables

    def get_idxs_from_name_list(self, variable_names: List[Union[str, int]]) -> np.ndarray:
        """
        Get a binary array of shape (variable_count,), where for each index the array value is 1 if the corresponding
        variable is named in `variable_names`, and 0 otherwise.
        """
        variables_to_query = np.zeros((len(self._variables),))
        # Look up indices of specified variables and mark as queriable.
        for variable_name in variable_names:
            # Cast name to string in case numeric names (e.g. question ids) have been input as integers.
            variable_name = str(variable_name)
            variable_idx = self.name_to_idx[variable_name]
            variables_to_query[variable_idx] = 1

        return variables_to_query

    def get_observable_groups(self, data_mask_row: np.ndarray, obs_mask_row: np.ndarray) -> List[int]:
        """
        Get list of indices for groups that are still observable in the current row
        Args:
            data_mask_row: 1D numpy array containing 1 for observed variables and 0 for unobserved in the underlying data
            obs_mask_row: 1D numpy array containing 1 for variables observed during active learning and 0 for ones unobserved

        Returns:
            list of indices of groups that can be observed, where the indices correspond to the corresponding group
            names in `self.group_names`.
        """
        observable_variables_idxs = self.get_observable_variable_idxs(data_mask_row, obs_mask_row)
        observable_groups_idxs: List[int] = []
        for group_idx, idxs in enumerate(self.group_idxs):
            if any(i in observable_variables_idxs for i in idxs):
                observable_groups_idxs.append(group_idx)
        return observable_groups_idxs

    def get_observable_variable_idxs(self, data_mask_row: np.ndarray, obs_mask_row: np.ndarray) -> List[int]:
        """
        Get list of variable idxs for variables that are still observable in the current row.
        Args:
            data_mask_row: 1D numpy array containing 1 for observed variables and 0 for unobserved in the underlying data
            obs_mask_row: 1D numpy array containing 1 for variables observed during active learning and 0 for ones unobserved

        Returns:
            observable_vars: List of indices of variables that can be observed.
        """
        if data_mask_row.ndim != 1:
            raise ValueError(f"Test mask should be 1D, had {data_mask_row.ndim} dims and shape {data_mask_row.shape}.")
        if obs_mask_row.ndim != 1:
            raise ValueError(
                f"Observation mask should be 1D, had {obs_mask_row.ndim} dims and shape {obs_mask_row.shape}."
            )
        if len(obs_mask_row) != len(data_mask_row) or len(data_mask_row) != len(self._variables):
            # One likely cause is accidentally passing 'processed' masks, which may be longer
            # if some variables are categorical.
            raise ValueError(
                f"Lengths of obs_mask_row {len(obs_mask_row)}, data_mask_row {len(data_mask_row)}, "
                f"and variables list {len(self._variables)} should all be the same."
            )
        # Get ids where there is an underlying data value (test_mask == 1) and that we haven't yet queried (obs_mask == 0)
        unobserved_idxs = np.where((data_mask_row == 1) & (obs_mask_row == 0))[0]

        # Intersection of these and query_var_idxs.
        observable_idx_set = set(unobserved_idxs).intersection(set(self.query_var_idxs))
        return list(observable_idx_set)

    def get_var_cols_from_data(self, var_idx, data):
        """
        Get data from an array for a single variable only.

        Args:
            var_idx: Index of variable we want data for.
            data (shape (batch_size, variable_count)): Array to get variable info from.

        Returns:
            var_data (shape (observed_count, processed_dim)): Values only for
                the corresponding variable.
        """
        return data[:, self.processed_cols[var_idx]]

    def get_variables_to_observe(self, data_mask: torch.Tensor) -> torch.Tensor:
        """
        Return a boolean tensor of length num_variables, where each element indicates whether the corresponding variable
        can be queried during active learning (i.e. the variable is queriable and has at least one observed value in
        the data).
        Args:
            data_mask (shape (batch_size, num_processed_cols)): Processed mask

        Returns:
            torch.Tensor (shape (variable_count,)): True where it's a query-able variable and we have at least one
            observed value
        """
        cols_with_data = data_mask.sum(dim=0).to(torch.bool)

        # data_mask may have multiple columns for a single variable, if it's a categorical variable. Pick first entry per variable
        ii = torch.tensor([cols[0] for cols in self.processed_cols], dtype=torch.long, device=cols_with_data.device)
        cols_with_data = torch.index_select(cols_with_data, 0, ii)
        is_query_id = torch.zeros(len(self), dtype=torch.bool, device=cols_with_data.device)
        is_query_id[
            tuple(self.query_var_idxs),
        ] = True
        return is_query_id * cols_with_data

    def _deduplicate_names(self):
        # Produce warning if var name is reused and add an increasing integer to the end until it is unique.
        var_names = set()
        for var in self._all_variables:
            i = 2
            original_name = var.name
            while var.name in var_names:
                new_name = f"{original_name}_{i}"
                var.name = new_name
                i += 1
            if var.name != original_name:
                # Do the warning in a separate block to the while loop so that we only raise one warning if we have to
                # try appending several different integers to the name.
                warnings.warn(
                    f"Name {original_name} has already been used, renaming to {var.name}",
                    UserWarning,
                )
            var_names.add(var.name)

    # TODO: Maybe create Variables.Utils for methods like the below one
    @staticmethod
    def create_empty_data(variables: Variables) -> np.ndarray:
        var_count = len(variables)
        empty_data = np.zeros((1, var_count), dtype=object)
        for i in range(var_count):
            if variables[i].type_ == "text":
                empty_data[:, i] = "empty str"
        return empty_data


class Variable:
    """
    Class representing a variable for the model.
    """

    def __init__(
        self,
        name: str,
        query: bool,
        type: str,  # pylint: disable=redefined-builtin
        lower: Optional[Union[float, int]] = None,
        upper: Optional[Union[float, int]] = None,
        group_name: Optional[str] = None,
        target: Optional[bool] = None,
        overwrite_processed_dim: Optional[int] = None,
        always_observed: Optional[bool] = None,
        is_latent: Optional[bool] = False,
    ) -> None:
        """
        Args:
            name: Name of the variable.
            query: Whether this variable can be queried or not during active learning.
            type: Type of variable - either "continuous", "binary" or "categorical".
            lower: Lower bound for the variable.
            upper: Upper bound for the variable.
            group_name: Name of the group of variables to which this variable belongs, if variables are being queried in
                groups during active learning.
            target: Whether this variable is an information acquisition target for active leaning algorithms.
                If unspecified, this is assumed to be true if the variable cannot be queried and vice-versa.
            overwrite_processed_dim: overwrites variable's processed dim
            is_latent: Indicates whether or not this variable is a latent variable.
        """
        self.query = query
        self.type_ = type
        self.name = name
        self.group_name = group_name or self.name
        if target is None:
            target = not query
        self.target = target
        self.overwrite_processed_dim = overwrite_processed_dim
        self.always_observed = always_observed
        self.is_latent = is_latent

        if self.type_ == "continuous":
            self.lower = self._try_assign(lower, "lower", float)
            self.upper = self._try_assign(upper, "upper", float)
        elif self.type_ == "binary":
            self.lower = 0
            self.upper = 1
        elif self.type_ == "categorical":
            self.lower = self._try_assign(lower, "lower", int)
            self.upper = self._try_assign(upper, "upper", int)
        elif self.type_ == "text":
            self.lower = -1
            self.upper = -1
            assert overwrite_processed_dim is not None
        else:
            raise ValueError(f"Invalid type {self.type_}")

        if self.type_ == "categorical" and self.processed_dim == 2:
            logging.info(f"Changing variable {self.name} from categorical to binary.")
            self.type_ = "binary"

        if not target and not query:
            warnings.warn(
                f"Variable {name} specified as neither queriable or a target variable and will be unused.", UserWarning
            )

    # TODO: Add generic type annotation to below method, so var_type is also the return type
    @staticmethod
    def _try_assign(var: Any, name: str, var_type: Any) -> Any:
        """
        Try to assign a variable with a given type. Throws an error if that value cannot be cast to the given type.
        Returns variable as given type.
        """
        if not isinstance(var, var_type):
            try:
                if var_type is bool:
                    # Direct converstion from string fails when type is bool.
                    var = strtobool(var)

                var = var_type(var)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"{name} must be of type {var_type} but was type {type(var)} and could not be converted."
                ) from exc
        return var

    def to_json(self) -> Dict[str, Any]:
        """
        Returns this class as a dict so it can be serialised to JSON.
        """
        var_dict = {
            "query": self.query,
            "target": self.target,
            "type": self.type_,
            "name": self.name,
            "group_name": self.group_name,
            "lower": self.lower,
            "upper": self.upper,
            "always_observed": self.always_observed,
            "is_latent": self.is_latent,
        }
        if self.overwrite_processed_dim is not None:  # As it is only used in VAEM(?), don't add if None
            var_dict["overwrite_processed_dim"] = self.overwrite_processed_dim
        return var_dict

    def __repr__(self):
        return str(self.to_json())

    @property
    def processed_dim(self) -> int:
        if self.overwrite_processed_dim is not None:
            return self.overwrite_processed_dim

        if self.type_ in {"continuous", "binary"}:
            return 1  # Binary or continuous only have a size of 1

        processed_dim = self.upper - self.lower + 1

        if not float.is_integer(float(processed_dim)):
            raise ValueError(f"Processed dim should be an integer, but was {processed_dim}.")
        return int(processed_dim)

    @property
    def unprocessed_dim(self) -> int:
        # TODO this is a hack, handle this problem more cleanly
        # If using a continuous variable with unprocessed dim >1 (e.g. a hack to encode a continuous vector as a
        # variable in VAEM), we have multiple columns in unprocessed data. However, we have single column in unprocessed
        # mask. As we never process/revert VAEM's data, but we process/revert VAEM's mask, we are always returning 1 here.
        return 1
