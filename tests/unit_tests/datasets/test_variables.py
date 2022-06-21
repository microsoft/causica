import numpy as np
import pytest
import torch

from causica.datasets.variables import Variable, Variables


# pylint: disable=redefined-outer-name
@pytest.fixture(scope="function")
def variables():
    return Variables(
        [
            Variable("categorical_input", True, "categorical", 1.0, 3.0),
            Variable("numeric_input", True, "continuous", 3, 13),
            Variable("numeric_target", False, "continuous", 2, 300),
        ]
    )


@pytest.fixture(scope="function")
def variables_overwrite_processed_dim():
    return Variables(
        [
            Variable("categorical_input", True, "categorical", 1.0, 3.0),
            Variable("numeric_input", True, "continuous", 3, 13, overwrite_processed_dim=2),
            Variable("numeric_target", False, "continuous", 2, 300),
        ]
    )


@pytest.fixture(scope="function")
def variables_with_text_var():
    return Variables(
        [
            Variable("categorical_input", True, "categorical", 1.0, 3.0),
            Variable("numeric_input", True, "continuous", 3, 13),
            Variable("text_input", True, "text", overwrite_processed_dim=5),
            Variable("numeric_target", False, "continuous", 2, 300),
        ]
    )


@pytest.fixture(scope="function")
def variables_with_groups():
    return Variables(
        [
            Variable("a", True, "continuous", 0, 1, group_name="Group 1"),
            Variable("b", True, "continuous", 0, 1, group_name="Group 1"),
            Variable("c", True, "continuous", 0, 1, group_name="Group 2"),
            Variable("d", True, "continuous", 0, 1, group_name="Group 2"),
        ]
    )


def test_variables(variables):
    assert len(variables) == 3
    assert variables.num_processed_cols == 5
    included_in_grouping = sorted(
        [
            idx
            for ranges_for_type in variables.processed_cols_by_type.values()
            for idx_range in ranges_for_type
            for idx in idx_range
        ]
    )
    assert included_in_grouping == list(range(variables.num_processed_cols))


def test_infer_variables_infer_datatype():
    data = np.array([[1, 1.1, 0, 0], [1, 2.2, 3, 0], [0, 3.3, 5, 0]])
    mask = np.ones_like(data, dtype=bool)
    variables = Variables.infer_from_data(data, mask)
    variables = variables["variables"]

    assert variables[0]["type"] == "binary"
    assert variables[1]["type"] == "continuous"
    assert variables[2]["type"] == "categorical"
    assert variables[3]["type"] == "binary"


def test_infer_variables_infer_always_observed_not_given():
    data = np.array([[1, 1.1, 0, 0], [1, 2.2, 3, 0], [0, 3.3, 5, 0]])
    mask = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 0, 1]])
    variables = Variables.infer_from_data(data, mask)
    variables = variables["variables"]

    assert variables[0]["always_observed"]
    assert variables[1]["always_observed"]
    assert not variables[2]["always_observed"]
    assert not variables[3]["always_observed"]


def test_infer_variables_infer_always_observed_is_given():
    data = np.array([[1, 1.1, 0, 0], [1, 2.2, 3, 0], [0, 3.3, 5, 0]])
    mask = np.array([[0, 1, 0, 0], [0, 1, 1, 0], [1, 1, 0, 1]])
    variables_dict = {
        "variables": [
            {"type": "binary", "always_observed": True},
            {"type": "continuous", "always_observed": False},
            {"type": "categorical", "always_observed": False},
            {"type": "binary", "always_observed": False},
        ]
    }
    variables = Variables.infer_from_data(data, mask, variables_dict)
    variables = variables["variables"]
    assert variables[0]["always_observed"]
    assert not variables[1]["always_observed"]
    assert not variables[2]["always_observed"]
    assert not variables[3]["always_observed"]


def test_infer_variables_proc_always_observed_list():
    data = np.array([[1, 1.1, 0, 0], [1, 2.2, 3, 0], [0, 3.3, 5, 0]])
    mask = np.array([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]])
    variables_dict = {
        "variables": [
            {
                "type": "binary",
            },
            {
                "type": "continuous",
            },
            {
                "type": "categorical",
            },
            {
                "type": "binary",
            },
        ]
    }
    variables = Variables.infer_from_data(data, mask, variables_dict)
    variables = Variables.create_from_dict(variables)
    always_observed_list = variables.proc_always_observed_list
    assert always_observed_list[0]
    assert always_observed_list[1]
    assert not any(always_observed_list[2:8])
    assert not always_observed_list[8]


def test_infer_variables_infer_min_max_val():
    data = np.arange(9).reshape(3, 3)
    mask = np.mod(data, 2)
    variables = Variables.infer_from_data(data, mask)
    variables = variables["variables"]

    assert variables[0]["lower"] == 3
    assert variables[1]["lower"] == 1
    assert variables[2]["lower"] == 5

    assert variables[0]["upper"] == 3
    assert variables[1]["upper"] == 7
    assert variables[2]["upper"] == 5


def test_infer_variables_infer_name():
    data = np.arange(9).reshape(3, 3)
    mask = np.ones((3, 3))
    variables = Variables.infer_from_data(data, mask)
    variables = variables["variables"]

    for i in range(data.shape[1]):
        assert variables[i]["name"] == f"Column {i}"


@pytest.mark.parametrize(
    "variables_list, expected_names",
    [
        ([{"type": "binary"}, {"type": "binary"}, {"type": "binary"}], ["10", "20", "30"]),
        (
            [
                {"type": "binary", "name": "name0"},
                {"type": "binary", "name": "name1"},
                {"type": "binary", "name": "name2"},
            ],
            ["name0", "name1", "name2"],
        ),
        ([], ["10", "20", "30"]),
        (None, ["10", "20", "30"]),
    ],
)
def test_infer_variables_infer_name_used_cols(variables_list, expected_names):
    variables_dict = {"used_cols": [10, 20, 30], "metadata_variables": []}
    if variables_list:
        variables_dict["variables"] = variables_list

    data = np.ones((10, 3))
    mask = np.ones((10, 3))

    variables = Variables.infer_from_data(data, mask, variables_dict=variables_dict)
    variables = variables["variables"]

    assert variables[0]["name"] == expected_names[0]
    assert variables[1]["name"] == expected_names[1]
    assert variables[2]["name"] == expected_names[2]


def test_infer_variables_infer_name_incorrect_used_cols_raises_error():
    variables_dict = {"used_cols": [10, 20, 30, 40], "metadata_variables": []}

    data = np.ones((10, 3))
    mask = np.ones((10, 3))

    with pytest.raises(AssertionError):
        _ = Variables.infer_from_data(data, mask, variables_dict=variables_dict)


def test_infer_variables_fully_specified():
    # Test that the input variables list is returned unchanged if it already specifies all fields.
    data = np.array([[1, 1.1], [1, 2.2], [0, 3.3]])
    mask = np.array([[0, 1], [0, 1], [1, 1]])
    # ["name", "type", "lower", "upper", "query", "target", "always_observed"]
    input_variables_dict = {
        "variables": [
            {
                "name": "var1",
                "type": "binary",
                "lower": 0,
                "upper": 1,
                "query": True,
                "target": False,
                "always_observed": False,
            },
            {
                "name": "var2",
                "type": "continuous",
                "lower": 1,
                "upper": 4,
                "query": False,
                "target": True,
                "always_observed": True,
            },
        ]
    }
    output_variables_dict = Variables.infer_from_data(data, mask, input_variables_dict)
    assert input_variables_dict["variables"] == output_variables_dict["variables"]


def test_name_to_idx(variables):
    assert variables.name_to_idx["categorical_input"] == 0
    assert variables.name_to_idx["numeric_input"] == 1
    assert variables.name_to_idx["numeric_target"] == 2
    with pytest.raises(KeyError):
        variables.name_to_idx["incorrect_name"]  # pylint: disable=pointless-statement


def test_get_idxs_from_name_list(variables):
    features_to_query = variables.get_idxs_from_name_list(variable_names=["categorical_input", "numeric_target"])
    assert np.array_equal(features_to_query, np.array([1, 0, 1]))


def test_get_idxs_from_name_list_from_used_cols():
    variables_dict = {
        "used_cols": [10, 20, 30, 40],
        "variables": [
            {"type": "binary"},
            {"type": "binary"},
            {"type": "binary"},
            {"type": "binary"},
        ],
        "metadata_variables": [],
    }
    data = np.ones((10, 4))
    mask = np.ones((10, 4))

    variables = Variables.create_from_data_and_dict(data, mask, variables_dict=variables_dict)

    features_to_query = variables.get_idxs_from_name_list(variable_names=["10", "30"])
    assert np.array_equal(features_to_query, np.array([1, 0, 1, 0]))


@pytest.mark.parametrize(
    "used_cols_specified, used_cols, expected_used_cols",
    [(False, None, None), (True, None, None), (True, [0, 2, 3], [0, 2, 3])],
)
def test_infer_variables_save_used_cols(used_cols_specified, used_cols, expected_used_cols):
    variables_dict = {"metadata_variables": [], "variables": [{}, {}, {}]}
    if used_cols_specified:
        variables_dict["used_cols"] = used_cols

    data = np.ones((5, 3))
    mask = np.ones((5, 3))

    processed_variables_dict = Variables.infer_from_data(data, mask, variables_dict=variables_dict)

    assert processed_variables_dict["used_cols"] == expected_used_cols


@pytest.mark.parametrize(
    "used_cols_specified, used_cols, expected_used_cols",
    [(False, None, [0, 1, 2]), (True, None, [0, 1, 2]), (True, [0, 2, 3], [0, 2, 3])],
)
def test_create_from_dict_save_used_cols(used_cols_specified, used_cols, expected_used_cols):
    variables_dict = {
        "variables": [
            {"name": "Column 0", "type": "binary", "lower": 0, "upper": 1, "query": True},
            {"name": "Column 1", "type": "binary", "lower": 0, "upper": 1, "query": True},
            {"name": "Column 2", "type": "binary", "lower": 0, "upper": 1, "query": True},
        ],
        "metadata_variables": [],
    }
    if used_cols_specified:
        variables_dict["used_cols"] = used_cols

    variables = Variables.create_from_dict(variables_dict)

    assert variables.used_cols == expected_used_cols


@pytest.mark.parametrize(
    "dict_specified, used_cols_specified, used_cols, expected_used_cols",
    [
        (False, None, None, [0, 1, 2]),
        (True, False, None, [0, 1, 2]),
        (True, True, None, [0, 1, 2]),
        (True, True, [0, 2, 3], [0, 2, 3]),
    ],
)
def test_create_from_data_and_dict_save_used_cols(dict_specified, used_cols_specified, used_cols, expected_used_cols):
    if dict_specified:
        variables_dict = {"metadata_variables": [], "variables": [{}, {}, {}]}
        if used_cols_specified:
            variables_dict["used_cols"] = used_cols
    else:
        variables_dict = None

    data = np.ones((5, 3))
    mask = np.ones((5, 3))

    variables = Variables.create_from_data_and_dict(data, mask, variables_dict)

    assert variables.used_cols == expected_used_cols


def test_processed_cols(variables):
    assert variables.processed_cols == [[0, 1, 2], [3], [4]]


def test_processed_cols_overwrite_processed_dim(variables_overwrite_processed_dim):
    assert variables_overwrite_processed_dim.processed_cols == [[0, 1, 2], [3, 4], [5]]


def test_unprocessed_cols(variables):
    assert variables.unprocessed_cols == [[0], [1], [2]]


def test_unprocessed_cols_overwrite_processed_dim(variables_overwrite_processed_dim):
    assert variables_overwrite_processed_dim.unprocessed_cols == [[0], [1], [2]]


def test_processed_cols_by_type(variables):
    assert variables.processed_cols_by_type == {"categorical": [[0, 1, 2]], "continuous": [[3], [4]]}


def test_processed_cols_by_type_processed_dim(variables_overwrite_processed_dim):
    assert variables_overwrite_processed_dim.processed_cols_by_type == {
        "categorical": [[0, 1, 2]],
        "continuous": [[3, 4], [5]],
    }


def test_processed_cols_by_type_with_text_var(variables_with_text_var):
    assert variables_with_text_var.processed_cols_by_type == {
        "categorical": [[0, 1, 2]],
        "continuous": [[3], [9]],
        "text": [[4, 5, 6, 7, 8]],
    }


def test_unprocessed_cols_by_type(variables):
    assert variables.unprocessed_cols_by_type == {"categorical": [0], "continuous": [1, 2]}


def test_unprocessed_cols_by_type_processed_dim(variables_overwrite_processed_dim):
    assert variables_overwrite_processed_dim.unprocessed_cols_by_type == {
        "categorical": [0],
        "continuous": [1, 2],
    }


def test_unprocessed_cols_by_type_with_text_var(variables_with_text_var: Variables):
    variables = variables_with_text_var
    assert variables.unprocessed_cols_by_type == {"categorical": [0], "continuous": [1, 3], "text": [2]}


def test_non_text_indices_without_txt(variables: Variables):
    assert np.all(variables.non_text_idxs == [True, True, True])


def test_non_text_indices_with_txt(variables_with_text_var: Variables):
    variables = variables_with_text_var
    assert np.all(variables.non_text_idxs == [True, True, False, True])


def test_num_unprocessed_cols(variables):
    assert variables.num_unprocessed_cols == 3


def test_num_unprocessed_cols_processed_dim(variables_overwrite_processed_dim):
    assert variables_overwrite_processed_dim.num_unprocessed_cols == 3


def test_num_unprocessed_cols_with_text_var(variables_with_text_var):
    assert variables_with_text_var.num_unprocessed_cols == 4


def test_var_idxs_by_type():
    variables = Variables(
        [
            Variable("continuous_input", True, "continuous", 3.1, 13.5),
            Variable("categorical_input", True, "categorical", 1, 3),
            Variable("binary_target", False, "binary", 0, 1),
        ]
    )
    assert variables.var_idxs_by_type["continuous"] == [0]
    assert variables.var_idxs_by_type["categorical"] == [1]
    assert variables.var_idxs_by_type["binary"] == [2]


def test_continuous_idxs():
    variables = Variables(
        [
            Variable("continuous_input", True, "continuous", 3.1, 13.5),
            Variable("categorical_input", True, "categorical", 1, 3),
            Variable("binary_target", False, "binary", 0, 1),
        ]
    )
    assert variables.continuous_idxs == [0]


def test_binary_idxs():
    variables = Variables(
        [
            Variable("continuous_input", True, "continuous", 3.1, 13.5),
            Variable("categorical_input", True, "categorical", 1, 3),
            Variable("binary_target", False, "binary", 0, 1),
        ]
    )
    assert variables.binary_idxs == [2]


def test_categorical_idxs():
    variables = Variables(
        [
            Variable("continuous_input", True, "continuous", 3.1, 13.5),
            Variable("categorical_input", True, "categorical", 1, 3),
            Variable("binary_target", False, "binary", 0, 1),
        ]
    )
    assert variables.categorical_idxs == [1]


def test_discrete_idxs():
    # Use multiple binary and categorical vars to ensure that the discrete_idxs property correctly sorts the list of
    # indices when combining the lists for categorical and binary variables.
    variables = Variables(
        [
            Variable("categorical_input_1", True, "categorical", 1, 3),
            Variable("continuous_input", True, "continuous", 3.1, 13.5),
            Variable("binary_input_1", True, "binary", 0, 1),
            Variable("categorical_input_2", True, "categorical", 1, 3),
            Variable("binary_input_2", True, "binary", 0, 1),
        ]
    )
    assert variables.discrete_idxs == [0, 2, 3, 4]


def test_get_variables_to_observe(variables):
    mask = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0]])
    assert variables.num_processed_cols == torch.numel(mask)
    variables_to_observe = variables.get_variables_to_observe(mask)
    assert torch.numel(variables_to_observe) == 3
    assert torch.equal(variables_to_observe, torch.tensor([False, True, False]))


def test_group_idxs1():
    variables = Variables(
        [
            Variable("binary_input_1", True, "binary", group_name="grp1"),
            Variable("binary_input_2", True, "binary", group_name="grp1"),
            Variable("binary_input_3", True, "binary", group_name="grp2"),
            Variable("binary_target", False, "binary", 0, 1),
        ]
    )
    assert variables.group_names == ["grp1", "grp2"]
    assert variables.group_idxs == [[0, 1], [2]]


def test_group_idxs2():
    variables = Variables(
        [
            Variable("binary_input_1", True, "binary", group_name="grp1"),
            Variable("binary_input_2", True, "binary", group_name="grp2"),
            Variable("binary_input_3", True, "binary", group_name="grp1"),
            Variable("binary_target", False, "binary", 0, 1),
        ]
    )
    assert variables.group_names == ["grp1", "grp2"]
    assert variables.group_idxs == [[0, 2], [1]]


def test_get_observable_groups():
    variables = Variables(
        [
            Variable("binary_input_1", True, "binary", 0, 1, group_name="grp1"),
            Variable("binary_input_2", True, "binary", 0, 1, group_name="grp1"),
            Variable("binary_input_3", True, "binary", 0, 1, group_name="grp2"),
            Variable("binary_input_4", True, "binary", 0, 1, group_name="grp2"),
            Variable("binary_input_5", True, "binary", 0, 1, group_name="grp3"),
            Variable("binary_input_6", True, "binary", 0, 1, group_name="grp3"),
            Variable("binary_input_7", True, "binary", 0, 1, group_name="grp4"),
            Variable("binary_input_8", True, "binary", 0, 1, group_name="grp4"),
            Variable("binary_target", False, "binary", 0, 1),
        ]
    )
    data_mask = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    obs_mask = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    observable_group_idxs = variables.get_observable_groups(data_mask, obs_mask)
    assert observable_group_idxs == [0, 3]


def test_target():
    variables = Variables(
        [
            Variable("continuous_input", True, "continuous", 3.1, 13.5),
            Variable("categorical_target", True, "categorical", 1, 3, target=True),
            Variable("categorical_input", False, "categorical", 1, 3, target=False),
            Variable("binary_target", False, "binary", 0, 1),
        ]
    )
    assert not variables[0].target
    assert variables[1].target
    assert not variables[2].target
    assert variables[3].target


def test_target_var_idxs():
    variables = Variables(
        [
            Variable("continuous_input", True, "continuous", 3.1, 13.5, target=False),
            Variable("categorical_target", True, "categorical", 1, 3, target=True),
            Variable("categorical_input", False, "categorical", 1, 3, target=False),
            Variable("binary_target", False, "binary", 0, 1, target=True),
        ]
    )
    assert variables.target_var_idxs == [1, 3]
    assert variables.not_target_var_idxs == [0, 2]


def test_query_var_idxs():
    variables = Variables(
        [
            Variable("continuous_input", True, "continuous", 3.1, 13.5, target=False),
            Variable("categorical_target", True, "categorical", 1, 3, target=True),
            Variable("categorical_input", False, "categorical", 1, 3, target=False),
            Variable("binary_target", False, "binary", 0, 1, target=True),
        ]
    )
    assert variables.query_var_idxs == [0, 1]
    assert variables.not_query_var_idxs == [2, 3]


def test_target_queriable_warning():
    with pytest.warns(UserWarning):
        _ = Variables(
            [
                Variable("continuous_input", True, "continuous", 3.1, 13.5, target=False),
                Variable("categorical_input", True, "categorical", 1, 3, target=False),
                Variable("categorical_input_2", True, "categorical", 1, 3, target=False),
                Variable("binary_target", True, "binary", 0, 1, target=True),
            ]
        )


def test_all_targets_queriable_warning():
    with pytest.warns(UserWarning):
        _ = Variables(
            [
                Variable("continuous_target", True, "continuous", 3.1, 13.5, target=True),
                Variable("categorical_target", True, "categorical", 1, 3, target=True),
                Variable("categorical_target_2", True, "categorical", 1, 3, target=True),
                Variable("binary_target", True, "binary", 0, 1, target=True),
            ]
        )


def test_not_query_or_target_warning():
    with pytest.warns(UserWarning):
        Variable("continuous_input", False, "continuous", 0, 10, target=False)


def test_create_txt_var_without_dim_given():
    with pytest.raises(AssertionError):
        Variables(
            [
                Variable("categorical_input", True, "categorical", 1.0, 3.0),
                Variable("numeric_input", True, "continuous", 3, 13),
                Variable("text_input", True, "text"),
                Variable("numeric_target", False, "continuous", 2, 300),
            ]
        )


def test_deduplicate_names():
    with pytest.warns(UserWarning):
        variables = Variables(
            [
                Variable("name", True, "categorical", 1.0, 3.0),
                Variable("name_2", True, "continuous", 3, 13),
                Variable("name_3", True, "continuous", 3, 13),
                Variable("name", False, "continuous", 2, 300),
            ]
        )
        assert variables[0].name == "name"
        assert variables[1].name == "name_2"
        assert variables[2].name == "name_3"
        assert variables[3].name == "name_4"


def test_group_mask(variables_with_groups):
    correct_group_mask = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=bool)
    computed_mask = variables_with_groups.group_mask
    assert (computed_mask == correct_group_mask).all()
