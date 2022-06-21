import numpy as np
import pytest

from causica.data_generation.csuite import utils


def test_to_counterfactual_dict_format():
    original_samples = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])

    intervention_samples = np.array([[10, 11, 12, 13, 14], [15, 16, 17, 18, 19]])

    reference_samples = np.array([[20, 21, 22, 23, 24], [25, 26, 27, 28, 29]])

    do_idx = 0
    do_value = 999
    reference_value = 111

    counterfactual_dicts = utils.to_counterfactual_dict_format(
        original_samples=original_samples,
        intervention_samples=intervention_samples,
        reference_samples=reference_samples,
        do_value=do_value,
        do_idx=do_idx,
        reference_value=reference_value,
    )

    counterfactual_dict = counterfactual_dicts[0]

    expected_counterfactual_dict = {
        "conditioning": original_samples,
        "intervention_samples": intervention_samples,
        "reference_samples": reference_samples,
        "effect_mask": np.array([False, False, False, False, False]),
        "intervention": np.array([999, np.nan, np.nan, np.nan, np.nan]),
        "reference": np.array([111, np.nan, np.nan, np.nan, np.nan]),
    }

    np.testing.assert_equal(counterfactual_dict["conditioning"], expected_counterfactual_dict["conditioning"])
    assert len(counterfactual_dict["conditioning"].shape) == 2

    np.testing.assert_equal(
        counterfactual_dict["intervention_samples"], expected_counterfactual_dict["intervention_samples"]
    )
    assert len(counterfactual_dict["intervention_samples"].shape) == 2

    np.testing.assert_equal(counterfactual_dict["reference_samples"], expected_counterfactual_dict["reference_samples"])
    assert len(counterfactual_dict["reference_samples"].shape) == 2

    np.testing.assert_equal(counterfactual_dict["effect_mask"], expected_counterfactual_dict["effect_mask"])
    assert len(counterfactual_dict["effect_mask"].shape) == 1

    np.testing.assert_equal(counterfactual_dict["intervention"], expected_counterfactual_dict["intervention"])
    assert len(counterfactual_dict["intervention"].shape) == 1

    np.testing.assert_equal(counterfactual_dict["reference"], expected_counterfactual_dict["reference"])
    assert len(counterfactual_dict["reference"].shape) == 1

    misshaped_original_samples = original_samples[:, :-1]
    with pytest.raises(AssertionError):

        counterfactual_dict = utils.to_counterfactual_dict_format(
            original_samples=misshaped_original_samples,
            intervention_samples=intervention_samples,
            reference_samples=reference_samples,
            do_value=do_value,
            do_idx=do_idx,
            reference_value=reference_value,
        )
