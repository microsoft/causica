# pylint: disable=E1123,E1120
from typing import Any, Dict

import tensorflow as tf
from octo.data.utils.data_utils import NormalizationType


def droid_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    pos_rot = trajectory["action_dict"]["cartesian_position"]
    last_element = tf.expand_dims(pos_rot[-1], axis=0)  # repeat last element to keep action length consistent
    pos_rot = tf.concat([pos_rot, last_element], axis=0)
    deltas = pos_rot[1:] - pos_rot[:-1]
    trajectory["action"] = tf.concat(  # pylint: disable=E1123,E1120
        (
            deltas,
            trajectory["action_dict"]["gripper_position"],
        ),
        axis=-1,
    )
    return trajectory


def make_droid_dataset_kwargs(
    dataset_name,
    dataset_dir,
    image_key: str = "image_primary",
    action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
):
    assert dataset_name in ["droid", "droid_100"], "Invalid dataset name for droid"

    if image_key == "image_primary":
        image_obs_keys = {
            "primary": "exterior_image_1_left",
        }
    elif image_key == "image_wrist":
        image_obs_keys = {
            "wrist": "wrist_image_left",
        }
    else:
        raise ValueError(f"Invalid image key {image_key}")

    dataset_kwargs = {
        "name": dataset_name,
        "data_dir": dataset_dir,
        "image_obs_keys": image_obs_keys,
        "state_obs_keys": ["cartesian_position", "gripper_position"],
        "language_key": "language_instruction",
        "norm_skip_keys": ["proprio"],
        "action_proprio_normalization_type": action_proprio_normalization_type,
        "absolute_action_mask": [False] * 6 + [True],  # droid_dataset_transform uses absolute actions
        "action_normalization_mask": [True] * 6 + [False],  # don't normalize final (gripper) dimension
        "standardize_fn": droid_dataset_transform,
    }
    return dataset_kwargs
