import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import Dataset


def preprocess_tf_video(video: tf.Tensor, target_height: int, target_width: int, pad: bool):
    """Prepares tensorflow tensor video for training with dynamicrafter.

    First resizes the video to the target height and width, with padding. Then normalizes the video to [-1, 1] and
    permutes the dimensions to [C, T, H, W].

    Args:
        video: The video tensor to preprocess with shape [T, H, W, C].
        target_height: The target height of the resized video.
        target_width: The target width of the resized video.

    Returns:
        The preprocessed video tensor with shape [C, T, H, W] in [-1, 1]
    """
    if pad:
        video = tf.image.resize_with_pad(video, target_height, target_width)
    else:
        video = center_crop_video(video, target_height, target_width)
    video = (tf.cast(video, tf.float32) / 255.0 - 0.5) * 2.0
    video = tf.transpose(video, [3, 0, 1, 2])
    return video


def convert_tf_to_torch(tf_batch):
    """
    Convert items in a batch from TensorFlow tensors to PyTorch tensors if they are TensorFlow tensors,
    otherwise copy the item as is.
    """
    pytorch_batch = {}
    for key, value in tf_batch.items():
        if isinstance(value, tf.Tensor):
            if value.dtype == tf.string:
                pytorch_batch[key] = value.numpy().decode("utf-8")
            else:
                numpy_value = value.numpy()
                if np.isscalar(numpy_value):
                    pytorch_batch[key] = numpy_value
                else:
                    pytorch_batch[key] = torch.from_numpy(value.numpy())
        else:
            pytorch_batch[key] = value
    return pytorch_batch


def center_crop_video(video, target_height, target_width):
    """Center crops a video to the target height and width.

    Args:
        video: The video tensor to center crop with shape [T, H, W, C].
        target_height: The target height of the center cropped video.
        target_width: The target width of the center cropped video.

    Returns:
        The center cropped video tensor with shape [T, target_height, target_width, C].
    """
    if not video.dtype == tf.uint8:
        raise TypeError("Tensor must be of type tf.uint8")

    video_shape = tf.shape(video)
    height = video_shape[1]
    width = video_shape[2]

    aspect_ratio = width / height
    target_aspect_ratio = target_width / target_height
    if aspect_ratio < target_aspect_ratio:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    video = tf.image.resize(video, (new_height, new_width))
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    video = video[:, top:bottom, left:right]
    return video


class TensorFlowDatasetWrapper(Dataset):
    def __init__(self, tf_dataset, dataset_len):
        """
        Initialize the dataset wrapper.
        """
        self.tf_dataset = tf_dataset
        self.tf_dataset_iter = iter(tf_dataset)
        self.dataset_len = dataset_len

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        """
        Fetch the next batch from the TensorFlow dataset iterator.
        """

        # Get the next batch from the iterator
        try:
            tf_batch = next(self.tf_dataset_iter)
        except StopIteration:
            # Reinitialize the iterator if all data has been consumed
            self.tf_dataset_iter = iter(self.tf_dataset)
            tf_batch = next(self.tf_dataset_iter)
        pytorch_batch = convert_tf_to_torch(tf_batch)
        return pytorch_batch
