import tensorflow as tf
from octo.data.dataset import make_single_dataset
from octo.data.oxe import make_oxe_dataset_kwargs
from torch.utils.data import DataLoader

from ldwma.datasets.droid import make_droid_dataset_kwargs
from ldwma.datasets.utils import TensorFlowDatasetWrapper, preprocess_tf_video


def decode_task(task):
    return task.numpy().tobytes().decode("utf-8")


def compute_action_avg(action, downsample, start_idx, traj_len):
    act = tf.reduce_mean(
        tf.reshape(action[start_idx : start_idx + downsample * traj_len], [traj_len, downsample, -1]), axis=1
    )
    return act


def prepare_rtx_example(
    data: dict,
    traj_len: int,
    target_height: int,
    target_width: int,
    fps: int,
    fs: int,
    pad: bool = True,
    downsample: int = 1,
    image_key: str = "image_primary",
    use_language: bool = True,
):
    """Creates an example with a random trajectory of length traj_len from the video"""
    start_idx = tf.random.uniform(
        (), maxval=tf.shape(data["observation"][image_key])[0] - (traj_len * downsample) + 1, dtype=tf.int32
    )

    if use_language:
        prefix = "Robot arm performs the task: "
        task = data["task"]["language_instruction"][0]
        task_description = tf.strings.join([tf.constant(prefix, dtype=tf.string), task])
    else:
        task_description = ""

    if downsample == 1:
        act = data["action"][start_idx : start_idx + traj_len, 0]
    else:
        act = compute_action_avg(data["action"][:, 0], downsample, start_idx, traj_len)

    data = {
        "video": preprocess_tf_video(
            data["observation"][image_key][start_idx : start_idx + downsample * traj_len : downsample, 0],
            target_height=target_height,
            target_width=target_width,
            pad=pad,
        ),
        "act": act,
        "caption": task_description,
        "fps": fps,
        "frame_stride": fs,
        "start_idx": start_idx,
    }
    return data


def get_rtx_tf_dataset(
    train: bool = True,
    seed: int = 0,
    deterministic: bool = False,
    train_split: str = "train[:95%]",
    val_split: str = "train[95%:]",
    downsample: int = 1,
    dataset_name: str = "fractal20220817_data",
    dataset_dir: str = "gs://gresearch/robotics",
    image_key: str = "image_primary",
    save_statistics_dir: str = None,
    pad: bool = True,
    target_height: int = 320,
    target_width: int = 512,
    shuffle_buffer: int = 1000,
    shuffle_seed: int = 0,
    traj_len: int = 10,
    use_language: bool = True,
    fps: int = 24,
    frame_stride: int = 24,
):
    """Get the RTX dataset as a TensorFlow dataset.

    Args:
        train: Whether to load the training or validation set.
        dataset_name: The name of the dataset.
        dataset_dir: The directory containing the dataset.
        save_statistics_dir: The directory that cached statistics are saved to.
        pad: whether to pad the images rather than center crop
        target_height: The target height of the resized video.
        target_width: The target width of the resized video.
        shuffle_buffer: The size of the shuffle buffer.
        traj_len: The length of each trajectory.
    """
    if "droid" not in dataset_name:
        dataset_kwargs = make_oxe_dataset_kwargs(dataset_name, dataset_dir)
    else:
        dataset_kwargs = make_droid_dataset_kwargs(dataset_name, dataset_dir, image_key)

    if deterministic:
        num_parallel_calls = 1
        num_parallel_reads = 1
        prefetch_buffer = tf.data.experimental.AUTOTUNE
    else:
        num_parallel_calls = tf.data.experimental.AUTOTUNE
        num_parallel_reads = tf.data.experimental.AUTOTUNE
        prefetch_buffer = tf.data.experimental.AUTOTUNE

    dataset_kwargs.update(
        {
            "save_statistics_dir": save_statistics_dir,
            "train_split": train_split,
            "val_split": val_split,
            "seed": seed,
            "num_parallel_calls": num_parallel_calls,
            "num_parallel_reads": num_parallel_reads,
            "deterministic": deterministic,
        }
    )
    dataset = make_single_dataset(
        dataset_kwargs,
        train=train,
        traj_transform_kwargs={"num_parallel_calls": num_parallel_calls, "deterministic": deterministic},
        frame_transform_kwargs={"num_parallel_calls": num_parallel_calls, "deterministic": deterministic},
    )
    dataset_len = dataset.dataset_statistics["num_trajectories"]

    # filter dataset to only include trajectories of length traj_len and resize images
    dataset = dataset.filter(lambda traj: tf.shape(traj["action"])[0] >= (traj_len * downsample))

    # prepare examples
    dataset = dataset.map(
        lambda traj: prepare_rtx_example(
            traj,
            traj_len=traj_len,
            target_height=target_height,
            target_width=target_width,
            fps=fps,
            fs=frame_stride,
            pad=pad,
            downsample=downsample,
            image_key=image_key,
            use_language=use_language,
        ),
        num_parallel_calls=num_parallel_calls,
        deterministic=deterministic,
    )

    # apply shuffle buffer
    dataset = dataset.shuffle(shuffle_buffer, seed=shuffle_seed)
    dataset = dataset.prefetch(prefetch_buffer)
    return dataset, dataset_len


if __name__ == "__main__":
    ds, length = get_rtx_tf_dataset()
    ds = TensorFlowDatasetWrapper(ds, length)
    dataloader = DataLoader(ds, batch_size=32, num_workers=0)
    ds = iter(dataloader)
    for _ in range(10000):
        batch = next(ds)
        print(batch["video"].shape)
        print(batch["act"].shape)
