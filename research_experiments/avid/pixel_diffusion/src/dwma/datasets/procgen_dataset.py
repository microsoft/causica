import os
import zipfile

import fsspec
import numpy as np
import torch
from fsspec import asyn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ProcgenDataLoader(DataLoader):
    """DataLoader class for Procgen data that filters out None values in the dataset.

    This is done because when using fsspec with caching and multiple workers, fsspec occasionally fails to successfully
    open the file and the dataset returns None. This DataLoader filters out these None values so that only valid data is
    passed to the training loop.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(collate_fn=self.default_collate_fn, *args, **kwargs)

    @staticmethod
    def default_collate_fn(batch):
        """Filters out None values in a batch"""
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)


class ProcgenDataset(Dataset):
    """Dataset class for Procgen data."""

    def __init__(
        self,
        data_folder: str,
        window_width: int = 15,
        fixed_episode_length: int | None = None,
        cache_files: bool = False,
        cache_dir: str = "~/.cache/procgen",
        precache_files: bool = False,
    ):
        """Initialize the dataset.

        Args:
            data_folder: folder containing the Procgen data of .npz files
            window_width: width of the window used to generate examples
            fixed_episode_length: known fixed length of each episode (if None, the actual length of each episode
                is first retrieved, but this may be slow)
            cache_files: whether to cache the files locally
            cache_dir: directory to store the cached files
            precache_files: whether to pre-cache the files
        """
        self.data_folder = data_folder
        self.window_width = window_width
        self.fixed_episode_length = fixed_episode_length

        # get the list of files in the data folder
        fs, _ = fsspec.url_to_fs(data_folder)
        self.files = fs.ls(self.data_folder)
        self.files = [f.split("/")[-1] for f in self.files if f.endswith(".npz")]

        # get the episode lengths and example indices
        self.episode_lengths = self.get_episode_lengths(self.files, self.fixed_episode_length)
        self.example_indices = self.get_example_indices(self.episode_lengths, self.window_width)

        # resolve issue whereby fsspec hangs with multiple workers
        # see github.com/fsspec/gcsfs/issues/379#issuecomment-1573494347
        os.register_at_fork(
            after_in_child=asyn.reset_lock,
        )

        # prepare caching options for fsspec
        self.cache_files = cache_files
        if cache_files:
            cache_dir = str(os.path.expanduser(cache_dir))
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            self.cache_options = {"simplecache": {"cache_storage": cache_dir, "same_names": True}}
            self.data_folder = "simplecache::" + self.data_folder
            self.cache_dir = cache_dir
        else:
            self.cache_options = {}

        # pre-cache the files
        if precache_files:
            self.precache_files()

    def precache_files(self):
        assert self.cache_files, "Caching is not enabled."
        cached_files = os.listdir(self.cache_dir)
        cached_files = [f for f in cached_files if f.endswith(".npz")]
        files_to_cache = list(set(self.files) - set(cached_files))

        # Read the required files to trigger caching
        print(f"{len(self.files) - len(files_to_cache)} files already cached. Pre-caching {len(files_to_cache)} files.")
        for file in tqdm(files_to_cache, desc="Pre-caching files"):
            with fsspec.open(os.path.join(self.data_folder, file), mode="rb", **self.cache_options) as opened_file:
                _ = opened_file.read()

    def get_episode_lengths(self, files: list[str], fixed_episode_length: int | None = None) -> np.ndarray:
        """Return a list containing the length of each episode in the dataset.

        Args:
            files: list of files containing the dataset
            fixed_episode_length: fixed length of each episode (if None, the actual length of each episode is used)

        Returns:
            1d array of episode lengths
        """
        if fixed_episode_length is None:
            episode_lengths = []
            for f in tqdm(files, desc="Getting dataset statistics"):
                with fsspec.open(os.path.join(self.data_folder, f), mode="rb", **self.cache_options) as opened_file:
                    data = np.load(opened_file)
                    episode_length = data["obs"].shape[0]
                    episode_lengths.append(episode_length)
            return np.array(episode_lengths)

        episode_lengths = [fixed_episode_length for _ in files]
        return np.array(episode_lengths)

    def get_example_indices(self, episode_lengths: np.ndarray, window_width: int) -> np.ndarray:
        """Return a numpy array containing the indices of the last example corresponding to each episode.

        Args:
            episode_lengths: list of episode lengths
            window_width: width of the window used to generate examples

        Returns:
            1d array of example indices
        """
        counter = 0
        example_indices = []
        for episode_length in episode_lengths:
            assert episode_length > window_width, "Episode length must be greater than window width."
            counter += episode_length - window_width
            example_indices.append(counter)
        return np.array(example_indices)

    def __len__(self):
        return self.example_indices[-1]

    def __getitem__(self, idx):
        """Get an example containing window_width observations, actions."""

        # find the correct episode and example index
        episode_idx = np.argmax(self.example_indices > idx)
        example_idx = idx - self.example_indices[episode_idx - 1] if episode_idx > 0 else idx

        # load episode and slice data
        try:
            with fsspec.open(
                os.path.join(self.data_folder, self.files[episode_idx]), mode="rb", **self.cache_options
            ) as opened_file:
                data = np.load(opened_file)
                episode_len = data["obs"].shape[0]
                obs = data["obs"][example_idx : example_idx + self.window_width]
                act = data["act"][example_idx : example_idx + self.window_width]
                example = {"obs": obs, "act": act}

        # workaround for issue with fsspec caching with multiple workers
        except (EOFError, zipfile.BadZipFile) as _:
            return None

        # check that the episode length matches the fixed episode length
        if self.fixed_episode_length is not None:
            assert episode_len == self.fixed_episode_length, "Episode length does not match fixed episode length."
        return example
