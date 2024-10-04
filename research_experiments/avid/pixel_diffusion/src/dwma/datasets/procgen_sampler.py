import os
from datetime import datetime

import fsspec
import gym
import numpy as np
from tqdm import tqdm


def sample_random_procgen_data(
    output_folder: str = "./output",
    env_name: str = "coinrun",
    num_levels: int = 10000,
    first_level: int = 0,
    steps_per_level: int = 1000,
    chunk_size: int = 1000,
):
    """Sample random data from the Procgen environment and save it to output folder.

    Samples steps_per_level steps of random data for each of the first num_levels levels of the Procgen environment.

    Args:
        output_folder: folder to save the data
        env_name: name of the Procgen environment
        num_levels: number of levels to sample
        start_level: level to start sampling from
        steps_per_level: number of steps to sample for each level
        chunk_size: number of steps to save in each npz file. Smaller chunks may help speed up dataloading.
    """

    if chunk_size < steps_per_level:
        assert steps_per_level % chunk_size == 0, "steps_per_level must be divisible by chunk_size"

    # iterate over the first num_levels levels of the environment
    for level in tqdm(range(first_level, first_level + num_levels)):
        env = gym.make(f"procgen:procgen-{env_name}-v0", start_level=level, num_levels=1)
        obs = env.reset()
        episode: dict[str, list] = {"obs": [], "act": [], "rew": [], "term": []}

        # save random data for each level
        for step in range(steps_per_level):
            random_act = env.action_space.sample()
            next_obs, _, term, *_ = env.step(random_act)
            episode["obs"].append(obs)
            episode["act"].append(random_act)

            obs = next_obs
            if term:
                obs = env.reset()

            # Save in chunks of chunk_size steps
            if (step + 1) % chunk_size == 0 or step == steps_per_level - 1:
                episode_np = {k: np.array(v) for k, v in episode.items()}
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
                chunk_index = step // chunk_size
                fname = f"{env_name}_lvl_{level}_chunk_{chunk_index}_{timestamp}.npz"
                with fsspec.open(os.path.join(output_folder, fname), "wb") as f:
                    np.savez(f, **episode_np)

                # Reset episode data for the next chunk
                episode = {"obs": [], "act": [], "rew": [], "term": []}
