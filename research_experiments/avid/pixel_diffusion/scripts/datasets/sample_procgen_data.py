"""This script samples random Procgen data according to the process defined in the Genie paper appendix. 

The data is stored as .npz files in the specified output folder.
"""

import argparse

from dwma.datasets.procgen_sampler import sample_random_procgen_data

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Sample random Procgen data.")
    parser.add_argument(
        "--output_folder",
        type=str,
        default="az://fvwm@azuastorage.blob.core.windows.net/genie/data/coinrun",
        help="The output folder path for storing data",
    )
    parser.add_argument(
        "--env_name", type=str, default="coinrun", help="The name of the environment to sample data from"
    )
    parser.add_argument("--num_levels", type=int, default=10000, help="The number of levels to run")
    parser.add_argument("--steps_per_level", type=int, default=1000, help="The number of steps per level")
    parser.add_argument("--first_level", type=int, default=0, help="The level to start from")
    parser.add_argument("--chunk_size", type=int, default=1000, help="The number of steps to save in each file")

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the parsed arguments
    sample_random_procgen_data(
        output_folder=args.output_folder,
        env_name=args.env_name,
        num_levels=args.num_levels,
        first_level=args.first_level,
        steps_per_level=args.steps_per_level,
        chunk_size=args.chunk_size,
    )
