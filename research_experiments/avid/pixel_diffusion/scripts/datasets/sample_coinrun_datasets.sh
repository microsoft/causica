#!/bin/bash

# List of Procgen environment names without coinrun
environments=("coinrun")

# Loop through each environment and save random data
for env_name in "${environments[@]}"
do
    echo "Processing environment: $env_name"
    python sample_procgen_data.py --output_folder /datasets/coinrun_500lvl_train --num_levels 500 --first_level 0 --env_name "$env_name" --chunk_size 50
    python sample_procgen_data.py --output_folder /datasets/coinrun_test --num_levels 1000 --first_level 10000 --env_name "$env_name" --chunk_size 50
done

echo "All environments processed."