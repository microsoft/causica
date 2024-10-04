#!/bin/bash

# List of Procgen environment names without coinrun
environments=("bigfish" "bossfight" "caveflyer" "chaser" "climber" "coinrun" "dodgeball" "fruitbot" "heist" "jumper" "leaper" "maze" "miner" "ninja" "plunder" "starpilot")

# Loop through each environment and save random data
for env_name in "${environments[@]}"
do
    echo "Processing environment: $env_name"
    python sample_procgen_data.py --output_folder /datasets/procgen_mixed_no_coinrun_train --num_levels 2000 --first_level 0 --env_name "$env_name" --chunk_size 50
    python sample_procgen_data.py --output_folder /datasets/procgen_mixed_no_coinrun_test --num_levels 100 --first_level 2000 --env_name "$env_name" --chunk_size 50
done

echo "All environments processed."