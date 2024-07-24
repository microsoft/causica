datasets=(
    triangle
    symprod_simpson
    large_backdoor
)

TOTAL_SEED=1

for dataset in "${datasets[@]}";
do
    for ((seed=1;seed<=TOTAL_SEED;seed++));
    do
        python -m fip.data_generation.normalizing_flow_data --dist_case $dataset --seed $seed
    done
done
