datasets=(
    lingauss
    linexp
    nonlingauss
    nonlin_simpson
    symprod_simpson
    large_backdoor
    weak_arrow
)

TOTAL_SEED=1

for dataset in "${datasets[@]}";
do
    for ((seed=1;seed<=TOTAL_SEED;seed++));
    do
        python -m fip.data_generation.csuite_data --dist_case $dataset --seed $seed
    done
done
