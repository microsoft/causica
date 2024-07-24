type_funcs=(
    linear
    rff
)

type_graphs=(
    er
    sf_in
    sf_out
    sbm
    ws

)

type_noises=(
    gaussian
    laplace
)

dist_cases=(
    in
    out
)


TOTAL_SEED=1

for func_type in "${type_funcs[@]}";
do  
    for graph_type in "${type_graphs[@]}";
    do 
        for noise_type in "${type_noises[@]}";
        do
            for dist_case in "${dist_cases[@]}";
            do
                for ((seed=1;seed<=TOTAL_SEED;seed++));
                do
                    python -m fip.data_generation.avici_data --func_type $func_type --graph_type $graph_type --noise_type $noise_type --dist_case $dist_case --seed $seed --data_dim 5 --num_interventions 5
                    python -m fip.data_generation.avici_data --func_type $func_type --graph_type $graph_type --noise_type $noise_type --dist_case $dist_case --seed $seed --data_dim 10 --num_interventions 10
                    python -m fip.data_generation.avici_data --func_type $func_type --graph_type $graph_type --noise_type $noise_type --dist_case $dist_case --seed $seed --data_dim 20 --num_interventions 20
                    python -m fip.data_generation.avici_data --func_type $func_type --graph_type $graph_type --noise_type $noise_type --dist_case $dist_case --seed $seed --data_dim 50 --num_interventions 50
                    python -m fip.data_generation.avici_data --func_type $func_type --graph_type $graph_type --noise_type $noise_type --dist_case $dist_case --seed $seed --data_dim 100 --num_interventions 100
                    python -m fip.data_generation.avici_data --func_type $func_type --graph_type $graph_type --noise_type $noise_type --dist_case $dist_case --seed $seed --data_dim 200
                done
            done
        done
    done
done

