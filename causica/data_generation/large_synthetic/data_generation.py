import numpy as np

from .data_utils import gen_dataset, get_name, save_data

if __name__ == "__main__":
    num_samples_train = 5000
    num_samples_test = 5000
    noise_mult_factor = 2
    expected_num_latent_confounders = 0

    N_interventions = 5
    generate_references = True
    discrete_dims_list = None

    max_parent_depth = 2
    use_quantile_references = True

    partial_ratio = 0.3

    dataset_folder = "causica/data_generation/large_synthetic/data/"

    # sem_type: 'linear', 'mlp', 'spline'
    # noise_type: uniform(-1,1), mlp, 'fixed'(N(0,1)), float <- max variance of Normal noise

    N_seeds = 5
    sem_noise_pairs = [("spline", "fixed"), ("spline", "mlp")]

    for base_seed in range(N_seeds):
        for graph_type in ["ER", "SF"]:
            for N in [16, 64, 128]:
                for E in [N, int(4 * N)]:
                    for sem_type, noise_type in sem_noise_pairs:

                        if isinstance(noise_type, float) and noise_type < 0.5:
                            noise_mult_factor_ = 10
                        else:
                            noise_mult_factor_ = noise_mult_factor

                            if expected_num_latent_confounders > 0:
                                name = "latent_" + get_name(graph_type, N, E, sem_type, noise_type, base_seed)
                            else:
                                name = get_name(graph_type, N, E, sem_type, noise_type, base_seed)
                            print(name)
                            (
                                X_train_test,
                                X_train,
                                X_test,
                                directed_matrix,
                                bidirected_matrix,
                                all_intervention_data,
                            ) = gen_dataset(
                                base_seed,
                                num_samples_train,
                                num_samples_test,
                                graph_type,
                                N,
                                E,
                                sem_type,
                                noise_type,
                                N_interventions=N_interventions,
                                generate_references=generate_references,
                                max_parent_depth=max_parent_depth,
                                use_quantile_references=use_quantile_references,
                                adj_matrix=None,
                                noise_mult_factor=noise_mult_factor,
                                discrete_dims_list=discrete_dims_list,
                                discrete_temperature=None,
                                expected_num_latent_confounders=expected_num_latent_confounders,
                            )

                            print(
                                X_train_test.shape,
                                X_train.shape,
                                X_test.shape,
                                directed_matrix.shape,
                                all_intervention_data.shape,
                            )

                            save_data(
                                dataset_folder,
                                name,
                                directed_matrix,
                                bidirected_matrix,
                                X_train_test,
                                X_train,
                                X_test,
                                all_intervention_data,
                            )

                            # https://stackoverflow.com/questions/47941079/can-i-make-random-mask-with-numpy
                            num_elements = np.prod(X_train.shape)
                            random_mask = np.zeros(num_elements, dtype=int)
                            random_mask[: int(num_elements * partial_ratio)] = 1
                            np.random.shuffle(random_mask)
                            random_mask = random_mask.astype(bool).reshape(X_train.shape)

                            X_train_partial = X_train[:]
                            X_train_partial[random_mask] = np.nan
                            X_train_test_partial = np.concatenate([X_train_partial, X_test], axis=0)

                            print(
                                X_train_test.shape,
                                X_train.shape,
                                X_test.shape,
                                directed_matrix.shape,
                                all_intervention_data.shape,
                            )

                            save_data(
                                dataset_folder,
                                name + "_partial",
                                directed_matrix,
                                bidirected_matrix,
                                X_train_test_partial,
                                X_train_partial,
                                X_test,
                                all_intervention_data,
                            )
