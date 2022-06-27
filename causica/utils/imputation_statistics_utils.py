import numpy as np
from sklearn.preprocessing import label_binarize


class ImputationStatistics:
    @classmethod
    def get_statistics(cls, data, variables, verbose=False):
        # get statistics on the marginal distributions
        # data: an np.array that has shape (sample_count, N_data, N_feature)
        stats = {}
        for var_idx, var in enumerate(variables):
            if var.type_ == "continuous":
                stats[var_idx] = cls._statistics_continuous(data[:, :, var_idx].astype(float), var)
            elif var.type_ == "binary":
                stats[var_idx] = cls._statistics_binary(data[:, :, var_idx].astype(float), var)
            elif var.type_ == "categorical":
                stats[var_idx] = cls._statistics_categorical(data[:, :, var_idx].astype(float), var)
            elif var.type_ == "text":
                stats[var_idx] = cls._statistics_text(var)
            else:
                raise ValueError(f"data statistics computation not supported for {var.type_} variable")

        if verbose:  # print statistics

            print("Data statistics")
            for var_idx, var in enumerate(variables):
                output = f"id: {var_idx}, "
                for key in stats[var_idx].keys():
                    val = stats[var_idx][key]
                    output += f"{key}: "
                    if isinstance(val, (float, np.float32, np.float64)):  # assume a float number
                        output += f"{val: .3f}, "
                    else:
                        output += f"{val}, "
                output = output[:-2]
                print(output)

        return stats

    @staticmethod
    def _statistics_categorical(feature, variable):
        # assume we collect the statistics of a categorical data
        # feature: an np.array has shape (sample_count, N_data)
        # variable: an object of variable class

        assert variable.type_ == "categorical", f"expecting a categorical variable, got {variable.type_}"
        stats = {"type": variable.type_}

        # convert to one-hot

        value_range = range(int(variable.lower), int(variable.upper + 1))
        sample_count, N_data = feature.shape
        stats["n_class"] = len(value_range)
        # If a feature is categorical, some methods such as mean imputing treating this as continous. Thus convert it to int first.
        int_feature = np.rint(feature.reshape((-1)))
        processed_feature = label_binarize(int_feature, classes=value_range, neg_label=0, pos_label=1).reshape(
            (sample_count, N_data, stats["n_class"])
        )

        # now compute statistics! assume processed_feature has shape (sample_count, N_data, n_class)
        stats["marginal_prob"] = np.mean(processed_feature, axis=0)
        stats["majority_vote"] = np.argmax(stats["marginal_prob"], axis=-1)
        stats["majority_prob"] = stats["marginal_prob"][np.arange(N_data), stats["majority_vote"]]
        stats["majority_vote"] += variable.lower  # shift the category back if variable.lower != 0
        stats["entropy"] = -np.sum(
            stats["marginal_prob"] * np.log(np.clip(stats["marginal_prob"], 1e-5, 1.0)),
            axis=-1,
        )

        return stats

    @staticmethod
    def _statistics_binary(feature, variable):
        # assume we collect the statistics of a binary data
        # feature: an np.array has shape (sample_count, N_data)
        # variable: an object of variable class

        assert variable.type_ == "binary", f"expecting a binary variable, got {variable.type_}"
        stats = {"type": variable.type_}

        # now compute statistics! assume feature has shape (sample_count, N_data) with entries in {0, 1}
        stats["n_class"] = 2
        prob = np.mean(feature, axis=0)  # probability of class 1
        stats["majority_vote"] = np.asarray(prob > 0.5, dtype="f")  # 1 or zero
        stats["majority_prob"] = stats["majority_vote"] * prob + (1 - stats["majority_vote"]) * (1 - prob)
        stats["entropy"] = -(
            prob * np.log(np.clip(prob, 1e-5, 1.0)) + (1 - prob) * np.log(np.clip(1 - prob, 1e-5, 1.0))
        )

        return stats

    @staticmethod
    def _statistics_continuous(feature, variable):
        # assume we collect the statistics of a continuous data
        # feature: an np.array has shape (sample_count, N_data)
        # variable: an object of variable class

        assert variable.type_ == "continuous", f"expecting a continuous variable, got {variable.type_}"
        stats = {"type": variable.type_}

        # now compute statistics! assume processed_masked_feature has shape (N_observed,)
        stats["variable_lower"] = variable.lower
        stats["variable_upper"] = variable.upper  # defined by the variable
        # stats for a box plot
        stats["min_val"] = np.min(feature, axis=0)  # the actual min value in data
        stats["max_val"] = np.max(feature, axis=0)  # the actual max value in data
        stats["mean"] = np.mean(feature, axis=0)
        # now compute quartiles, 25%, 50% (median), 75%
        stats["quartile_1"], stats["median"], stats["quartile_3"] = np.quantile(feature, [0.25, 0.5, 0.75], axis=0)

        return stats

    @staticmethod
    def _statistics_text(variable):
        assert variable.type_ == "text", f"expecting a text variable, got {variable.type_}"
        stats = {"type": variable.type_}

        # TODO #18598: Add metrics for text variable
        # To do so, we probably need to add decoding capability to text embedder first

        return stats
