import pandas as pd

from causica.experiment.steps.aggregation_step import _get_summary_dict_from_dataframe, _get_summary_dicts_from_groupby


def get_df():

    return pd.DataFrame(
        [
            {"model_config.random_seed": model_seed, "dataset_config.random_seed": dataset_seed, "results.x": x}
            for model_seed, dataset_seed, x in [(0, 0, 0.0), (0, 1, 1.0), (2, 0, 2.0), (2, 1, 10.0)]
        ]
    )


def test_summary_dict_from_dataframe():
    df = get_df()
    summary = _get_summary_dict_from_dataframe(df)
    print(summary)
    assert summary == {
        "model_config": {"random_seed": {"mean": 1.0, "std": 1.0, "num_samples": 4}},
        "dataset_config": {"random_seed": {"mean": 0.5, "std": 0.5, "num_samples": 4}},
        "results": {"x": {"mean": 3.25, "std": 3.960744879438715, "num_samples": 4}},
    }


def test_summary_dict_from_groupby():
    df = get_df()
    grouped = df.groupby("dataset_config.random_seed")
    summary = _get_summary_dicts_from_groupby(grouped)
    assert summary == {
        0: {
            "model_config": {"random_seed": {"mean": 1, "num_samples": 2, "std": 1.0}},
            "results": {"x": {"mean": 1.0, "num_samples": 2, "std": 1.0}},
        },
        1: {
            "model_config": {"random_seed": {"mean": 1, "num_samples": 2, "std": 1.0}},
            "results": {"x": {"mean": 5.5, "num_samples": 2, "std": 4.5}},
        },
    }
