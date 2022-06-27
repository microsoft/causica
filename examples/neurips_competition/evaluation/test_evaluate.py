import os
from tempfile import TemporaryDirectory

import numpy as np

from .evaluate import evaluate_and_write_scores


def test_evaluate_and_write_score():
    true_ate = np.arange(6).reshape(2, 3)
    predicted_ate = np.arange(6).reshape(2, 3)
    predicted_ate[1] += 1

    true_adj = np.array([[True, False, False], [False, False, True], [False, False, False]], dtype=bool)[
        np.newaxis, :, :
    ]
    predicted_adj = np.array([[True, False, True], [False, False, True], [False, False, False]], dtype=bool)[
        np.newaxis, np.newaxis, :, :
    ]

    with TemporaryDirectory() as tmpdirname:
        input_dir = os.path.join(tmpdirname, "submission")
        score_dir = os.path.join(tmpdirname, "scores")

        os.makedirs(os.path.join(input_dir, "ref"))
        os.makedirs(os.path.join(input_dir, "res"))
        os.makedirs(score_dir)

        np.save(os.path.join(input_dir, "ref", "ate_estimate.npy"), true_ate)
        np.save(os.path.join(input_dir, "res", "ate_estimate.npy"), predicted_ate)

        np.save(os.path.join(input_dir, "ref", "adj_matrix.npy"), true_adj)
        np.save(os.path.join(input_dir, "res", "adj_matrices.npy"), predicted_adj)

        evaluate_and_write_scores(
            os.path.join(input_dir, "ref"), os.path.join(input_dir, "res"), score_dir, summarise=False
        )

        with open(os.path.join(score_dir, "scores.txt"), "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert lines[0].startswith("orientation_fscore_dataset_0: 0.66")
        assert lines[1].strip() == "ate_rmse_dataset_0: 0.0"
        assert lines[2].strip() == "ate_rmse_dataset_1: 1.0"


if __name__ == "__main__":
    test_evaluate_and_write_score()
