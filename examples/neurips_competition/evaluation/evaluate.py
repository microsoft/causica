import argparse
import os
from typing import Any, Dict, List, Optional

import numpy as np
from adjacency_utils import edge_prediction_metrics_multisample

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<p>Detailed Score{page_header_section}</p>
</br>
<p>Overall score: {avg_score}</p>
</br>
{detailed_score_section}
</html>
"""


def load_submitted_adj(prediction_dir: str) -> np.ndarray:
    """Loads submitted adjacency matrix from the given directory.

    Args:
        prediction_dir (str): Submission directory with adj_matrices.npy file.

    Returns:
        np.ndarray: Loaded adjacency matrix of shape [num_datasets, num_matrices, variables, variables]
    """

    prediction_file = os.path.join(prediction_dir, "adj_matrix.npy")

    adj_matrices = np.load(prediction_file)

    assert adj_matrices.ndim in (3, 4), "Needs to be a 4D or 3D array"

    if adj_matrices.ndim == 3:
        adj_matrices = adj_matrices[:, np.newaxis, :, :]

    return adj_matrices.astype(bool)


def load_true_adj_matrix(reference_dir: str) -> np.ndarray:
    """Loads true adjacency matrix from the given directory.

    Args:
        reference_dir (str): Reference directory with adj_matrix.npy file.

    Returns:
        np.ndarray: Loaded adjacency matrix of shape [num_datasets, variables, variables]
    """

    reference_file = os.path.join(reference_dir, "adj_matrix.npy")

    return np.load(reference_file).astype(bool)


def load_predicted_ate_estimate(prediction_dir: str) -> np.ndarray:
    """Loads predicted ATE estimate from the given directory.

    Args:
        prediction_dir (str): Prediction directory with ate_estimate.npy file.

    Returns:
        np.ndarray: Loaded ATE estimate of shape [num_datasets, num_interventions]
    """

    prediction_file = os.path.join(prediction_dir, "cate_estimate.npy")

    ate_predictions = np.load(prediction_file)

    assert ate_predictions.ndim == 2

    return ate_predictions


def load_true_ate_estimate(reference_dir: str) -> np.ndarray:
    """Loads true ATE estimate from the given directory.

    Args:
        reference_dir (str): Reference directory with ate_estimate.npy file.

    Returns:
        np.ndarray: Loaded ATE estimate of shape [num_datasets, num_interventions]
    """

    reference_file = os.path.join(reference_dir, "cate_estimate.npy")

    ate_reference = np.load(reference_file)

    assert ate_reference.ndim == 2

    return ate_reference


def write_score_file(
    score_dir: str,
    score_dict: Dict[str, Any],
    summarise: bool = True,
    page_header_name: str = "",
    detailed_dict: Optional[Dict[str, Any]] = None,
    write_html: bool = False,
) -> None:
    """Writes the score dictionary to the given file.
    Args:
        score_dir (str): Dir to write the score dictionary to.
        score_dict (dict[str, Any]): Score dictionary to write.
        summarise (bool): Whether to summarise the score dictionary.
        page_header_name (str): Name of the page header.
        detailed_dict (dict[str, Any]): Detailed score dictionary to write to HTML output.
    """

    score_file = os.path.join(score_dir, "scores.txt")

    if summarise:
        avg_score = np.mean(list(score_dict.values()))

        with open(score_file, "w", encoding="utf-8") as f:
            f.write(f"score:{avg_score}\n")

        if write_html:
            # output detailed results
            detail_file = os.path.join(score_dir, "scores.html")

            if len(page_header_name) > 0:
                page_header_name = f": {page_header_name}"

            detailed_score_section = f"<p>Detailed Score{page_header_name}</p>\n</br>\n"
            detailed_dict = detailed_dict or score_dict
            for k, v in detailed_dict.items():
                detailed_score_section += f"<p>{k}: {v}</p>\n"

            with open(detail_file, "w", encoding="utf-8") as f:
                detail_output = HTML_TEMPLATE.format(
                    page_header_section=page_header_name,
                    avg_score=avg_score,
                    detailed_score_section=detailed_score_section,
                )
                f.write(detail_output)
    else:
        with open(score_file, "w", encoding="utf-8") as f:
            for key, value in score_dict.items():
                f.write(f"{key}: {value}\n")


def evaluate_adjacency(solution_dir: str, prediction_dir: str) -> List[Any]:
    """Evaluate adjacency matrices

    Args:
        solution_dir (str): Directory with true solutions.
        prediction_dir (str): Directory with predictions.

    Returns:
        Dict[str, Any]: Metrics dictionary
    """
    true_adj_matrix = load_true_adj_matrix(solution_dir)
    submitted_adj_matrix = load_submitted_adj(prediction_dir)

    assert true_adj_matrix.shape[0] == submitted_adj_matrix.shape[0], "Need to submit adjacency for each dataset."

    adj_metrics = []
    for i in range(true_adj_matrix.shape[0]):
        adj_metrics.append(edge_prediction_metrics_multisample(true_adj_matrix[i], submitted_adj_matrix[i]))

    return adj_metrics


def evaluate_ate(solution_dir: str, prediction_dir: str) -> List[Dict[str, Any]]:
    """Evaluate adjacency matrices

    Args:
        solution_dir (str): Directory with true solutions.
        prediction_dir (str): Directory with predictions.

    Returns:
        Dict[str, Any]: Metrics dictionary
    """
    true_adj_matrix = load_true_ate_estimate(solution_dir)
    submitted_adj_matrix = load_predicted_ate_estimate(prediction_dir)

    assert true_adj_matrix.shape == submitted_adj_matrix.shape, "Need to submit ATE for each dataset and intervention."

    # calculating RMSE across each intervention for each dataset.
    rmse = np.sqrt(np.mean(np.square(true_adj_matrix - submitted_adj_matrix), axis=1))

    return [{"ate_rmse": rmse_slice} for rmse_slice in rmse]


def evaluate_and_write_scores(
    solution_dir: str,
    prediction_dir: str,
    score_dir: str,
    eval_adjacency: bool = True,
    eval_ate: bool = True,
    summarise: bool = True,
    page_header_name: str = "",
    write_html: bool = False,
    score_multiplier: float = 1.0,
) -> None:
    """Run main evaluation and write scores to file.

    Args:
        solution_dir (str): Directory with true solutions.
        prediction_dir (str): Directory with predictions.
        score_dir (str): Directory to write scores to.
        evaluate_adjacency (bool, optional): Whether to evaluate adjacency matrix. Defaults to True.
        evaluate_ate (bool, optional): Whether to evalaute ATEs. Defaults to True.
        summarise (bool, optional): Whether to summarise scores. Defaults to True.
        page_header_name (str, optional): Name of the page header. Defaults to "".
        write_html (bool, optional): Whether to write the detailed HTML file.
        score_multiplier: (float, optional): factor to use for multiplying the scores with.
    """
    if summarise and eval_adjacency and eval_ate:
        raise Exception("Cannot evaluate adjacency and ATE at the same time.")

    score_dict = {}
    if eval_adjacency:
        detailed_metrics = evaluate_adjacency(solution_dir, prediction_dir)

        for i, adj_metric in enumerate(detailed_metrics):
            score_dict[f"orientation_fscore_dataset_{i}"] = adj_metric["orientation_fscore"] * score_multiplier

    if eval_ate:
        detailed_metrics = evaluate_ate(solution_dir, prediction_dir)

        for i, ate_metric in enumerate(detailed_metrics):
            score_dict[f"neg_ate_rmse_dataset_{i}"] = -ate_metric["ate_rmse"] * score_multiplier

    if summarise:
        # convert list of dicts to flat dict
        detailed_metrics_dict = {f"{k}_dataset_{i}": v for i, d in enumerate(detailed_metrics) for k, v in d.items()}
        write_score_file(
            score_dir,
            score_dict,
            summarise=True,
            page_header_name=page_header_name,
            detailed_dict=detailed_metrics_dict,
            write_html=write_html,
        )
    else:
        write_score_file(score_dir, score_dict, summarise=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the submitted adjacency matrices.")
    parser.add_argument("--submission_dir", type=str, help="Directory with the input.")
    parser.add_argument("--reference_dir", type=str, help="Directory with the reference files.")
    parser.add_argument("--score_dir", type=str, help="Directory to write the scores to.")

    parser.add_argument("--evaluate_adjacency", action="store_true", help="Evaluate adjacency matrices.")
    parser.add_argument("--evaluate_ate", action="store_true", help="Evaluate ATE.")

    parser.add_argument("--summarise", action="store_true", help="Summarise scores.")
    parser.add_argument("--write_html", action="store_true", help="Whether to write detailed scores.")
    parser.add_argument("--score_multiplier", type=float, help="Multiplier for the score.", default=1.0)

    args = parser.parse_args()
    args.page_name = "Detailed Results"

    evaluate_and_write_scores(
        args.reference_dir,
        args.submission_dir,
        args.score_dir,
        args.evaluate_adjacency,
        args.evaluate_ate,
        args.summarise,
        args.page_name,
        args.write_html,
        args.score_multiplier,
    )


if __name__ == "__main__":
    main()
