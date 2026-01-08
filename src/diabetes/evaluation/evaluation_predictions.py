import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

# ChatGPT helped take what we had in the problem set 3 claims data and rework
# it for the data science question I had which was binary classification.
# There are some differences because I don't have a Tweedie Distribution just
# Binomial distribution.


def evaluate_predictions(
    df: pd.DataFrame,
    outcome_column: str,
    preds_column: str,
) -> pd.DataFrame:
    """
    Evaluate predicted probabilities against a binary outcome.

    Parameters
    ----
    df (pd.DataFrame):
        DataFrame containing outcomes and predictions.
    outcome_column (str):
        Name of the binary outcome column (0/1).
    preds_column (str):
        Name of the column containing predicted probabilities for class 1.

    Returns
    ----
    pd.DataFrame:
        DataFrame (index=metric names) containing evaluation metrics.
    """

    y_true = df[outcome_column].astype(float).to_numpy()
    preds = df[preds_column].astype(float).to_numpy()

    preds = np.clip(preds, 1e-15, 1 - 1e-15)

    mean_preds = float(np.average(preds))
    mean_outcome = float(np.average(y_true))

    abs_bias = mean_preds - mean_outcome
    bias = float("nan")
    if mean_outcome != 0:
        bias = float((mean_preds - mean_outcome) / mean_outcome)

    mse = float(np.average((preds - y_true) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.average(np.abs(preds - y_true)))

    ll = float(log_loss(y_true, preds))
    brier = float(brier_score_loss(y_true, preds))

    try:
        auc = float(roc_auc_score(y_true, preds))
    except ValueError:
        auc = float("nan")

    gini = float(2 * auc - 1) if np.isfinite(auc) else float("nan")

    evals: dict[str, float] = {
        "mean_preds": mean_preds,
        "mean_outcome": mean_outcome,
        "abs_bias": abs_bias,
        "bias": bias,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "log_loss": ll,
        "brier": brier,
        "auc": auc,
        "gini": gini,
    }

    metrics = pd.DataFrame.from_dict(evals, orient="index", columns=["value"])
    return metrics
