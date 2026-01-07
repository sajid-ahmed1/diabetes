import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import roc_auc_score


def value_counts(df: pd.DataFrame, column: str) -> None:
    """
    Checks the value counts of the variable with a plot
    """

    print(f"The values within this {column} are : \n{df[column].value_counts()}")
    print("-" * 30)
    df[column].value_counts().plot(kind="bar")
    return None


def plot_predicted_vs_actual(
    df: pd.DataFrame,
    pred_col: str,
    target_col: str,
    n_bins: int = 10,
    title: str | None = None,
) -> Figure:
    """
    Plot mean predicted probability vs. observed event rate.

    Predictions are binned into quantiles; for each bin the mean prediction
    and mean outcome are plotted.

    Parameters
    ----
    df (pd.DataFrame):
        Dataframe containing prediction and target columns.
    pred_col (str):
        Column name with predicted probabilities.
    target_col (str):
        Column name with outcomes.
    n_bins (int):
        Number of quantile bins to use.
    title (str | None):
        Optional title override.

    Returns
    ----
    Figure:
        The generated Matplotlib figure.
    """
    tmp = df[[pred_col, target_col]].copy()
    tmp["bin"] = pd.qcut(tmp[pred_col], q=n_bins, duplicates="drop")

    grouped = (
        tmp.groupby("bin")
        .agg(
            mean_pred=(pred_col, "mean"),
            mean_actual=(target_col, "mean"),
            count=(target_col, "size"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(grouped["mean_pred"], grouped["mean_actual"], marker="o")
    ax.plot([0, 1], [0, 1], "--", color="gray")  # perfect calibration line
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed event rate")
    ax.set_title(title or "Predicted vs Actual")
    ax.grid(True)

    return fig


def lorenz_curve(
    df: pd.DataFrame,
    outcome_column: str,
    preds_column: str | None = None,
    title: str | None = None,
    preds_columns: dict[str, str] | None = None,
    include_oracle: bool = False,
) -> Figure:
    """
    Plot a Lorenz (cumulative gains) curve for the same inputs used by
    `evaluate_predictions`.

    Parameters
    ----
    df (pd.DataFrame):
        Dataframe containing predictions and binary outcomes.
    outcome_column (str):
        Name of the binary outcome column.
    preds_column (str | None):
        Column containing predicted probabilities or scores. Kept for backward
        compatibility when plotting a single model.
    preds_columns (dict[str, str] | None):
        Optional mapping of legend label -> column name to plot multiple models
        on the same figure.
    include_oracle (bool):
        Whether to include an oracle curve (sorting by the true outcome) for
        reference.
    title (str | None):
        Optional custom title.

    Returns
    ----
    Figure:
        Matplotlib figure showing the Lorenz curve.
    """
    column_map = {str(label): col for label, col in (preds_columns or {}).items()}
    if preds_column:
        fallback_label = preds_column
        if fallback_label in column_map:
            suffix = 1
            new_label = f"{fallback_label}_{suffix}"
            while new_label in column_map:
                suffix += 1
                new_label = f"{fallback_label}_{suffix}"
            fallback_label = new_label
        column_map[fallback_label] = preds_column

    if not column_map:
        raise ValueError(
            "Provide at least one predictions column via `preds_column` or "
            "`preds_columns`."
        )

    subset_columns = [outcome_column, *column_map.values()]
    tmp = df[subset_columns].dropna().copy()
    if tmp.empty:
        msg = (
            "No rows are available to plot after dropping NA values "
            f"from {', '.join(subset_columns)!r}."
        )
        raise ValueError(msg)

    total_events = tmp[outcome_column].sum()
    if total_events <= 0:
        raise ValueError(
            f"Outcome column {outcome_column!r} contains no positive events."
        )

    fig, ax = plt.subplots(figsize=(6, 6))
    unique_outcomes = tmp[outcome_column].nunique()

    def _curve(sorted_outcomes: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        cumulative_population = np.arange(
            1, len(sorted_outcomes) + 1, dtype=float
        ) / len(sorted_outcomes)
        cumulative_events = sorted_outcomes.cumsum().to_numpy() / total_events
        x_vals = np.concatenate(([0.0], cumulative_population))
        y_vals = np.concatenate(([0.0], cumulative_events))
        return x_vals, y_vals

    for label, column in column_map.items():
        sorted_tmp = tmp.sort_values(column, ascending=False)
        x_values, y_values = _curve(sorted_tmp[outcome_column])

        gini_value = float("nan")
        if unique_outcomes > 1:
            auc = float(roc_auc_score(tmp[outcome_column], tmp[column]))
            gini_value = 2 * auc - 1

        display_label = label
        if np.isfinite(gini_value):
            display_label = f"{label} (Gini={gini_value:.3f})"

        ax.plot(x_values, y_values, label=display_label)

    if include_oracle:
        oracle_sorted = tmp.sort_values(outcome_column, ascending=False)
        x_values, y_values = _curve(oracle_sorted[outcome_column])
        ax.plot(x_values, y_values, linestyle=":", color="black", label="Oracle")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")

    ax.set_xlabel("Cumulative share of population")
    ax.set_ylabel("Cumulative share of positive outcomes")
    ax.set_title(title or "Lorenz Curve")
    ax.legend()
    ax.grid(True)

    return fig
