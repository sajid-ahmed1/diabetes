import os

# Fix for segmentation fault on macOS with LightGBM/OpenMP
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from diabetes.data import load_model
from diabetes.evaluation import evaluate_predictions


def _predict_positive(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Get the probability of the positive class (1).
    Handles the API difference between LGBMClassifier and GeneralizedLinearRegressor.
    """
    if hasattr(model, "predict_proba"):
        # LGBMClassifier / sklearn classifiers
        return model.predict_proba(X)[:, 1]  # type: ignore[no-any-return]
    # GeneralizedLinearRegressor (glum)
    return model.predict(X)  # type: ignore[no-any-return]


ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"

# %%
# Load and split data
# Load split data directly to ensure consistency with training
print("\n=== Loading Data for Evaluation ===")
df = pd.read_parquet(DATA_DIR / "data_split.parquet")

train_idx = np.where(df["sample"] == "train")[0]
test_idx = np.where(df["sample"] == "test")[0]

df_train = df.iloc[train_idx].copy()
df_test = df.iloc[test_idx].copy()
print(f"Loaded {len(df_train)} train samples and {len(df_test)} test samples.")

# %%
# Feature definitions (shared with training/visualisation scripts)
categoricals: list[str] = []

numericals = [
    "Pregnancies",
    "Glucose",
    "BMI",
    "Age",
    "Insulin_log",
    "DPF_log",
    "BloodPressure_log",
    "Glucose_Insulin_Interaction",
    "Age_Insulin_Interaction",
]

predictors = categoricals + numericals

X_train = df_train[predictors]
y_train = df_train["Outcome"]

X_test = df_test[predictors]
y_test = df_test["Outcome"]

# %%
# Load saved pipelines from model_training
print("\n=== Loading Trained Models ===")
pipelines = {
    "glm_baseline": load_model("glm_baseline"),
    "lgbm_baseline": load_model("lgbm_baseline"),
    "glm_best": load_model("glm_best_pipeline"),
    "lgbm_best": joblib.load(MODELS_DIR / "lgbm_best_pipeline.pkl"),
}

# Sanity Check: Train a model on shuffled targets
# If our models are learning real signal, they should significantly outperform this.
print("\n=== Training Shuffled Target Sanity Check ===")
y_shuffled = np.random.permutation(y_train)
shuffled_pipeline = clone(pipelines["lgbm_baseline"])
shuffled_pipeline.fit(X_train, y_shuffled)
pipelines["shuffled_lgbm"] = shuffled_pipeline

# %%
# Evaluate each model on the test split and collect metrics
pd.set_option("display.float_format", lambda x: "%.4f" % x)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
df_eval = pd.DataFrame({"Outcome": y_test})
df_eval_train = pd.DataFrame({"Outcome": y_train})
metrics_tables: list[pd.DataFrame] = []
print("\n=== Calculating Metrics ===")

for label, model in pipelines.items():
    print(f"Processing {label}...")

    # Test Metrics
    pred_col = f"p_{label}"
    df_eval[pred_col] = _predict_positive(model, X_test)

    metrics = evaluate_predictions(
        df_eval,
        outcome_column="Outcome",
        preds_column=pred_col,
    ).rename(columns={"value": f"{label}_test"})
    metrics_tables.append(metrics)

    # Train Metrics
    pred_col_train = f"p_{label}_train"
    df_eval_train[pred_col_train] = _predict_positive(model, X_train)

    metrics_train = evaluate_predictions(
        df_eval_train,
        outcome_column="Outcome",
        preds_column=pred_col_train,
    ).rename(columns={"value": f"{label}_train"})
    metrics_tables.append(metrics_train)

summary = pd.concat(metrics_tables, axis=1)
print("\n=== Final Evaluation Summary ===")
print(summary)

# %%
