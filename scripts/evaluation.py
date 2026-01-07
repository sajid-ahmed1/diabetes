import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from diabetes.data import create_sample_split, load_model, load_parquet
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


# %%
# Load and split data
df_model = load_parquet().copy()
df = create_sample_split(df_model, id_column="Id")

train_idx = np.where(df["sample"] == "train")[0]
test_idx = np.where(df["sample"] == "test")[0]

df_train = df.iloc[train_idx].copy()
df_test = df.iloc[test_idx].copy()

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

X_test = df_test[predictors]
y_test = df_test["Outcome"]

# %%
# Load saved pipelines from model_training
pipelines = {
    "glm_baseline": load_model("glm_baseline"),
    "lgbm_baseline": load_model("lgbm_baseline"),
    "glm_best": load_model("glm_best_pipeline"),
    "lgbm_best": load_model("lgbm_best_pipeline"),
}

# %%
# Evaluate each model on the test split and collect metrics
pd.set_option("display.float_format", lambda x: "%.4f" % x)
df_eval = pd.DataFrame({"Outcome": y_test})
metrics_tables: list[pd.DataFrame] = []

for label, model in pipelines.items():
    print(f"Evaluating {label}...")
    pred_col = f"p_{label}"
    df_eval[pred_col] = _predict_positive(model, X_test)

    metrics = evaluate_predictions(
        df_eval,
        outcome_column="Outcome",
        preds_column=pred_col,
    ).rename(columns={"value": label})
    metrics_tables.append(metrics)

summary = pd.concat(metrics_tables, axis=1)
print("Evaluation metrics by model:")
print(summary)

# %%
