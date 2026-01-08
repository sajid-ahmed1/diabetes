import os

# Fix for segmentation fault on macOS with LightGBM/OpenMP
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path

import dalex as dx
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from diabetes.data import load_model, save_image
from diabetes.evaluation import evaluate_predictions
from diabetes.visualisation import lorenz_curve, plot_predicted_vs_actual

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"

# %%
# load data
# Load split data directly to ensure consistency with training
df = pd.read_parquet(DATA_DIR / "data_split.parquet")

train_idx = np.where(df["sample"] == "train")[0]
test_idx = np.where(df["sample"] == "test")[0]

df_train = df.iloc[train_idx].copy()
df_test = df.iloc[test_idx].copy()

# %%
# assign categorical and numerical columns (mirrors slow_sessions.py)
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
# load best pipelines trained in scripts/model_training.py
best_glm = load_model("glm_best_pipeline")
best_lgbm = joblib.load(MODELS_DIR / "lgbm_best_pipeline.pkl")

# %%
# predictions from tuned models
df_test = df_test.copy()
df_test["p_glm"] = best_glm.predict(X_test)
df_test["p_lgbm"] = best_lgbm.predict_proba(X_test)[:, 1]

pd.set_option("display.float_format", lambda x: "%.3f" % x)
print("GLM Evaluation:")
print(
    evaluate_predictions(
        df_test,
        outcome_column="Outcome",
        preds_column="p_glm",
    )
)
print("\nLGBM Evaluation:")
print(
    evaluate_predictions(
        df_test,
        outcome_column="Outcome",
        preds_column="p_lgbm",
    )
)

# %%
# Lorenz curves
fig_lorenz = lorenz_curve(
    df_test,
    outcome_column="Outcome",
    preds_columns={
        "GLM": "p_glm",
        "LGBM": "p_lgbm",
    },
    title="Lorenz Curve - GLM vs LGBM",
    include_oracle=True,
)
save_image(fig_lorenz, "lorenz_curve_glm_vs_lgbm_best.png")
plt.close(fig_lorenz)

# %%
# plotting predicted vs actual for GLM and LGBM
fig_glm = plot_predicted_vs_actual(
    df_test,
    pred_col="p_glm",
    target_col="Outcome",
    title="GLM: Predicted vs Actual",
)
save_image(fig_glm, "glm_pred_vs_actual.png")
plt.close(fig_glm)

fig_lgbm = plot_predicted_vs_actual(
    df_test,
    pred_col="p_lgbm",
    target_col="Outcome",
    title="LGBM: Predicted vs Actual",
)
save_image(fig_lgbm, "lgbm_pred_vs_actual.png")
plt.close(fig_lgbm)

# %%
# plotting the top 5 most important features that GLM found
model_glm = best_glm.named_steps["model"]

top5_glm = (
    pd.Series(np.abs(model_glm.coef_), index=model_glm.feature_names_)
    .sort_values(ascending=False)
    .head(5)
)

print(top5_glm)
top5_glm_plot = top5_glm.plot(kind="bar", title="Top 5 GLM (abs|coef|)")
top5_glm_plot.set_ylabel("|Coefficient|")
save_image(top5_glm_plot, "top5_glm_features.png")
plt.close(top5_glm_plot.figure)  # type: ignore[arg-type]

# %%
# plotting the top 5 most important features that LGBM found
feature_names = best_lgbm.named_steps["preprocess"].get_feature_names_out()

importances = best_lgbm.named_steps["model"].booster_.feature_importance(
    importance_type="gain"
)

top5_lgbm = (
    pd.Series(importances, index=feature_names).sort_values(ascending=False).head(5)
)

print(top5_lgbm)
top5_lgbm_plot = top5_lgbm.plot(kind="bar", title="Top 5 LGBM (gain)")
top5_lgbm_plot.set_ylabel("Feature importance (gain)")
save_image(top5_lgbm_plot, "top5_lgbm_features.png")
plt.close(top5_lgbm_plot.figure)  # type: ignore[arg-type]

# %%
# model-agnostic permutation importance using tuned LGBM
perm = permutation_importance(
    best_lgbm,
    X_test,
    y_test,
    scoring="neg_log_loss",
    n_repeats=10,
    random_state=42,
)

top5_perm = (
    pd.Series(perm.importances_mean, index=X_test.columns)
    .sort_values(ascending=False)
    .head(5)
)

print(top5_perm)
top5_perm_plot = top5_perm.plot(kind="bar", title="Top 5 permutation importance")
top5_perm_plot.set_ylabel("Mean importance (permutation)")
save_image(top5_perm_plot, "top5_perm_features.png")
plt.close(top5_perm_plot.figure)  # type: ignore[arg-type]

# %%
# plotting PDP plot using Dalex for the LGBM model
num_top5 = (
    pd.Series(perm.importances_mean, index=X_test.columns)
    .loc[numericals]
    .sort_values(ascending=False)
    .head(5)
)

top5_pdp_features = num_top5.index.tolist()
exp = dx.Explainer(
    best_lgbm,
    X_test,
    y_test,
    predict_function=lambda m, X: m.predict_proba(X)[:, 1],
    verbose=False,
)

pdp = exp.model_profile(
    variables=top5_pdp_features,
    type="partial",
)

pdp.plot()
save_image(None, "pdp_lgbm.png")

# %%
# SHAP beeswarm for tuned LGBM pipeline (handles categoricals)
X_shap = X_test.sample(n=min(len(X_test), 1000), random_state=42)
X_bg = X_train.sample(n=min(len(X_train), 200), random_state=42)

preprocess = best_lgbm.named_steps["preprocess"]
model = best_lgbm.named_steps["model"]

X_shap_trans = np.asarray(preprocess.transform(X_shap))
X_bg_trans = np.asarray(preprocess.transform(X_bg))

try:
    shap_feature_names = preprocess.get_feature_names_out()
except Exception:
    shap_feature_names = [f"feature_{i}" for i in range(X_shap_trans.shape[1])]

explainer = shap.TreeExplainer(model, data=X_bg_trans)

shap_values = explainer.shap_values(X_shap_trans)
if isinstance(shap_values, list) and len(shap_values) == 2:
    shap_values = shap_values[1]

shap.summary_plot(
    shap_values,
    X_shap_trans,
    feature_names=shap_feature_names,
    max_display=20,
    show=False,
)
save_image(None, "shap_beeswarm_lgbm.png")
plt.close()

# %%
