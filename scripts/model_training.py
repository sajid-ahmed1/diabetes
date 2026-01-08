# %%
# importing libraries

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupShuffleSplit

from diabetes.data import read_data, save_model
from diabetes.modelling import glm_pipeline, glm_search, lgbm_pipeline, lgbm_search
from diabetes.preprocessing import clean_data

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"

# Ensure scikit-learn transformers output pandas DataFrames
# This prevents "X does not have valid feature names" warnings in LightGBM
sklearn.set_config(transform_output="pandas")
# %%
# load data
print("\n=== Loading Data ===")
df_model = read_data()
print(f"Data shape: {df_model.shape}")
print(df_model.head())

# %%
# Apply train/test split
print("\n=== Splitting Data (Group-Based) ===")
df_model = df_model.sort_values("Id")

# Use GroupShuffleSplit to ensure no ID leakage between train and test
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_inds, test_inds = next(splitter.split(df_model, groups=df_model["Id"]))

df_model["sample"] = "test"
df_model.loc[df_model.index[train_inds], "sample"] = "train"
df = df_model.copy()

# Split into temp train/test to learn stats safely
train_mask = df["sample"] == "train"
df_train_temp = df[train_mask].copy()
df_test_temp = df[~train_mask].copy()

# Clean Train (learns stats)
print("Cleaning training data...")
df_train_clean, stats = clean_data(df_train_temp)

# Clean Test (applies stats from train)
print("Cleaning test data (using training stats)...")
df_test_clean, _ = clean_data(df_test_temp, stats=stats)

# Combine back for saving
df = pd.concat([df_train_clean, df_test_clean]).sort_values("Id")

# Save the split dataframe for other scripts to ensure consistency
df.to_parquet(DATA_DIR / "data_split.parquet", index=False)
print(f"Train set size: {len(df_train_clean)}")
print(f"Test set size: {len(df_test_clean)}")

# %%
# separate train and test sets
train_idx = np.where(df["sample"] == "train")[0]
test_idx = np.where(df["sample"] == "test")[0]

df_train = df.iloc[train_idx].copy()
df_test = df.iloc[test_idx].copy()

# %%
# assign cateogrical and numerical columns
# session id taken out as an identifier not a feature to model with
# taken out os version due to high cardinality
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

# combing both categoricals and numericals together
predictors = categoricals + numericals

# spliting away features and the targets away
X_train = df_train[predictors]
y_train = df_train["Outcome"]
groups_train = df_train["Id"]
print("\n=== Feature Selection ===")
print(f"Features used ({len(predictors)}): {predictors}")
print("First 5 rows of training features:")
print(X_train.head())

# Check for potential data leakage
print("\nChecking feature correlations with target (to spot leakage)...")
correlations = X_train.corrwith(y_train).abs().sort_values(ascending=False)
print(correlations.head())

# %%
# getting pipelines made from modelling scripts
glm_pipe = glm_pipeline(numericals, categoricals)
lgbm_pipe = lgbm_pipeline(numericals, categoricals)

# %%
# establishing baseline models for GLM and LGBM
# pkl file captures the full pipeline (num and cat) and the trained model
print("\n=== Training Baseline Models ===")
glm_baseline = glm_pipe.fit(X_train, y_train)
save_model(glm_baseline, "glm_baseline")
print("GLM Baseline saved.")

lgbm_baseline = lgbm_pipe.fit(X_train, y_train)
save_model(lgbm_baseline, "lgbm_baseline")
print("LGBM Baseline saved.")
# %%
# tuning (CV) + fitted best estimators
# The model is trained when we call .fit(...) on the search object.
# Reduce warning noise during CV + smoke-tests
warnings.filterwarnings("ignore", category=FutureWarning)

# Even with transform_output="pandas", LightGBM can sometimes raise this warning
# during cross-validation (likely due to joblib parallel workers or internal wrapping).
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)

# ---- GLM tuned ----
print("\n=== Tuning GLM (Grid Search) ===")
glm_search_obj = glm_search(glm_pipe)
glm_search_obj.fit(X_train, y_train, groups=groups_train)
print(f"GLM Best CV Score: {glm_search_obj.best_score_:.4f}")
print(f"GLM Best Params: {glm_search_obj.best_params_}")
glm_best_pipeline = glm_search_obj.best_estimator_
save_model(glm_best_pipeline, "glm_best_pipeline")

# ---- LGBM tuned ----
print("\n=== Tuning LightGBM (Grid Search) ===")
lgbm_search_obj = lgbm_search(lgbm_pipe)
lgbm_search_obj.fit(X_train, y_train, groups=groups_train)
print(f"LGBM Best CV Score: {lgbm_search_obj.best_score_:.4f}")
print(f"LGBM Best Params: {lgbm_search_obj.best_params_}")
lgbm_best_pipeline = lgbm_search_obj.best_estimator_
save_model(lgbm_best_pipeline, "lgbm_best_pipeline", compress=3)

# %%
