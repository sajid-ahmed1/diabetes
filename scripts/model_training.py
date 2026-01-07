# %%
# importing libraries

import warnings
from pathlib import Path

import joblib
import numpy as np

from diabetes.data import create_sample_split, load_parquet, save_model
from diabetes.modelling import glm_pipeline, glm_search, lgbm_pipeline, lgbm_search

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
# %%
# load data
df_model = load_parquet().copy()
df_model.head()

# %%
# Apply train/test split
df = create_sample_split(df_model, id_column="Id")

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

# %%
# getting pipelines made from modelling scripts
glm_pipe = glm_pipeline(numericals, categoricals)
lgbm_pipe = lgbm_pipeline(numericals, categoricals)

# %%
# establishing baseline models for GLM and LGBM
# pkl file captures the full pipeline (num and cat) and the trained model
glm_baseline = glm_pipe.fit(X_train, y_train)
save_model(glm_baseline, "glm_baseline")

lgbm_baseline = lgbm_pipe.fit(X_train, y_train)
save_model(lgbm_baseline, "lgbm_baseline")
# %%
# tuning (CV) + fitted best estimators
# The model is trained when we call .fit(...) on the search object.
# Reduce warning noise during CV + smoke-tests
warnings.filterwarnings("ignore", category=FutureWarning)

# This warning is raised when an estimator was fitted with feature names
# (e.g., pandas columns) but later receives a NumPy array without names.
# It does not affect predictions here; suppress to keep logs clean.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)

# ---- GLM tuned ----
glm_search_obj = glm_search(glm_pipe)
glm_search_obj.fit(X_train, y_train)
glm_best_pipeline = glm_search_obj.best_estimator_
save_model(glm_best_pipeline, "glm_best_pipeline")

# ---- LGBM tuned ----
lgbm_search_obj = lgbm_search(lgbm_pipe)
lgbm_search_obj.fit(X_train, y_train)
lgbm_best_pipeline = lgbm_search_obj.best_estimator_
model_path = MODELS_DIR / "lgbm_best_pipeline.pkl"
joblib.dump(
    lgbm_best_pipeline, model_path, compress=3
)  # could not save using save_model function


# %%
