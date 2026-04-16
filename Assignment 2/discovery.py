import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from lightgbm import LGBMClassifier
from pycaret.classification import compare_models, plot_model, save_model, setup


"""
SYNTHESIS

For this project, the low-code PyCaret workflow was more efficient for model discovery,
because it automated several repetitive steps that normally take time in a manual
scikit-learn process. With only a few lines of code, PyCaret handled the experiment setup,
preprocessing configuration, cross-validation, and model comparison across many algorithms.
That makes it especially useful in an early-stage business setting where a data engineer wants
to quickly identify strong candidate models before committing to deeper tuning or deployment.
The confusion matrix and saved pipeline were also easy to generate.

The manual scikit-learn workflow required more effort, but it offered better transparency and
control. I had to explicitly define the train-test split, identify categorical versus numeric
columns, create preprocessing pipelines, select the estimator, and generate the final
classification report. This takes longer, but it also makes each design choice visible and easier
to explain. That is valuable in production environments where maintainability and reproducibility
matter.

The results may differ slightly between PyCaret and scikit-learn even when the underlying model
is similar. Small differences can come from random train-test splits, cross-validation behavior,
default hyperparameters, preprocessing details, missing-value handling, or thresholding choices.
So even if both approaches point to the same best model family, their exact metrics may not be
identical.
"""


DATA_PATH = Path("adult.data")
MODEL_SAVE_NAME = "best_pipeline"

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS, skipinitialspace=True)

    # Replace missing markers with numpy.nan, not pandas.NA
    df = df.replace("?", np.nan)

    # Strip spaces from object columns only
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    return df


def get_manual_model(best_model):
    model_type = type(best_model).__name__
    print(f"\nBest PyCaret model type: {model_type}")

    model_map = {
        "LGBMClassifier": LGBMClassifier(random_state=42),
        "RandomForestClassifier": RandomForestClassifier(random_state=42),
        "ExtraTreesClassifier": ExtraTreesClassifier(random_state=42),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42),
    }

    if model_type not in model_map:
        raise ValueError(
            f"No manual sklearn mapping written yet for {model_type}. "
            "Run once, check the printed model type, then add it to model_map."
        )

    return model_map[model_type]


if __name__ == "__main__":
    df = load_data()
    print("Dataset shape:", df.shape)
    print("\nTarget distribution:")
    print(df["income"].value_counts(dropna=False))

    # -------------------------
    # 1) PYCARET WORKFLOW
    # -------------------------
    setup(
        data=df,
        target="income",
        session_id=42,
        train_size=0.8,
        normalize=True,
        fold=5,
        verbose=False,
        html=False,
    )

    top3 = compare_models(n_select=3)

    print("\nTop 3 PyCaret models:")
    for i, model in enumerate(top3, start=1):
        print(f"{i}. {type(model).__name__}")

    best_model = top3[0]

    # Save confusion matrix
    plot_model(best_model, plot="confusion_matrix", save=True)

    # Save PyCaret model pipeline
    save_model(best_model, MODEL_SAVE_NAME)

    # Save feature names for API reference
    feature_columns = [col for col in df.columns if col != "income"]
    with open("feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, indent=2)

    # -------------------------
    # 2) SCIKIT-LEARN WORKFLOW
    # -------------------------
    X = df.drop(columns=["income"])
    y = df["income"]

    categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    manual_estimator = get_manual_model(best_model)

    manual_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", manual_estimator),
        ]
    )

    manual_pipeline.fit(X_train, y_train)
    y_pred = manual_pipeline.predict(X_test)

    print("\nScikit-learn classification report:")
    print(classification_report(y_test, y_pred))