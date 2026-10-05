from pathlib import Path
import sys
import warnings
from xgboost import XGBClassifier


warnings.filterwarnings("ignore")



PROJECT_DIR = Path(__file__).resolve().parent.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    PROCESSED_DATA_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)

# MODEL FEATURES

TARGET = "delayed"

# Never include these during training.
LEAKAGE_COLUMNS = {
    "project_id",
    "delayed",
    "delay_days",
}

CATEGORICAL_COLUMNS = [
    "state",
    "sector",
    "rural_urban",
]

NUMERIC_COLUMNS = [
    "sanctioned_cost_cr",
    "land_required_ha",
    "land_acquired_pct",
    "num_landowners",
    "affected_families",
    "litigation_flag",
    "litigation_cases",
    "days_since_notification",
    "compensation_disbursed_pct",
    "award_completed_pct",
    "possession_pct",
    "rehabilitation_progress_pct",
    "approval_pending_days",
    "documentation_completion_pct",
    "stakeholder_response_score",
    "grievances_count",
    "previous_delay_rate",
    "data_quality_score",
]

FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


# PREPROCESSING PIPELINE

def make_preprocessor():
    """
    Build preprocessing that is saved inside the final model pipeline.

    - Categorical fields: fill missing values and one-hot encode.
    - Numeric fields: fill missing values and standardize.
    - handle_unknown='ignore' prevents errors for a new State/UT or category.
    """
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_COLUMNS,
            ),
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_COLUMNS,
            ),
        ],
        remainder="drop",
    )


# TRAIN AND EVALUATE

def evaluate_model(name, pipeline, x_train, x_test, y_train, y_test):
    """Fit one pipeline and return both metrics and fitted pipeline."""

    print(f"\n{'=' * 60}")
    print(f"Training {name}")
    print(f"{'=' * 60}")

    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            digits=3,
            zero_division=0,
        )
    )

    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    return metrics, pipeline


def main():
    """Load data, train model candidates, select best pipeline, and save it."""

    data_path = Path(PROCESSED_DATA_PATH)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {data_path}\n"
            "Run `python generate_dataset.py` first."
        )

    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    required_columns = FEATURE_COLUMNS + [TARGET]
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset is missing required column(s): "
            + ", ".join(missing_columns)
        )

    x = df[FEATURE_COLUMNS].copy()
    y = df[TARGET].astype(int)

    print(f"\nDataset rows: {len(df):,}")
    print(f"Model features: {len(FEATURE_COLUMNS)}")
    print(f"Delayed-project rate: {y.mean():.1%}")
    print(f"Leakage columns excluded: {sorted(LEAKAGE_COLUMNS)}")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training rows: {len(x_train):,}")
    print(f"Testing rows: {len(x_test):,}")

    model_candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost Classifier": XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    
        "Histogram Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.06,
            max_leaf_nodes=24,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
    }

    results = []
    fitted_models = {}

    for model_name, estimator in model_candidates.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
                ("model", estimator),
            ]
        )

        metrics, trained_pipeline = evaluate_model(
            model_name,
            pipeline,
            x_train,
            x_test,
            y_train,
            y_test,
        )

        results.append(metrics)
        fitted_models[model_name] = trained_pipeline

    results_df = pd.DataFrame(results)

    # For an early-warning system:
    # 1. Recall is most important — avoid missing potentially delayed projects.
    # 2. F1 balances precision and recall.
    # 3. ROC-AUC measures ranking/discrimination quality.
    results_df = results_df.sort_values(
        by=["recall", "f1", "roc_auc"],
        ascending=False,
    ).reset_index(drop=True)

    best_model_name = results_df.iloc[0]["model"]
    best_pipeline = fitted_models[best_model_name]

    print(f"\n{'=' * 60}")
    print("MODEL COMPARISON")
    print(f"{'=' * 60}")
    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    print(f"\nSelected Model: {best_model_name}")

    model_path = Path(MODEL_PATH)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "pipeline": best_pipeline,
        "model": best_pipeline.named_steps["model"],
        "feature_columns": FEATURE_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
        "target_column": TARGET,
        "selected_model": best_model_name,
        "metrics": results_df.to_dict(orient="records"),
        "risk_note": (
            "Prototype trained on synthetic data; it is not validated "
            "for operational government decision-making."
        ),
    }

    joblib.dump(artifact, model_path)

    reports_dir = PROJECT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_path = reports_dir / "model_comparison.csv"
    results_df.to_csv(report_path, index=False)

    print(f"\nSaved model artifact: {model_path}")
    print(f"Saved model comparison report: {report_path}")

    print("\nModel artifact keys:")
    print(list(artifact.keys()))

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()