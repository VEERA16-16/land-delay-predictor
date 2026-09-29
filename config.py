import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, "raw_delayed_projects.csv")
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "processed_dataset.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "delay_model.pkl")

FEATURE_COLUMNS = [
    "state",
    "sector",
    "sanctioned_cost_cr",
    "land_required_ha",
    "land_acquired_pct",
    "num_landowners",
    "litigation_flag",
    "days_since_notification",
    "compensation_disbursed_pct",
]

CATEGORICAL_COLUMNS = ["state", "sector"]
NUMERIC_COLUMNS = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]

TARGET_COLUMN = "delayed"

RISK_THRESHOLDS = {
    "low": 0.4,
    "medium": 0.7,
}

RANDOM_STATE = 42
TEST_SIZE = 0.2