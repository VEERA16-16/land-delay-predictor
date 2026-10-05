import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
ASSETS_DIR = BASE_DIR / "assets"

RAW_DATA_PATH = RAW_DATA_DIR / "raw_delayed_projects.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "land_acquisition_data.csv"
MODEL_PATH = MODELS_DIR / "delay_model.pkl"

CATEGORICAL_COLUMNS = ["state", "sector", "rural_urban"]

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
TARGET_COLUMN = "delayed"

RISK_THRESHOLDS = {
    "low": 0.40,
    "medium": 0.70,
}

RANDOM_STATE = 42
TEST_SIZE = 0.20