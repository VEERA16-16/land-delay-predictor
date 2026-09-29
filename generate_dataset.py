"""Generate a realistic synthetic training dataset for SIH26017.

IMPORTANT: This is synthetic prototype data, not government data.
The generator creates only information that could reasonably be known at the
prediction point; final delay_days is retained only as an outcome/reference
field and is NOT used as a model feature.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from config import PROCESSED_DATA_PATH, PROCESSED_DATA_DIR
except ImportError:
    PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
    PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "land_acquisition_data.csv"

SEED = 42
N = 12000
rng = np.random.default_rng(SEED)

STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu",
    "Delhi", "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry"
]

SECTORS = ["Highways", "Railways", "Irrigation", "Power", "Industrial", "Other"]
RURAL_URBAN = ["Rural", "Urban", "Mixed"]

# Mild geographic effect only; do not let state dominate the target.
STATE_EFFECT = {s: float(rng.normal(0, 0.06)) for s in STATES}
SECTOR_EFFECT = {
    "Highways": 0.03, "Railways": 0.05, "Irrigation": 0.01,
    "Power": 0.04, "Industrial": -0.01, "Other": 0.00,
}

rows = []
for i in range(N):
    state = str(rng.choice(STATES))
    sector = str(rng.choice(SECTORS, p=[0.25, 0.16, 0.13, 0.14, 0.12, 0.20]))
    rural_urban = str(rng.choice(RURAL_URBAN, p=[0.56, 0.24, 0.20]))

    sanctioned_cost_cr = float(np.clip(rng.lognormal(mean=4.7, sigma=0.85), 10, 2500))
    land_required_ha = float(np.clip(rng.lognormal(mean=3.4, sigma=0.9), 2, 2500))
    num_landowners = int(np.clip(rng.lognormal(mean=4.3, sigma=0.85), 5, 2500))
    affected_families = int(np.clip(num_landowners * rng.uniform(0.65, 1.45), 3, 3500))

    days_since_notification = int(rng.integers(30, 1001))

    # Progress tends to increase with time, but with substantial variation.
    expected_acq = 15 + 0.065 * days_since_notification
    land_acquired_pct = float(np.clip(rng.normal(expected_acq, 18), 1, 100))
    compensation_disbursed_pct = float(np.clip(
        rng.normal(land_acquired_pct * rng.uniform(0.72, 0.98), 13), 0, 100
    ))
    award_completed_pct = float(np.clip(
        rng.normal(land_acquired_pct * rng.uniform(0.82, 1.02), 10), 0, 100
    ))
    possession_pct = float(np.clip(
        rng.normal(min(land_acquired_pct, award_completed_pct) * rng.uniform(0.70, 0.96), 12), 0, 100
    ))
    rehabilitation_progress_pct = float(np.clip(
        rng.normal(compensation_disbursed_pct * rng.uniform(0.60, 0.95), 15), 0, 100
    ))

    litigation_flag = int(rng.random() < np.clip(0.10 + 0.00020 * num_landowners + 0.05 * (days_since_notification > 365), 0.05, 0.55))
    litigation_cases = int(rng.poisson(max(0.2, litigation_flag * (1.5 + 0.012 * num_landowners))))
    grievances_count = int(rng.poisson(max(0.2, affected_families * 0.015 + litigation_cases * 0.4)))
    approval_pending_days = int(np.clip(rng.gamma(shape=2.0, scale=18.0) + (25 if litigation_flag else 0), 0, 365))
    documentation_completion_pct = float(np.clip(
        rng.normal(88 - 0.00010 * affected_families - 0.05 * approval_pending_days, 9), 35, 100
    ))
    stakeholder_response_score = float(np.clip(
        rng.normal(75 - 0.35 * grievances_count - 5 * litigation_flag, 12), 10, 100
    ))
    previous_delay_rate = float(np.clip(
        rng.beta(2.2, 5.0) + 0.08 * litigation_flag + 0.04 * (sector == "Power"), 0, 0.95
    ))

    # Data completeness proxy for operational quality; not an outcome.
    data_quality_score = float(np.clip(rng.normal(88 - 0.08 * grievances_count, 8), 45, 100))

    # Nonlinear latent risk with noise. The outcome is generated from the
    # current-stage factors only, not from future delay_days.
    progress_gap = 1 - land_acquired_pct / 100
    comp_gap = 1 - compensation_disbursed_pct / 100
    approval_gap = min(approval_pending_days / 180, 1)
    docs_gap = 1 - documentation_completion_pct / 100
    stakeholder_gap = 1 - stakeholder_response_score / 100
    owner_pressure = min(num_landowners / 700, 1)
    time_pressure = min(days_since_notification / 900, 1)
    litigation_pressure = min(litigation_cases / 12, 1)
    grievance_pressure = min(grievances_count / 80, 1)
    historical_pressure = previous_delay_rate

    risk_logit = (
        -2.35
        + 1.55 * progress_gap
        + 1.10 * comp_gap
        + 0.95 * approval_gap
        + 0.70 * docs_gap
        + 0.65 * stakeholder_gap
        + 0.75 * owner_pressure
        + 0.55 * time_pressure
        + 0.95 * litigation_flag
        + 0.65 * litigation_pressure
        + 0.45 * grievance_pressure
        + 0.85 * historical_pressure
        + STATE_EFFECT[state]
        + SECTOR_EFFECT[sector]
        + rng.normal(0, 0.65)
    )
    delay_probability = 1 / (1 + np.exp(-risk_logit))
    delayed = int(rng.random() < delay_probability)

    # Outcome/reference field for analytics only. DO NOT feed it to the model.
    delay_days = int(np.clip(
        rng.normal(30 + 360 * delay_probability, 45),
        0, 750
    )) if delayed else int(rng.uniform(0, 75))

    rows.append({
        "project_id": f"PRJ{i+1:05d}",
        "state": state,
        "sector": sector,
        "rural_urban": rural_urban,
        "sanctioned_cost_cr": round(sanctioned_cost_cr, 2),
        "land_required_ha": round(land_required_ha, 2),
        "land_acquired_pct": round(land_acquired_pct, 1),
        "num_landowners": num_landowners,
        "affected_families": affected_families,
        "litigation_flag": litigation_flag,
        "litigation_cases": litigation_cases,
        "days_since_notification": days_since_notification,
        "compensation_disbursed_pct": round(compensation_disbursed_pct, 1),
        "award_completed_pct": round(award_completed_pct, 1),
        "possession_pct": round(possession_pct, 1),
        "rehabilitation_progress_pct": round(rehabilitation_progress_pct, 1),
        "approval_pending_days": approval_pending_days,
        "documentation_completion_pct": round(documentation_completion_pct, 1),
        "stakeholder_response_score": round(stakeholder_response_score, 1),
        "grievances_count": grievances_count,
        "previous_delay_rate": round(previous_delay_rate, 3),
        "data_quality_score": round(data_quality_score, 1),
        "delayed": delayed,
        "delay_days": delay_days,
    })

df = pd.DataFrame(rows)
PROCESSED_DATA_DIR = Path(PROCESSED_DATA_DIR)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(PROCESSED_DATA_PATH, index=False)

print(f"Saved {len(df):,} rows to {PROCESSED_DATA_PATH}")
print("Delayed rate:", f"{df['delayed'].mean():.1%}")
print("States:", df['state'].nunique())
print("Sectors:", df['sector'].nunique())
print("Columns:", len(df.columns))
print("Target distribution:")
print(df['delayed'].value_counts())
