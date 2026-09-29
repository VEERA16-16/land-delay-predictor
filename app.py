import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent if APP_DIR.name == "app" else APP_DIR
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import MODEL_PATH, PROCESSED_DATA_PATH

st.set_page_config(
    page_title="LAND-AI | Land Acquisition Delay Intelligence",
    page_icon="⚠️",
    layout="wide",
)

CATEGORICAL_COLUMNS = ["state", "sector", "rural_urban"]
NUMERIC_COLUMNS = [
    "sanctioned_cost_cr", "land_required_ha", "land_acquired_pct",
    "num_landowners", "affected_families", "litigation_flag",
    "litigation_cases", "days_since_notification",
    "compensation_disbursed_pct", "award_completed_pct", "possession_pct",
    "rehabilitation_progress_pct", "approval_pending_days",
    "documentation_completion_pct", "stakeholder_response_score",
    "grievances_count", "previous_delay_rate", "data_quality_score",
]
DEFAULT_FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
RISK_LOW_THRESHOLD = 0.40
RISK_MEDIUM_THRESHOLD = 0.70

INDIAN_STATES_AND_UTS = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand",
    "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan",
    "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh",
    "Uttarakhand", "West Bengal", "Andaman and Nicobar Islands", "Chandigarh",
    "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Jammu and Kashmir",
    "Ladakh", "Lakshadweep", "Puducherry",
]
SECTORS = ["Highways", "Railways", "Irrigation", "Power", "Industrial", "Other"]
RURAL_URBAN_OPTIONS = ["Rural", "Urban", "Mixed"]


def get_risk_level(probability):
    if probability < RISK_LOW_THRESHOLD:
        return "Low"
    if probability < RISK_MEDIUM_THRESHOLD:
        return "Medium"
    return "High"


def risk_message(level):
    if level == "High":
        return "HIGH RISK — immediate intervention and monitoring are recommended."
    if level == "Medium":
        return "MEDIUM RISK — monitor closely and resolve major bottlenecks."
    return "LOW RISK — continue routine monitoring and milestone review."


@st.cache_resource
def load_artifact(model_path):
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


@st.cache_data
def load_demo_data(data_path):
    if not os.path.exists(data_path):
        return None
    return pd.read_csv(data_path)


def predict_probability(artifact, input_df):
    feature_columns = artifact.get("feature_columns", DEFAULT_FEATURE_COLUMNS)
    return float(
        artifact["pipeline"].predict_proba(input_df[feature_columns])[0, 1]
    )


def predict_portfolio(artifact, portfolio_df):
    feature_columns = artifact.get("feature_columns", DEFAULT_FEATURE_COLUMNS)
    scored = portfolio_df.copy()
    probabilities = artifact["pipeline"].predict_proba(
        scored[feature_columns]
    )[:, 1]
    scored["delay_probability"] = probabilities
    scored["delay_probability_pct"] = (probabilities * 100).round(1)
    scored["risk_level"] = scored["delay_probability"].apply(get_risk_level)
    return scored


def safe_number(row, column, default=0):
    value = row.get(column, default)
    if pd.isna(value):
        return default
    return float(value)


def scenario_definitions(row):
    scenarios = []

    if safe_number(row, "litigation_flag") == 1 or safe_number(row, "litigation_cases") > 0:
        changed = row.copy()
        changed["litigation_flag"] = 0
        changed["litigation_cases"] = 0
        scenarios.append((
            "Resolve active litigation",
            "Legal disputes can block acquisition, compensation, and possession activities.",
            changed,
        ))

    if safe_number(row, "compensation_disbursed_pct") < 95:
        changed = row.copy()
        changed["compensation_disbursed_pct"] = 100
        scenarios.append((
            "Accelerate compensation disbursement",
            "Timely compensation can reduce landowner resistance and acquisition bottlenecks.",
            changed,
        ))

    if safe_number(row, "approval_pending_days") > 0:
        changed = row.copy()
        changed["approval_pending_days"] = 0
        scenarios.append((
            "Clear pending administrative approvals",
            "Pending approvals can delay awards, possession, and field execution.",
            changed,
        ))

    if safe_number(row, "land_acquired_pct") < 95:
        changed = row.copy()
        changed["land_acquired_pct"] = min(100, safe_number(row, "land_acquired_pct") + 25)
        scenarios.append((
            "Accelerate land acquisition progress",
            "Low acquisition progress is a direct early-warning indicator of schedule risk.",
            changed,
        ))

    if safe_number(row, "documentation_completion_pct") < 95:
        changed = row.copy()
        changed["documentation_completion_pct"] = 100
        scenarios.append((
            "Complete pending documentation",
            "Incomplete records can delay approvals, awards, and possession transfer.",
            changed,
        ))

    if safe_number(row, "grievances_count") > 0 or safe_number(row, "stakeholder_response_score") < 60:
        changed = row.copy()
        changed["grievances_count"] = 0
        changed["stakeholder_response_score"] = max(
            90, safe_number(row, "stakeholder_response_score")
        )
        scenarios.append((
            "Resolve stakeholder grievances",
            "Reducing grievances improves stakeholder cooperation and field coordination.",
            changed,
        ))

    if safe_number(row, "rehabilitation_progress_pct") < 95:
        changed = row.copy()
        changed["rehabilitation_progress_pct"] = 100
        scenarios.append((
            "Accelerate rehabilitation and resettlement",
            "Incomplete rehabilitation can delay possession and create stakeholder resistance.",
            changed,
        ))

    return scenarios


def optimize_project(artifact, row, scored_probability=None):
    feature_columns = artifact.get("feature_columns", DEFAULT_FEATURE_COLUMNS)
    base_row = row.copy()
    base_risk = (
        float(scored_probability)
        if scored_probability is not None
        else float(predict_probability(artifact, pd.DataFrame([base_row])))
    )

    risk_level = get_risk_level(base_risk)
    if risk_level == "Low":
        return {
            "current_risk_pct": round(base_risk * 100, 1),
            "recommendation": "Continue routine monitoring",
            "reason": "The project is Low Risk. No immediate intervention is required.",
            "estimated_risk_after_action_pct": round(base_risk * 100, 1),
            "estimated_risk_reduction_pct": 0.0,
            "optimizer_table": pd.DataFrame(),
        }

    results = []
    for action, reason, changed_row in scenario_definitions(base_row):
        changed_df = pd.DataFrame([changed_row])
        new_risk = predict_probability(artifact, changed_df)
        results.append({
            "Intervention": action,
            "Reason": reason,
            "Current Risk (%)": round(base_risk * 100, 1),
            "Simulated Risk (%)": round(new_risk * 100, 1),
            "Risk Reduction (points)": round((base_risk - new_risk) * 100, 1),
        })

    if not results:
        return {
            "current_risk_pct": round(base_risk * 100, 1),
            "recommendation": "Conduct officer review",
            "reason": "Elevated risk exists, but no intervention scenario was triggered.",
            "estimated_risk_after_action_pct": round(base_risk * 100, 1),
            "estimated_risk_reduction_pct": 0.0,
            "optimizer_table": pd.DataFrame(),
        }

    optimizer_df = pd.DataFrame(results).sort_values(
        "Risk Reduction (points)", ascending=False
    ).reset_index(drop=True)
    best = optimizer_df.iloc[0]

    return {
        "current_risk_pct": round(base_risk * 100, 1),
        "recommendation": best["Intervention"],
        "reason": best["Reason"],
        "estimated_risk_after_action_pct": best["Simulated Risk (%)"],
        "estimated_risk_reduction_pct": best["Risk Reduction (points)"],
        "optimizer_table": optimizer_df,
    }


def add_portfolio_recommendations(artifact, scored_df):
    recommendation_rows = []
    for _, row in scored_df.iterrows():
        result = optimize_project(
            artifact,
            row.to_dict(),
            scored_probability=row["delay_probability"],
        )
        recommendation_rows.append({
            "ai_recommendation": result["recommendation"],
            "recommendation_reason": result["reason"],
            "estimated_risk_after_action_pct": result["estimated_risk_after_action_pct"],
            "estimated_risk_reduction_pct": result["estimated_risk_reduction_pct"],
        })
    return pd.concat(
        [scored_df.reset_index(drop=True), pd.DataFrame(recommendation_rows)],
        axis=1,
    )


def build_sample_portfolio():
    return pd.DataFrame([
        {
            "project_id": "LAND_001", "state": "Bihar", "sector": "Highways", "rural_urban": "Rural",
            "sanctioned_cost_cr": 250, "land_required_ha": 120, "land_acquired_pct": 35,
            "num_landowners": 350, "affected_families": 420, "litigation_flag": 1,
            "litigation_cases": 12, "days_since_notification": 420, "compensation_disbursed_pct": 30,
            "award_completed_pct": 40, "possession_pct": 25, "rehabilitation_progress_pct": 20,
            "approval_pending_days": 125, "documentation_completion_pct": 55,
            "stakeholder_response_score": 45, "grievances_count": 38,
            "previous_delay_rate": 0.55, "data_quality_score": 72,
        },
        {
            "project_id": "LAND_002", "state": "Tamil Nadu", "sector": "Irrigation", "rural_urban": "Rural",
            "sanctioned_cost_cr": 85, "land_required_ha": 45, "land_acquired_pct": 82,
            "num_landowners": 70, "affected_families": 88, "litigation_flag": 0,
            "litigation_cases": 0, "days_since_notification": 160, "compensation_disbursed_pct": 86,
            "award_completed_pct": 90, "possession_pct": 78, "rehabilitation_progress_pct": 75,
            "approval_pending_days": 22, "documentation_completion_pct": 95,
            "stakeholder_response_score": 88, "grievances_count": 3,
            "previous_delay_rate": 0.12, "data_quality_score": 94,
        },
        {
            "project_id": "LAND_003", "state": "Karnataka", "sector": "Power", "rural_urban": "Mixed",
            "sanctioned_cost_cr": 500, "land_required_ha": 200, "land_acquired_pct": 48,
            "num_landowners": 420, "affected_families": 510, "litigation_flag": 1,
            "litigation_cases": 8, "days_since_notification": 560, "compensation_disbursed_pct": 42,
            "award_completed_pct": 50, "possession_pct": 35, "rehabilitation_progress_pct": 40,
            "approval_pending_days": 105, "documentation_completion_pct": 65,
            "stakeholder_response_score": 55, "grievances_count": 29,
            "previous_delay_rate": 0.44, "data_quality_score": 79,
        },
    ])


artifact = load_artifact(MODEL_PATH)
if artifact is None:
    st.error("Model artifact not found. Run `python generate_dataset_v2.py` and `python train_v2.py`.")
    st.stop()

pipeline = artifact["pipeline"]
feature_columns = artifact.get("feature_columns", DEFAULT_FEATURE_COLUMNS)
selected_model = artifact.get("selected_model", "ML Pipeline")
model_metrics = artifact.get("metrics", [])
demo_data = load_demo_data(PROCESSED_DATA_PATH)

st.title("LAND-AI: Land Acquisition Delay Intelligence")
st.caption("SIH26017 · Predict → Explain → Simulate → Recommend")
st.info(
    "Prototype notice: this proof of concept uses synthetic training data. "
    "Recommendations are decision-support outputs and require human officer review."
)

st.subheader("Executive Dashboard")
if demo_data is not None and "delayed" in demo_data.columns:
    monitored_projects = len(demo_data)
    historical_delayed = int(demo_data["delayed"].sum())
    historical_rate = demo_data["delayed"].mean() * 100
else:
    monitored_projects, historical_delayed, historical_rate = 0, 0, 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Projects Monitored", f"{monitored_projects:,}")
c2.metric("Historical Delayed Cases", f"{historical_delayed:,}")
c3.metric("Historical Delay Rate", f"{historical_rate:.1f}%")
c4.metric("Selected Model", selected_model)

if model_metrics:
    st.subheader("Model Selection")
    metrics_df = pd.DataFrame(model_metrics)
    columns = [c for c in ["model", "accuracy", "precision", "recall", "f1", "roc_auc"] if c in metrics_df]
    st.dataframe(metrics_df[columns], use_container_width=True, hide_index=True)

with st.sidebar:
    st.header("Project Risk Input")
    with st.form("land_ai_prediction_form"):
        state = st.selectbox("State / Union Territory", INDIAN_STATES_AND_UTS)
        sector = st.selectbox("Project Sector", SECTORS)
        rural_urban = st.selectbox("Project Setting", RURAL_URBAN_OPTIONS)
        sanctioned_cost_cr = st.number_input("Sanctioned Cost (INR Crore)", min_value=1.0, value=250.0)
        land_required_ha = st.number_input("Land Required (Hectares)", min_value=1.0, value=120.0)
        num_landowners = st.number_input("Number of Landowners", min_value=1, value=350)
        affected_families = st.number_input("Affected Families", min_value=1, value=420)
        land_acquired_pct = st.slider("Land Acquired (%)", 0, 100, 35)
        compensation_disbursed_pct = st.slider("Compensation Disbursed (%)", 0, 100, 30)
        award_completed_pct = st.slider("Award Completion (%)", 0, 100, 40)
        possession_pct = st.slider("Possession Progress (%)", 0, 100, 25)
        rehabilitation_progress_pct = st.slider("Rehabilitation / R&R Progress (%)", 0, 100, 20)
        litigation_text = st.selectbox("Ongoing Litigation?", ["Yes", "No"])
        litigation_flag = int(litigation_text == "Yes")
        litigation_cases = st.number_input("Number of Litigation Cases", min_value=0, value=12)
        days_since_notification = st.number_input("Days Since Notification", min_value=0, value=420)
        approval_pending_days = st.number_input("Approval Pending Days", min_value=0, value=125)
        documentation_completion_pct = st.slider("Documentation Completion (%)", 0, 100, 55)
        stakeholder_response_score = st.slider("Stakeholder Response Score", 0, 100, 45)
        grievances_count = st.number_input("Number of Grievances", min_value=0, value=38)
        previous_delay_rate = st.slider("Historical Delay Rate", 0.0, 1.0, 0.55, 0.01)
        data_quality_score = st.slider("Data Quality Score", 0, 100, 72)
        predict_button = st.form_submit_button("Predict Delay Risk", use_container_width=True)

if predict_button:
    project_df = pd.DataFrame([{
        "state": state, "sector": sector, "rural_urban": rural_urban,
        "sanctioned_cost_cr": sanctioned_cost_cr, "land_required_ha": land_required_ha,
        "land_acquired_pct": land_acquired_pct, "num_landowners": num_landowners,
        "affected_families": affected_families, "litigation_flag": litigation_flag,
        "litigation_cases": litigation_cases, "days_since_notification": days_since_notification,
        "compensation_disbursed_pct": compensation_disbursed_pct,
        "award_completed_pct": award_completed_pct, "possession_pct": possession_pct,
        "rehabilitation_progress_pct": rehabilitation_progress_pct,
        "approval_pending_days": approval_pending_days,
        "documentation_completion_pct": documentation_completion_pct,
        "stakeholder_response_score": stakeholder_response_score,
        "grievances_count": grievances_count, "previous_delay_rate": previous_delay_rate,
        "data_quality_score": data_quality_score,
    }])
    try:
        probability = predict_probability(artifact, project_df)
        st.session_state["single_project_df"] = project_df
        st.session_state["single_probability"] = probability
    except Exception as error:
        st.error(f"Prediction failed: {error}")

if "single_project_df" in st.session_state:
    project_df = st.session_state["single_project_df"]
    probability = st.session_state["single_probability"]
    level = get_risk_level(probability)

    st.divider()
    st.subheader("AI Risk Assessment")
    c1, c2, c3 = st.columns(3)
    c1.metric("Delay Probability", f"{probability * 100:.1f}%")
    c2.metric("Risk Level", level)
    c3.metric("Model", selected_model)
    if level == "High":
        st.error(risk_message(level))
    elif level == "Medium":
        st.warning(risk_message(level))
    else:
        st.success(risk_message(level))

    st.subheader("AI Recommended Actions")
    recommendation = optimize_project(artifact, project_df.iloc[0].to_dict(), probability)
    st.info(
        f"**Primary recommendation:** {recommendation['recommendation']}\n\n"
        f"**Reason:** {recommendation['reason']}\n\n"
        f"**Estimated risk after action:** {recommendation['estimated_risk_after_action_pct']:.1f}%\n\n"
        f"**Estimated reduction:** {recommendation['estimated_risk_reduction_pct']:.1f} percentage points"
    )

# Portfolio upload
st.divider()
st.subheader("Portfolio Risk Analysis")
st.write("Upload a CSV to predict risk and generate an AI recommendation for every project.")

sample = build_sample_portfolio()
st.download_button(
    "Download Sample Portfolio CSV",
    sample.to_csv(index=False).encode("utf-8"),
    "land_ai_sample_portfolio.csv",
    "text/csv",
)

uploaded_file = st.file_uploader("Upload Portfolio CSV", type=["csv"])
if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
        missing = [column for column in feature_columns if column not in uploaded_df.columns]
        if missing:
            st.error("Your CSV is missing required fields: " + ", ".join(missing))
            st.code(",".join(feature_columns))
            st.stop()

        portfolio_df = add_portfolio_recommendations(
            artifact,
            predict_portfolio(artifact, uploaded_df),
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Projects Analysed", len(portfolio_df))
        c2.metric("High-Risk Projects", int((portfolio_df["risk_level"] == "High").sum()))
        c3.metric("Average Portfolio Risk", f"{portfolio_df['delay_probability'].mean() * 100:.1f}%")

        st.subheader("Portfolio Priority Table")
        display_columns = [
            "project_id", "state", "sector", "land_acquired_pct",
            "compensation_disbursed_pct", "litigation_cases",
            "delay_probability_pct", "risk_level", "ai_recommendation",
            "estimated_risk_reduction_pct",
        ]
        display_columns = [c for c in display_columns if c in portfolio_df.columns]
        st.dataframe(
            portfolio_df[display_columns].sort_values("delay_probability_pct", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("AI Recommendations for Uploaded Portfolio")
        st.caption(
            "Each recommendation is generated by testing eligible interventions through the trained model. "
            "Low-risk projects receive a monitoring recommendation instead of an unnecessary intervention."
        )
        for _, row in portfolio_df.sort_values("delay_probability_pct", ascending=False).iterrows():
            title = f"{row['project_id']} — {row['risk_level']} Risk ({row['delay_probability_pct']:.1f}%)"
            message = (
                f"**AI recommendation:** {row['ai_recommendation']}\n\n"
                f"**Why:** {row['recommendation_reason']}\n\n"
                f"**Estimated risk after action:** {row['estimated_risk_after_action_pct']:.1f}%  \n"
                f"**Estimated risk reduction:** {row['estimated_risk_reduction_pct']:.1f} percentage points"
            )
            if row["risk_level"] == "High":
                st.error(f"**{title}**\n\n{message}")
            elif row["risk_level"] == "Medium":
                st.warning(f"**{title}**\n\n{message}")
            else:
                st.success(f"**{title}**\n\n{message}")

        st.subheader("Intervention Ranking by Project")
        selected_project = st.selectbox(
            "Select a project to see all tested interventions",
            portfolio_df["project_id"].astype(str).tolist(),
        )
        selected_row = portfolio_df.loc[
            portfolio_df["project_id"].astype(str) == str(selected_project)
        ].iloc[0]
        selected_result = optimize_project(
            artifact,
            selected_row.to_dict(),
            selected_row["delay_probability"],
        )
        if selected_result["optimizer_table"].empty:
            st.success("Low risk: no intervention ranking is necessary; continue routine monitoring.")
        else:
            st.dataframe(
                selected_result["optimizer_table"],
                use_container_width=True,
                hide_index=True,
            )
            st.success(
                f"Priority action: {selected_result['recommendation']} — "
                f"estimated reduction of {selected_result['estimated_risk_reduction_pct']:.1f} percentage points."
            )

        risk_chart_df = (
            portfolio_df["risk_level"]
            .value_counts()
            .reindex(["Low", "Medium", "High"], fill_value=0)
            .rename_axis("Risk Level")
            .reset_index(name="Projects")
        )
        chart = px.bar(
            risk_chart_df,
            x="Risk Level",
            y="Projects",
            color="Risk Level",
            color_discrete_map={"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"},
            title="Portfolio Risk Distribution",
        )
        st.plotly_chart(chart, use_container_width=True)

        st.download_button(
            "Download Portfolio Prediction Results",
            portfolio_df.to_csv(index=False).encode("utf-8"),
            "land_ai_portfolio_predictions_with_recommendations.csv",
            "text/csv",
        )

        st.info(
            "Responsible AI note: recommendations are model-based simulations for officer review. "
            "They do not replace legal, financial, or administrative decisions."
        )

    except Exception as error:
        st.error(f"Could not process the portfolio CSV: {error}")

st.divider()
st.caption("LAND-AI prototype | Predict → Explain → Simulate → Recommend | Synthetic-data proof of concept")