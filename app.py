import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
import shap
import matplotlib.pyplot as plt

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent if APP_DIR.name == "app" else APP_DIR
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import MODEL_PATH, PROCESSED_DATA_PATH

# 1. PAGE CONFIG MUST BE THE ABSOLUTE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="LAND-AI | Land Acquisition Intelligence",
    page_icon="🏢",
    layout="wide",
)

def apply_enterprise_theme():
    custom_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        * { font-family: 'Inter', sans-serif; }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .block-container { padding-top: 2rem !important; padding-bottom: 3rem !important; max-width: 1280px; }
        
        .portal-nav {
            display: flex; align-items: center; justify-content: space-between;
            padding: 1rem 1.5rem; background-color: #ffffff;
            border: 1px solid #E2E8F0; border-radius: 12px; margin-bottom: 2rem;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        }
        .portal-brand { font-size: 1.25rem; font-weight: 700; color: #0F172A; display: flex; align-items: center; gap: 0.5rem; }
        .portal-badge { background-color: #EFF6FF; color: #1D4ED8; font-size: 0.75rem; font-weight: 600; padding: 0.25rem 0.65rem; border-radius: 9999px; border: 1px solid #DBEAFE; }
        
        div[data-testid="stMetric"] { background-color: #ffffff; border: 1px solid #E2E8F0; padding: 1.25rem; border-radius: 12px; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.04); }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# 2. APPLY UI THEME
apply_enterprise_theme()

st.markdown("""
<div class="portal-nav">
    <div class="portal-brand">
        <span>🏢</span> BHOOMIGUARD AI Platform
        <span class="portal-badge">SIH26017 Engine</span>
    </div>
    <div style="font-size: 0.85rem; color: #64748B; font-weight: 500;">
        Model Stack: <b>XGBoost + TreeSHAP</b>
    </div>
</div>
""", unsafe_allow_html=True)

CATEGORICAL_COLUMNS = ["state", "sector", "rural_urban"]
NUMERIC_COLUMNS = [
    "sanctioned_cost_cr", "land_required_ha", "land_acquired_pct", "num_landowners", "affected_families", 
    "litigation_flag", "litigation_cases", "days_since_notification", "compensation_disbursed_pct", 
    "award_completed_pct", "possession_pct", "rehabilitation_progress_pct", "approval_pending_days",
    "documentation_completion_pct", "stakeholder_response_score", "grievances_count", "previous_delay_rate", 
    "data_quality_score"
]
DEFAULT_FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
RISK_LOW_THRESHOLD = 0.40
RISK_MEDIUM_THRESHOLD = 0.70

INDIAN_STATES_AND_UTS = ["Andhra Pradesh", "Bihar", "Gujarat", "Karnataka", "Maharashtra", "Tamil Nadu", "Uttar Pradesh", "Other"]
SECTORS = ["Highways", "Railways", "Irrigation", "Power", "Industrial", "Other"]
RURAL_URBAN_OPTIONS = ["Rural", "Urban", "Mixed"]

def get_risk_level(probability):
    if probability < RISK_LOW_THRESHOLD: return "Low"
    if probability < RISK_MEDIUM_THRESHOLD: return "Medium"
    return "High"

@st.cache_resource
def load_artifact(model_path):
    if not os.path.exists(model_path): return None
    return joblib.load(model_path)

@st.cache_data
def load_demo_data(data_path):
    if not os.path.exists(data_path): return None
    return pd.read_csv(data_path)

def predict_probability(artifact, input_df):
    feature_columns = artifact.get("feature_columns", DEFAULT_FEATURE_COLUMNS)
    return float(artifact["pipeline"].predict_proba(input_df[feature_columns])[0, 1])

def predict_portfolio(artifact, portfolio_df):
    feature_columns = artifact.get("feature_columns", DEFAULT_FEATURE_COLUMNS)
    scored = portfolio_df.copy()
    probabilities = artifact["pipeline"].predict_proba(scored[feature_columns])[:, 1]
    scored["delay_probability"] = probabilities
    scored["delay_probability_pct"] = (probabilities * 100).round(1)
    scored["risk_level"] = scored["delay_probability"].apply(get_risk_level)
    return scored

def safe_number(row, column, default=0):
    value = row.get(column, default)
    if pd.isna(value): return default
    return float(value)

def scenario_definitions(row):
    scenarios = []
    if safe_number(row, "litigation_flag") == 1 or safe_number(row, "litigation_cases") > 0:
        changed = row.copy()
        changed["litigation_flag"] = 0
        changed["litigation_cases"] = 0
        scenarios.append(("Resolve active litigation", "Legal disputes block acquisition activities.", changed))
    if safe_number(row, "compensation_disbursed_pct") < 95:
        changed = row.copy()
        changed["compensation_disbursed_pct"] = 100
        scenarios.append(("Accelerate compensation", "Timely compensation reduces resistance.", changed))
    return scenarios

def optimize_project(artifact, row, scored_probability=None):
    base_risk = float(scored_probability) if scored_probability is not None else float(predict_probability(artifact, pd.DataFrame([row])))
    if get_risk_level(base_risk) == "Low":
        return {"recommendation": "Continue routine monitoring", "reason": "Project is Low Risk.", "estimated_risk_after_action_pct": base_risk * 100, "estimated_risk_reduction_pct": 0.0}
    
    results = []
    for action, reason, changed_row in scenario_definitions(row):
        new_risk = predict_probability(artifact, pd.DataFrame([changed_row]))
        results.append({"Intervention": action, "Reason": reason, "Simulated Risk (%)": new_risk * 100, "Risk Reduction (points)": (base_risk - new_risk) * 100})
    
    if not results:
        return {"recommendation": "Conduct officer review", "reason": "No automated intervention triggered.", "estimated_risk_after_action_pct": base_risk * 100, "estimated_risk_reduction_pct": 0.0}
    
    best = pd.DataFrame(results).sort_values("Risk Reduction (points)", ascending=False).iloc[0]
    return {"recommendation": best["Intervention"], "reason": best["Reason"], "estimated_risk_after_action_pct": best["Simulated Risk (%)"], "estimated_risk_reduction_pct": best["Risk Reduction (points)"]}

def add_portfolio_recommendations(artifact, scored_df):
    recommendation_rows = []
    for _, row in scored_df.iterrows():
        result = optimize_project(artifact, row.to_dict(), scored_probability=row["delay_probability"])
        recommendation_rows.append({
            "ai_recommendation": result["recommendation"],
            "recommendation_reason": result["reason"],
            "estimated_risk_after_action_pct": result["estimated_risk_after_action_pct"],
            "estimated_risk_reduction_pct": result["estimated_risk_reduction_pct"],
        })
    return pd.concat([scored_df.reset_index(drop=True), pd.DataFrame(recommendation_rows)], axis=1)

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
        }
    ])

def render_decision_card(level, probability, recommendation, reason, reduction):
    theme = {
        "High": {"bg": "#FEF2F2", "border": "#DC2626", "text": "#991B1B", "tag": "CRITICAL INTERVENTION"},
        "Medium": {"bg": "#FFFBEB", "border": "#D97706", "text": "#92400E", "tag": "CLOSE MONITORING"},
        "Low": {"bg": "#F0FDF4", "border": "#16A34A", "text": "#166534", "tag": "ROUTINE REVIEW"}
    }.get(level, {"bg": "#F0FDF4", "border": "#16A34A", "text": "#166534", "tag": "ROUTINE REVIEW"})

    st.markdown(f"""
    <div style="background-color: {theme['bg']}; border-left: 6px solid {theme['border']}; border-radius: 8px; padding: 1.25rem; margin-bottom: 1.5rem;">
        <span style="background-color: {theme['border']}; color: #fff; font-size: 0.75rem; font-weight: 700; padding: 0.2rem 0.6rem; border-radius: 4px;">{theme['tag']}</span>
        <h3 style="color: {theme['text']}; margin: 10px 0 5px 0;">Delay Probability: {probability * 100:.1f}%</h3>
        <p style="margin: 0; color: #334155;"><b>AI Recommendation:</b> {recommendation}<br><b>Reason:</b> {reason}<br><b>Projected Risk Drop:</b> {reduction:.1f}% points</p>
    </div>
    """, unsafe_allow_html=True)

def render_shap_waterfall(artifact, project_df):
    pipeline = artifact["pipeline"]
    model, preprocessor = pipeline.named_steps["model"], pipeline.named_steps["preprocessor"]
    x_transformed = preprocessor.transform(project_df)
    
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(x_transformed)
        shap_values.feature_names = list(preprocessor.get_feature_names_out())
        
        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(shap_values[0], max_display=7, show=False)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
    except Exception as e:
        st.caption(f"SHAP explanation fallback: {e}")

artifact = load_artifact(MODEL_PATH)
if artifact is None:
    st.error("Model artifact not found. Please run the training script.")
    st.stop()

# 3. TAB ARCHITECTURE IMPLEMENTED HERE
tab_overview, tab_single, tab_portfolio = st.tabs(["📊 Executive Summary", "🎯 Single Project Assessor", "📂 Portfolio Batch Audit"])

with tab_overview:
    demo_data = load_demo_data(PROCESSED_DATA_PATH)
    monitored_projects = len(demo_data) if demo_data is not None else 0
    historical_delayed = int(demo_data["delayed"].sum()) if demo_data is not None else 0
    historical_rate = (demo_data["delayed"].mean() * 100) if demo_data is not None else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Projects Monitored", f"{monitored_projects:,}")
    c2.metric("Critical Delays", f"{historical_delayed:,}")
    c3.metric("Mean Portfolio Risk", f"{historical_rate:.1f}%")
    c4.metric("Active Engine", artifact.get("selected_model", "XGBoost"))
    
    st.markdown("### Model Governance Metrics")
    metrics_df = pd.DataFrame(artifact.get("metrics", []))
    if not metrics_df.empty:
        st.dataframe(metrics_df, width="stretch", hide_index=True)

with tab_single:
    with st.sidebar:
        st.header("Project Parameters")
        state = st.selectbox("State", INDIAN_STATES_AND_UTS)
        sector = st.selectbox("Sector", SECTORS)
        rural_urban = st.selectbox("Setting", RURAL_URBAN_OPTIONS)
        sanctioned_cost_cr = st.number_input("Cost (Cr)", value=250.0)
        land_required_ha = st.number_input("Land (Ha)", value=120.0)
        num_landowners = st.number_input("Landowners", value=350)
        affected_families = st.number_input("Affected Families", value=420)
        land_acquired_pct = st.slider("Land Acquired (%)", 0, 100, 35)
        compensation_disbursed_pct = st.slider("Compensation (%)", 0, 100, 30)
        award_completed_pct = st.slider("Award Completion (%)", 0, 100, 40)
        possession_pct = st.slider("Possession (%)", 0, 100, 25)
        rehabilitation_progress_pct = st.slider("Rehabilitation (%)", 0, 100, 20)
        litigation_cases = st.number_input("Litigation Cases", value=12)
        litigation_flag = 1 if litigation_cases > 0 else 0
        days_since_notification = st.number_input("Days Since Notification", value=420)
        approval_pending_days = st.number_input("Approval Pending Days", value=125)
        documentation_completion_pct = st.slider("Docs Completion (%)", 0, 100, 55)
        stakeholder_response_score = st.slider("Stakeholder Score", 0, 100, 45)
        grievances_count = st.number_input("Grievances", value=38)
        previous_delay_rate = st.slider("Delay Rate", 0.0, 1.0, 0.55)
        data_quality_score = st.slider("Data Quality", 0, 100, 72)
        
        predict_button = st.button("Predict Delay Risk", type="primary", width="stretch")

    if predict_button:
        # DATA VALIDATION GUARD
        if land_acquired_pct == 100 and compensation_disbursed_pct < 20:
            st.warning("⚠️ Data Anomaly: Land acquisition is marked 100% complete, but compensation disbursement is under 20%. Please verify field records.")
            st.stop()
            
        project_df = pd.DataFrame([{
            "state": state, "sector": sector, "rural_urban": rural_urban, "sanctioned_cost_cr": sanctioned_cost_cr,
            "land_required_ha": land_required_ha, "land_acquired_pct": land_acquired_pct, "num_landowners": num_landowners,
            "affected_families": affected_families, "litigation_flag": litigation_flag, "litigation_cases": litigation_cases,
            "days_since_notification": days_since_notification, "compensation_disbursed_pct": compensation_disbursed_pct,
            "award_completed_pct": award_completed_pct, "possession_pct": possession_pct, "rehabilitation_progress_pct": rehabilitation_progress_pct,
            "approval_pending_days": approval_pending_days, "documentation_completion_pct": documentation_completion_pct,
            "stakeholder_response_score": stakeholder_response_score, "grievances_count": grievances_count,
            "previous_delay_rate": previous_delay_rate, "data_quality_score": data_quality_score
        }])
        
        probability = predict_probability(artifact, project_df)
        level = get_risk_level(probability)
        rec = optimize_project(artifact, project_df.iloc[0].to_dict(), probability)
        
        c1, c2 = st.columns([1, 1.5])
        with c1:
            render_decision_card(level, probability, rec['recommendation'], rec['reason'], rec['estimated_risk_reduction_pct'])
        with c2:
            st.markdown("#### Driver Attribution (TreeSHAP)")
            render_shap_waterfall(artifact, project_df)
            
            # JSON REPORT EXPORT
            assessment_summary = {
                "project_id": "MANUAL_EVAL",
                "state": state,
                "sector": sector,
                "delay_probability_pct": round(probability * 100, 2),
                "risk_level": level,
                "primary_recommendation": rec["recommendation"],
                "action_rationale": rec["reason"],
                "projected_risk_after_intervention_pct": round(rec["estimated_risk_after_action_pct"], 2)
            }

            st.download_button(
                label="📥 Export Assessment Summary (JSON)",
                data=pd.Series(assessment_summary).to_json(indent=2),
                file_name="bhoomiguard_assessment.json",
                mime="application/json",
                width="stretch"
            )

with tab_portfolio:
    st.write("Upload a CSV to predict risk and generate AI recommendations for an entire portfolio.")
    sample = build_sample_portfolio()
    st.download_button("Download Sample Portfolio CSV", sample.to_csv(index=False).encode("utf-8"), "land_ai_sample.csv", "text/csv")
    
    uploaded_file = st.file_uploader("Upload Portfolio CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            portfolio_df = add_portfolio_recommendations(artifact, predict_portfolio(artifact, uploaded_df))
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Projects Analysed", len(portfolio_df))
            c2.metric("High-Risk Projects", int((portfolio_df["risk_level"] == "High").sum()))
            c3.metric("Average Risk", f"{portfolio_df['delay_probability'].mean() * 100:.1f}%")

            st.subheader("Priority Table")
            display_columns = ["project_id", "delay_probability_pct", "risk_level", "ai_recommendation"]
            display_columns = [c for c in display_columns if c in portfolio_df.columns]
            st.dataframe(portfolio_df[display_columns].sort_values("delay_probability_pct", ascending=False), width="stretch", hide_index=True)
            
            risk_chart_df = portfolio_df["risk_level"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0).rename_axis("Risk Level").reset_index(name="Projects")
            chart = px.bar(risk_chart_df, x="Risk Level", y="Projects", color="Risk Level", color_discrete_map={"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"})
            st.plotly_chart(chart, width="stretch")
        except Exception as error:
            st.error(f"Could not process the CSV: {error}")