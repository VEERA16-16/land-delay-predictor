# 🏢 BHOOMIGUARD AI: Infrastructure Land Acquisition Delay Intelligence

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bhoomiguard-ai.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Live Deployment:** [bhoomiguard-ai.streamlit.app](https://bhoomiguard-ai.streamlit.app/)  
> **Problem Statement Track:** Smart India Hackathon (SIH26017) — Ministry of Rural Development

---

## 📌 Executive Overview
Land acquisition bottlenecks cause major infrastructure schedule overruns across India. **BHOOMIGUARD AI** shifts monitoring from reactive tracking to **predictive early-warning decision support**. 

By evaluating 21 administrative, financial, legal, and operational milestone indicators, the system:
1. **Predicts** schedule delay probability using optimized ensemble decision trees.
2. **Explains** local feature attribution via **TreeSHAP** waterfall diagnostics.
3. **Simulates** counterfactual operational interventions to prescribe specific risk-reduction actions.

---

## 🏗️ System Architecture

```text
Synthetic Domain Engine (21 Features)
               │
               ▼
  Scikit-Learn ColumnTransformer Pipeline
  (SimpleImputer + StandardScaler + OneHotEncoder)
               │
               ▼
      XGBoost / HistGradientBoosting
         (Optimized for High Recall)
               │
       ┌───────┴───────┐
       ▼               ▼
 TreeSHAP Attribution   Counterfactual Prescriptive Engine
 (Local Waterfall Plot) (Simulated Interventions)
       └───────┬───────┘
               ▼
 Streamlit Cloud Enterprise Decision Interface