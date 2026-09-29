# Land Acquisition Delay Predictor - SIH26017

## Problem
Predictive Analytics System for Early Detection of Land Acquisition Delays (Ministry of Rural Development)

## Folder Structure
land-delay-predictor/
├── data/raw/
├── data/processed/
├── notebooks/
├── src/
├── app/
├── models/
├── assets/
├── reports/
├── config.py
├── requirements.txt
├── setup_windows.bat
├── setup_mac_linux.sh
├── generate_dataset.py
└── README.md


## Setup (do this first)

### Windows
Double-click `setup_windows.bat`, or run:

### Mac/Linux


## Daily activation (every time you open a new terminal)
- Windows: `venv\Scripts\activate`
- Mac/Linux: `source venv/bin/activate`

## Run the Streamlit app

## Data sources
1. https://www.data.gov.in/resource/project-wise-details-some-major-projects-delayed-due-land-acquisition-01-04-2024
   → Download CSV into `data/raw/raw_delayed_projects.csv`

2. https://larr.dolr.gov.in/faces/public/projectwise.xhtml
   → Reference for real field structure

## 8-Hour Plan
- Hour 1-2: Generate dataset, quick EDA
- Hour 2-4: Train model (`python src/train.py`)
- Hour 4-6: Build Streamlit app
- Hour 6-7: Add dashboard, polish UI
- Hour 7-8: Prepare presentation, rehearse demo