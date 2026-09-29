@echo off
echo Setting up Land Acquisition Delay Predictor project...
python -m venv venv
call venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
echo Setup complete. Activate with: venv\Scripts\activate
pause