#!/bin/bash
echo "Setting up Land Acquisition Delay Predictor project..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Setup complete. Activate with: source venv/bin/activate"