#!/bin/bash
set -e

pip install -r requirements.txt
python3 app/pipeline.py
streamlit run ui/streamlit_app.py
