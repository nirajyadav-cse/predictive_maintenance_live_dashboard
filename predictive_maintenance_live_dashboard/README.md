# Predictive Maintenance for Industrial Equipment

A Python and Streamlit dashboard for exploring equipment sensor readings and demonstrating machine-learning-based failure-risk estimation.

## Features
- Interactive dashboard with summary metrics and charts
- Predict failure risk for an individual machine using sensor readings
- Explore and download predictions as CSV
- Upload your own labelled sensor dataset
- Random Forest classifier with accuracy, precision, recall, F1-score, confusion matrix, and feature importance

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit will print a local address, usually `http://localhost:8501`.

## CSV format for uploading data

Your CSV must include these columns:

- `temperature_c`
- `vibration_mm_s`
- `operating_hours`
- `pressure_psi`
- `days_since_maintenance`
- `failure` (`0` = no failure, `1` = failure)

The uploaded dataset must contain both 0 and 1 in the `failure` column.

## Deploy a live dashboard

1. Push this folder to a public GitHub repository.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Choose **Create app** and select your repository and `main` branch.
4. Set **Main file path** to `app.py`.
5. Click **Deploy**. Streamlit will provide a public `*.streamlit.app` URL.
6. Add that URL to this README and share it.

## Important limitation

The built-in dataset is synthetic and intended for academic demonstration only. Predictions are not a safety guarantee. Use validated, representative real equipment data before drawing operational conclusions.
