# Predictive Maintenance for Industrial Equipment 🏭

## About the Project

Machine failures can cause unexpected downtime and increase maintenance costs. I built this project to explore how machine learning can help identify equipment that may be at risk of failure.

The application uses Python and a Random Forest model to analyze equipment sensor readings and estimate failure risk. I also created a Streamlit dashboard where users can explore the data, check model performance, and test different machine conditions.

This project is part of my learning journey in Python, machine learning, and data analytics.

## What Can You Do With It?

- **Predict failure risk:** Enter sensor readings and estimate the risk for a machine.
- **Explore equipment data:** View sensor readings through interactive charts.
- **Evaluate the model:** Check accuracy, precision, recall, and other performance metrics.
- **Upload your own data:** Use a CSV file with the required sensor columns.
- **Download results:** Export predictions for further analysis.

## Tools and Technologies

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Random Forest Classification

## Project Structure

```text
predictive_maintenance_live_dashboard/
├── app.py
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/nirajyadav-cse/predictive_maintenance_live_dashboard.git
cd predictive_maintenance_live_dashboard
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL printed in your terminal, usually `http://localhost:8501`.

## Dataset

The application generates sample equipment data so the dashboard can run without an external dataset.

You can also upload your own CSV file containing these columns:

- `temperature_c`
- `vibration_mm_s`
- `operating_hours`
- `pressure_psi`
- `days_since_maintenance`
- `failure` — use `0` for no failure and `1` for failure.

Your uploaded dataset must contain both failure labels.


## 🚀 Live Dashboard

Try the application here:

[Open Predictive Maintenance Dashboard](https://predictivemaintenancelivedashboard-4m8nn38sfkcu68vu53g6bm.streamlit.app)


## What I Learned

While building this project, I practised working with sensor data, training a machine learning model, evaluating predictions, and building a dashboard with Streamlit. I also learned how to organize a Python project and prepare it for GitHub.

## A Note About Predictions

This project is for learning and demonstration purposes. The built-in dataset is synthetic, so the results do not represent verified industrial performance. The predictions should not be used alone to make real equipment safety or maintenance decisions.

## Author

**Niraj Yadav**  
B.Tech CSE — Artificial Intelligence and Machine Learning

[GitHub Profile](https://github.com/nirajyadav-cse)
