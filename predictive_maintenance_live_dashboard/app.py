from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🏭",
    layout="wide",
)

st.title("🏭 Predictive Maintenance for Industrial Equipment")
st.caption(
    "An educational machine-learning dashboard that estimates equipment failure risk "
    "from sensor readings. The built-in dataset is synthetic."
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "temperature_c",
    "vibration_mm_s",
    "operating_hours",
    "pressure_psi",
    "days_since_maintenance",
]
LABEL = "failure"


@st.cache_data
def make_sample_data(rows: int = 1500, random_state: int = 42) -> pd.DataFrame:
    """Create realistic-looking illustrative data, not real industrial records."""
    rng = np.random.default_rng(random_state)
    temperature = rng.normal(70, 12, rows).clip(30, 120)
    vibration = rng.normal(4.5, 1.8, rows).clip(0.1, 12)
    operating_hours = rng.uniform(100, 12000, rows)
    pressure = rng.normal(30, 5, rows).clip(10, 55)
    days_since_maintenance = rng.uniform(1, 365, rows)

    score = (
        0.045 * (temperature - 65)
        + 0.55 * (vibration - 4)
        + 0.00008 * (operating_hours - 5000)
        + 0.006 * (days_since_maintenance - 150)
        - 0.035 * (pressure - 30)
    )
    probability = 1 / (1 + np.exp(-score))
    failure = rng.binomial(1, probability)

    return pd.DataFrame({
        "temperature_c": temperature.round(2),
        "vibration_mm_s": vibration.round(2),
        "operating_hours": operating_hours.round().astype(int),
        "pressure_psi": pressure.round(2),
        "days_since_maintenance": days_since_maintenance.round().astype(int),
        "failure": failure,
    })


@st.cache_resource
def train_model(data: pd.DataFrame):
    X = data[FEATURES]
    y = data[LABEL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "y_test": y_test,
        "predictions": predictions,
    }
    return model, metrics


with st.sidebar:
    st.header("Dashboard controls")
    data_source = st.radio(
        "Choose data source",
        ["Built-in sample data", "Upload CSV"],
        help="Uploaded CSV must include the sensor columns listed below and a failure column (0 or 1).",
    )
    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload equipment CSV", type=["csv"])
    st.divider()
    st.markdown("**Required CSV columns**")
    st.code(
        "temperature_c\nvibration_mm_s\noperating_hours\npressure_psi\ndays_since_maintenance\nfailure",
        language="text",
    )
    st.caption("failure: 0 = no failure, 1 = failure")

if data_source == "Upload CSV" and uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        missing = [c for c in FEATURES + [LABEL] if c not in data.columns]
        if missing:
            st.error("CSV is missing required columns: " + ", ".join(missing))
            st.stop()
        data = data[FEATURES + [LABEL]].dropna()
        data[LABEL] = pd.to_numeric(data[LABEL], errors="coerce")
        data = data.dropna()
        data[LABEL] = data[LABEL].astype(int)
        if not set(data[LABEL].unique()).issubset({0, 1}) or data[LABEL].nunique() < 2:
            st.error("The failure column must contain both 0 and 1 values.")
            st.stop()
        data_source_label = "Uploaded dataset"
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        st.stop()
elif data_source == "Upload CSV":
    st.info("Upload a CSV to continue. Or select Built-in sample data.")
    st.stop()
else:
    data = make_sample_data()
    data_source_label = "Synthetic sample dataset"

model, metrics = train_model(data)
data = data.copy()
data["predicted_failure_probability"] = model.predict_proba(data[FEATURES])[:, 1]
data["predicted_failure"] = model.predict(data[FEATURES])
data["risk_level"] = pd.cut(
    data["predicted_failure_probability"],
    bins=[-0.001, 0.35, 0.65, 1.001],
    labels=["Low", "Medium", "High"],
)

st.info(
    f"Data source: **{data_source_label}** · {len(data):,} usable records. "
    "Predictions are estimates and should be reviewed by a qualified maintenance professional."
)

# KPI cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Model accuracy", f"{metrics['accuracy']:.1%}")
c2.metric("Precision", f"{metrics['precision']:.1%}")
c3.metric("Recall", f"{metrics['recall']:.1%}")
c4.metric("High-risk records", f"{(data['risk_level'] == 'High').sum():,}")

tab_overview, tab_predict, tab_data, tab_model = st.tabs(
    ["📊 Overview", "🔎 Predict a machine", "📁 Data explorer", "🧠 Model evaluation"]
)

with tab_overview:
    left, right = st.columns(2)
    with left:
        st.subheader("Failure labels in the dataset")
        label_counts = data[LABEL].map({0: "No failure", 1: "Failure"}).value_counts()
        fig, ax = plt.subplots()
        ax.bar(label_counts.index, label_counts.values)
        ax.set_ylabel("Number of records")
        ax.set_title("Actual failure labels")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("Predicted risk levels")
        risk_counts = data["risk_level"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
        fig, ax = plt.subplots()
        ax.bar(risk_counts.index.astype(str), risk_counts.values)
        ax.set_ylabel("Number of records")
        ax.set_title("Model-estimated risk")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Sensor readings")
    sensor = st.selectbox("Select sensor", FEATURES, format_func=lambda x: x.replace("_", " ").title())
    fig, ax = plt.subplots()
    ax.hist(data[sensor], bins=25)
    ax.set_xlabel(sensor.replace("_", " ").title())
    ax.set_ylabel("Records")
    ax.set_title(f"Distribution of {sensor.replace('_', ' ')}")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Highest estimated risk")
    display_cols = FEATURES + ["predicted_failure_probability", "risk_level"]
    high_risk = data.sort_values("predicted_failure_probability", ascending=False)[display_cols].head(10)
    st.dataframe(high_risk, use_container_width=True, hide_index=True)

with tab_predict:
    st.subheader("Estimate failure risk for one machine")
    st.write("Enter the current sensor readings. The model will return an estimated probability.")
    col_a, col_b = st.columns(2)
    with col_a:
        temperature = st.slider("Temperature (°C)", 30.0, 120.0, 75.0, 0.5)
        vibration = st.slider("Vibration (mm/s)", 0.1, 12.0, 5.0, 0.1)
        operating_hours = st.number_input("Operating hours", min_value=0, max_value=100000, value=6000, step=100)
    with col_b:
        pressure = st.slider("Pressure (psi)", 10.0, 55.0, 30.0, 0.5)
        maintenance_days = st.slider("Days since maintenance", 1, 365, 180)
    if st.button("Predict failure risk", type="primary"):
        machine = pd.DataFrame([{
            "temperature_c": temperature,
            "vibration_mm_s": vibration,
            "operating_hours": operating_hours,
            "pressure_psi": pressure,
            "days_since_maintenance": maintenance_days,
        }])
        risk_probability = float(model.predict_proba(machine[FEATURES])[0, 1])
        predicted = int(model.predict(machine[FEATURES])[0])
        st.metric("Estimated failure probability", f"{risk_probability:.1%}")
        if risk_probability >= 0.65:
            st.error("High estimated risk — arrange an inspection and verify sensor readings.")
        elif risk_probability >= 0.35:
            st.warning("Medium estimated risk — review operating conditions and maintenance history.")
        else:
            st.success("Low estimated risk under this model — continue routine monitoring.")
        st.caption(
            f"Classifier output: {'Failure risk flagged' if predicted else 'Failure not flagged'}. "
            "This is a demonstration model trained on synthetic data, not a safety guarantee."
        )

with tab_data:
    st.subheader("Filter and download records")
    min_risk = st.slider("Minimum estimated failure probability", 0.0, 1.0, 0.0, 0.05)
    filtered = data[data["predicted_failure_probability"] >= min_risk]
    st.write(f"Showing {len(filtered):,} of {len(data):,} records.")
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered predictions CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="maintenance_risk_predictions.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download dataset CSV",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="equipment_sensor_data.csv",
        mime="text/csv",
    )

with tab_model:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(metrics["y_test"], metrics["predictions"])
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=["No failure", "Failure"]).plot(ax=ax, colorbar=False)
    ax.set_title("Test-set confusion matrix")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Classification report")
    report = classification_report(
        metrics["y_test"], metrics["predictions"],
        target_names=["No failure", "Failure"],
        output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

    st.subheader("Feature importance")
    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots()
    ax.barh(importances.index.str.replace("_", " "), importances.values)
    ax.set_xlabel("Importance")
    ax.set_title("Which sensor features influenced the model most?")
    st.pyplot(fig)
    plt.close(fig)

st.divider()
st.caption(
    "Academic project demo. Synthetic data is used by default. Validate on representative, labelled real-world data "
    "before considering any operational use."
)
