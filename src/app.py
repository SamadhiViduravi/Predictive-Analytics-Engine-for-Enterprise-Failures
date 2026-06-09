import streamlit as st
import requests
from datetime import datetime

st.set_page_config(page_title="Enterprise Predictive Maintenance Engine", layout="wide")

st.title("🏭 Multi-Class Predictive Maintenance System")
st.subheader("Decoupled Enterprise Architecture — Telemetry Analysis Terminal")

FASTAPI_URL = "http://127.0.0.1:8000/predict"

# Set up user input fields split into clean operational layout columns
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📊 Short-Term Metrics (3-Hour Window)")
    volt_mean_3h = st.number_input("Voltage Mean (3h)", value=170.0)
    rotate_mean_3h = st.number_input("Rotation Speed Mean (3h)", value=450.0)
    pressure_mean_3h = st.number_input("Pressure Mean (3h)", value=100.0)
    vibration_mean_3h = st.number_input("Vibration Mean (3h)", value=40.0)

with col2:
    st.markdown("### 📈 Long-Term Wear Metrics (24-Hour Window)")
    volt_mean_24h = st.number_input("Voltage Mean (24h)", value=170.0)
    rotate_mean_24h = st.number_input("Rotation Speed Mean (24h)", value=450.0)
    pressure_mean_24h = st.number_input("Pressure Mean (24h)", value=100.0)
    vibration_mean_24h = st.number_input("Vibration Mean (24h)", value=40.0)

with col3:
    st.markdown("### ⚙️ Asset Metadata & Historical Errors")
    model_type = st.selectbox("Machine Model Type", ["model1", "model2", "model3", "model4"])
    age = st.slider("Machine Age (Years in Service)", min_value=0, max_value=25, value=10)

    st.markdown("**Error Logs Thrown (Past 24 Hours)**")
    error1 = st.checkbox("error1 triggered")
    error2 = st.checkbox("error2 triggered")
    error3 = st.checkbox("error3 triggered")
    error4 = st.checkbox("error4 triggered")
    error5 = st.checkbox("error5 triggered")

st.markdown("---")

if st.button("Run Diagnostic Analyzer via REST API", type="primary"):
    # Capture the exact current timestamp to feed the backend's feature extractor
    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Construct standard JSON payload matching FastAPI validation structures perfectly
    payload = {
        "datetime": current_timestamp,  # 🌟 Added to satisfy the API validation contract
        "volt_mean_3h": float(volt_mean_3h), "volt_std_3h": 2.1,
        "volt_mean_24h": float(volt_mean_24h), "volt_std_24h": 2.5,
        "rotate_mean_3h": float(rotate_mean_3h), "rotate_std_3h": 45.0,
        "rotate_mean_24h": float(rotate_mean_24h), "rotate_std_24h": 48.0,
        "pressure_mean_3h": float(pressure_mean_3h), "pressure_std_3h": 10.0,
        "pressure_mean_24h": float(pressure_mean_24h), "pressure_std_24h": 11.0,
        "vibration_mean_3h": float(vibration_mean_3h), "vibration_std_3h": 4.0,
        "vibration_mean_24h": float(vibration_mean_24h), "vibration_std_24h": 4.5,
        "error1": 1 if error1 else 0, "error2": 1 if error2 else 0,
        "error3": 1 if error3 else 0, "error4": 1 if error4 else 0,
        "error5": 1 if error5 else 0, "age": int(age),
        "model": str(model_type)
    }

    with st.spinner("Transmitting inference vectors to backend microservice..."):
        try:
            response = requests.post(FASTAPI_URL, json=payload)

            if response.status_code == 200:
                result = response.json()

                prediction_data = result["prediction"]
                risk_distribution = result["risk_distribution"]

                st.markdown("### 🔍 Diagnostic Risk Allocation Analysis")

                # Check if system output detects an imminent failure risk
                if prediction_data["class_index"] == 0:
                    st.success(
                        f"**Asset Status Recommendation:** {prediction_data['label']} (Confidence: {prediction_data['confidence_score'] * 100:.2f}%)")
                else:
                    st.sidebar.error("⚠️ CRITICAL ALERT: IMMINENT FAILURE PATHWAY IDENTIFIED")
                    st.error(
                        f"**Critical Alert:** Maintenance recommended! **{prediction_data['label']}** isolated. (Confidence: {prediction_data['confidence_score'] * 100:.2f}%)")

                # Display structural diagnostic progress bars
                for class_label, confidence in risk_distribution.items():
                    st.progress(float(confidence))
                    st.write(f"{class_label}: **{confidence * 100:.2f}%**")

            else:
                st.error(f"Backend Server returned an unhandled exception state: {response.status_code}")
                st.code(response.text)
        except Exception as err:
            st.error(
                f"Network transport level connection error: Unable to talk to FastAPI microservice. Details: {err}")