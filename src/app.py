import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
# Initialize FastAPI application
app = FastAPI(
    title="Enterprise Predictive Maintenance Analytics Engine",
    description="Production-grade API serving real-time machine failure classifications.",
    version="1.0.0"
)

# Core Artifact Paths
DATA_DIR = "dataset"
MODEL_PATH = os.path.join(DATA_DIR, "predictive_model.pkl")
SCALER_PATH = os.path.join(DATA_DIR, "numerical_scaler.pkl")

# Global variables to hold loaded models in memory
model = None
scaler = None


@app.on_event("startup")
def load_artifacts():
    """Load serialized model and scaling parameters when the server starts."""
    global model, scaler
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise RuntimeError("Serialized ML artifacts missing. Ensure pipeline scripts have been executed.")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Production model artifacts successfully cached in memory.")


# Define the expected incoming JSON schema using Pydantic
class TelemetryPayload(BaseModel):
    Air_Temperature_K: float = Field(..., example=301.2)
    Process_Temperature_K: float = Field(..., example=311.5)
    Rotational_Speed_RPM: int = Field(..., example=1420)
    Torque_Nm: float = Field(..., example=48.5)
    Tool_Wear_Mins: int = Field(..., example=120)


@app.get("/")
def health_check():
    """Simple API status confirmation."""
    return {"status": "operational", "engine": "XGBoost-v1.0"}


@app.post("/predict")
def predict_failure(payload: TelemetryPayload):
    """Accept raw sensor arrays, execute feature engineering, and evaluate breakdown risk."""
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Inference engine components uninitialized.")

    try:
        # 1. Convert incoming JSON structure to a structured dictionary
        input_data = payload.dict()

        # 2. Replicate Production Feature Engineering Step
        # The model relies heavily on this thermodynamic property we calculated during preprocessing
        input_data['Temp_Difference_K'] = input_data['Process_Temperature_K'] - input_data['Air_Temperature_K']

        # 3. Shape input data into a DataFrame with identical ordering as training columns
        input_df = pd.DataFrame([input_data])

        # 4. Standardize features using the exact training distribution scaler
        scaled_features = scaler.transform(input_df)
        scaled_df = pd.DataFrame(scaled_features, columns=input_df.columns)

        # 5. Run Predictive Inference
        prediction = int(model.predict(scaled_df)[0])
        probability = float(model.predict_proba(scaled_df)[0][1])

        # 6. Structuring Response Payload
        return {
            "failure_predicted": bool(prediction),
            "breakdown_probability": round(probability, 4),
            "risk_assessment": "CRITICAL - Maintenance Required" if prediction == 1 else "NOMINAL - Safe Operation"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference Pipeline Error: {str(e)}")