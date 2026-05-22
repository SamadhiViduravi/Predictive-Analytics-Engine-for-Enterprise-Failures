from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI(
    title="Predictive Analytics Engine for Enterprise Failures",
    description="Production REST API exposing multi-class industrial component diagnostic vectors.",
    version="1.0.0"
)

TARGET_NAMES = [
    "Nominal Operational State (Healthy)",
    "Component 1 Failure Risk",
    "Component 2 Failure Risk",
    "Component 3 Failure Risk",
    "Component 4 Failure Risk"
]

MODEL_PATH = "model/xgb_multi_model.pkl"
FEATURES_PATH = "model/feature_columns.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    raise RuntimeError("Model compilation artifacts missing. Run 'python src/train.py' first.")

model = joblib.load(MODEL_PATH)
expected_features = joblib.load(FEATURES_PATH)


class TelemetryPayload(BaseModel):
    # Incoming payload structure from Streamlit or external enterprise systems
    datetime: str = Field(..., example="2015-01-01 06:00:00")
    volt_mean_3h: float = Field(..., example=170.5)
    volt_std_3h: float = Field(2.1, example=2.1)
    volt_mean_24h: float = Field(..., example=170.8)
    volt_std_24h: float = Field(2.5, example=2.5)

    rotate_mean_3h: float = Field(..., example=450.2)
    rotate_std_3h: float = Field(45.0, example=45.0)
    rotate_mean_24h: float = Field(..., example=451.1)
    rotate_std_24h: float = Field(48.0, example=48.0)

    pressure_mean_3h: float = Field(..., example=100.1)
    pressure_std_3h: float = Field(10.0, example=10.0)
    pressure_mean_24h: float = Field(..., example=100.4)
    pressure_std_24h: float = Field(11.0, example=11.0)

    vibration_mean_3h: float = Field(..., example=40.2)
    vibration_std_3h: float = Field(4.0, example=4.0)
    vibration_mean_24h: float = Field(..., example=40.5)
    vibration_std_24h: float = Field(4.5, example=4.5)

    error1: int = Field(0, ge=0, le=1, example=0)
    error2: int = Field(0, ge=0, le=1, example=0)
    error3: int = Field(0, ge=0, le=1, example=0)
    error4: int = Field(0, ge=0, le=1, example=0)
    error5: int = Field(0, ge=0, le=1, example=0)

    age: int = Field(..., ge=0, le=25, example=10)
    model: str = Field(..., example="model1")


@app.get("/")
def read_root():
    return {"status": "ONLINE", "engine": "XGBoost Multi-Class SMOTE-NC Core"}


@app.post("/predict")
def predict_diagnostics(payload: TelemetryPayload):
    try:
        raw_data = payload.dict()
        input_df = pd.DataFrame([raw_data])

        # =========================================================================
        # MATCH TRAINING PIPELINE: DYNAMIC FEATURE EXTRACTION
        # =========================================================================
        datetime_series = pd.to_datetime(input_df["datetime"])
        input_df["hour"] = datetime_series.dt.hour.astype(np.uint8)
        input_df["day_of_week"] = datetime_series.dt.dayofweek.astype(np.uint8)

        # Ensure the 'model' column is treated explicitly as a categorical element
        input_df["model"] = input_df["model"].astype("category")

        # Enforce exact column alignment as compiled during training
        input_df = input_df[expected_features]
        # =========================================================================

        probabilities = model.predict_proba(input_df)[0]

        # Industrial threshold: flag any failure class exceeding 25% probability.
        # In a real plant, a 37% component failure risk warrants immediate inspection —
        # defaulting to "nominal" because nominal is still 59% would be dangerous.
        FAILURE_THRESHOLD = 0.25
        failure_probs = probabilities[1:]  # classes 1-4 are failure types
        max_failure_idx = int(np.argmax(failure_probs))
        max_failure_prob = float(failure_probs[max_failure_idx])

        if max_failure_prob >= FAILURE_THRESHOLD:
            predicted_class_idx = max_failure_idx + 1  # shift back to label index 1-4
        else:
            predicted_class_idx = 0

        return {
            "prediction": {
                "class_index": predicted_class_idx,
                "label": TARGET_NAMES[predicted_class_idx],
                "confidence_score": float(probabilities[predicted_class_idx])
            },
            "risk_distribution": {
                TARGET_NAMES[i]: float(probabilities[i]) for i in range(len(TARGET_NAMES))
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference execution failure: {str(e)}")