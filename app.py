from fastapi import FastAPI
from pydantic import BaseModel

import numpy as np
import joblib

model = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")

app = FastAPI(
    title="Fraud Detection API"
)

class TransactionData(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount_Log: float


@app.get("/")
def home():

    return {
        "message": "Fraud Detection API is running"
    }

@app.post("/predict")
def predict(data: TransactionData):

    input_data = np.array([[
        data.Time,
        data.V1,
        data.V2,
        data.V3,
        data.V4,
        data.V5,
        data.V6,
        data.V7,
        data.V8,
        data.V9,
        data.V10,
        data.V11,
        data.V12,
        data.V13,
        data.V14,
        data.V15,
        data.V16,
        data.V17,
        data.V18,
        data.V19,
        data.V20,
        data.V21,
        data.V22,
        data.V23,
        data.V24,
        data.V25,
        data.V26,
        data.V27,
        data.V28,
        data.Amount_Log
    ]])

    # Scale ALL Features
    input_data_scaled = scaler.transform(
        input_data
    )

    # Prediction
    prediction = model.predict(
        input_data_scaled
    )[0]

    # Prediction Probability
    prediction_probability = model.predict_proba(
        input_data_scaled
    )[0][1]

    # Prediction Label
    result = (
        "Fraud"
        if prediction == 1
        else "Non-Fraud"
    )


    # Risk Level
    if prediction_probability < 0.30:
        risk_level = "Low"

    elif prediction_probability < 0.70:
        risk_level = "Medium"

    else:
        risk_level = "High"


    return {
        "prediction": result,
        "fraud_probability": round(
            float(prediction_probability),
            4
        ),
        "risk_level": risk_level
    }