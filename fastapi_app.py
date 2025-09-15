from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import pandas as pd
from joblib import load
import os

app = FastAPI(title="Credit Card Fraud Detection API")

MODEL_PATH = "best_model.joblib"
EXPECTED_COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

class Transaction(BaseModel):
    data: Dict[str, float]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(tx: Transaction):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=400, detail="Model not found. Run train_model.py first.")
    model_bundle = load(MODEL_PATH)

    # Ensure correct column order
    df = pd.DataFrame([[tx.data[c] for c in EXPECTED_COLS]], columns=EXPECTED_COLS)
    proba = float(model_bundle["pipeline"].predict_proba(df)[:, 1][0])
    pred = int(proba >= model_bundle["threshold"])
    return {
        "fraud_probability": round(proba, 6),
        "fraud_flag": pred,
        "threshold": model_bundle["threshold"],
        "model": model_bundle["model_name"]
    }
