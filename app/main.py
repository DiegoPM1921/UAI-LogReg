import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.features import impute_knn, apply_quartile_bins, apply_woe

app = FastAPI()

bundle = joblib.load("models/model.pkl")

model          = bundle["model"]
quartile_bins  = bundle["quartile_bins"]
woe_maps       = bundle["woe_maps"]
features_final = bundle["features_final"]
raw_features   = bundle["raw_features"]

MISSING_ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def fit_by_medical_info(df):

    df = df.copy()

    df["Glucose_bin"] = pd.cut(df["Glucose"], [-np.inf, 100, 125, np.inf],
                               labels=["normal", "prediabetes", "diabetes"])
    df["BMI_bin"] = pd.cut(df["BMI"], [-np.inf, 24, 30, np.inf],
                           labels=["normal", "overweight", "obese"])
    df["Age_bin"] = pd.cut(df["Age"], [-np.inf, 30, 45, np.inf],
                           labels=["young", "adult", "senior"])
    df["Pregnancies_bin"] = pd.cut(df["Pregnancies"], [-np.inf, 0, 2, np.inf],
                                   labels=["none", "low", "high"])
    return df

class PredictInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.post("/predict")
def predict(data: dict):

    try:

        X = pd.DataFrame([data.model_dump()])

        for col in MISSING_ZERO_COLS:

            X[f"{col}_missing_ind"] = (X[col] == 0).astype(int)
            X.loc[X[col] == 0, col] = np.nan
        
        if X[MISSING_ZERO_COLS].isna().all(axis=1).any():
            X[MISSING_ZERO_COLS] = X[MISSING_ZERO_COLS].fillna(0)

        _, X_knn = impute_knn(X, X, raw_features, n_neighbors=5)

        X_knn = fit_by_medical_info(X_knn)

        X_knn = apply_quartile_bins(X_knn, quartile_bins)

        for feat, mapping in woe_maps.items():
            X_knn = apply_woe(X_knn, feat, mapping)

        X_final = X_knn[features_final].fillna(0.0)

        proba = model.predict_proba(X_final)[0, 1]
        pred = int(proba >= 0.5)

        return {
            "prediction": pred,
            "probability": float(proba)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))