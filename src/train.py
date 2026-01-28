import os
import joblib
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from features import impute_knn, fit_quartile_bins, apply_quartile_bins, compute_woe_table, apply_woe


# Definir como se trataran las variables

TARGET = "Outcome"

RAW_FEATURES      = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness","Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
MISSING_ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
QUARTILE_FEAT     = ["BloodPressure", "SkinThickness", "Insulin", "DiabetesPedigreeFunction"]

# Funciones utiles

def fit_by_medical_info(df):
    """
    Categoriza las columnas especificadas de los DataFrames de entrenamiento y prueba en función de valores médicos típicos

    Parámetros:
    df (DataFrame): El DataFrame en que se realizará la categorización.
    features (list): Lista de nombres de columnas donde se categorizarán los valores según información médica.

    Retorna:
    tuple: Una tupla que contiene los DataFrames modificados de entrenamiento y prueba con las columnas categorizadas según información médica.
    """

    ret = df.copy()

    ret["Glucose_bin"] = pd.cut(ret["Glucose"], bins=[-np.inf, 100, 125, np.inf], labels=["normal", "prediabetes", "diabetes"])

    ret["BMI_bin"] = pd.cut(ret["BMI"], bins=[-np.inf, 24, 30, np.inf], labels=["normal", "overweight", "obese"])

    ret["Age_bin"] = pd.cut(ret["Age"], bins=[-np.inf, 30, 45, np.inf], labels=["young", "adult", "senior"])

    ret["Pregnancies_bin"] = pd.cut(ret["Pregnancies"], bins=[-np.inf, 0, 2, np.inf], labels=["none", "low", "high"])

    return ret

def main():
    
    df = pd.read_csv("data/diabetes.csv")

    X = df[RAW_FEATURES].copy()
    y = df[TARGET].copy()

    # Informacion faltante

    for col in MISSING_ZERO_COLS:

        X[f"{col}_missing_ind"] = (X[col] == 0).astype(int)
        X.loc[X[col] == 0, col] = np.nan

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # KNN 
    X_train_knn, X_test_knn = impute_knn(X_train, X_test, RAW_FEATURES, n_neighbors=5)

    train_df = pd.concat([X_train_knn, y_train], axis=1)
    test_df  = pd.concat([X_test_knn, y_test], axis=1)

    # Bins médicos
    train_df = fit_by_medical_info(train_df)
    test_df  = fit_by_medical_info(test_df)

    # Bins de cuartiles
    quartile_bins = fit_quartile_bins(train_df, QUARTILE_FEAT)
    train_df      = apply_quartile_bins(train_df, quartile_bins)
    test_df       = apply_quartile_bins(test_df, quartile_bins)

    # WOE
    binned_feats = [
        "Glucose_bin", "BMI_bin", "Age_bin", "Pregnancies_bin",
        "BloodPressure_qbin", "SkinThickness_qbin",
        "Insulin_qbin", "DiabetesPedigreeFunction_qbin"
    ]

    woe_maps = {}

    for feat in binned_feats:

        _, mapping, _ = compute_woe_table(train_df, feat, TARGET)

        woe_maps[feat] = mapping

        train_df = apply_woe(train_df, feat, mapping)
        test_df  = apply_woe(test_df, feat, mapping)

    woe_cols    = [c for c in train_df.columns if c.endswith("_woe")]
    missing_ind = [c for c in train_df.columns if c.endswith("_missing_ind")]

    FEATURES_FINAL = woe_cols + missing_ind

    # Modelo final
    model = LogisticRegression(class_weight="balanced", max_iter=5000)
    model.fit(train_df[FEATURES_FINAL], y_train)

    proba = model.predict_proba(test_df[FEATURES_FINAL])[:, 1]

    # Métricas
    auc       = roc_auc_score(y_test, proba)
    accuracy  = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, zero_division=0)
    recall    = recall_score(y_test, pred, zero_division=0)
    f1        = f1_score(y_test, pred, zero_division=0)

    print(f"ROC-AUC test : {auc:.4f}")
    print(f"Accuracy     : {accuracy:.4f}")
    print(f"Precision    : {precision:.4f}")
    print(f"Recall       : {recall:.4f}")
    print(f"F1-score     : {f1:.4f}")


    bundle = {
        "model": model,
        "quartile_bins": quartile_bins,
        "woe_maps": woe_maps,
        "features_final": FEATURES_FINAL,
        "raw_features": RAW_FEATURES,
    }

    os.makedirs("models", exist_ok=True)
    joblib.dump(bundle, "models/model.pkl")

if __name__ == "__main__":
    main()