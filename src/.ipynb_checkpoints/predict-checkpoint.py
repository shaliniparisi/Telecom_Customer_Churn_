from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "models"

def predict_churn(customer_data: pd.DataFrame) -> pd.DataFrame:
    """Predict churn from a DataFrame containing all required customer features."""
    if not isinstance(customer_data, pd.DataFrame):
        raise TypeError("customer_data must be a pandas DataFrame.")
    model_path = MODEL_DIR / "telecom_churn_xgboost.pkl"
    threshold_path = MODEL_DIR / "churn_threshold.pkl"
    columns_path = MODEL_DIR / "feature_columns.json"
    if not all(p.exists() for p in [model_path, threshold_path, columns_path]):
        raise FileNotFoundError("Run the notebook and save model artifacts first.")
    model = joblib.load(model_path)
    threshold = float(joblib.load(threshold_path))
    with open(columns_path, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)
    missing = [c for c in feature_columns if c not in customer_data.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    features = customer_data.loc[:, feature_columns].copy()
    if features.isna().any().any():
        raise ValueError("Input contains missing values.")
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return pd.DataFrame({
        "Churn_Probability": probabilities,
        "Predicted_Churn": predictions,
        "Churn_Status": np.where(predictions == 1, "Likely to Churn", "Unlikely to Churn")
    }, index=customer_data.index)
