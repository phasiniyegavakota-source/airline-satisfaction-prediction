"""
predict.py

Loads the trained model + preprocessing artifacts and predicts satisfaction
for one or more new passenger records.

Run standalone for a demo prediction on a hand-crafted example:
    python src/predict.py

Or import `predict_satisfaction(sample_dict)` / `predict_from_dataframe(df)`
from other code.
"""

import os

import joblib
import pandas as pd

from data_preprocessing import (
    CATEGORICAL_COLS,
    TARGET_COL,
    clean_data,
    engineer_features,
)

MODELS_DIR = "models"
DEFAULT_MODEL_NAME = "random_forest.joblib"  # swap to "logistic_regression.joblib" if preferred


def load_artifacts(model_name: str = DEFAULT_MODEL_NAME):
    model = joblib.load(os.path.join(MODELS_DIR, model_name))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    encoders = joblib.load(os.path.join(MODELS_DIR, "label_encoders.joblib"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.joblib"))
    return model, scaler, encoders, feature_names


def _encode_with_fitted_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df = df.copy()
    cat_cols = CATEGORICAL_COLS + ["Age Group"]
    for col in cat_cols:
        if col not in df.columns:
            continue
        le = encoders[col]
        df[col] = df[col].astype(str).map(
            lambda v, le=le: le.transform([v])[0] if v in le.classes_ else -1
        )
    return df


def predict_from_dataframe(raw_df: pd.DataFrame, model_name: str = DEFAULT_MODEL_NAME):
    """Predict satisfaction for one or more raw (unprocessed) passenger rows."""
    model, scaler, encoders, feature_names = load_artifacts(model_name)

    df = raw_df.copy()
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])

    # Reuse the same cleaning/feature-engineering steps as training.
    # clean_data() drops rows with missing target, which won't exist here,
    # so we skip target-related steps and just impute delay + clip ratings.
    if "Arrival Delay in Minutes" in df.columns:
        df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(0)

    df = engineer_features(df)
    df = _encode_with_fitted_encoders(df, encoders)

    # Align columns to the exact training feature order (fill any missing
    # engineered columns with 0 as a safe default)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    X_scaled = scaler.transform(df)
    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else None

    target_encoder = encoders[TARGET_COL]
    labels = target_encoder.inverse_transform(preds)

    result = raw_df.copy()
    result["predicted_satisfaction"] = labels
    if probs is not None:
        # probability of the class encoded as 1 (check which label that is)
        satisfied_label_idx = list(target_encoder.classes_).index("satisfied") if "satisfied" in target_encoder.classes_ else 1
        result["probability_satisfied"] = probs if satisfied_label_idx == 1 else 1 - probs

    return result


def predict_satisfaction(sample: dict, model_name: str = DEFAULT_MODEL_NAME):
    """Convenience wrapper: predict satisfaction for a single passenger given as a dict."""
    df = pd.DataFrame([sample])
    result = predict_from_dataframe(df, model_name=model_name)
    return result.iloc[0]["predicted_satisfaction"], (
        result.iloc[0]["probability_satisfied"] if "probability_satisfied" in result.columns else None
    )


if __name__ == "__main__":
    # Example: a business traveler with strong service ratings and no delays
    sample_happy = {
        "Gender": "Female",
        "Customer Type": "Loyal Customer",
        "Age": 35,
        "Type of Travel": "Business travel",
        "Class": "Business",
        "Flight Distance": 1200,
        "Inflight wifi service": 5,
        "Departure/Arrival time convenient": 4,
        "Ease of Online booking": 5,
        "Gate location": 4,
        "Food and drink": 4,
        "Online boarding": 5,
        "Seat comfort": 5,
        "Inflight entertainment": 4,
        "On-board service": 5,
        "Leg room service": 4,
        "Baggage handling": 5,
        "Checkin service": 5,
        "Inflight service": 5,
        "Cleanliness": 5,
        "Departure Delay in Minutes": 0,
        "Arrival Delay in Minutes": 0,
    }

    # Example: an economy passenger with poor service ratings and a big delay
    sample_unhappy = {
        "Gender": "Male",
        "Customer Type": "disloyal Customer",
        "Age": 24,
        "Type of Travel": "Personal Travel",
        "Class": "Eco",
        "Flight Distance": 400,
        "Inflight wifi service": 1,
        "Departure/Arrival time convenient": 2,
        "Ease of Online booking": 1,
        "Gate location": 2,
        "Food and drink": 1,
        "Online boarding": 1,
        "Seat comfort": 2,
        "Inflight entertainment": 1,
        "On-board service": 2,
        "Leg room service": 1,
        "Baggage handling": 2,
        "Checkin service": 2,
        "Inflight service": 1,
        "Cleanliness": 2,
        "Departure Delay in Minutes": 120,
        "Arrival Delay in Minutes": 135,
    }

    for name, sample in [("Happy business traveler", sample_happy), ("Frustrated economy traveler", sample_unhappy)]:
        label, prob = predict_satisfaction(sample)
        prob_str = f" (P(satisfied)={prob:.3f})" if prob is not None else ""
        print(f"{name}: predicted = '{label}'{prob_str}")
