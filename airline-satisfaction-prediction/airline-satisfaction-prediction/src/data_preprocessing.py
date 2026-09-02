"""
data_preprocessing.py

Loads the raw airline passenger satisfaction CSV, cleans it, engineers a
handful of extra features, and encodes everything into a numeric matrix
ready for modeling.

Can be run standalone (`python src/data_preprocessing.py`) to sanity-check
the pipeline, or imported by train_model.py / predict.py.
"""

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

RAW_DATA_PATH = os.path.join("data", "airline_passenger_satisfaction.csv")

# Columns that are pure identifiers / not predictive
DROP_COLS = ["id", "Unnamed: 0"]

TARGET_COL = "satisfaction"

CATEGORICAL_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]

SERVICE_RATING_COLS = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Inflight service",
    "Cleanliness",
]


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV from disk."""
    df = pd.read_csv(path)
    existing_drops = [c for c in DROP_COLS if c in df.columns]
    if existing_drops:
        df = df.drop(columns=existing_drops)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and obvious data issues."""
    df = df.copy()

    # Arrival Delay in Minutes has a small number of missing values in both
    # the real Kaggle dataset and our synthetic generator -> impute with the
    # median (delay distributions are right-skewed, so median is more robust
    # than mean).
    if "Arrival Delay in Minutes" in df.columns:
        median_delay = df["Arrival Delay in Minutes"].median()
        df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(median_delay)

    # Drop any fully-duplicated rows
    df = df.drop_duplicates()

    # Clip service ratings to the expected 0-5 Likert range in case of
    # stray bad values
    for col in SERVICE_RATING_COLS:
        if col in df.columns:
            df[col] = df[col].clip(0, 5)

    # Drop rows missing the target
    df = df.dropna(subset=[TARGET_COL])

    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features on top of the cleaned raw columns."""
    df = df.copy()

    # Combine departure + arrival delay into a single total-delay feature,
    # plus a boolean flag for "flight was delayed at all"
    df["Total Delay in Minutes"] = (
        df["Departure Delay in Minutes"] + df["Arrival Delay in Minutes"]
    )
    df["Was Delayed"] = (df["Total Delay in Minutes"] > 0).astype(int)

    # Average of all service-quality ratings -> a single "service score"
    present_service_cols = [c for c in SERVICE_RATING_COLS if c in df.columns]
    df["Avg Service Rating"] = df[present_service_cols].mean(axis=1)

    # Bucket age into life-stage groups (captures non-linear age effects
    # for the linear model)
    df["Age Group"] = pd.cut(
        df["Age"],
        bins=[0, 18, 30, 45, 60, 100],
        labels=["Teen", "Young Adult", "Adult", "Middle Age", "Senior"],
    )

    # Long-haul flag
    df["Long Flight"] = (df["Flight Distance"] > 1500).astype(int)

    return df


def encode_features(df: pd.DataFrame, encoders: dict | None = None):
    """
    Label-encode categorical columns (including the new Age Group and the
    target). Returns the encoded dataframe and the dict of fitted encoders
    so the same mapping can be reused at inference time.
    """
    df = df.copy()
    cat_cols = CATEGORICAL_COLS + ["Age Group"]
    fit_new = encoders is None
    if fit_new:
        encoders = {}

    for col in cat_cols:
        if col not in df.columns:
            continue
        if fit_new:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = df[col].astype(str).map(
                lambda v, le=le: le.transform([v])[0] if v in le.classes_ else -1
            )

    if TARGET_COL in df.columns:
        if fit_new:
            le_target = LabelEncoder()
            df[TARGET_COL] = le_target.fit_transform(df[TARGET_COL].astype(str))
            encoders[TARGET_COL] = le_target
        else:
            le_target = encoders[TARGET_COL]
            df[TARGET_COL] = le_target.transform(df[TARGET_COL].astype(str))

    return df, encoders


def preprocess(path: str = RAW_DATA_PATH):
    """Full pipeline: load -> clean -> engineer -> encode. Returns (X, y, encoders, feature_names)."""
    df = load_raw_data(path)
    df = clean_data(df)
    df = engineer_features(df)
    df, encoders = encode_features(df)

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    return X, y, encoders, list(X.columns)


if __name__ == "__main__":
    X, y, encoders, feature_names = preprocess()
    print(f"Loaded and preprocessed data: {X.shape[0]} rows, {X.shape[1]} features")
    print("Feature columns:", feature_names)
    print("\nTarget distribution:")
    print(y.value_counts())
    print("\nSample rows:")
    print(X.head())
