✈️ AI-Based Airline Customer Satisfaction Prediction
=====================================================

![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Data-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

A machine learning pipeline that predicts whether an airline passenger is
**satisfied** or **neutral / dissatisfied** with their flight, based on
demographics, trip details, and 14 in-flight service ratings. Covers the
full applied ML workflow end to end: data cleaning, exploratory data
analysis, feature engineering, model training/comparison, and inference on
new passenger records.

> **Data note:** This project is modeled on the public [Airline Passenger
> Satisfaction dataset on Kaggle](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction).
> The real CSV wasn't available while rebuilding this project, so
> `generate_data.py` creates a synthetic dataset with the exact same column
> schema and a hand-designed, plausible feature → satisfaction relationship.
> All results below are from this synthetic data — see
> [Reproducing on real data](#reproducing-on-the-real-dataset) to swap in
> the actual Kaggle CSV.

## Tech Stack

- **Language:** Python 3
- **Data Processing:** pandas, NumPy
- **Machine Learning:** scikit-learn (Logistic Regression, Random Forest,
  StandardScaler, train/test split, classification metrics)
- **Visualization:** matplotlib, seaborn
- **Model Persistence:** joblib

## Features

- 🧹 **Data Preprocessing** — missing-value imputation, duplicate removal,
  categorical encoding
- 📊 **Exploratory Data Analysis** — 6 saved plots covering satisfaction
  distribution, correlations, and service-rating breakdowns
- ⚙️ **Feature Engineering** — delay aggregates, a composite service-quality
  score, age bucketing, long-flight flag
- 🤖 **Model Training & Comparison** — Logistic Regression vs. Random Forest,
  evaluated on accuracy, precision, recall, and F1
- 🔮 **Inference Script** — loads the saved model and predicts satisfaction
  for new passenger records

## Application Structure

```
airline-satisfaction-prediction/
├── generate_data.py                 # Creates the synthetic dataset
├── requirements.txt
├── data/
│   └── airline_passenger_satisfaction.csv
├── src/
│   ├── data_preprocessing.py        # Load, clean, engineer features, encode
│   ├── eda.py                       # Exploratory analysis -> saves plots
│   ├── train_model.py               # Train + evaluate LogReg & Random Forest
│   └── predict.py                   # Load saved model, predict on new data
├── models/                          # Saved model & preprocessing artifacts
└── outputs/                         # Saved plots + metrics report
```

## How to Run

```bash
pip install -r requirements.txt

# Generate the synthetic dataset (skip if using the real Kaggle CSV)
python generate_data.py

# Run EDA — saves plots to outputs/
python src/eda.py

# Train and evaluate models — saves models to models/, metrics to outputs/
python src/train_model.py

# Predict on new sample passengers
python src/predict.py
```

All scripts run from the project root.

### Reproducing on the real dataset

Download `train.csv`/`test.csv` from the
[Kaggle dataset page](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction),
save as `data/airline_passenger_satisfaction.csv` (column names already
match), skip `generate_data.py`, and run the rest of the pipeline as-is.
Public notebooks on the real data commonly report 90%+ accuracy with
tree-based models — meaningfully higher than the synthetic numbers below.

## Methodology

1. **Preprocessing** — drop identifier columns, median-impute missing delay
   values, clip ratings to a valid 0–5 range, drop duplicates, label-encode
   categoricals.
2. **Feature engineering** — `Total Delay`, `Was Delayed` flag,
   `Avg Service Rating` (composite of all 14 ratings), `Age Group` buckets,
   `Long Flight` flag.
3. **Modeling** — 80/20 stratified split, features scaled with
   `StandardScaler`, Logistic Regression (interpretable baseline) vs.
   Random Forest (captures non-linear interactions).
4. **Inference** — loads the saved model + scaler + encoders and predicts
   on new passenger dicts/DataFrames.

## Results / Outcomes

- ✅ Built and ran the full pipeline end to end — data generation through
  inference — with zero errors
- ✅ Logistic Regression: **76.6%** accuracy, 0.755 F1
- ✅ Random Forest: **76.9%** accuracy, **0.758 F1** (best model)
- ✅ Top predictors identified: Avg Service Rating, Inflight wifi service,
  Online boarding, Cleanliness, Flight Distance — consistent with public
  analyses of the real dataset
- ✅ Verified example predictions: happy business traveler → satisfied
  (96.0% confidence); frustrated economy traveler → dissatisfied (98.6%
  confidence)

~77% accuracy is a modest, honest result on synthetic data that
deliberately mixes signal with noise — this is not a claim about
real-world performance, which is expected to be substantially higher (see
above).

## Possible Extensions

- Hyperparameter tuning (`GridSearchCV`/`RandomizedSearchCV`)
- Additional models: Gradient Boosting/XGBoost, SVM
- SHAP values for per-prediction explainability
- A small Flask/Streamlit front end for interactive predictions

---

Developer: **Hasini Prasad** | [LinkedIn](https://linkedin.com/in/hasini-prasad-b761b0286)
