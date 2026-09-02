"""
train_model.py

Trains and evaluates classification models (Logistic Regression and Random
Forest) to predict airline passenger satisfaction, using the preprocessing
pipeline in data_preprocessing.py.

Saves:
  - models/logistic_regression.joblib
  - models/random_forest.joblib
  - models/scaler.joblib          (StandardScaler fit on training data)
  - models/label_encoders.joblib  (categorical encoders, incl. target)
  - models/feature_names.joblib   (ordered list of feature columns)
  - outputs/metrics_report.txt    (accuracy/precision/recall/F1 for both models)
  - outputs/confusion_matrix_logreg.png
  - outputs/confusion_matrix_rf.png
  - outputs/feature_importance_rf.png

Run with: python src/train_model.py
"""

import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_preprocessing import preprocess

MODELS_DIR = "models"
OUTPUT_DIR = "outputs"
RANDOM_STATE = 42


def ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def evaluate_model(name, model, X_test, y_test, target_names):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n===== {name} =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(report)

    # Save confusion matrix plot
    fig, ax = plt.subplots(figsize=(5.5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    fname = f"confusion_matrix_{'logreg' if 'Logistic' in name else 'rf'}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150)
    plt.close()

    return {
        "name": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "report": report,
    }


def plot_feature_importance(model, feature_names):
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    top = importances.head(15)
    plt.figure(figsize=(8, 7))
    sns.barplot(x=top.values, y=top.index, hue=top.index, palette="viridis", legend=False)
    plt.title("Top 15 Feature Importances (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance_rf.png"), dpi=150)
    plt.close()
    return importances


def main():
    ensure_dirs()

    print("Loading and preprocessing data...")
    X, y, encoders, feature_names = preprocess()
    target_names = list(encoders["satisfaction"].classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # Scale features (mainly benefits Logistic Regression; harmless for RF
    # since we fit RF on the same scaled data for a consistent pipeline)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    # ---- Logistic Regression -------------------------------------------------
    logreg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    logreg.fit(X_train_scaled, y_train)
    results.append(evaluate_model("Logistic Regression", logreg, X_test_scaled, y_test, target_names))

    # ---- Random Forest ---------------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    results.append(evaluate_model("Random Forest", rf, X_test_scaled, y_test, target_names))

    importances = plot_feature_importance(rf, feature_names)

    # ---- Save models & artifacts ------------------------------------------
    joblib.dump(logreg, os.path.join(MODELS_DIR, "logistic_regression.joblib"))
    joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump(encoders, os.path.join(MODELS_DIR, "label_encoders.joblib"))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.joblib"))

    # ---- Write metrics report ------------------------------------------------
    report_path = os.path.join(OUTPUT_DIR, "metrics_report.txt")
    with open(report_path, "w") as f:
        f.write("Airline Passenger Satisfaction - Model Evaluation Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}\n")
        f.write(f"Number of features: {len(feature_names)}\n\n")
        for r in results:
            f.write(f"----- {r['name']} -----\n")
            f.write(f"Accuracy : {r['accuracy']:.4f}\n")
            f.write(f"Precision: {r['precision']:.4f}\n")
            f.write(f"Recall   : {r['recall']:.4f}\n")
            f.write(f"F1 Score : {r['f1']:.4f}\n")
            f.write(r["report"] + "\n")
        f.write("\nTop 10 Random Forest Feature Importances:\n")
        f.write(importances.head(10).to_string() + "\n")

    print(f"\nSaved models to {MODELS_DIR}/ and metrics report to {report_path}")
    best = max(results, key=lambda r: r["f1"])
    print(f"\nBest model by F1 score: {best['name']} (F1={best['f1']:.4f})")


if __name__ == "__main__":
    main()
