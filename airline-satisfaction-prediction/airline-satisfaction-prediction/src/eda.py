"""
eda.py

Exploratory Data Analysis: produces a handful of saved plots into outputs/
summarizing the dataset and the relationships between key features and the
satisfaction target.

Run with: python src/eda.py
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data_preprocessing import (
    RAW_DATA_PATH,
    SERVICE_RATING_COLS,
    clean_data,
    engineer_features,
    load_raw_data,
)

OUTPUT_DIR = "outputs"
sns.set_theme(style="whitegrid")


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_satisfaction_distribution(df: pd.DataFrame):
    plt.figure(figsize=(6, 4.5))
    order = df["satisfaction"].value_counts().index
    ax = sns.countplot(x="satisfaction", data=df, order=order, hue="satisfaction",
                        palette="viridis", legend=False)
    ax.set_title("Distribution of Customer Satisfaction")
    ax.set_xlabel("Satisfaction")
    ax.set_ylabel("Number of Passengers")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "satisfaction_distribution.png"), dpi=150)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include="number")
    plt.figure(figsize=(14, 11))
    corr = numeric_df.corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, linewidths=0.3)
    plt.title("Correlation Heatmap of Numeric Features")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()


def plot_class_vs_satisfaction(df: pd.DataFrame):
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(x="Class", hue="satisfaction", data=df, palette="mako")
    ax.set_title("Satisfaction by Travel Class")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Passengers")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "satisfaction_by_class.png"), dpi=150)
    plt.close()


def plot_service_ratings_vs_satisfaction(df: pd.DataFrame):
    present_cols = [c for c in SERVICE_RATING_COLS if c in df.columns]
    means = (
        df.groupby("satisfaction")[present_cols]
        .mean()
        .T.sort_values(by=df["satisfaction"].unique()[0])
    )
    means.plot(kind="barh", figsize=(9, 8), colormap="viridis")
    plt.title("Average Service Rating by Satisfaction Level")
    plt.xlabel("Average Rating (0-5)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "service_ratings_by_satisfaction.png"), dpi=150)
    plt.close()


def plot_delay_vs_satisfaction(df: pd.DataFrame):
    plt.figure(figsize=(7, 5))
    sns.boxplot(x="satisfaction", y="Total Delay in Minutes", data=df, showfliers=False,
                hue="satisfaction", palette="crest", legend=False)
    plt.title("Total Delay vs Satisfaction")
    plt.ylabel("Total Delay (Departure + Arrival, minutes)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "delay_vs_satisfaction.png"), dpi=150)
    plt.close()


def plot_age_distribution(df: pd.DataFrame):
    plt.figure(figsize=(7, 5))
    sns.histplot(data=df, x="Age", hue="satisfaction", kde=True, bins=30, palette="viridis", element="step")
    plt.title("Age Distribution by Satisfaction")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "age_distribution.png"), dpi=150)
    plt.close()


def run_eda(path: str = RAW_DATA_PATH):
    ensure_output_dir()
    df = load_raw_data(path)
    df = clean_data(df)
    df = engineer_features(df)

    plot_satisfaction_distribution(df)
    plot_correlation_heatmap(df)
    plot_class_vs_satisfaction(df)
    plot_service_ratings_vs_satisfaction(df)
    plot_delay_vs_satisfaction(df)
    plot_age_distribution(df)

    print(f"EDA plots saved to {OUTPUT_DIR}/")
    print(df.describe(include="all").T)


if __name__ == "__main__":
    run_eda()
