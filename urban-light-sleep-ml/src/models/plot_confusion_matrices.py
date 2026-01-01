"""src/models/plot_confusion_matrices.py

Generate confusion-matrix figures for baseline models.

Goal
- Train Logistic Regression and Random Forest after a stratified train/test split.
- Plot the confusion matrix as a seaborn heatmap and save PNG outputs.

Input
- data/processed/final_model_dataset.csv

Output
- outputs/figures/confusion_matrices/cm_logistic_regression.png
- outputs/figures/confusion_matrices/cm_random_forest.png

Note
- Confusion matrix convention: rows = true class, columns = predicted class.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final_model_dataset.csv"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures" / "confusion_matrices"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "night_light_avg",
    "centroid_lat",
    "centroid_lon",
    "peak_hour",
]
TARGET = "high_noise_risk"


def plot_cm(y_true, y_pred, title, filename):
    # Confusion matrix hesapla ve okunabilir bir heatmap olarak kaydet
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Low risk", "High risk"],
        yticklabels=["Low risk", "High risk"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def main():
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # 1) Logistic Regression (ölçekleme + lineer model)
    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    logreg.fit(X_train, y_train)
    y_pred_lr = logreg.predict(X_test)

    plot_cm(
        y_test,
        y_pred_lr,
        "Confusion Matrix - Logistic Regression",
        FIG_DIR / "cm_logistic_regression.png"
    )

    # 2) Random Forest (ağaç tabanlı)
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    plot_cm(
        y_test,
        y_pred_rf,
        "Confusion Matrix - Random Forest",
        FIG_DIR / "cm_random_forest.png"
    )

    print("✅ Confusion matrices saved:")
    print(FIG_DIR)


if __name__ == "__main__":
    main()
