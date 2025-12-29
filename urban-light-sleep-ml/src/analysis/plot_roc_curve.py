"""src/analysis/plot_roc_curve.py

Bu script, baseline modellerin ROC eğrilerini tek grafikte karşılaştırır.

Amaç
- Logistic Regression ve Random Forest için `predict_proba` ile skor üretmek.
- ROC curve (FPR vs TPR) çizmek ve AUC değerlerini legend'da göstermek.

Girdi
- data/processed/final_model_dataset.csv

Çıktı
- outputs/figures/roc_curve.png

Not
- `stratify=y` ile train/test sınıf oranı korunur.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final_model_dataset.csv"
OUT_PATH = PROJECT_ROOT / "outputs" / "figures" / "roc_curve.png"

FEATURES = ["night_light_avg", "centroid_lat", "centroid_lon", "peak_hour"]
TARGET = "high_noise_risk"

def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 1) Logistic Regression (ölçekleme + lineer)
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    lr.fit(X_train, y_train)
    lr_prob = lr.predict_proba(X_test)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_prob)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)

    # 2) Random Forest (ağaç tabanlı)
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_prob)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)

    # ROC plot: rastgele sınıflandırıcı referansı (diagonal)
    plt.figure(figsize=(7,5))
    plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC={lr_auc:.2f})")
    plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={rf_auc:.2f})")
    plt.plot([0,1], [0,1], linestyle="--", label="Random (AUC=0.50)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Nighttime Noise Risk Classification")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=300)
    plt.close()

    print(f"✅ Saved ROC curve: {OUT_PATH}")

if __name__ == "__main__":
    main()
