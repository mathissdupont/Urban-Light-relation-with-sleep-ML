"""src/models/train_baseline.py

Bu script, nihai veri seti üzerinde iki temel (baseline) sınıflandırıcı eğitir ve metriklerini raporlar.

Amaç
- `final_model_dataset.csv` içinden feature/target seçip train/test split yapmak.
- Baseline olarak:
    1) Logistic Regression (ölçekleme + lineer model)
    2) Random Forest (non-lineer, ağaç tabanlı model)
- Accuracy, F1 ve ROC-AUC metriklerini yazdırmak.

Girdi
- data/processed/final_model_dataset.csv

Çıktı
- Konsola metrik tablosu + classification_report.

Notlar
- Logistic Regression için StandardScaler kullanılır.
- `stratify=y` ile sınıf oranları train/test'te korunur.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final_model_dataset.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    # Modelde kullanılan feature seti:
    # - night_light_avg: VIIRS ortalama ışık
    # - centroid_lat/lon: mekansal konum bilgisi (kabaca mahalle etkileri)
    # - peak_hour: hücrede şikayetlerin en yoğun olduğu saat (yoksa -1)
    FEATURES = [
        "night_light_avg",
        "centroid_lat",
        "centroid_lon",
        "peak_hour",
    ]
    TARGET = "high_noise_risk"

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # -------------------
    # 1) Logistic Regression
    # -------------------
    # Ölçekleme + lineer model pipeline'ı.
    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    logreg.fit(X_train, y_train)
    y_pred_lr = logreg.predict(X_test)
    y_prob_lr = logreg.predict_proba(X_test)[:, 1]

    # -------------------
    # 2) Random Forest
    # -------------------
    # Ağaç tabanlı model ölçeklemeye ihtiyaç duymaz.
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]

    # -------------------
    # Değerlendirme
    # -------------------
    # ROC-AUC için sınıf olasılıklarını (predict_proba) kullanıyoruz.
    results = {
        "Model": ["LogisticRegression", "RandomForest"],
        "Accuracy": [
            accuracy_score(y_test, y_pred_lr),
            accuracy_score(y_test, y_pred_rf),
        ],
        "F1": [
            f1_score(y_test, y_pred_lr),
            f1_score(y_test, y_pred_rf),
        ],
        "ROC_AUC": [
            roc_auc_score(y_test, y_prob_lr),
            roc_auc_score(y_test, y_prob_rf),
        ],
    }

    res_df = pd.DataFrame(results)
    print(res_df)

    print("\nLogistic Regression Report")
    print(classification_report(y_test, y_pred_lr))

    print("\nRandom Forest Report")
    print(classification_report(y_test, y_pred_rf))


if __name__ == "__main__":
    main()
