"""src/analysis/plot_feature_importance_rf.py

Bu script, Random Forest modelinin feature importance değerlerini görselleştirir.

Amaç
- Aynı feature setiyle RandomForest eğitmek.
- `feature_importances_` değerlerini sıralayıp bar chart çizmek.

Girdi
- data/processed/final_model_dataset.csv

Çıktı
- outputs/figures/rf_feature_importance.png

Not
- Feature importance, ağaçlarda kullanılan split kazançlarından türetilen göreli bir ölçüdür.
    Nedensellik iddiası değildir; sadece modelin hangi feature'ı daha çok kullandığını gösterir.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final_model_dataset.csv"
OUT_PATH = PROJECT_ROOT / "outputs" / "figures" / "rf_feature_importance.png"

FEATURES = ["night_light_avg", "centroid_lat", "centroid_lon", "peak_hour"]
TARGET = "high_noise_risk"

def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Random Forest eğitimi
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # Feature importance -> büyükten küçüğe sırala
    importances = rf.feature_importances_
    pairs = sorted(zip(FEATURES, importances), key=lambda x: x[1], reverse=True)
    labels = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]

    plt.figure(figsize=(6.5,4.5))
    plt.bar(labels, vals)
    plt.ylabel("Feature Importance")
    plt.title("Random Forest Feature Importance")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=300)
    plt.close()

    print(f"✅ Saved feature importance: {OUT_PATH}")
    print("Order:", pairs)

if __name__ == "__main__":
    main()
