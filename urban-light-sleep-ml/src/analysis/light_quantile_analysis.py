from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final_model_dataset.csv"
OUT_PATH = PROJECT_ROOT / "outputs" / "tables" / "light_quantile_summary.csv"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# 1) Night light quantiles (5 seviye)
df["light_quantile"] = pd.qcut(
    df["night_light_avg"],
    q=5,
    labels=["Very Low", "Low", "Medium", "High", "Very High"]
)

# 2) Her quantile için özet istatistikler
summary = (
    df
    .groupby("light_quantile")
    .agg(
        cells=("cell_id", "count"),
        avg_light=("night_light_avg", "mean"),
        avg_noise_count=("noise_night_count", "mean"),
        high_risk_ratio=("high_noise_risk", "mean")
    )
    .reset_index()
)

summary["high_risk_ratio"] = summary["high_risk_ratio"] * 100

print(summary)

summary.to_csv(OUT_PATH, index=False)
print(f"\nSaved summary to {OUT_PATH}")
