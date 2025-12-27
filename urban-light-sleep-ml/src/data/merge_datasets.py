from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

IN_COUNTS = PROCESSED_DIR / "grid_311_counts.csv"
IN_LIGHT = PROCESSED_DIR / "grid_viirs_light.csv"

OUT_FINAL = PROCESSED_DIR / "final_model_dataset.csv"


def main():
    if not IN_COUNTS.exists():
        raise FileNotFoundError(f"Missing: {IN_COUNTS}")
    if not IN_LIGHT.exists():
        raise FileNotFoundError(f"Missing: {IN_LIGHT}")

    counts = pd.read_csv(IN_COUNTS)
    light = pd.read_csv(IN_LIGHT)

    # Merge on cell_id
    df = counts.merge(light, on="cell_id", how="left")

    # Basic cleaning
    df["night_light_avg"] = pd.to_numeric(df["night_light_avg"], errors="coerce")

    # Fill missing light with 0 (outside coverage) OR median.
    # For VIIRS, missing usually means no data. We'll use median to be safer.
    median_light = df["night_light_avg"].median(skipna=True)
    df["night_light_avg"] = df["night_light_avg"].fillna(median_light)

    # Target engineering:
    # We will create a binary "high_noise_risk" label using top 20% complaint counts.
    # This avoids heavy imbalance and makes classification meaningful.
    q80 = df["noise_night_count"].quantile(0.80)
    df["high_noise_risk"] = (df["noise_night_count"] >= q80).astype(int)

    # Also keep log-transformed count as a regression-friendly target (optional)
    df["noise_night_count_log1p"] = np.log1p(df["noise_night_count"])

    # Save
    df.to_csv(OUT_FINAL, index=False)

    print("✅ Done.")
    print(f"Saved: {OUT_FINAL}")
    print(f"Rows: {len(df)}")
    print(f"High risk threshold (80th percentile) = {q80:.0f}")
    print(df["high_noise_risk"].value_counts())


if __name__ == "__main__":
    main()
