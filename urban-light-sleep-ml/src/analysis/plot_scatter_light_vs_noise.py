from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final_model_dataset.csv"
OUT_PATH = PROJECT_ROOT / "outputs" / "figures" / "scatter_light_vs_noise.png"

def main():
    df = pd.read_csv(DATA_PATH)

    x = df["night_light_avg"].astype(float)
    y = df["noise_night_count"].astype(float)

    # log scale for y to reduce skew
    y_log = np.log1p(y)

    plt.figure(figsize=(7,5))
    plt.scatter(x, y_log, s=8, alpha=0.35)
    plt.xlabel("Nighttime Light Intensity (VIIRS avg radiance)")
    plt.ylabel("log(1 + Nighttime Noise Complaints)")
    plt.title("Nighttime Light Intensity vs Nighttime Noise Complaints")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=300)
    plt.close()

    print(f"✅ Saved scatter: {OUT_PATH}")

if __name__ == "__main__":
    main()
