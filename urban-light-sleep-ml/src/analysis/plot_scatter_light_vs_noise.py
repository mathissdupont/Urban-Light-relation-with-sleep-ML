"""src/analysis/plot_scatter_light_vs_noise.py

Bu script, grid hücresi bazında gece ışığı ile gece şikayet yoğunluğu arasındaki ilişkiyi dağılım grafiğiyle gösterir.

Amaç
- X ekseni: `night_light_avg` (VIIRS ortalama radiance)
- Y ekseni: `log(1 + noise_night_count)` (şikayet sayısı çok çarpık olduğu için log ölçek)

Girdi
- data/processed/final_model_dataset.csv

Çıktı
- outputs/figures/scatter_light_vs_noise.png
"""

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

    # Y çok sağa çarpık olduğu için log(1+y) ile sıkıştırıyoruz.
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
