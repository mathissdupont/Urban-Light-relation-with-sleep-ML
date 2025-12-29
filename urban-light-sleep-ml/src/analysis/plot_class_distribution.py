from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final_model_dataset.csv"
OUT_PATH = PROJECT_ROOT / "outputs" / "figures" / "class_distribution.png"

def main():
    df = pd.read_csv(DATA_PATH)
    counts = df["high_noise_risk"].value_counts().sort_index()

    plt.figure(figsize=(5,4))
    plt.bar(["Low risk (0)", "High risk (1)"], counts.values)
    plt.ylabel("Number of Grid Cells")
    plt.title("Class Distribution of Nighttime Noise Risk")
    for i, v in enumerate(counts.values):
        plt.text(i, v + max(counts.values)*0.01, str(int(v)), ha="center")
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=300)
    plt.close()
    print(f"✅ Saved class distribution: {OUT_PATH}")

if __name__ == "__main__":
    main()
