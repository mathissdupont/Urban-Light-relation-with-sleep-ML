from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "outputs" / "tables" / "light_quantile_summary.csv"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Sıra garantisi
order = ["Very Low", "Low", "Medium", "High", "Very High"]
df["light_quantile"] = pd.Categorical(df["light_quantile"], categories=order, ordered=True)
df = df.sort_values("light_quantile")

# Grafik
plt.figure(figsize=(8, 5))
bars = plt.bar(
    df["light_quantile"],
    df["high_risk_ratio"],
)

# Bar üstü değerler
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 1,
        f"{height:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.ylabel("High Noise Risk Ratio (%)")
plt.xlabel("Nighttime Light Intensity Quantile")
plt.title("Relationship Between Nighttime Light Intensity and Noise Risk")

plt.ylim(0, max(df["high_risk_ratio"]) * 1.15)
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
out_path = FIG_DIR / "light_quantile_vs_noise_risk.png"
plt.savefig(out_path, dpi=300)
plt.close()

print(f"Saved figure to {out_path}")
