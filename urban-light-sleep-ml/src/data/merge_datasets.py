"""src/data/merge_datasets.py

Bu script, model eğitiminde kullanılacak nihai tabloyu üretir.

Amaç
- 311 şikayet sayımları (grid_311_counts.csv) ile VIIRS gece ışığı özetlerini (grid_viirs_light.csv)
    `cell_id` üzerinden birleştirmek.
- Eksik değerleri temizlemek/doldurmak.
- Sınıflandırma için hedef label üretmek: `high_noise_risk`.

Girdiler
- data/processed/grid_311_counts.csv
- data/processed/grid_viirs_light.csv

Çıktı
- data/processed/final_model_dataset.csv

Hedef (label) tanımı
- `high_noise_risk = 1` : hücrenin `noise_night_count` değeri üst %20'lik dilimdeyse.
    (80. persentil eşiği)

Notlar
- `night_light_avg` eksikse median ile doldurulur (VIIRS coverage / nodata etkilerini yumuşatmak için).
- Ek olarak regresyon için kullanılabilecek `noise_night_count_log1p` kolonu da eklenir.
"""

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

    # `cell_id` ortak anahtar: 311 sayımları + ışık özetleri
    df = counts.merge(light, on="cell_id", how="left")

    # Tip dönüşümü / temel temizlik
    df["night_light_avg"] = pd.to_numeric(df["night_light_avg"], errors="coerce")

    # Işık verisi eksikse doldurma stratejisi:
    # - 0 ile doldurmak coverage dışını "karanlık" varsayar.
    # - Median ile doldurmak daha muhafazakar bir yaklaşım (uç değer etkisini azaltır).
    # Burada median kullanıyoruz.
    median_light = df["night_light_avg"].median(skipna=True)
    df["night_light_avg"] = df["night_light_avg"].fillna(median_light)

    # Label üretimi:
    # Üst %20'lik şikayet yoğunluğunu "yüksek risk" olarak işaretle.
    # Böylece sınıf dengesizliği yönetilebilir seviyede kalır ve sınıflandırma anlamlı olur.
    q80 = df["noise_night_count"].quantile(0.80)
    df["high_noise_risk"] = (df["noise_night_count"] >= q80).astype(int)

    # Regresyon / görselleştirme için log(1+x) dönüşümü (sağa çarpıklığı azaltır)
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
