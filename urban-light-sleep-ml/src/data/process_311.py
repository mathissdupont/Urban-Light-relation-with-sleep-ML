"""src/data/process_311.py

Bu script, NYC 311 (Noise - Residential) kayıtlarını grid hücrelerine sayısal feature'lara dönüştürür.

Amaç
- Ham 311 CSV'yi okuyup minimum temizlik yapmak (lat/lon/tarih zorunlu).
- Noktasal şikayetleri grid hücrelerine mekansal join ile atamak.
- Her grid hücresi için toplam gece şikayet sayısını üretmek (`noise_night_count`).
- Ek, dayanıklı bir zaman özeti olarak hücre bazında en sık görülen saat bilgisini üretmek (`peak_hour`).

Girdiler
- data/processed/grid_cells.geojson (create_grid.py çıktısı)
- data/raw/nyc_311_noise_night_2020_present.csv (download_311_filtered.py ile indirilen/gece filtreli CSV)

Çıktı
- data/processed/grid_311_counts.csv

Önemli not
- Mekansal işlemler (grid + noktalar) metre CRS'te yapılır: `EPSG:32618`.
- `peak_hour` hiç şikayet olmayan hücrelerde -1 olarak doldurulur.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

GRID_PATH = PROCESSED_DIR / "grid_cells.geojson"
NYC_311_PATH = RAW_DIR / "nyc_311_noise_night_2020_present.csv"

OUT_COUNTS = PROCESSED_DIR / "grid_311_counts.csv"

# Same metric CRS used in grid generation
METRIC_CRS = "EPSG:32618"


def main():
    if not GRID_PATH.exists():
        raise FileNotFoundError(f"Missing grid: {GRID_PATH}")
    if not NYC_311_PATH.exists():
        raise FileNotFoundError(f"Missing 311 CSV: {NYC_311_PATH}")

    print(f"Reading grid: {GRID_PATH}")
    grid = gpd.read_file(GRID_PATH).to_crs(METRIC_CRS)

    print(f"Reading 311: {NYC_311_PATH}")
    df = pd.read_csv(NYC_311_PATH)

    # Temel temizlik:
    # - konum ve tarih alanları boş olanları at
    # - tarih parse edilemeyenleri ele
    df = df.dropna(subset=["latitude", "longitude", "created_date"]).copy()
    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    df = df.dropna(subset=["created_date"])

    # Opsiyonel: borough kolonu varsa boş borough'ları ele
    if "borough" in df.columns:
        df = df[df["borough"].notna()]

    # Noktaları GeoDataFrame'e çevir:
    # - ham lat/lon ile CRS=EPSG:4326
    # - sonra grid ile uyum için METRIC_CRS'e projeksiyon
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    ).to_crs(METRIC_CRS)

    # Mekansal join: her şikayet noktasını içinde kaldığı grid hücresine ata
    print("Spatial join: assigning each complaint to a grid cell...")
    joined = gpd.sjoin(
        gdf[["created_date", "geometry"]],
        grid[["cell_id", "geometry"]],
        how="inner",
        predicate="within",
    )

    # Hücre bazında toplam şikayet sayısı
    print("Aggregating counts per cell...")
    counts = joined.groupby("cell_id").size().reset_index(name="noise_night_count")

    # Zaman bazlı yardımcı kolonlar (feature mühendisliği):
    # Bu scriptte sadece `peak_hour` üretmek için kullanılıyor.
    joined["hour"] = joined["created_date"].dt.hour
    joined["year"] = joined["created_date"].dt.year
    joined["month"] = joined["created_date"].dt.month

    # Hücre bazında saat histogramı -> en yoğun saat (peak_hour)
    # Not: İleri geliştirmede ay/yıl bazlı time-aware feature'lar da üretilebilir.
    hourly = joined.groupby(["cell_id", "hour"]).size().reset_index(name="cnt")
    hour_peak = hourly.sort_values(["cell_id", "cnt"], ascending=[True, False]).drop_duplicates("cell_id")
    hour_peak = hour_peak[["cell_id", "hour"]].rename(columns={"hour": "peak_hour"})

    # Grid'e geri merge:
    # - her hücre için centroid lat/lon'ı koruyoruz
    # - sayım olmayan hücreleri 0 ile dolduruyoruz
    out = grid[["cell_id", "centroid_lat", "centroid_lon"]].merge(counts, on="cell_id", how="left")
    out = out.merge(hour_peak, on="cell_id", how="left")
    out["noise_night_count"] = out["noise_night_count"].fillna(0).astype(int)

    # Şikayet sayısı 0 olan hücrelerde peak_hour NaN kalır; -1 ile işaretle
    out["peak_hour"] = out["peak_hour"].fillna(-1).astype(int)

    print(f"Saving: {OUT_COUNTS}")
    out.to_csv(OUT_COUNTS, index=False)

    print("✅ Done.")
    print(f"Saved grid counts to: {OUT_COUNTS}")
    print(f"Cells with >=1 complaint: {(out['noise_night_count']>0).sum()} / {len(out)}")


if __name__ == "__main__":
    main()
