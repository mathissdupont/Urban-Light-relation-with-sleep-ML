from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

GRID_PATH = PROCESSED_DIR / "grid_cells.geojson"
VIIRS_TIF = RAW_DIR / "NYC_VIIRS_Night_Lights_2020_present.tif"

OUT_LIGHT = PROCESSED_DIR / "grid_viirs_light.csv"

# CRS used for grid
METRIC_CRS = "EPSG:32618"


def zonal_mean(raster, geom):
    """Compute mean raster value within a geometry."""
    try:
        out_image, out_transform = mask(
            raster,
            [geom],
            crop=True,
            filled=True,
            nodata=raster.nodata,
        )
        data = out_image[0]
        data = data[data > 0]  # ignore zero / nodata
        if data.size == 0:
            return np.nan
        return float(np.mean(data))
    except Exception:
        return np.nan


def main():
    if not GRID_PATH.exists():
        raise FileNotFoundError(f"Missing grid: {GRID_PATH}")
    if not VIIRS_TIF.exists():
        raise FileNotFoundError(f"Missing VIIRS tif: {VIIRS_TIF}")

    print(f"Reading grid: {GRID_PATH}")
    grid = gpd.read_file(GRID_PATH).to_crs(METRIC_CRS)

    print(f"Opening VIIRS raster: {VIIRS_TIF}")
    with rasterio.open(VIIRS_TIF) as src:

        # Reproject grid to raster CRS if needed
        if grid.crs != src.crs:
            grid = grid.to_crs(src.crs)

        light_vals = []
        for idx, row in grid.iterrows():
            mean_val = zonal_mean(src, row.geometry)
            light_vals.append(mean_val)

    out = pd.DataFrame({
        "cell_id": grid["cell_id"].values,
        "night_light_avg": light_vals
    })

    print(f"Saving: {OUT_LIGHT}")
    out.to_csv(OUT_LIGHT, index=False)

    print("✅ Done.")
    print(f"Saved VIIRS grid light values to: {OUT_LIGHT}")
    print(f"Non-null light cells: {out['night_light_avg'].notna().sum()} / {len(out)}")


if __name__ == "__main__":
    main()
