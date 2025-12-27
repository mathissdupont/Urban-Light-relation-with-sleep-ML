from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import box


# -----------------------------
# Config
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

BOROUGHS_GEOJSON = RAW_DIR / "new-york-city-boroughs.geojson"

OUT_GRID_GEOJSON = PROCESSED_DIR / "grid_cells.geojson"

# Grid size in meters (VIIRS is ~500m resolution-ish, so 500m grid is a good match)
GRID_SIZE_M = 500

# Use a projected CRS in meters for NYC area.
# EPSG:3857 is okay; for more accurate local distances use UTM Zone 18N (EPSG:32618).
METRIC_CRS = "EPSG:32618"


def create_fishnet(bounds, cell_size_m: int):
    """
    Create grid polygons (fishnet) over bounds in a metric CRS.
    bounds = (minx, miny, maxx, maxy)
    """
    minx, miny, maxx, maxy = bounds

    # snap bounds to grid
    minx = math.floor(minx / cell_size_m) * cell_size_m
    miny = math.floor(miny / cell_size_m) * cell_size_m
    maxx = math.ceil(maxx / cell_size_m) * cell_size_m
    maxy = math.ceil(maxy / cell_size_m) * cell_size_m

    xs = np.arange(minx, maxx, cell_size_m)
    ys = np.arange(miny, maxy, cell_size_m)

    cells = []
    cell_id = 0
    for x in xs:
        for y in ys:
            geom = box(x, y, x + cell_size_m, y + cell_size_m)
            cells.append({"cell_id": cell_id, "geometry": geom})
            cell_id += 1

    return gpd.GeoDataFrame(cells, crs=METRIC_CRS)


def main():
    if not BOROUGHS_GEOJSON.exists():
        raise FileNotFoundError(f"Missing file: {BOROUGHS_GEOJSON}")

    print(f"Reading boroughs: {BOROUGHS_GEOJSON}")
    boroughs = gpd.read_file(BOROUGHS_GEOJSON)

    # Ensure valid geometry
    boroughs = boroughs[boroughs.geometry.notnull()].copy()
    boroughs["geometry"] = boroughs["geometry"].buffer(0)

    # Reproject to metric CRS
    boroughs_m = boroughs.to_crs(METRIC_CRS)

    # Union all borough polygons (NYC boundary)
    nyc_union = boroughs_m.unary_union
    nyc_bounds = nyc_union.bounds

    print("Creating fishnet grid...")
    grid = create_fishnet(nyc_bounds, GRID_SIZE_M)

    # Clip grid to NYC boundary
    print("Clipping grid to NYC boundary...")
    grid_clipped = gpd.clip(grid, nyc_union)

    # Remove tiny slivers (optional) - keep only cells with meaningful area
    grid_clipped["area_m2"] = grid_clipped.geometry.area
    grid_clipped = grid_clipped[grid_clipped["area_m2"] >= (GRID_SIZE_M * GRID_SIZE_M * 0.20)].copy()

    # Add centroid lat/lon for easy joins/debugging later
    centroids = grid_clipped.geometry.centroid
    grid_clipped["centroid_x"] = centroids.x
    grid_clipped["centroid_y"] = centroids.y

    # Also store WGS84 centroids (lat/lon)
    grid_wgs84 = grid_clipped.to_crs("EPSG:4326")
    centroids_wgs = grid_wgs84.geometry.centroid
    grid_clipped["centroid_lon"] = centroids_wgs.x
    grid_clipped["centroid_lat"] = centroids_wgs.y

    # Save
    print(f"Saving grid: {OUT_GRID_GEOJSON}")
    grid_clipped.drop(columns=["area_m2"]).to_file(OUT_GRID_GEOJSON, driver="GeoJSON")

    print("✅ Done.")
    print(f"Grid cells saved to: {OUT_GRID_GEOJSON}")
    print(f"Grid cell count: {len(grid_clipped)}")


if __name__ == "__main__":
    main()
