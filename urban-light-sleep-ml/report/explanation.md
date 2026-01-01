# Urban Light + Sleep ML — Code Overview

This document provides an end-to-end overview of the project, explaining what each script does and how data flows through the pipeline.

## 1) High-Level Architecture

This project combines two primary NYC-scale data sources on a shared spatial grid and builds a baseline ML model for **nighttime noise risk** classification:

- **311 (Noise - Residential)** complaints (point event data)
- **VIIRS Night Lights** (raster) nighttime light intensity

Join key: `cell_id` (grid cell identifier).

Primary outputs:
- Grid: `data/processed/grid_cells.geojson`
- 311 counts per grid cell: `data/processed/grid_311_counts.csv`
- VIIRS light summary per grid cell: `data/processed/grid_viirs_light.csv`
- Final modeling dataset: `data/processed/final_model_dataset.csv`

## 2) Data Pipeline

### A) Grid creation

Dosya: [src/data/create_grid.py](../src/data/create_grid.py)

What it does:
- Loads NYC borough boundaries.
- Reprojects to a meter-based CRS (`EPSG:32618`).
- Creates a 500m square fishnet over the bounding box.
- Clips the grid to NYC boundaries.
- Assigns a `cell_id` and computes centroid lat/lon for each cell.

Output:
- `data/processed/grid_cells.geojson`

### B) Download 311 data and apply nighttime filter

Dosya: [download_311_filtered.py](../download_311_filtered.py)

What it does:
- Downloads 311 data in chunks via Socrata (LIMIT/OFFSET).
- Keeps only `Noise - Residential` records from 2020 onward.
- Applies a nighttime filter: 22:00–06:00.

Output:
- `nyc_311_noise_night_2020_present.csv` (saved to the folder where the script is executed)

Note:
- This file should be placed under `urban-light-sleep-ml/data/raw/`.

### C) Aggregate 311 events to the grid

Dosya: [src/data/process_311.py](../src/data/process_311.py)

What it does:
- Loads `grid_cells.geojson` and `nyc_311_noise_night_2020_present.csv`.
- Converts events to a GeoDataFrame.
- Assigns each complaint to a grid cell via spatial join.
- Produces per-cell features:
  - `noise_night_count` (total number of complaints)
  - `peak_hour` (most frequent hour; -1 if unavailable)

Output:
- `data/processed/grid_311_counts.csv`

### D) Summarize VIIRS raster values per grid cell

Dosya: [src/data/extract_viirs.py](../src/data/extract_viirs.py)

What it does:
- Loads `grid_cells.geojson` and `NYC_VIIRS_Night_Lights_2020_present.tif`.
- Masks raster values per cell.
- Computes mean intensity excluding 0 and nodata values.

Output:
- `data/processed/grid_viirs_light.csv` (`cell_id`, `night_light_avg`)

### E) Merge datasets and create label

Dosya: [src/data/merge_datasets.py](../src/data/merge_datasets.py)

What it does:
- Merges 311 counts and VIIRS light by `cell_id`.
- Fills missing `night_light_avg` with the median.
- Creates the label:
  - `high_noise_risk = 1` if `noise_night_count` is at or above the 80th percentile
- Adds an extra feature:
  - `noise_night_count_log1p = log(1 + noise_night_count)`

Output:
- `data/processed/final_model_dataset.csv`

## 3) Modeling (Baseline)

### A) Baseline training and metrics

Dosya: [src/models/train_baseline.py](../src/models/train_baseline.py)

- Features:
  - `night_light_avg`
  - `centroid_lat`, `centroid_lon`
  - `peak_hour`
- Target:
  - `high_noise_risk`

Models:
- Logistic Regression (StandardScaler + LogisticRegression)
- Random Forest

Output:
- Prints Accuracy / F1 / ROC-AUC + classification_report to the console.

### B) Confusion matrix figures

Dosya: [src/models/plot_confusion_matrices.py](../src/models/plot_confusion_matrices.py)

Output:
- `outputs/figures/confusion_matrices/cm_logistic_regression.png`
- `outputs/figures/confusion_matrices/cm_random_forest.png`

## 4) Analysis / Visualization

### A) Light quantile analysis (table)

Dosya: [src/analysis/light_quantile_analysis.py](../src/analysis/light_quantile_analysis.py)

What it does:
- Splits `night_light_avg` into 5 quantiles.
- Produces summary statistics per quantile.

Output:
- `outputs/tables/light_quantile_summary.csv`

### B) Quantile figure

Dosya: [src/analysis/plot_light_quantile.py](../src/analysis/plot_light_quantile.py)

Output:
- `outputs/figures/light_quantile_vs_noise_risk.png`

### C) Scatter plot: light vs noise

Dosya: [src/analysis/plot_scatter_light_vs_noise.py](../src/analysis/plot_scatter_light_vs_noise.py)

Output:
- `outputs/figures/scatter_light_vs_noise.png`

### D) ROC curves

Dosya: [src/analysis/plot_roc_curve.py](../src/analysis/plot_roc_curve.py)

Output:
- `outputs/figures/roc_curve.png`

### E) RF feature importance

Dosya: [src/analysis/plot_feature_importance_rf.py](../src/analysis/plot_feature_importance_rf.py)

Output:
- `outputs/figures/rf_feature_importance.png`

### F) Class distribution

Dosya: [src/analysis/plot_class_distribution.py](../src/analysis/plot_class_distribution.py)

Output:
- `outputs/figures/class_distribution.png`

## 5) Dashboard

Dosya: [src/visualization/dashboard_app.py](../src/visualization/dashboard_app.py)

What it does:
- Joins `grid_cells.geojson` geometries with `final_model_dataset.csv`.
- Provides filtering via the Streamlit sidebar.
- Visualizes risk using centroids on a Folium map.

Run:
- `streamlit run src/visualization/dashboard_app.py`

## 6) Recommended Run Order

Following this order helps manage dependencies between scripts correctly:

1. Create grid: `python src/data/create_grid.py`
2. Download 311: `python download_311_filtered.py` → move output under `data/raw/`
3. Process 311: `python src/data/process_311.py`
4. Extract VIIRS: `python src/data/extract_viirs.py`
5. Merge datasets: `python src/data/merge_datasets.py`
6. Train baselines: `python src/models/train_baseline.py`
7. Figures/tables:
   - `python src/models/plot_confusion_matrices.py`
   - `python src/analysis/plot_roc_curve.py`
   - `python src/analysis/plot_feature_importance_rf.py`
   - `python src/analysis/plot_class_distribution.py`
   - `python src/analysis/plot_scatter_light_vs_noise.py`
   - `python src/analysis/light_quantile_analysis.py`
   - `python src/analysis/plot_light_quantile.py`

---

İstersen bir sonraki adım olarak: bu dokümanı “Rapor” formatına (giriş/veri/özellik/sonuçlar) daha akademik hale de getirebilirim.
