# Urban Light + Sleep ML

This project builds a simple, reproducible ML pipeline (CRISP-DM style) to classify **nighttime noise risk** in NYC grid cells using features derived from:
- NYC **311 Noise - Residential** complaints (nighttime filtered)
- **VIIRS Night Lights** intensity

## How to run

Run commands from the project root folder (`urban-light-sleep-ml/`).

### Train baseline models

- `python src/models/train_baseline.py`

### Train extended model set (paper-ready metrics + plots)

This script trains and evaluates:
- Logistic Regression, Random Forest (for comparison)
- XGBoost (if installed) or GradientBoosting (fallback)
- KNN (GridSearchCV)
- GaussianNB

Run:
- `python src/models/train_extended_models.py`

Outputs:
- `outputs/tables/metrics_all_models.csv`
- `outputs/figures/cm_<model>.png`
- `outputs/figures/roc_curve_all_models.png`
