"""src/models/train_extended_models.py

Train and evaluate multiple classification models on the project's tabular dataset.

Models
- Logistic Regression (baseline)
- Random Forest (baseline)
- XGBoost (XGBClassifier) if available; otherwise GradientBoostingClassifier fallback
- K-Nearest Neighbors (GridSearchCV over a small hyperparameter grid)
- Gaussian Naive Bayes

Metrics (binary classification, positive class = 1)
- Accuracy
- Precision
- Recall (Sensitivity)
- Specificity (TN / (TN + FP))
- F1
- ROC-AUC (from predicted probabilities when available)

Inputs
- data/processed/final_model_dataset.csv

Outputs
- outputs/tables/metrics_all_models.csv
- outputs/figures/cm_<model>.png
- outputs/figures/roc_curve_all_models.png

Run from project root:
  python src/models/train_extended_models.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final_model_dataset.csv"

OUT_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
OUT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

TARGET_COL = "high_noise_risk"


def _make_one_hot_encoder() -> OneHotEncoder:
    """Create a OneHotEncoder compatible across sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # Older scikit-learn uses `sparse` instead of `sparse_output`
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def compute_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Specificity = TN / (TN + FP)."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    denom = tn + fp
    return float(tn / denom) if denom > 0 else 0.0


def plot_confusion_matrix_png(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    """Save a labeled confusion matrix PNG with counts."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0 (Low risk)", "1 (High risk)"])
    ax.set_yticklabels(["0 (Low risk)", "1 (High risk)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    # Annotate counts
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(int(value)), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def infer_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    """Infer feature columns, excluding obvious non-features."""
    drop_cols = {target_col, "geometry"}
    # Common identifier columns to avoid leaking IDs into the model
    for maybe_id in ["cell_id", "id", "ID"]:
        if maybe_id in df.columns:
            drop_cols.add(maybe_id)

    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Drop all-null columns
    feature_cols = [c for c in feature_cols if not df[c].isna().all()]

    if not feature_cols:
        raise ValueError("No feature columns found after filtering.")

    return feature_cols


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Build a ColumnTransformer with scaling for numeric and OHE for categorical."""
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_cols, categorical_cols


def safe_predict_proba(estimator: Any, X_test: pd.DataFrame) -> Optional[np.ndarray]:
    """Return P(y=1) scores if available; otherwise None."""
    if hasattr(estimator, "predict_proba"):
        prob = estimator.predict_proba(X_test)
        if prob is None:
            return None
        return np.asarray(prob)[:, 1]

    if hasattr(estimator, "decision_function"):
        # Convert decision scores to [0,1] via a sigmoid-like mapping
        scores = np.asarray(estimator.decision_function(X_test))
        # Numerical stability
        scores = np.clip(scores, -50, 50)
        return 1.0 / (1.0 + np.exp(-scores))

    return None


@dataclass
class ModelResult:
    name: str
    metrics: Dict[str, float]
    best_params: Optional[Dict[str, Any]] = None
    roc_points: Optional[Tuple[np.ndarray, np.ndarray, float]] = None  # (fpr, tpr, auc)


def evaluate_model(
    name: str,
    estimator: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    figures_dir: Path,
) -> ModelResult:
    estimator.fit(X_train, y_train)

    y_pred = estimator.predict(X_test)
    y_score = safe_predict_proba(estimator, X_test)

    metrics: Dict[str, float] = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "Specificity": float(compute_specificity(y_test.to_numpy(), np.asarray(y_pred))),
        "F1": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "ROC_AUC": float("nan"),
    }

    roc_points: Optional[Tuple[np.ndarray, np.ndarray, float]] = None
    if y_score is not None:
        auc = float(roc_auc_score(y_test, y_score))
        fpr, tpr, _ = roc_curve(y_test, y_score)
        metrics["ROC_AUC"] = auc
        roc_points = (fpr, tpr, auc)

    # Confusion matrix
    cm_path = figures_dir / f"cm_{slugify_model_name(name)}.png"
    plot_confusion_matrix_png(
        y_true=y_test.to_numpy(),
        y_pred=np.asarray(y_pred),
        title=f"Confusion Matrix — {name}",
        out_path=cm_path,
    )

    best_params = getattr(estimator, "best_params_", None)
    return ModelResult(name=name, metrics=metrics, best_params=best_params, roc_points=roc_points)


def slugify_model_name(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )


def plot_roc_curves(results: List[ModelResult], out_path: Path) -> None:
    plt.figure(figsize=(7.2, 5.2))

    any_plotted = False
    for r in results:
        if r.roc_points is None:
            continue
        fpr, tpr, auc = r.roc_points
        plt.plot(fpr, tpr, label=f"{r.name} (AUC={auc:.3f})")
        any_plotted = True

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random (AUC=0.500)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — All Models")
    plt.grid(alpha=0.3)
    if any_plotted:
        plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    # --- I/O prep
    OUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    print(f"📄 Loading dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset columns.")

    feature_cols = infer_feature_columns(df, TARGET_COL)
    X = df[feature_cols]
    y = df[TARGET_COL]

    print(f"✅ Dataset shape: {df.shape}")
    print(f"✅ Features used ({len(feature_cols)}): {feature_cols}")

    # Class distribution
    class_counts = y.value_counts(dropna=False).sort_index()
    class_props = (class_counts / len(y)).round(4)
    print("📊 Class distribution (full):")
    for cls in class_counts.index.tolist():
        print(f"   y={cls}: n={int(class_counts.loc[cls])} ({float(class_props.loc[cls]):.4f})")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    print(f"✅ Train size: {X_train.shape}, Test size: {X_test.shape}")

    train_counts = y_train.value_counts().sort_index()
    test_counts = y_test.value_counts().sort_index()
    print("📊 Class distribution (train/test):")
    print(f"   Train: {train_counts.to_dict()}")
    print(f"   Test : {test_counts.to_dict()}")

    # scale_pos_weight for XGBoost from training data
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = float(neg / pos) if pos > 0 else 1.0

    # Preprocessing
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train)
    print(f"🧩 Numeric columns ({len(numeric_cols)}): {numeric_cols}")
    print(f"🧩 Categorical columns ({len(categorical_cols)}): {categorical_cols}")

    # --- Model definitions
    models: List[Tuple[str, Any]] = []

    # Baselines for comparison
    lr = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )
    models.append(("Logistic Regression", lr))

    rf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    models.append(("Random Forest", rf))

    # XGBoost (or fallback)
    used_xgb = False
    try:
        from xgboost import XGBClassifier  # type: ignore

        xgb = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=400,
                        learning_rate=0.05,
                        max_depth=4,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.0,
                        random_state=42,
                        eval_metric="logloss",
                        scale_pos_weight=scale_pos_weight,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        models.append(("XGBoost", xgb))
        used_xgb = True
        print(f"🧠 Using XGBoost with scale_pos_weight={scale_pos_weight:.3f}")

    except ImportError:
        print("⚠️  xgboost is not installed. Falling back to GradientBoostingClassifier.")
        gb = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "clf",
                    GradientBoostingClassifier(random_state=42),
                ),
            ]
        )
        models.append(("GradBoost", gb))

    # KNN with small search (scaling is enforced by the preprocessor)
    knn_pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", KNeighborsClassifier()),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    knn_grid = GridSearchCV(
        estimator=knn_pipe,
        param_grid={
            "clf__n_neighbors": [3, 5, 7, 11],
            "clf__weights": ["uniform", "distance"],
        },
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    models.append(("KNN (GridSearch)", knn_grid))

    # Gaussian Naive Bayes
    gnb = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", GaussianNB()),
        ]
    )
    models.append(("GaussianNB", gnb))

    # --- Train + evaluate
    results: List[ModelResult] = []
    print("\n🚀 Training / evaluating models...")
    for name, estimator in models:
        print(f"\n--- {name} ---")
        # Print params (lightweight)
        try:
            params = estimator.get_params(deep=False)
            # Keep it readable
            keys = [k for k in params.keys() if k in {"clf", "n_jobs", "scoring"}]
            if keys:
                print("Params:", {k: params[k] for k in keys})
        except Exception:
            pass

        r = evaluate_model(
            name=name,
            estimator=estimator,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            figures_dir=OUT_FIGURES_DIR,
        )
        results.append(r)

        if r.best_params:
            print(f"Best params: {r.best_params}")

        print(
            "Metrics:",
            {k: (f"{v:.3f}" if np.isfinite(v) else "NA") for k, v in r.metrics.items()},
        )
        print(f"Saved CM: {OUT_FIGURES_DIR / f'cm_{slugify_model_name(name)}.png'}")

    # --- Save metrics table
    rows: List[Dict[str, Any]] = []
    for r in results:
        row = {"Model": r.name}
        row.update(r.metrics)
        rows.append(row)

    metrics_df = pd.DataFrame(rows)

    # Sort by ROC_AUC (NaN last)
    metrics_df["_roc_sort"] = metrics_df["ROC_AUC"].fillna(-1)
    metrics_df = metrics_df.sort_values("_roc_sort", ascending=False).drop(columns=["_roc_sort"])

    out_metrics_csv = OUT_TABLES_DIR / "metrics_all_models.csv"
    metrics_df.to_csv(out_metrics_csv, index=False)

    print(f"\n✅ Saved metrics table: {out_metrics_csv}")

    # --- Save combined ROC plot
    out_roc_png = OUT_FIGURES_DIR / "roc_curve_all_models.png"
    plot_roc_curves(results, out_roc_png)
    print(f"✅ Saved ROC plot: {out_roc_png}")

    if used_xgb:
        print("ℹ️  XGBoost was used (xgboost installed).")
    else:
        print("ℹ️  GradientBoosting fallback was used.")


if __name__ == "__main__":
    main()
