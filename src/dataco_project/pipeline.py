"""Leakage-aware data preparation and baseline modeling helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TARGET = "Late_delivery_risk"
DATE = "order date (DateOrders)"
ORDER_ID = "Order Id"
POST_OUTCOME_OR_ID = {
    TARGET,
    "Days for shipping (real)",
    "Delivery Status",
    "shipping date (DateOrders)",
    "Order Status",
    "Order Id",
    "Order Item Id",
    "Order Customer Id",
    "Customer Id",
    "Customer Email",
    "Customer Fname",
    "Customer Lname",
    "Customer Password",
    "Customer Street",
    "Customer Zipcode",
    "Order Zipcode",
    "Product Description",
    "Product Image",
}
MODEL_COLUMNS = [
    TARGET, DATE, ORDER_ID,
    "Shipping Mode", "Market", "Order Region", "Category Name", "Customer Segment",
    "Days for shipment (scheduled)", "Order Item Quantity", "Order Item Product Price",
]


def load_data(path: str | Path, usecols: list[str] | None = None) -> pd.DataFrame:
    """Read the dataset, accommodating common Kaggle CSV encodings."""
    path = Path(path)
    errors: list[str] = []
    for encoding in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, usecols=usecols, low_memory=True)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError("Could not decode CSV. " + " | ".join(errors))


def clean_target(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET not in df.columns:
        raise ValueError(f"Required target column {TARGET!r} is missing.")
    result = df.copy()
    result[TARGET] = pd.to_numeric(result[TARGET], errors="coerce")
    result = result[result[TARGET].isin([0, 1])].copy()
    result[TARGET] = result[TARGET].astype("int8")
    if DATE in result.columns:
        result[DATE] = pd.to_datetime(result[DATE], errors="coerce")
    return result


def build_audit(df: pd.DataFrame) -> dict[str, Any]:
    data = clean_target(df)
    summary: dict[str, Any] = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_rows_after_valid_target_filter": int(len(data)),
        "target_rate": float(data[TARGET].mean()) if len(data) else None,
        "target_counts": {str(k): int(v) for k, v in data[TARGET].value_counts().sort_index().items()},
        "missing_cells": int(df.isna().sum().sum()),
        "columns_all_missing": [str(c) for c in df.columns[df.isna().all()]],
        "columns_with_missing_values": int((df.isna().sum() > 0).sum()),
        "date_min": str(data[DATE].min()) if DATE in data and data[DATE].notna().any() else None,
        "date_max": str(data[DATE].max()) if DATE in data and data[DATE].notna().any() else None,
        "unique_orders": int(df[ORDER_ID].nunique(dropna=True)) if ORDER_ID in df else None,
    }
    if "Delivery Status" in data:
        summary["target_matches_late_delivery_status"] = bool(
            (data[TARGET] == data["Delivery Status"].eq("Late delivery").astype("int8")).all()
        )
    return summary


def chronological_group_split(df: pd.DataFrame, test_fraction: float = 0.2):
    """Split by order date; ensure an Order Id cannot cross the cutoff."""
    data = clean_target(df)
    required = [DATE, ORDER_ID]
    missing = [c for c in required if c not in data]
    if missing:
        raise ValueError(f"Chronological grouped split requires columns: {missing}")
    data = data[data[DATE].notna() & data[ORDER_ID].notna()].copy()
    order_dates = data.groupby(ORDER_ID, observed=True)[DATE].min().sort_values()
    if order_dates.nunique() < 3:
        raise ValueError("Not enough distinct order dates for a chronological split.")
    cutoff_pos = max(1, min(len(order_dates) - 1, int(np.floor(len(order_dates) * (1 - test_fraction)))))
    cutoff = order_dates.iloc[cutoff_pos]
    train_ids = set(order_dates[order_dates < cutoff].index)
    test_ids = set(order_dates[order_dates >= cutoff].index)
    train = data[data[ORDER_ID].isin(train_ids)].copy()
    test = data[data[ORDER_ID].isin(test_ids)].copy()
    if train.empty or test.empty or train[ORDER_ID].isin(test[ORDER_ID]).any():
        raise ValueError("Could not create non-overlapping chronological order split.")
    return train, test, cutoff


def make_preprocessor(X: pd.DataFrame, scale_numeric: bool, sparse_output: bool = True):
    # Keep scikit-learn imports out of the audit path. Some locked-down Windows
    # installations block its compiled extensions even when pandas is usable.
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20, max_categories=50,
                                  sparse_output=sparse_output)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ], remainder="drop", sparse_threshold=0 if not sparse_output else 0.3)


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Keep the full scikit-learn run compact. High-cardinality city, customer,
    # product-name and transaction fields add memory cost to this initial model.
    allowed = [
        "Shipping Mode", "Market", "Order Region", "Category Name", "Customer Segment",
        "Days for shipment (scheduled)", "Order Item Quantity", "Order Item Product Price",
    ]
    X = df[[c for c in allowed if c in df.columns]].copy()
    # Order calendar features are available at the proposed order-placement point.
    if DATE in df:
        dates = pd.to_datetime(df[DATE], errors="coerce")
        X["order_year"] = dates.dt.year
        X["order_month"] = dates.dt.month
        X["order_dayofweek"] = dates.dt.dayofweek
    # Discard all-empty and constant fields; they add no signal and may break imputers.
    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.nunique(dropna=False) > 1]
    return X


def evaluate_model(name: str, model: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    predicted = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    return {
        "model": name,
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)) if y.nunique() == 2 else None,
        "pr_auc": float(average_precision_score(y, probabilities)) if y.nunique() == 2 else None,
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y, predicted, zero_division=0, output_dict=True),
        "predicted_label": predicted,
        "predicted_probability": probabilities,
    }


def train_models(df: pd.DataFrame, random_state: int = 42):
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    train, test, cutoff = chronological_group_split(df)
    X_train, X_test = feature_frame(train), feature_frame(test)
    # Align features when categories/columns differ across time windows.
    X_test = X_test.reindex(columns=X_train.columns)
    y_train, y_test = train[TARGET], test[TARGET]
    models: dict[str, Pipeline] = {}
    models["majority_baseline"] = Pipeline([
        ("prep", make_preprocessor(X_train, scale_numeric=False)),
        ("model", DummyClassifier(strategy="most_frequent")),
    ])
    models["logistic_regression"] = Pipeline([
        ("prep", make_preprocessor(X_train, scale_numeric=True)),
        ("model", LogisticRegression(max_iter=1200, class_weight="balanced", random_state=random_state)),
    ])
    models["random_forest"] = Pipeline([
        ("prep", make_preprocessor(X_train, scale_numeric=False)),
        ("model", RandomForestClassifier(
            n_estimators=100, max_depth=14, min_samples_leaf=10, class_weight="balanced_subsample",
            n_jobs=1, random_state=random_state,
        )),
    ])
    models["extra_trees"] = Pipeline([
        ("prep", make_preprocessor(X_train, scale_numeric=False)),
        ("model", ExtraTreesClassifier(
            n_estimators=160, max_depth=16, min_samples_leaf=10,
            class_weight="balanced", n_jobs=1, random_state=random_state,
        )),
    ])
    models["hist_gradient_boosting"] = Pipeline([
        ("prep", make_preprocessor(X_train, scale_numeric=False, sparse_output=False)),
        ("model", HistGradientBoostingClassifier(
            max_iter=160, learning_rate=0.08, max_leaf_nodes=15,
            l2_regularization=1.0, class_weight="balanced",
            random_state=random_state,
        )),
    ])
    metrics, predictions = [], []
    for name, model in models.items():
        model.fit(X_train, y_train)
        result = evaluate_model(name, model, X_test, y_test)
        predictions.append(pd.DataFrame({
            "order_id": test[ORDER_ID].to_numpy(),
            "order_date": test[DATE].to_numpy(),
            "actual": y_test.to_numpy(),
            "predicted": result.pop("predicted_label"),
            "risk_score": result.pop("predicted_probability"),
            "model": name,
        }))
        metrics.append(result)
    # Store selected model for reproducibility; user should select based on use-case metrics.
    selected_name = "logistic_regression"
    metadata = {
        "test_cutoff_date": str(cutoff),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_orders": int(train[ORDER_ID].nunique()),
        "test_orders": int(test[ORDER_ID].nunique()),
        "overlapping_order_ids": int(len(set(train[ORDER_ID]) & set(test[ORDER_ID]))),
        "excluded_fields": sorted(POST_OUTCOME_OR_ID),
        "selected_pipeline_name_for_inspection_only": selected_name,
    }
    return metrics, pd.concat(predictions, ignore_index=True), models[selected_name], metadata


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
