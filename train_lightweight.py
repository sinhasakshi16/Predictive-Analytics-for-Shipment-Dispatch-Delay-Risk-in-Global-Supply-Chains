"""Lightweight category-rate benchmark using pandas only (no scikit-learn)."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
DATA_CANDIDATES = [ROOT / "data/raw/DataCoSupplyChainDataset.csv", PROJECT_ROOT / "DataCoSupplyChainDataset.csv"]
DATA = next((path for path in DATA_CANDIDATES if path.exists()), DATA_CANDIDATES[0])
OUT = ROOT / "data/processed"
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from dataco_project.pipeline import TARGET, DATE, ORDER_ID

if not DATA.exists():
    raise SystemExit(f"Dataset not found: {DATA}")

# Load only order-time features required for this small benchmark.
usecols = [TARGET, DATE, ORDER_ID, "Shipping Mode", "Market", "Order Region",
           "Category Name", "Customer Segment", "Days for shipment (scheduled)"]
df = pd.read_csv(DATA, encoding="latin-1", usecols=usecols, low_memory=True)
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df[DATE] = pd.to_datetime(df[DATE], errors="coerce")
df = df[df[TARGET].isin([0, 1]) & df[DATE].notna() & df[ORDER_ID].notna()].copy()
df[TARGET] = df[TARGET].astype("int8")
df["order_month"] = df[DATE].dt.month.astype(str)
df["scheduled_days"] = df["Days for shipment (scheduled)"].astype(str)

# Chronological split based on unique orders; orders cannot cross the cutoff.
order_dates = df.groupby(ORDER_ID)[DATE].min().sort_values()
cutoff = order_dates.iloc[int(len(order_dates) * 0.8)]
train_ids = set(order_dates[order_dates < cutoff].index)
test_ids = set(order_dates[order_dates >= cutoff].index)
train = df[df[ORDER_ID].isin(train_ids)].copy()
test = df[df[ORDER_ID].isin(test_ids)].copy()
if set(train[ORDER_ID]) & set(test[ORDER_ID]):
    raise RuntimeError("Order leakage across time split")

features = ["Shipping Mode", "Market", "Order Region", "Category Name", "Customer Segment", "order_month", "scheduled_days"]
prior = float(train[TARGET].mean())
smooth = 20.0
score = np.zeros(len(test), dtype=float)
for col in features:
    rates = train.groupby(col, dropna=False)[TARGET].agg(["sum", "count"])
    rates["risk_rate"] = (rates["sum"] + smooth * prior) / (rates["count"] + smooth)
    mapped = test[col].map(rates["risk_rate"]).fillna(prior).to_numpy()
    score += mapped
score /= len(features)
y = test[TARGET].to_numpy()
pred = (score >= 0.5).astype(int)

tp = int(((pred == 1) & (y == 1)).sum())
tn = int(((pred == 0) & (y == 0)).sum())
fp = int(((pred == 1) & (y == 0)).sum())
fn = int(((pred == 0) & (y == 1)).sum())
precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
rank = pd.Series(score).rank(method="average").to_numpy()
n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
roc_auc = float((rank[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)) if n_pos and n_neg else None
order = np.argsort(-score)
sorted_y = y[order]
cum_tp = np.cumsum(sorted_y == 1)
precision_at_k = cum_tp / np.arange(1, len(y) + 1)
pr_auc = float(precision_at_k[sorted_y == 1].mean()) if n_pos else None

metrics = {
    "model": "smoothed_category_rate_average",
    "note": "Lightweight benchmark; scores average smoothed train-set late rates for order-time feature groups.",
    "test_cutoff_date": str(cutoff),
    "train_rows": int(len(train)), "test_rows": int(len(test)),
    "train_orders": int(len(train_ids)), "test_orders": int(len(test_ids)),
    "overlapping_order_ids": 0,
    "test_target_rate": float(y.mean()),
    "accuracy": float((pred == y).mean()),
    "precision": precision, "recall": recall, "f1": f1,
    "roc_auc": roc_auc, "pr_auc": pr_auc,
    "confusion_matrix_tn_fp_fn_tp": [[tn, fp], [fn, tp]],
    "features": features,
    "excluded_post_outcome_fields": ["Days for shipping (real)", "Delivery Status", "shipping date (DateOrders)"],
}

# Aggregate line-item scores to one ranking per order for a capacity-based view.
order_predictions = test[[ORDER_ID, DATE, TARGET]].copy()
order_predictions["risk_score"] = score
order_predictions = order_predictions.groupby(ORDER_ID, as_index=False).agg(
    order_date=(DATE, "min"), actual=(TARGET, "max"),
    risk_score=("risk_score", "mean"), line_items=(TARGET, "size"),
    label_values=(TARGET, "nunique"),
)
if (order_predictions["label_values"] > 1).any():
    raise ValueError("Target label varies within an order; order-level evaluation is ambiguous.")
order_predictions = order_predictions.drop(columns="label_values")
order_predictions["rank"] = order_predictions["risk_score"].rank(method="first", ascending=False).astype(int)
order_y = order_predictions["actual"].to_numpy()
capacity_results = []
for capacity in (0.05, 0.10, 0.20):
    k = max(1, int(np.ceil(len(order_predictions) * capacity)))
    selected = order_predictions.nsmallest(k, "rank")
    caught = int(selected["actual"].sum())
    positives = int(order_y.sum())
    capacity_results.append({
        "capacity_fraction": capacity,
        "orders_flagged": k,
        "orders_total": int(len(order_predictions)),
        "late_orders_caught": caught,
        "late_orders_total": positives,
        "late_order_recall_at_capacity": caught / positives if positives else 0.0,
        "precision_at_capacity": caught / k if k else 0.0,
        "late_order_baseline_rate": positives / len(order_predictions) if len(order_predictions) else 0.0,
    })
order_metrics = {
    "unit": "order",
    "test_orders": int(len(order_predictions)),
    "test_late_order_rate": float(order_y.mean()),
    "capacity_results": capacity_results,
    "score_aggregation": "mean line-item risk score per order",
    "label_aggregation": "maximum line-item target; consistency validated",
}
OUT.mkdir(parents=True, exist_ok=True)
order_predictions.to_csv(OUT / "lightweight_order_test_predictions.csv", index=False)
pd.DataFrame(capacity_results).to_csv(OUT / "lightweight_capacity_metrics.csv", index=False)
metrics["order_level_evaluation"] = order_metrics
pd.DataFrame([metrics]).drop(columns=["features", "excluded_post_outcome_fields", "order_level_evaluation"]).to_csv(OUT / "lightweight_model_metrics.csv", index=False)
predictions = test[[ORDER_ID, DATE, TARGET]].copy()
predictions["risk_score"] = score
predictions["predicted"] = pred
predictions.to_csv(OUT / "lightweight_test_predictions.csv", index=False)
(OUT / "lightweight_model_results.json").write_text(__import__("json").dumps(metrics, indent=2), encoding="utf-8")
print(__import__("json").dumps(metrics, indent=2))
print(f"\nOutputs written to {OUT}")
