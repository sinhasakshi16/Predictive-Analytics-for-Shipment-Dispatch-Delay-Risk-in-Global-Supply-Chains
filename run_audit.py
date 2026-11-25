from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import json
import pandas as pd
from dataco_project.pipeline import DATE, TARGET, ORDER_ID, write_json

DATA_CANDIDATES = [ROOT / "data/raw/DataCoSupplyChainDataset.csv", ROOT.parent / "DataCoSupplyChainDataset.csv"]
DATA = next((path for path in DATA_CANDIDATES if path.exists()), DATA_CANDIDATES[0])
OUT = ROOT / "data/processed"

if not DATA.exists():
    raise SystemExit(f"Dataset not found: {DATA}\nDownload the Kaggle CSV and place it there.")

OUT.mkdir(parents=True, exist_ok=True)
missing = None
target_counts = {0: 0, 1: 0}
valid_target_rows = 0
total_rows = 0
order_ids = set()
row_hashes = set()
date_min = date_max = None
monthly = {}
status_target = {}
target_matches_status = True
columns = None

# Stream the file in chunks so the audit works in memory-limited environments.
for chunk in pd.read_csv(DATA, encoding="latin-1", chunksize=15000, low_memory=True):
    columns = chunk.columns.tolist()
    total_rows += len(chunk)
    chunk_missing = chunk.isna().sum()
    missing = chunk_missing if missing is None else missing.add(chunk_missing, fill_value=0)
    row_hashes.update(pd.util.hash_pandas_object(chunk, index=False).tolist())
    if ORDER_ID in chunk:
        order_ids.update(chunk[ORDER_ID].dropna().unique().tolist())
    if TARGET in chunk:
        target = pd.to_numeric(chunk[TARGET], errors="coerce")
        valid = target.isin([0, 1])
        valid_target_rows += int(valid.sum())
        counts = target[valid].value_counts()
        for label in (0, 1):
            target_counts[label] += int(counts.get(label, 0))
        if "Delivery Status" in chunk:
            matches = (target[valid].astype(int) == chunk.loc[valid, "Delivery Status"].eq("Late delivery").astype(int))
            target_matches_status = target_matches_status and bool(matches.all())
    if DATE in chunk:
        dates = pd.to_datetime(chunk[DATE], errors="coerce")
        if dates.notna().any():
            local_min, local_max = dates.min(), dates.max()
            date_min = local_min if date_min is None else min(date_min, local_min)
            date_max = local_max if date_max is None else max(date_max, local_max)
        if TARGET in chunk:
            good = dates.notna() & pd.to_numeric(chunk[TARGET], errors="coerce").isin([0, 1])
            months = dates[good].dt.to_period("M").astype(str)
            pairs = pd.DataFrame({"month": months, "target": pd.to_numeric(chunk.loc[good, TARGET]).astype(int).to_numpy()})
            grouped = pairs.groupby("month")["target"].agg(["size", "sum"])
            for month, row in grouped.iterrows():
                prior = monthly.setdefault(month, [0, 0])
                prior[0] += int(row["size"])
                prior[1] += int(row["sum"])
    if "Delivery Status" in chunk and TARGET in chunk:
        cross = pd.crosstab(chunk["Delivery Status"].fillna("<missing>"), chunk[TARGET].fillna("<missing>"))
        for status, row in cross.iterrows():
            for label, count in row.items():
                key = (str(status), str(label))
                status_target[key] = status_target.get(key, 0) + int(count)

summary = {
    "rows": total_rows,
    "columns": len(columns or []),
    "duplicate_rows_by_row_hash": total_rows - len(row_hashes),
    "target_rows_after_valid_target_filter": valid_target_rows,
    "target_rate": target_counts[1] / valid_target_rows if valid_target_rows else None,
    "target_counts": {str(k): v for k, v in target_counts.items()},
    "missing_cells": int(missing.sum()) if missing is not None else 0,
    "columns_all_missing": [str(c) for c, n in missing.items() if n == total_rows] if missing is not None else [],
    "columns_with_missing_values": int((missing > 0).sum()) if missing is not None else 0,
    "date_min": str(date_min) if date_min is not None else None,
    "date_max": str(date_max) if date_max is not None else None,
    "unique_orders": len(order_ids) if ORDER_ID in (columns or []) else None,
    "target_matches_late_delivery_status": target_matches_status if "Delivery Status" in (columns or []) else None,
}
write_json(summary, OUT / "audit_summary.json")
missing.rename("missing_count").to_csv(OUT / "missing_values.csv", header=True)
if status_target:
    status_table = pd.DataFrame.from_dict(status_target, orient="index", columns=["count"])
    status_table.index = pd.MultiIndex.from_tuples(status_table.index, names=["Delivery Status", TARGET])
    status_table.unstack(fill_value=0).to_csv(OUT / "target_by_delivery_status.csv")
if monthly:
    monthly_df = pd.DataFrame.from_dict(monthly, orient="index", columns=["order_line_rows", "late_count"])
    monthly_df.index.name = "month"
    monthly_df["late_rate"] = monthly_df["late_count"] / monthly_df["order_line_rows"]
    monthly_df.drop(columns="late_count").sort_index().to_csv(OUT / "monthly_delivery_rate.csv")
print(f"Audit written to {OUT}")
