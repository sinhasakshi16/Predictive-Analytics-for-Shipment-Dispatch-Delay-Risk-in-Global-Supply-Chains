# DataCo Late Delivery Risk Project

An initial, reproducible project for exploring delivery performance and testing whether order information can identify late deliveries. The project is designed for the DataCo Smart Supply Chain dataset available from [Kaggle](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) and [Mendeley Data](https://data.mendeley.com/datasets/8gx2fvg2k6/5).

## Research question

Can information available when an order is placed help identify orders that will be delivered late, and which groups might benefit from additional monitoring?

## Project structure

```text
dataco-late-delivery/
├── data/
│   ├── raw/                 # Put the downloaded CSV here; not committed
│   └── processed/           # Generated audit and model outputs
├── src/dataco_project/
│   ├── __init__.py
│   └── pipeline.py          # Data loading, leakage controls, validation, modeling
├── scripts/
│   ├── run_audit.py
│   └── train_model.py
├── requirements.txt
└── README.md
```

## Setup

Use Python 3.10 or newer. From this folder:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download `DataCoSupplyChainDataset.csv` from Kaggle and place it at `data/raw/DataCoSupplyChainDataset.csv`. If available, also place `DescriptionDataCoSupplyChain.csv` beside it. The loader tries UTF-8 and Latin-1 encodings.

The audit and training scripts also check for `DataCoSupplyChainDataset.csv` in the parent project folder, which is useful when this code is stored in a `late delivery/` subfolder beside the downloaded CSV.

## Run the initial data audit

```bash
python scripts/run_audit.py
```

This writes `data/processed/audit_summary.json`, `missing_values.csv`, `target_by_delivery_status.csv`, and `monthly_delivery_rate.csv`. Review these before modeling, especially the target/status cross-tab and date coverage.

The audit only needs pandas and NumPy. Scikit-learn is imported only by the training command, so a Windows policy that blocks its compiled DLLs will not prevent the audit from running. If the training command is blocked, use an organization-approved Python environment or ask IT to allow the scikit-learn installation; changing Python versions alone may not override Application Control.

In VS Code, open this folder, select the project virtual environment as the Python interpreter, then use **Terminal → Run Task** and choose `DataCo: Run data audit` or `DataCo: Train baseline models`.

The audit reads the CSV in chunks to keep memory use low.

## Train and compare initial models

```bash
python scripts/train_model.py
```

The script excludes outcome and post-shipment fields, holds out the latest 20% of order dates for testing, and keeps each order ID in one split. It compares a majority-class baseline, logistic regression, random forest, Extra Trees, and histogram gradient boosting. Results, test predictions, and the fitted logistic-regression preprocessing/model pipeline are saved under `data/processed/`.
The script also writes `model_comparison.csv`, which combines these models with the lightweight benchmark results when that benchmark has been run.

If scikit-learn is unavailable, run `python scripts/train_lightweight.py` instead. This pandas/NumPy benchmark uses smoothed training-set late rates across order-time feature groups and the same chronological, order-grouped holdout. It is an initial ranking baseline, not a replacement for the multi-model comparison.

The lightweight run also writes `lightweight_capacity_metrics.csv` and `lightweight_order_test_predictions.csv`. These report top 5%, 10%, and 20% order-level monitoring scenarios by mean line-item risk score. The order-level table checks that all line items in an order share the same target before calculating capacity results. These scenarios describe ranking performance; they do not estimate the effect or cost of an intervention.

## Leakage and interpretation guardrails

- `Days for shipping (real)`, `Delivery Status`, `shipping date (DateOrders)`, `Late_delivery_risk`, and `Order Status` are excluded from model inputs. Some are outcome fields or are only available after the prediction point.
- `Late_delivery_risk` may be closely defined by delivery outcome fields. The audit reports its relationship with `Delivery Status` so this can be checked explicitly.
- The data is line-item level. Splitting by `Order Id` prevents lines from a single order appearing in both train and test sets.
- The chronological test estimates performance on later records from this dataset, not on today's logistics operations. Treat the model as a retrospective study until evaluated on current operational data.
- Feature importance describes model associations, not causes. Cost savings require external, defensible cost assumptions.
- This dataset does not provide stock-on-hand, replenishment lead time, holding cost, or stockout outcomes, so it cannot validate inventory optimization.

## Initial scope

1. Validate and document the downloaded files.
2. Explore delivery rates over time and across shipping mode, region, category, and customer segment.
3. Establish simple baselines and compare classification models using recall, precision, F1, ROC-AUC, and PR-AUC.
4. Inspect false positives/negatives and run a transparent top-risk monitoring scenario.

The current code is a starting point. Confirm the dataset version, target semantics, and intended prediction point before making conclusions.
