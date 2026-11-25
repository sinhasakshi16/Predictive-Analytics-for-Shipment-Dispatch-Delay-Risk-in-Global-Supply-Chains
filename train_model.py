from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import json
import joblib
import pandas as pd
import sklearn
import platform
import sys
from dataco_project.pipeline import MODEL_COLUMNS, load_data, train_models, write_json

DATA_CANDIDATES = [ROOT / "data/raw/DataCoSupplyChainDataset.csv", ROOT.parent / "DataCoSupplyChainDataset.csv"]
DATA = next((path for path in DATA_CANDIDATES if path.exists()), DATA_CANDIDATES[0])
OUT = ROOT / "data/processed"

if not DATA.exists():
    raise SystemExit(f"Dataset not found: {DATA}\nDownload the Kaggle CSV and place it there.")

df = load_data(DATA, usecols=MODEL_COLUMNS)
metrics, predictions, model, metadata = train_models(df)
metadata["environment"] = {
    "python": sys.version.split()[0],
    "pandas": pd.__version__,
    "numpy": __import__("numpy").__version__,
    "scikit_learn": sklearn.__version__,
    "platform": platform.platform(),
}
OUT.mkdir(parents=True, exist_ok=True)
pd.DataFrame([{k: v for k, v in row.items() if k != "classification_report" and k != "confusion_matrix"}
              for row in metrics]).to_csv(OUT / "model_metrics.csv", index=False)
comparison_rows = [{k: row[k] for k in ("model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")}
                   for row in metrics]
lightweight_results = OUT / "lightweight_model_results.json"
if lightweight_results.exists():
    lightweight = json.loads(lightweight_results.read_text(encoding="utf-8"))
    comparison_rows.append({
        "model": lightweight["model"],
        **{key: lightweight.get(key) for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")},
    })
pd.DataFrame(comparison_rows).sort_values("pr_auc", ascending=False).to_csv(
    OUT / "model_comparison.csv", index=False
)
write_json({"metadata": metadata, "models": metrics}, OUT / "model_results.json")
predictions.to_csv(OUT / "test_predictions.csv", index=False)
joblib.dump(model, OUT / "logistic_regression_pipeline.joblib")
print(pd.DataFrame([{k: row[k] for k in ("model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")}
                    for row in metrics]).to_string(index=False))
print(f"\nOutputs written to {OUT}")
