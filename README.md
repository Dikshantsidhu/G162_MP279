# G162 — MP279: Ration Shop Distribution (Surat)

SCSE3040 MLOps Mini Project for the **Surat Food and Civil Supplies Office** — predicting daily `grain_kg` distributed by fair price (ration) shops so stock can be replenished in time.

## Group Information

- **Group ID:** G162
- **Problem ID:** MP279
- **Faculty:** Ravi Pratap (Batch 79)

### Team Members

| Enrolment No. | Name / GitHub Handle | Role |
|---|---|---|
| `S24CSEU0861` | DIKSHANT. (`Dikshantsidhu`) | Data Lead |
| `S24CSEU0946` | vanshdhingra | Model Lead |
| `S24CSEU0890` | BhaveshUpadhyay | Team Member |
| `S24CSEU0758` | HIMANSHUYADAV | Team Member |

## Project Structure

```text
├── README.md
├── requirements.txt
├── params.yaml
├── data/
│   └── raw/
│       ├── train.csv.dvc          # DVC-tracked training data (Jan–Jun 2026)
│       ├── later.csv.dvc          # DVC-tracked later data (Jul–Aug 2026)
│       └── holdout_features.csv   # Holdout features (Sep 2026)
├── src/
│   └── mp279/
│       ├── __init__.py
│       ├── config.py              # Loads params.yaml via yaml.safe_load
│       ├── data_checks.py         # Automated detection of the 7 data issues
│       ├── cleaning.py            # Reproducible data cleaning pipeline
│       ├── train.py               # Model training, MLflow tracking & registry
│       └── predict.py             # Holdout predictions using @champion
├── tests/
│   ├── test_data_checks.py
│   ├── test_cleaning.py
│   ├── test_training.py
│   └── test_predict.py
├── models/
│   └── model.joblib               # Exported @champion model artifact
├── mlflow.db                      # MLflow tracking & model registry SQLite DB
├── reports/
│   └── findings.yaml              # Structured findings for marking
└── predictions/
    └── holdout_predictions.csv    # Holdout predictions (record_id,prediction)
```

## Setup & Usage (Data Pipeline)

1. Create a Python 3.12 virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Pull DVC-tracked raw datasets (when working from a fresh clone with remote storage configured) or verify local `.dvc` files:
   ```bash
   dvc status
   ```

3. Run the seven data quality checks on `data/raw/train.csv`:
   ```bash
   PYTHONPATH=src python -m mp279.data_checks
   ```

4. Run the data cleaning pipeline to generate `data/processed/train_cleaned.csv`:
   ```bash
   PYTHONPATH=src python -m mp279.cleaning
   ```

5. Run unit tests (uses synthetic in-memory DataFrames; does not require raw CSVs):
   ```bash
   PYTHONPATH=src pytest -v
   ```

## Setup & Usage (Model Pipeline)

1. Train candidate models, log all runs to `mlflow.db`, register v1 with alias `@champion`, and export `models/model.joblib`:
   ```bash
   PYTHONPATH=src python -m mp279.train
   ```

2. Generate `predictions/holdout_predictions.csv` using the registered `@champion` model:
   ```bash
   PYTHONPATH=src python -m mp279.predict
   ```
