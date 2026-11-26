"""Holdout prediction using the @champion model from MLflow Model Registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from mp279.cleaning import clean_dataframe
from mp279.config import load_params


def load_champion_model(
    params: dict[str, Any],
    tracking_uri_override: str | None = None,
    artifact_path_override: str | Path | None = None,
) -> Any:
    """Load the registered @champion model from MLflow, falling back to joblib."""
    mlflow_cfg = params["mlflow"]
    model_cfg = params["model"]

    tracking_uri = tracking_uri_override or mlflow_cfg["tracking_uri"]
    registered_model_name = mlflow_cfg["registered_model_name"]
    champion_alias = mlflow_cfg.get("champion_alias", "champion")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    # Verify the @champion alias exists in the MLflow registry
    client.get_model_version_by_alias(registered_model_name, champion_alias)

    model_uri = f"models:/{registered_model_name}@{champion_alias}"
    try:
        return mlflow.sklearn.load_model(model_uri)
    except Exception:
        fallback_path = Path(artifact_path_override or model_cfg["artifact_path"])
        if not fallback_path.exists():
            raise
        return joblib.load(fallback_path)


def predict_holdout(
    params_path: str | Path = "params.yaml",
    tracking_uri_override: str | None = None,
    artifact_path_override: str | Path | None = None,
    output_path_override: str | Path | None = None,
) -> pd.DataFrame:
    """Predict grain_kg for holdout_features.csv and save record_id,prediction CSV."""
    params = load_params(params_path)
    data_cfg = params["data"]
    model_cfg = params["model"]
    cleaning_cfg = params.get("cleaning", {})

    id_col = cleaning_cfg.get("id_col", "record_id")
    features = list(model_cfg["features"])

    holdout_raw = pd.read_csv(data_cfg["holdout"])
    holdout_clean = clean_dataframe(holdout_raw, params=params, drop_duplicates=False)

    model = load_champion_model(
        params,
        tracking_uri_override=tracking_uri_override,
        artifact_path_override=artifact_path_override,
    )
    preds = np.clip(model.predict(holdout_clean[features]), 0.0, None)

    out_df = pd.DataFrame(
        {
            "record_id": holdout_clean[id_col].astype(str),
            "prediction": preds.astype(float),
        }
    )

    out_path = Path(
        output_path_override or params.get("prediction", {}).get("output_path", "predictions/holdout_predictions.csv")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(out_df)} holdout predictions to {out_path}")
    return out_df


def main() -> None:
    """CLI entry point for generating holdout predictions."""
    predict_holdout()


if __name__ == "__main__":
    main()
