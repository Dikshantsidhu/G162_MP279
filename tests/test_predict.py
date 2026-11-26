"""Unit tests for @champion holdout prediction (synthetic data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from mp279.config import load_params
from mp279.predict import predict_holdout
from mp279.train import train_and_register


def test_predict_holdout_outputs_required_schema(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    n_train = 36
    train_df = pd.DataFrame(
        {
            "record_id": [f"R{i:04d}" for i in range(n_train)],
            "date": pd.date_range("2026-01-01", periods=n_train, freq="D").strftime("%Y-%m-%d"),
            "cards_attached": rng.integers(300, 1500, size=n_train),
            "working_hours": rng.uniform(4.0, 9.0, size=n_train),
            "biometric_failures": rng.integers(0, 50, size=n_train),
            "temp_c": rng.uniform(18.0, 40.0, size=n_train),
            "stock_kg_morning": rng.uniform(600.0, 2200.0, size=n_train),
            "days_to_month_end": rng.integers(0, 30, size=n_train),
            "shop_area": ["urban", "rural", "tribal"] * (n_train // 3),
            "block": ["Block North", "Block South", "Block East"] * (n_train // 3),
            "grain_kg": rng.uniform(400.0, 1200.0, size=n_train),
        }
    )
    train_csv = tmp_path / "train.csv"
    train_df.to_csv(train_csv, index=False)

    n_hold = 6
    hold_df = train_df.drop(columns=["grain_kg"]).iloc[:n_hold].copy()
    hold_df["record_id"] = [f"H{i:04d}" for i in range(n_hold)]
    hold_csv = tmp_path / "holdout.csv"
    hold_df.to_csv(hold_csv, index=False)

    params = load_params("params.yaml")
    db_path = tmp_path / "test_mlflow.db"
    artifact_path = tmp_path / "model.joblib"
    out_path = tmp_path / "holdout_predictions.csv"

    params["data"]["raw_train"] = str(train_csv)
    params["data"]["holdout"] = str(hold_csv)
    params["model"]["artifact_path"] = str(artifact_path)
    params["model"]["cv_folds"] = 3
    params["model"]["candidates"] = [
        {"name": "ridge_1", "type": "ridge", "params": {"alpha": 1.0}},
    ]
    params["mlflow"]["tracking_uri"] = f"sqlite:///{db_path.as_posix()}"
    params["mlflow"]["experiment_name"] = "test-predict-exp"
    params["mlflow"]["registered_model_name"] = "test-predict-model"
    params["prediction"]["output_path"] = str(out_path)

    params_path = tmp_path / "params.yaml"
    params_path.write_text(yaml.safe_dump(params), encoding="utf-8")

    train_and_register(params_path)
    preds_df = predict_holdout(params_path=params_path)

    assert out_path.exists()
    loaded = pd.read_csv(out_path)
    assert list(loaded.columns) == ["record_id", "prediction"]
    assert len(loaded) == n_hold
    assert loaded["prediction"].notna().all()
    assert (loaded["prediction"] >= 0).all()
    pd.testing.assert_frame_equal(preds_df.reset_index(drop=True), loaded)
