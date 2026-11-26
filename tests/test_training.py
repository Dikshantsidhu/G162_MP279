"""Unit tests for model training and MLflow tracking/registry (synthetic data)."""

from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient

from mp279.config import load_params
from mp279.data_checks import compute_file_sha256
from mp279.train import build_pipeline, train_and_register


def _write_synthetic_project(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    rng = np.random.default_rng(42)
    n_train = 45
    blocks = ["Block North", "Block South", "Block East"] * (n_train // 3)
    areas = ["urban", "rural", "tri."] * (n_train // 3)

    cards = rng.integers(300, 1800, size=n_train)
    hours = rng.uniform(3.5, 9.5, size=n_train)
    bio = rng.integers(0, 60, size=n_train)
    bio[2] = 9999
    temp = rng.uniform(15.0, 42.0, size=n_train)
    stock = rng.uniform(500.0, 2500.0, size=n_train)
    days = rng.integers(0, 30, size=n_train)
    grain = 400.0 + 0.3 * cards + 20.0 * hours - 10.0 * temp + 0.1 * stock + rng.normal(0, 5, n_train)

    train_df = pd.DataFrame(
        {
            "record_id": [f"R{i:04d}" for i in range(n_train)],
            "date": pd.date_range("2026-01-01", periods=n_train, freq="D").strftime("%Y-%m-%d"),
            "cards_attached": cards,
            "working_hours": hours,
            "biometric_failures": bio,
            "temp_c": temp,
            "stock_kg_morning": stock,
            "days_to_month_end": days,
            "shop_area": areas,
            "block": blocks,
            "kg_entered_in_epos_at_close": grain + 1.0,
            "grain_kg": grain,
        }
    )
    train_df.loc[train_df["block"] == "Block South", "stock_kg_morning"] *= 1000.0

    train_csv = tmp_path / "train.csv"
    train_df.to_csv(train_csv, index=False)

    n_hold = 9
    hold_df = pd.DataFrame(
        {
            "record_id": [f"H{i:04d}" for i in range(n_hold)],
            "date": pd.date_range("2026-09-01", periods=n_hold, freq="D").strftime("%Y-%m-%d"),
            "cards_attached": rng.integers(300, 1800, size=n_hold),
            "working_hours": rng.uniform(3.5, 9.5, size=n_hold),
            "biometric_failures": rng.integers(0, 60, size=n_hold),
            "temp_c": rng.uniform(15.0, 42.0, size=n_hold),
            "stock_kg_morning": rng.uniform(500.0, 2500.0, size=n_hold),
            "days_to_month_end": rng.integers(0, 30, size=n_hold),
            "shop_area": ["urban", "rural", "tribal"] * 3,
            "block": ["Block North", "Block South", "Block East"] * 3,
        }
    )
    holdout_csv = tmp_path / "holdout.csv"
    hold_df.to_csv(holdout_csv, index=False)

    base_params = load_params("params.yaml")
    db_path = tmp_path / "test_mlflow.db"
    artifact_path = tmp_path / "model.joblib"
    output_csv = tmp_path / "holdout_predictions.csv"

    base_params["data"]["raw_train"] = str(train_csv)
    base_params["data"]["holdout"] = str(holdout_csv)
    base_params["model"]["artifact_path"] = str(artifact_path)
    base_params["model"]["cv_folds"] = 3
    base_params["model"]["candidates"] = [
        {"name": "ridge_1", "type": "ridge", "params": {"alpha": 1.0}},
        {"name": "ridge_10", "type": "ridge", "params": {"alpha": 10.0}},
        {"name": "poly2_ridge_10", "type": "poly_ridge", "params": {"degree": 2, "alpha": 10.0}},
    ]
    base_params["mlflow"]["tracking_uri"] = f"sqlite:///{db_path.as_posix()}"
    base_params["mlflow"]["experiment_name"] = "test-mp279-exp"
    base_params["mlflow"]["registered_model_name"] = "test-mp279-model"
    base_params["prediction"]["output_path"] = str(output_csv)

    params_path = tmp_path / "params.yaml"
    params_path.write_text(yaml.safe_dump(base_params), encoding="utf-8")
    return params_path, train_csv, db_path, artifact_path


def test_build_pipeline_handles_supported_types() -> None:
    num = ["cards_attached", "temp_c"]
    cat = ["shop_area", "block"]
    for cfg in (
        {"name": "r", "type": "ridge", "params": {"alpha": 1.0}},
        {"name": "p", "type": "poly_ridge", "params": {"degree": 2, "alpha": 5.0}},
        {"name": "g", "type": "gradient_boosting", "params": {"n_estimators": 10}},
    ):
        pipe = build_pipeline(cfg, num, cat, random_state=42)
        assert len(pipe.steps) == 2


def test_train_and_register_logs_runs_and_sets_champion(tmp_path: Path) -> None:
    params_path, train_csv, db_path, artifact_path = _write_synthetic_project(tmp_path)
    expected_sha = compute_file_sha256(train_csv)

    summary = train_and_register(params_path)

    assert len(summary["runs"]) >= 3
    assert summary["data_sha256"] == expected_sha
    assert summary["champion_version"] == 1
    assert artifact_path.exists()

    tracking_uri = f"sqlite:///{db_path.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name("test-mp279-exp")
    assert exp is not None

    runs = client.search_runs([exp.experiment_id])
    assert len(runs) >= 3
    for run in runs:
        assert run.data.params["data_sha256"] == expected_sha
        assert any("mae" in k.lower() for k in run.data.metrics)

    champ_mv = client.get_model_version_by_alias("test-mp279-model", "champion")
    assert int(champ_mv.version) == 1
    assert champ_mv.run_id == summary["best_run_id"]

    model = joblib.load(artifact_path)
    sample = pd.DataFrame(
        [
            {
                "cards_attached": 1000,
                "working_hours": 6.5,
                "biometric_failures": 20.0,
                "temp_c": 25.0,
                "stock_kg_morning": 1500.0,
                "days_to_month_end": 10,
                "shop_area": "urban",
                "block": "Block North",
            }
        ]
    )
    preds = model.predict(sample)
    assert preds.shape == (1,)
