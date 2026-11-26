"""Model training, MLflow tracking, and model registry management for MP279."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from mp279.cleaning import clean_dataframe, compute_imputation_medians
from mp279.config import load_params
from mp279.data_checks import compute_file_sha256


def _make_ridge(cfg: dict[str, Any], _rs: int) -> Ridge:
    return Ridge(alpha=float(cfg.get("alpha", 1.0)))


def _make_huber(cfg: dict[str, Any], _rs: int) -> HuberRegressor:
    return HuberRegressor(
        alpha=float(cfg.get("alpha", 0.0001)),
        max_iter=int(cfg.get("max_iter", 500)),
    )


def _make_gbr(cfg: dict[str, Any], rs: int) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=int(cfg.get("n_estimators", 200)),
        learning_rate=float(cfg.get("learning_rate", 0.05)),
        max_depth=int(cfg.get("max_depth", 3)),
        random_state=rs,
    )


def _make_rf(cfg: dict[str, Any], rs: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=int(cfg.get("n_estimators", 200)),
        max_depth=cfg.get("max_depth", 12),
        random_state=rs,
    )


ESTIMATOR_FACTORIES: dict[str, Callable[[dict[str, Any], int], Any]] = {
    "linear": lambda _cfg, _rs: LinearRegression(),
    "ridge": _make_ridge,
    "poly_ridge": _make_ridge,
    "huber": _make_huber,
    "gradient_boosting": _make_gbr,
    "random_forest": _make_rf,
}


def build_pipeline(
    candidate: dict[str, Any],
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int = 42,
) -> Pipeline:
    """Build a preprocessing + regression Pipeline for a candidate configuration."""
    model_type = candidate["type"]
    if model_type not in ESTIMATOR_FACTORIES:
        raise ValueError(f"Unsupported model type: {model_type}")

    model_params = candidate.get("params", {})
    num_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if model_type == "poly_ridge":
        degree = int(model_params.get("degree", 2))
        num_steps.append(("poly", PolynomialFeatures(degree=degree, include_bias=False)))

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    estimator = ESTIMATOR_FACTORIES[model_type](model_params, random_state)
    return Pipeline([("preprocessor", preprocessor), ("regressor", estimator)])


def evaluate_candidate(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_full: pd.DataFrame,
    y_full: pd.Series,
    cv_folds: int,
    random_state: int,
) -> tuple[Pipeline, dict[str, float]]:
    """Evaluate a candidate via holdout validation and K-fold CV, then fit on full data."""
    val_pipe = clone(pipeline)
    val_pipe.fit(X_train, y_train)

    train_preds = val_pipe.predict(X_train)
    val_preds = val_pipe.predict(X_val)

    n_splits = max(2, min(cv_folds, len(X_full)))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_scores = -cross_val_score(
        clone(pipeline),
        X_full,
        y_full,
        cv=cv,
        scoring="neg_mean_absolute_error",
    )

    metrics = {
        "mae_cv": float(np.mean(cv_scores)),
        "mae_cv_std": float(np.std(cv_scores)),
        "mae_val": float(mean_absolute_error(y_val, val_preds)),
        "mae_train": float(mean_absolute_error(y_train, train_preds)),
        "rmse_val": float(root_mean_squared_error(y_val, val_preds)),
        "r2_val": float(r2_score(y_val, val_preds)),
    }

    full_pipe = clone(pipeline)
    full_pipe.fit(X_full, y_full)
    return full_pipe, metrics


def load_and_clean_training_data(params: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    """Load raw training CSV, compute its SHA-256, and apply the cleaning pipeline."""
    data_cfg = params["data"]
    raw_train_path = Path(data_cfg["raw_train"])
    data_sha256 = compute_file_sha256(raw_train_path)

    raw_df = pd.read_csv(raw_train_path)
    deduped = raw_df.drop_duplicates()

    sent_cfg = params.get("cleaning", {}).get("sentinel", {})
    medians = compute_imputation_medians(
        deduped,
        sentinel_col=sent_cfg.get("column", "biometric_failures"),
        sentinel_val=sent_cfg.get("value", 9999),
    )
    cleaned_df = clean_dataframe(
        deduped,
        params=params,
        imputation_medians=medians,
        drop_duplicates=False,
    )
    return cleaned_df, data_sha256


def train_and_register(
    params_path: str | Path = "params.yaml",
    tracking_uri_override: str | None = None,
    artifact_path_override: str | Path | None = None,
) -> dict[str, Any]:
    """Train candidate models, log runs to MLflow, register v1, and set @champion."""
    params = load_params(params_path)
    model_cfg = params["model"]
    mlflow_cfg = params["mlflow"]

    cleaned_df, data_sha256 = load_and_clean_training_data(params)

    target_col = model_cfg["target"]
    features = list(model_cfg["features"])
    numeric_features = list(model_cfg["numeric_features"])
    categorical_features = list(model_cfg["categorical_features"])
    random_state = int(model_cfg.get("random_state", 42))
    test_size = float(model_cfg.get("test_size", 0.2))
    cv_folds = int(model_cfg.get("cv_folds", 5))

    X_full = cleaned_df[features]
    y_full = cleaned_df[target_col]
    X_train, X_val, y_train, y_val = train_test_split(
        X_full,
        y_full,
        test_size=test_size,
        random_state=random_state,
    )

    tracking_uri = tracking_uri_override or mlflow_cfg["tracking_uri"]
    experiment_name = mlflow_cfg["experiment_name"]
    registered_model_name = mlflow_cfg["registered_model_name"]
    champion_alias = mlflow_cfg.get("champion_alias", "champion")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    run_records: list[dict[str, Any]] = []
    best_record: dict[str, Any] | None = None
    best_pipeline: Pipeline | None = None

    for candidate in model_cfg["candidates"]:
        cand_name = candidate["name"]
        cand_type = candidate["type"]
        cand_params = candidate.get("params", {})

        pipeline = build_pipeline(
            candidate,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            random_state=random_state,
        )
        fitted_full, metrics = evaluate_candidate(
            pipeline,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_full=X_full,
            y_full=y_full,
            cv_folds=cv_folds,
            random_state=random_state,
        )

        with mlflow.start_run(run_name=cand_name) as run:
            mlflow.log_param("data_sha256", data_sha256)
            mlflow.log_param("model_name", cand_name)
            mlflow.log_param("model_type", cand_type)
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("n_rows", len(cleaned_df))
            mlflow.log_param("n_features", len(features))
            for k, v in cand_params.items():
                mlflow.log_param(f"hp_{k}", v)

            mlflow.log_metrics(metrics)
            model_info = mlflow.sklearn.log_model(
                fitted_full,
                name="model",
                serialization_format="cloudpickle",
            )
            run_id = run.info.run_id

        record = {
            "name": cand_name,
            "type": cand_type,
            "run_id": run_id,
            "model_uri": model_info.model_uri,
            **metrics,
        }
        run_records.append(record)

        if best_record is None or (
            record["mae_cv"],
            record["mae_val"],
        ) < (
            best_record["mae_cv"],
            best_record["mae_val"],
        ):
            best_record = record
            best_pipeline = fitted_full

    assert best_record is not None and best_pipeline is not None

    mv = mlflow.register_model(best_record["model_uri"], registered_model_name)
    client = MlflowClient(tracking_uri=tracking_uri)
    client.set_registered_model_alias(
        name=registered_model_name,
        alias=champion_alias,
        version=str(mv.version),
    )

    artifact_path = Path(artifact_path_override or model_cfg["artifact_path"])
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, artifact_path)

    return {
        "experiment_name": experiment_name,
        "best_run_id": best_record["run_id"],
        "best_model_name": best_record["name"],
        "best_mae_cv": best_record["mae_cv"],
        "best_mae_val": best_record["mae_val"],
        "registered_model_name": registered_model_name,
        "champion_version": int(mv.version),
        "artifact_path": str(artifact_path),
        "data_sha256": data_sha256,
        "runs": run_records,
    }


def main() -> None:
    """CLI entry point for model training and MLflow registration."""
    summary = train_and_register()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
