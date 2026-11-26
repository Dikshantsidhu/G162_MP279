"""Reproducible data cleaning pipeline for MP279."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mp279.config import load_params

DEFAULT_SPELLING_MAP = {
    "tri.": "tribal",
    "rur.": "rural",
    "urb.": "urban",
}


def compute_imputation_medians(
    df: pd.DataFrame,
    sentinel_col: str = "biometric_failures",
    sentinel_val: int | float = 9999,
) -> dict[str, float]:
    """Compute median of valid (non-sentinel) values for imputation."""
    if sentinel_col not in df.columns:
        return {}
    valid = df.loc[df[sentinel_col] != sentinel_val, sentinel_col].dropna()
    if valid.empty:
        return {}
    return {sentinel_col: float(valid.median())}


def clean_dataframe(
    df: pd.DataFrame,
    params: dict[str, Any] | None = None,
    imputation_medians: dict[str, float] | None = None,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """Clean raw MP279 records using rules configured in params.yaml.

    Safe and idempotent across train, later, and holdout datasets:
    - Drops post-event leaking columns if present.
    - Converts Block South stock_kg_morning from grams to kg only when values
      exceed the unit-mixup threshold (> 10000).
    - Replaces sentinel value (9999 in biometric_failures) with NaN and imputes
      with the training median of valid values.
    - Optionally drops exact duplicate rows.
    - Normalizes shop_area spelling variants to {tribal, rural, urban}.
    """
    cleaned = df.copy()
    cleaning_cfg = (params or {}).get("cleaning", {})

    # 1. Drop leaking columns (only known after closing)
    leaking_cols = cleaning_cfg.get("leaking_columns", ["kg_entered_in_epos_at_close"])
    cols_to_drop = [c for c in leaking_cols if c in cleaned.columns]
    if cols_to_drop:
        cleaned = cleaned.drop(columns=cols_to_drop)

    # 2. Fix unit mixup (grams -> kg) only for inflated values in affected group
    unit_cfg = cleaning_cfg.get("unit_mixup", {})
    u_col = unit_cfg.get("column", "stock_kg_morning")
    u_grp_col = unit_cfg.get("group_col", "block")
    u_grp_val = unit_cfg.get("group_value", "Block South")
    u_factor = float(unit_cfg.get("factor", 1000))
    u_thresh = float(unit_cfg.get("threshold", 10000.0))

    if u_col in cleaned.columns and u_grp_col in cleaned.columns:
        cleaned[u_col] = cleaned[u_col].astype(float)
        mask = (cleaned[u_grp_col] == u_grp_val) & (cleaned[u_col] > u_thresh)
        cleaned.loc[mask, u_col] = cleaned.loc[mask, u_col] / u_factor

    # 3. Replace sentinel value with NaN and impute with valid median
    sent_cfg = cleaning_cfg.get("sentinel", {})
    s_col = sent_cfg.get("column", "biometric_failures")
    s_val = sent_cfg.get("value", 9999)

    if s_col in cleaned.columns:
        cleaned[s_col] = cleaned[s_col].astype(float)
        cleaned.loc[cleaned[s_col] == s_val, s_col] = np.nan
        if imputation_medians and s_col in imputation_medians:
            fill_val = imputation_medians[s_col]
        else:
            valid = cleaned[s_col].dropna()
            fill_val = float(valid.median()) if not valid.empty else 0.0
        cleaned[s_col] = cleaned[s_col].fillna(fill_val)

    # 4. Drop exact duplicate rows if requested
    if drop_duplicates:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # 5. Standardize spelling variants in categorical column
    spell_cfg = cleaning_cfg.get("spelling", {})
    sp_col = spell_cfg.get("column", "shop_area")
    sp_map = spell_cfg.get("mapping", DEFAULT_SPELLING_MAP)

    if sp_col in cleaned.columns:
        normalized = cleaned[sp_col].astype(str).str.strip().str.lower()
        cleaned[sp_col] = normalized.replace(sp_map)

    return cleaned


def clean_and_save(params_path: str | Path = "params.yaml") -> pd.DataFrame:
    """Load raw train data, clean it, and write cleaned CSV to disk."""
    params = load_params(params_path)
    data_cfg = params["data"]
    raw_train = pd.read_csv(data_cfg["raw_train"])

    deduped = raw_train.drop_duplicates()
    sent_cfg = params.get("cleaning", {}).get("sentinel", {})
    medians = compute_imputation_medians(
        deduped,
        sentinel_col=sent_cfg.get("column", "biometric_failures"),
        sentinel_val=sent_cfg.get("value", 9999),
    )
    cleaned = clean_dataframe(
        deduped,
        params=params,
        imputation_medians=medians,
        drop_duplicates=False,
    )

    out_path = Path(data_cfg["cleaned_train"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_path, index=False)
    print(f"Saved cleaned training data ({len(cleaned)} rows, {len(cleaned.columns)} cols) to {out_path}")
    return cleaned


if __name__ == "__main__":
    clean_and_save()
