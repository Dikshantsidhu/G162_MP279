"""Unit tests for data cleaning and params loading (using synthetic data)."""

from __future__ import annotations

import pandas as pd

from mp279.cleaning import clean_dataframe, compute_imputation_medians
from mp279.config import load_params


def test_load_params_reads_yaml() -> None:
    params = load_params("params.yaml")
    assert params["data"]["raw_train"] == "data/raw/train.csv"
    assert params["cleaning"]["unit_mixup"]["factor"] == 1000
    assert params["cleaning"]["sentinel"]["value"] == 9999


def test_clean_dataframe_fixes_all_issues_and_is_idempotent() -> None:
    params = load_params("params.yaml")
    raw = pd.DataFrame(
        {
            "record_id": ["R1", "R2", "R3", "R3"],
            "date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-03"],
            "cards_attached": [500, 600, 700, 700],
            "working_hours": [6.0, 7.0, 8.0, 8.0],
            "biometric_failures": [10, 9999, 30, 30],
            "temp_c": [25.0, 26.0, 27.0, 27.0],
            "stock_kg_morning": [1500.0, 2000000.0, 1800.0, 1800.0],
            "days_to_month_end": [10, 9, 8, 8],
            "shop_area": ["tri.", " Rural ", "URB.", "URB."],
            "block": ["Block North", "Block South", "Block East", "Block East"],
            "kg_entered_in_epos_at_close": [501.0, 602.0, 703.0, 703.0],
            "grain_kg": [500.0, 600.0, 700.0, 700.0],
        }
    )

    medians = compute_imputation_medians(raw.drop_duplicates())
    assert medians["biometric_failures"] == 20.0

    cleaned = clean_dataframe(raw, params=params, imputation_medians=medians)
    assert "kg_entered_in_epos_at_close" not in cleaned.columns
    assert len(cleaned) == 3
    assert cleaned.loc[cleaned["block"] == "Block South", "stock_kg_morning"].iloc[0] == 2000.0
    assert (cleaned["biometric_failures"] != 9999).all()
    assert cleaned.loc[1, "biometric_failures"] == 20.0
    assert set(cleaned["shop_area"]) == {"tribal", "rural", "urban"}

    # Applying cleaning again does not re-divide Block South stock_kg_morning
    cleaned_twice = clean_dataframe(cleaned, params=params, imputation_medians=medians)
    pd.testing.assert_frame_equal(cleaned, cleaned_twice)
