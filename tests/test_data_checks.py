"""Unit tests for the seven automated data quality checks (using synthetic data)."""

from __future__ import annotations

import pandas as pd

from mp279.data_checks import (
    check_duplicate_rows,
    check_leaking_column,
    check_sentinel_value,
    check_spelling_variants,
    check_stuck_sensor,
    check_target_typos,
    check_unit_mixup,
    run_all_checks,
)


def _make_clean_df(n: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=n, freq="D").strftime("%Y-%m-%d")
    blocks = ["Block North", "Block South", "Block East"] * (n // 3)
    areas = ["tribal", "rural", "urban"] * (n // 3)
    return pd.DataFrame(
        {
            "record_id": [f"R{i:05d}" for i in range(n)],
            "date": dates,
            "cards_attached": [500 + i * 20 for i in range(n)],
            "working_hours": [4.0 + (i % 6) for i in range(n)],
            "biometric_failures": [10 + (i % 25) for i in range(n)],
            "temp_c": [20.0 + (i % 15) for i in range(n)],
            "stock_kg_morning": [1000.0 + i * 30 for i in range(n)],
            "days_to_month_end": [i % 30 for i in range(n)],
            "shop_area": areas,
            "block": blocks,
            "grain_kg": [400.0 + i * 15 for i in range(n)],
        }
    )


def test_leaking_column_detected_and_absent() -> None:
    clean_df = _make_clean_df()
    holdout_df = clean_df.drop(columns=["grain_kg"]).copy()
    assert check_leaking_column(clean_df, holdout_df) == {
        "present": False,
        "column": None,
    }

    leaky_df = clean_df.copy()
    leaky_df["kg_entered_in_epos_at_close"] = leaky_df["grain_kg"] + 1.5
    assert check_leaking_column(leaky_df, holdout_df) == {
        "present": True,
        "column": "kg_entered_in_epos_at_close",
    }


def test_unit_mixup_detected_and_absent() -> None:
    clean_df = _make_clean_df()
    assert check_unit_mixup(clean_df) == {
        "present": False,
        "column": None,
        "group_value": None,
        "factor": None,
    }

    mixed_df = clean_df.copy()
    mixed_df.loc[mixed_df["block"] == "Block South", "stock_kg_morning"] *= 1000
    assert check_unit_mixup(mixed_df) == {
        "present": True,
        "column": "stock_kg_morning",
        "group_value": "Block South",
        "factor": 1000,
    }


def test_sentinel_value_detected_and_absent() -> None:
    clean_df = _make_clean_df()
    assert check_sentinel_value(clean_df) == {
        "present": False,
        "column": None,
        "value": None,
        "count": None,
    }

    sent_df = clean_df.copy()
    sent_df.loc[:5, "biometric_failures"] = 9999
    assert check_sentinel_value(sent_df) == {
        "present": True,
        "column": "biometric_failures",
        "value": 9999,
        "count": 6,
    }


def test_duplicate_rows_and_stuck_sensor_and_target_typos() -> None:
    clean_df = _make_clean_df()
    assert check_duplicate_rows(clean_df) == {"present": False, "count": None}
    dup_df = pd.concat([clean_df, clean_df.iloc[:2]], ignore_index=True)
    assert check_duplicate_rows(dup_df) == {"present": True, "count": 2}

    assert check_stuck_sensor(clean_df)["present"] is False
    stuck_df = clean_df.copy()
    stuck_df.loc[5:16, "temp_c"] = 31.4
    stuck_res = check_stuck_sensor(stuck_df)
    assert stuck_res["present"] is True
    assert stuck_res["column"] == "temp_c"

    assert check_target_typos(clean_df) == {"present": False, "count": None}
    typo_df = clean_df.copy()
    typo_df.loc[0:1, "grain_kg"] *= 10.0
    assert check_target_typos(typo_df) == {"present": True, "count": 2}


def test_spelling_variants_and_run_all_checks() -> None:
    clean_df = _make_clean_df()
    holdout_df = clean_df.drop(columns=["grain_kg"]).copy()
    later_df = clean_df.copy()

    messy_df = clean_df.copy()
    messy_df["kg_entered_in_epos_at_close"] = messy_df["grain_kg"] + 0.5
    messy_df.loc[messy_df["block"] == "Block South", "stock_kg_morning"] *= 1000
    messy_df.loc[:5, "biometric_failures"] = 9999
    messy_df.loc[0, "shop_area"] = "tri."
    messy_df.loc[1, "shop_area"] = " Rural "

    sp_res = check_spelling_variants(messy_df, reference_df=later_df)
    assert sp_res == {"present": True, "column": "shop_area", "true_levels": 3}

    summary = run_all_checks(messy_df, holdout_df=holdout_df, later_df=later_df)
    assert summary["leaking_column"]["present"] is True
    assert summary["unit_mixup"]["present"] is True
    assert summary["sentinel_value"]["present"] is True
    assert summary["duplicate_rows"]["present"] is False
    assert summary["stuck_sensor"]["present"] is False
    assert summary["target_typos"]["present"] is False
    assert summary["spelling_variants"]["present"] is True
