"""Automated detection for the seven data quality issues in MP279."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mp279.config import load_params

VALID_UNIT_FACTORS = (10, 12, 60, 100, 1000)
SENTINEL_CANDIDATES = (-9999, -999, -99, -1, 999, 9999, 99999)
ABBREVIATION_MAP = {
    "tri.": "tribal",
    "rur.": "rural",
    "urb.": "urban",
}


def compute_file_sha256(file_path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with Path(file_path).open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check_leaking_column(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame | None = None,
    target_col: str = "grain_kg",
    id_col: str = "record_id",
    corr_threshold: float = 0.98,
) -> dict[str, Any]:
    """Detect a feature only known post-event that leaks the target."""
    numeric_cols = [
        c
        for c in train_df.select_dtypes(include=[np.number]).columns
        if c not in {target_col, id_col}
    ]
    missing_in_holdout: set[str] = set()
    if holdout_df is not None:
        missing_in_holdout = set(train_df.columns) - set(holdout_df.columns) - {target_col}

    best_col: str | None = None
    best_corr = 0.0
    for col in numeric_cols:
        series = train_df[col]
        if series.nunique(dropna=True) <= 1:
            continue
        corr = abs(float(series.corr(train_df[target_col])))
        in_missing = col in missing_in_holdout if holdout_df is not None else True
        if corr >= corr_threshold and in_missing and corr > best_corr:
            best_corr = corr
            best_col = col

    if best_col is None and missing_in_holdout:
        # Fallback if correlation wasn't linear or numeric
        for col in sorted(missing_in_holdout):
            if col in numeric_cols:
                best_col = col
                break

    return {
        "present": best_col is not None,
        "column": best_col,
    }


def check_unit_mixup(
    df: pd.DataFrame,
    group_col: str = "block",
    target_col: str = "grain_kg",
    valid_factors: tuple[int, ...] = VALID_UNIT_FACTORS,
) -> dict[str, Any]:
    """Detect whether one group recorded a numeric column in a scaled unit."""
    if group_col not in df.columns or df[group_col].nunique(dropna=True) < 2:
        return {"present": False, "column": None, "group_value": None, "factor": None}

    numeric_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c != target_col
    ]

    for col in numeric_cols:
        group_medians = df.groupby(group_col)[col].median().dropna()
        if len(group_medians) < 2:
            continue

        for grp_val, grp_med in group_medians.items():
            other_med = float(group_medians.drop(index=grp_val).median())
            if other_med <= 0 or grp_med <= 0:
                continue
            ratio = float(grp_med) / other_med
            for factor in valid_factors:
                if 0.75 * factor <= ratio <= 1.35 * factor:
                    return {
                        "present": True,
                        "column": col,
                        "group_value": str(grp_val),
                        "factor": int(factor),
                    }

    return {"present": False, "column": None, "group_value": None, "factor": None}


def check_sentinel_value(
    df: pd.DataFrame,
    target_col: str = "grain_kg",
    candidates: tuple[int | float, ...] = SENTINEL_CANDIDATES,
    min_count: int = 5,
) -> dict[str, Any]:
    """Detect a repeated impossible sentinel value in any numeric column."""
    numeric_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c != target_col
    ]

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        vc = series.value_counts()

        # Check explicit sentinel candidates that sit outside the normal bulk
        q1, q3 = float(series.quantile(0.25)), float(series.quantile(0.75))
        iqr = max(q3 - q1, 1e-9)
        for cand in candidates:
            cnt = int(vc.get(cand, 0))
            if cnt >= min_count and (cand < q1 - 3 * iqr or cand > q3 + 3 * iqr):
                val: int | float = int(cand) if float(cand).is_integer() else float(cand)
                return {
                    "present": True,
                    "column": col,
                    "value": val,
                    "count": cnt,
                }

    return {"present": False, "column": None, "value": None, "count": None}


def check_duplicate_rows(df: pd.DataFrame) -> dict[str, Any]:
    """Detect exact duplicate rows in the dataset."""
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        return {"present": True, "count": dup_count}
    return {"present": False, "count": None}


def check_stuck_sensor(
    df: pd.DataFrame,
    date_col: str = "date",
    target_col: str = "grain_kg",
    min_run_length: int = 10,
    min_days: int = 3,
) -> dict[str, Any]:
    """Detect a prolonged consecutive run of identical values when sorted by date."""
    if date_col not in df.columns or df.empty:
        return {"present": False, "column": None, "start_date": None, "end_date": None}

    sorted_df = df.sort_values(date_col).reset_index(drop=True)
    numeric_cols = [
        c
        for c in sorted_df.select_dtypes(include=[np.number]).columns
        if c != target_col
    ]

    for col in numeric_cols:
        s = sorted_df[col]
        run_ids = (s != s.shift()).cumsum()
        runs = sorted_df.groupby(run_ids).agg(
            cnt=(col, "size"),
            start=(date_col, "first"),
            end=(date_col, "last"),
        )
        long_runs = runs[runs["cnt"] >= min_run_length]
        for _, row in long_runs.sort_values("cnt", ascending=False).iterrows():
            day_span = (pd.to_datetime(row["end"]) - pd.to_datetime(row["start"])).days
            if day_span >= min_days:
                return {
                    "present": True,
                    "column": col,
                    "start_date": str(row["start"]),
                    "end_date": str(row["end"]),
                }

    return {"present": False, "column": None, "start_date": None, "end_date": None}


def check_target_typos(
    df: pd.DataFrame,
    target_col: str = "grain_kg",
    gap_ratio: float = 3.0,
) -> dict[str, Any]:
    """Detect target values typed ~10x too large with a large gap below them."""
    if target_col not in df.columns or len(df) < 10:
        return {"present": False, "count": None}

    sorted_vals = np.sort(df[target_col].dropna().to_numpy(dtype=float))
    if len(sorted_vals) < 10:
        return {"present": False, "count": None}

    # Look for a jump where an upper tail sits >= gap_ratio times above the point below it
    pos_mask = sorted_vals > 0
    vals = sorted_vals[pos_mask]
    if len(vals) < 10:
        return {"present": False, "count": None}

    ratios = vals[1:] / vals[:-1]
    max_idx = int(np.argmax(ratios))
    max_ratio = float(ratios[max_idx])
    tail_count = len(vals) - (max_idx + 1)

    if max_ratio >= gap_ratio and 1 <= tail_count <= max(1, int(0.1 * len(vals))):
        return {"present": True, "count": int(tail_count)}

    return {"present": False, "count": None}


def _normalize_category(value: Any, mapping: dict[str, str]) -> str:
    cleaned = str(value).strip().lower()
    return mapping.get(cleaned, cleaned)


def check_spelling_variants(
    df: pd.DataFrame,
    reference_df: pd.DataFrame | None = None,
    ignore_cols: tuple[str, ...] = ("record_id", "date"),
    mapping: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Detect categorical columns with casing/whitespace/abbreviation variants."""
    abbrev = mapping if mapping is not None else ABBREVIATION_MAP
    text_cols = [
        c
        for c in df.select_dtypes(include=["object", "string", "category"]).columns
        if c not in ignore_cols
    ]

    for col in text_cols:
        raw_levels = set(df[col].dropna().astype(str).unique())
        normalized = df[col].dropna().map(lambda v: _normalize_category(v, abbrev))
        clean_levels = set(normalized.unique())

        ref_levels = (
            set(reference_df[col].dropna().astype(str).unique())
            if reference_df is not None and col in reference_df.columns
            else clean_levels
        )

        if len(raw_levels) > len(clean_levels) or raw_levels != ref_levels:
            return {
                "present": True,
                "column": col,
                "true_levels": int(len(ref_levels)),
            }

    return {"present": False, "column": None, "true_levels": None}


def run_all_checks(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame | None = None,
    later_df: pd.DataFrame | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run all seven data quality checks and return structured findings."""
    cleaning_cfg = (params or {}).get("cleaning", {})
    target_col = cleaning_cfg.get("target_col", "grain_kg")
    date_col = cleaning_cfg.get("date_col", "date")
    id_col = cleaning_cfg.get("id_col", "record_id")
    group_col = cleaning_cfg.get("unit_mixup", {}).get("group_col", "block")
    spelling_map = cleaning_cfg.get("spelling", {}).get("mapping", ABBREVIATION_MAP)

    return {
        "leaking_column": check_leaking_column(
            train_df, holdout_df=holdout_df, target_col=target_col, id_col=id_col
        ),
        "unit_mixup": check_unit_mixup(
            train_df, group_col=group_col, target_col=target_col
        ),
        "sentinel_value": check_sentinel_value(train_df, target_col=target_col),
        "duplicate_rows": check_duplicate_rows(train_df),
        "stuck_sensor": check_stuck_sensor(
            train_df, date_col=date_col, target_col=target_col
        ),
        "target_typos": check_target_typos(train_df, target_col=target_col),
        "spelling_variants": check_spelling_variants(
            train_df,
            reference_df=later_df,
            ignore_cols=(id_col, date_col),
            mapping=spelling_map,
        ),
    }


def main(params_path: str | Path = "params.yaml") -> None:
    """Load datasets from params.yaml, run all 7 checks, and print findings."""
    params = load_params(params_path)
    data_cfg = params["data"]
    train_df = pd.read_csv(data_cfg["raw_train"])
    later_df = pd.read_csv(data_cfg["raw_later"])
    holdout_df = pd.read_csv(data_cfg["holdout"])

    results = run_all_checks(
        train_df=train_df,
        holdout_df=holdout_df,
        later_df=later_df,
        params=params,
    )
    sha256 = compute_file_sha256(data_cfg["raw_train"])
    print(f"raw_train SHA-256: {sha256}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
