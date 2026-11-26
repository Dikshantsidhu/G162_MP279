"""Configuration loading utilities."""

from pathlib import Path
from typing import Any

import yaml


def load_params(params_path: str | Path = "params.yaml") -> dict[str, Any]:
    """Load pipeline parameters from YAML using yaml.safe_load."""
    path = Path(params_path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data
