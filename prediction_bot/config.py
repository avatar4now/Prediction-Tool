"""Configuration loading from YAML."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration from a YAML file.

    Looks in this order:
    1. Explicit path argument
    2. PREDBOT_CONFIG environment variable
    3. config.yaml in the project root
    """
    if path is None:
        path = os.environ.get("PREDBOT_CONFIG", _DEFAULT_CONFIG_PATH)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def get_nested(cfg: dict, *keys: str, default: Any = None) -> Any:
    """Safely get a nested config value."""
    current = cfg
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
        if current is default:
            return default
    return current
