"""Configuration loading (paths are relative to the repository root)."""
from __future__ import annotations
from pathlib import Path
import copy
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROCESSED = DATA_DIR / "processed" / "works.csv.gz"
USER_DATA = DATA_DIR / "user" / "works_user.csv.gz"
ACTIVE = DATA_DIR / "active.json"


def load_config(path: Path | None = None) -> dict:
    """Load config.yaml and return a fresh dict."""
    with open(path or ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        return copy.deepcopy(yaml.safe_load(f))
