"""Data loading (synthetic on first run, or user data) and the detector pipeline."""
from __future__ import annotations
import json
from typing import Callable
import pandas as pd
from detectors import REGISTRY
from . import clean, synthetic
from .config import ACTIVE, PROCESSED, USER_DATA, load_config

DATE_COLS = ["recommended_date", "sanctioned_date", "last_progress_date", "completion_date"]


def active_meta() -> dict:
    """{'source': 'synthetic'|'user', 'as_of': 'YYYY-MM-DD'} (defaults to synthetic)."""
    if ACTIVE.exists():
        return json.loads(ACTIVE.read_text())
    return {"source": "synthetic", "as_of": load_config()["as_of"]}


def set_active(source: str, as_of: str) -> None:
    ACTIVE.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE.write_text(json.dumps({"source": source, "as_of": as_of}))


def ensure_synthetic(force: bool = False) -> None:
    """Create the synthetic dataset if missing (or when ``force``)."""
    if force or not PROCESSED.exists():
        PROCESSED.parent.mkdir(parents=True, exist_ok=True)
        cfg = load_config()
        df = clean.clean(synthetic.generate(seed=cfg["seed"], as_of=cfg["as_of"]))
        df.to_csv(PROCESSED, index=False, compression="gzip")


def signature() -> str:
    m = active_meta()
    p = USER_DATA if m["source"] == "user" else PROCESSED
    ensure_synthetic()
    return f"{m['source']}|{m['as_of']}|{p.stat().st_mtime_ns}"


def load_works() -> tuple[pd.DataFrame, dict]:
    """Return (works, meta) for the active dataset."""
    ensure_synthetic()
    m = active_meta()
    p = USER_DATA if m["source"] == "user" and USER_DATA.exists() else PROCESSED
    d = pd.read_csv(p, parse_dates=DATE_COLS)
    return d, m


def run_all(works: pd.DataFrame, cfg: dict, progress: Callable[[float, str], None] | None = None) -> pd.DataFrame:
    """Run all five detectors; returns the long frame work_id/detector/score/reason/evidence."""
    parts = []
    for i, (name, fn) in enumerate(REGISTRY.items()):
        if progress:
            progress(i / len(REGISTRY), f"Running {name} detector…")
        parts.append(fn(works, cfg))
    if progress:
        progress(1.0, "Done")
    return pd.concat(parts, ignore_index=True)
