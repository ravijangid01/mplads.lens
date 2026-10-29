"""Shared helpers. Every detector module exposes NAME and run(works, cfg) -> DataFrame[work_id, detector, score, reason, evidence]."""
from __future__ import annotations
import numpy as np
import pandas as pd

COLUMNS = ["work_id", "detector", "score", "reason", "evidence"]


def frame(name: str, work_ids, scores, reasons, evidence) -> pd.DataFrame:
    """Build the standard detector output frame (scores clipped to 0-100)."""
    out = pd.DataFrame({"work_id": list(work_ids), "score": np.clip(np.asarray(list(scores), dtype=float), 0, 100),
                        "reason": list(reasons), "evidence": list(evidence)})
    out.insert(1, "detector", name)
    return out[COLUMNS]


def empty(name: str) -> pd.DataFrame:
    return frame(name, [], [], [], [])


def scale(x, lo: float, span: float, floor: float = 50.0, ceil: float = 100.0):
    """Map x=lo -> floor and x=lo+span -> ceil (linear, clipped)."""
    return np.clip(floor + (np.asarray(x, dtype=float) - lo) / span * (ceil - floor), floor, ceil)


def lakh(x: float) -> str:
    return f"{x / 1e5:,.1f}"
