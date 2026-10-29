"""Detector 3 - stalled work: old, funded, not completed, no recent progress."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import base

NAME = "stalled"


def mask(works: pd.DataFrame, cfg: dict) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (stalled mask, sanction age in days, days without progress)."""
    c = cfg["stalled"]
    asof = pd.Timestamp(cfg["as_of"])
    sdate = works["sanctioned_date"].fillna(works["recommended_date"])
    age = (asof - sdate).dt.days
    noprog = (asof - works["last_progress_date"]).dt.days.fillna(age)
    m = (works["status"] != "completed") & (age > c["min_sanction_age_days"]) & (works["released_amount"] > 0) & (noprog > c["min_no_progress_days"])
    return m, age, noprog


def run(works: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    m, age, noprog = mask(works, cfg)
    f = works[m]
    if f.empty:
        return base.empty(NAME)
    yrs = noprog[m] / 365.25
    sc = np.clip(60 + 25 * (yrs - 1), 60, 100)
    rs, ev = [], []
    for (_, r), y in zip(f.iterrows(), yrs):
        none = pd.isna(r.last_progress_date)
        rs.append(f"Funded Rs {base.lakh(r.released_amount)} lakh and no progress recorded for {y:.1f} years" + (" (no update ever recorded)." if none else "."))
        ev.append({"recommended": r.recommended_date, "sanctioned": r.sanctioned_date, "last_progress": r.last_progress_date, "as_of": pd.Timestamp(cfg["as_of"]),
                   "released": float(r.released_amount), "sanctioned_amount": float(r.sanctioned_amount), "years_stalled": float(y), "completion_percent": r.completion_percent})
    return base.frame(NAME, f["work_id"], sc, rs, ev)
