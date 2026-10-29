"""Detector 5 - process delay: late sanction (guideline threshold) or completion time far above the peer 90th percentile."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import base

NAME = "delay"


def run(works: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    c = cfg["delay"]
    thr = c["sanction_delay_days"]
    d = works[["work_id", "work_category", "district_class", "sanction_delay_days", "completion_days"]].copy()
    ok = d["completion_days"].notna() & (d["completion_days"] > 0)
    g = d[ok].groupby(["work_category", "district_class"])["completion_days"]
    p_cd = g.quantile(c["percentile"] / 100).rename("p_cd")
    p_c = d[ok].groupby("work_category")["completion_days"].quantile(c["percentile"] / 100).rename("p_c")
    n_cd = g.size().rename("n_cd")
    d = d.join(p_cd, on=["work_category", "district_class"]).join(n_cd, on=["work_category", "district_class"]).join(p_c, on="work_category")
    d["p90"] = np.where(d["n_cd"].fillna(0) >= 8, d["p_cd"], d["p_c"])
    late = d["sanction_delay_days"] > thr
    over = ok & (d["completion_days"] > d["p90"] * c["overrun_factor"])
    f = d[late | over]
    if f.empty:
        return base.empty(NAME)
    s1 = np.where(f["sanction_delay_days"] > thr, base.scale(f["sanction_delay_days"], thr, 2 * thr), 0)
    ratio = (f["completion_days"] / f["p90"]).fillna(0)
    s2 = np.where(ratio > c["overrun_factor"], base.scale(ratio, c["overrun_factor"], 1.0), 0)
    rs, ev = [], []
    for (_, r), a, b in zip(f.iterrows(), s1, s2):
        parts = []
        if a > 0:
            parts.append(f"Sanctioned {int(r.sanction_delay_days)} days after recommendation (guideline threshold {thr})")
        if b > 0:
            parts.append(f"Completion took {int(r.completion_days)} days vs peer 90th percentile {r.p90:.0f}")
        rs.append("; ".join(parts) + ".")
        ev.append({"sanction_delay_days": None if pd.isna(r.sanction_delay_days) else float(r.sanction_delay_days), "threshold": thr,
                   "completion_days": None if pd.isna(r.completion_days) else float(r.completion_days), "peer_p90": None if pd.isna(r.p90) else float(r.p90)})
    return base.frame(NAME, f["work_id"], np.maximum(s1, s2), rs, ev)
