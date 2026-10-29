"""Detector 1 - cost outlier: robust z-score of log(sanctioned amount) within a peer group (median/MAD)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pipeline import peer_groups
from . import base

NAME = "cost"


def compute(works: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per-work peer group info, median, robust sigma and z (all works). Index matches ``works``."""
    c = cfg["cost"]
    pg = peer_groups.assign(works, c["min_peer_size"])
    d = pd.concat([works[["work_id", "sanctioned_amount"]], pg], axis=1)
    d["logc"] = np.log(d["sanctioned_amount"].clip(lower=1))
    g = d.groupby(["pg_level", "pg_key"])["logc"]
    d["med"] = g.transform("median")
    d["mad"] = (d["logc"] - d["med"]).abs().groupby([d["pg_level"], d["pg_key"]]).transform("median") * 1.4826
    d["sigma"] = d["mad"].clip(lower=c["mad_floor"])
    d["z"] = (d["logc"] - d["med"]) / d["sigma"]
    return d


def run(works: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    d = compute(works, cfg)
    thr = cfg["cost"]["z_threshold"]
    f = d[(d["z"] >= thr) & (d["pg_n"] >= cfg["cost"]["min_peer_size"])]
    if f.empty:
        return base.empty(NAME)
    ratio = np.exp(f["logc"] - f["med"])
    scores = base.scale(f["z"], thr, 3.5)
    reasons, ev = [], []
    for (_, r), rt in zip(f.iterrows(), ratio):
        lab = peer_groups.label(int(r.pg_level), r.pg_key)
        note = "" if r.pg_level == 0 else f" Peer group widened to {peer_groups.LEVEL_NAMES[int(r.pg_level)]}."
        reasons.append(f"Costs {rt:.1f}x the median of {int(r.pg_n)} similar works ({lab}).{note}")
        ev.append({"level": int(r.pg_level), "level_name": peer_groups.LEVEL_NAMES[int(r.pg_level)], "key": r.pg_key, "label": lab,
                   "n": int(r.pg_n), "median": float(np.exp(r.med)), "ratio": float(np.exp(r.logc - r.med)), "z": float(r.z)})
    return base.frame(NAME, f["work_id"], scores, reasons, ev)
