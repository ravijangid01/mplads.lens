"""Combine detector scores into a final 0-100 score with severity."""
from __future__ import annotations
import numpy as np
import pandas as pd

DETECTORS = ["cost", "duplicate", "stalled", "agency", "delay"]


def wide(dets: pd.DataFrame, work_ids: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot long detector output into (scores, reasons) frames indexed by work_id."""
    idx = pd.Index(work_ids, name="work_id")
    s = dets.pivot_table(index="work_id", columns="detector", values="score", aggfunc="max").reindex(idx).reindex(columns=DETECTORS).fillna(0.0)
    r = dets.drop_duplicates(["work_id", "detector"]).pivot(index="work_id", columns="detector", values="reason").reindex(idx).reindex(columns=DETECTORS).fillna("")
    return s, r


def severity(score: pd.Series, cfg: dict) -> pd.Series:
    sv = cfg["severity"]
    return pd.Series(np.where(score >= sv["high"], "High", np.where(score >= sv["medium"], "Medium", "Low")), index=score.index)


def combine(scores: pd.DataFrame, reasons: pd.DataFrame, weights: dict, cfg: dict) -> pd.DataFrame:
    """Weighted sum (weights rescaled so the largest is 1) + multi-detector bonus, capped at 100."""
    w = np.array([max(float(weights.get(d, 0)), 0.0) for d in DETECTORS])
    rel = w / w.max() if w.max() > 0 else w
    contrib = scores.to_numpy() * rel
    fired = (scores.to_numpy() > 0) & (rel > 0)
    n = fired.sum(axis=1)
    final = np.minimum(100.0, contrib.sum(axis=1) + np.where(n >= 2, cfg["multi_detector_bonus"], 0))
    top = contrib.argmax(axis=1)
    reason = reasons.to_numpy()[np.arange(len(top)), top]
    out = pd.DataFrame({"score": np.round(final, 1), "n_detectors": n, "top_detector": np.array(DETECTORS)[top], "reason": reason}, index=scores.index)
    out.loc[final == 0, ["top_detector", "reason"]] = ""
    out["severity"] = severity(out["score"], cfg)
    out["flagged"] = out["score"] >= cfg["flag_threshold"]
    for i, d in enumerate(DETECTORS):
        out[f"c_{d}"] = contrib[:, i]
        out[f"s_{d}"] = scores[d].to_numpy()
    return out.reset_index()
