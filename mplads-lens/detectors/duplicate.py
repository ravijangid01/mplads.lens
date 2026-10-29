"""Detector 2 - near-duplicate works at the same block: TF-IDF cosine + rapidfuzz token_set_ratio within a date window."""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from . import base

NAME = "duplicate"
STOP = set("a an the of at near in on for and to with by from construction improvement providing repair work new".split())


def normalise(text: object) -> str:
    """Lowercase; strip numbers, punctuation and stopwords."""
    t = re.sub(r"[^a-z\s]", " ", str(text).lower())
    return " ".join(w for w in t.split() if w not in STOP)


def run(works: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    c = cfg["duplicate"]
    w = works.reset_index(drop=True)
    norm = w["work_description"].fillna(w["work_title"]).map(normalise).to_numpy()
    when = w["sanctioned_date"].fillna(w["recommended_date"])
    if len(w) < 2:
        return base.empty(NAME)
    X = TfidfVectorizer(min_df=1).fit_transform(norm)
    best: dict[int, tuple] = {}
    for _, idx in w.groupby(["block", "work_category"]).indices.items():
        if len(idx) < 2:
            continue
        S = (X[idx] @ X[idx].T).toarray()
        ii, jj = np.where(np.triu(S >= c["cosine_min"], 1))
        for a, b in zip(ii, jj):
            i, j = idx[a], idx[b]
            gap = abs((when.iloc[i] - when.iloc[j]).days) if pd.notna(when.iloc[i]) and pd.notna(when.iloc[j]) else 10**6
            if gap > c["window_days"]:
                continue
            fz = fuzz.token_set_ratio(norm[i], norm[j])
            if fz < c["fuzz_min"]:
                continue
            first, second = (i, j) if (when.iloc[i], w.work_id.iloc[i]) <= (when.iloc[j], w.work_id.iloc[j]) else (j, i)
            same_village = w.block_or_village.iloc[i] == w.block_or_village.iloc[j]
            sc = float(base.scale(S[a, b], c["cosine_min"], 1 - c["cosine_min"], 55, 100)) - (0 if same_village else 10)
            if second not in best or sc > best[second][0]:
                best[second] = (sc, first, float(S[a, b]), float(fz), int(gap), bool(same_village))
    if not best:
        return base.empty(NAME)
    ids, sc, rs, ev = [], [], [], []
    for sec, (s, first, cos, fz, gap, sv) in best.items():
        ids.append(w.work_id.iloc[sec]); sc.append(s)
        where = "at this location" if sv else "in this block"
        rs.append(f"Same description appears twice {where}, sanctioned {gap} days apart.")
        ev.append({"pair_work_id": w.work_id.iloc[first], "cosine": cos, "fuzz": fz, "days_apart": gap, "same_village": sv})
    return base.frame(NAME, ids, sc, rs, ev)
