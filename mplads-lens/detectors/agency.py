"""Detector 4 - agency / district pattern: IsolationForest on aggregate features, explained by the top-2 deviating features."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from . import base, cost, stalled

NAME = "agency"
# key -> (label, floor of robust scale, formatter)
FEATURES = {"completion_rate": ("Completion rate", .05, lambda v: f"{v * 100:.0f}%"), "share_stalled": ("Share stalled", .03, lambda v: f"{v * 100:.0f}%"),
            "mean_delay": ("Mean sanction delay", 5.0, lambda v: f"{v:.0f} days"), "spend_ratio": ("Spend-to-release ratio", .03, lambda v: f"{v * 100:.0f}%"),
            "mean_cost_z": ("Mean cost z-score", .15, lambda v: f"{v:+.2f}")}


def entity_features(works: pd.DataFrame, cfg: dict, by: str, z: pd.Series | None = None) -> pd.DataFrame:
    """Aggregate features per agency or district."""
    z = (cost.compute(works, cfg)["z"] if z is None else z).clip(-6, 6)
    st, _, _ = stalled.mask(works, cfg)
    d = pd.DataFrame({by: works[by], "done": works["status"].eq("completed"), "stalled": st, "delay": works["sanction_delay_days"], "z": z,
                      "rel": works["released_amount"], "exp": works["expenditure_amount"]})
    g = d.groupby(by)
    out = pd.DataFrame({"n_works": g.size(), "completion_rate": g["done"].mean(), "share_stalled": g["stalled"].mean(), "mean_delay": g["delay"].mean(),
                        "spend_ratio": g["exp"].sum() / g["rel"].sum().replace(0, np.nan), "mean_cost_z": g["z"].mean()})
    out["avg_amount"] = works.groupby(by)["sanctioned_amount"].mean()
    return out.fillna({"mean_delay": 0, "spend_ratio": 1})


def score_entities(feat: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return feat plus dev_* (robust deviations), outlier flag, score and top-2 explanation."""
    c = cfg["agency"]
    f = feat.copy()
    keys = list(FEATURES)
    med = f[keys].median()
    dev = pd.DataFrame({k: (f[k] - med[k]) / np.maximum(1.4826 * (f[k] - med[k]).abs().median(), FEATURES[k][1]) for k in keys})
    elig = f["n_works"] >= c["min_works"]
    f["flag"], f["score"], f["max_dev"] = False, 0.0, dev.abs().max(axis=1)
    if elig.sum() >= 5:
        iso = IsolationForest(n_estimators=200, contamination=c["contamination"], random_state=c["seed"]).fit(dev[elig])
        pred = pd.Series(iso.predict(dev[elig]) == -1, index=dev[elig].index)
        f.loc[pred.index, "flag"] = pred & (f.loc[pred.index, "max_dev"] >= c["min_dev_z"])
    f["score"] = np.where(f["flag"], base.scale(f["max_dev"], c["min_dev_z"], 5, 50, 100), 0.0)
    f["explain"] = ""
    for e in f.index[f["flag"]]:
        top = dev.loc[e].abs().sort_values(ascending=False).index[:2]
        f.at[e, "explain"] = "; ".join(f"{FEATURES[k][0]} {FEATURES[k][2](f.at[e, k])} vs peer median {FEATURES[k][2](med[k])}" for k in top)
    return f.join(dev.add_prefix("dev_")), med


def run(works: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = {}
    zall = cost.compute(works, cfg)["z"]
    for by, kind in [("implementing_agency", "agency"), ("district", "district")]:
        f, med = score_entities(entity_features(works, cfg, by, zall), cfg)
        for e in f.index[f["flag"]]:
            top = f.loc[e, [f"dev_{k}" for k in FEATURES]].abs().sort_values(ascending=False).index[:2]
            ev = {"kind": kind, "entity": e, "features": {k: float(f.at[e, k]) for k in FEATURES}, "peer_median": {k: float(med[k]) for k in FEATURES},
                  "top": [t[4:] for t in top]}
            rows[e] = (kind, float(f.at[e, "score"]), f.at[e, "explain"], ev)
    ids, sc, rs, ev = [], [], [], []
    cols = {"agency": "implementing_agency", "district": "district"}
    best: dict[str, tuple] = {}
    for kind, col in cols.items():
        for e, (k, s, r, evd) in rows.items():
            if k != kind:
                continue
            for wid in works.loc[works[col] == e, "work_id"]:
                if wid not in best or s > best[wid][0]:
                    best[wid] = (s, f"{k.title()} pattern: {r}.", evd)
    for wid, (s, r, evd) in best.items():
        ids.append(wid); sc.append(s); rs.append(r); ev.append(evd)
    return base.frame(NAME, ids, sc, rs, ev)
