"""Synthetic MPLADS-like works generator (fixed seed). All names are invented.

Anomalies are planted in the hidden column ``_planted_type``; detectors never read it.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

CLASSES = ["plain", "hill", "desert", "tribal", "coastal"]
CLASS_MULT = {"plain": 1.0, "hill": 1.25, "desert": 1.2, "tribal": 1.1, "coastal": 1.05}
CATEGORIES = {  # median cost (rupees), sampling weight, item names
    "road": (12e5, .28, ["CC road", "Interlocking paver block road", "Link road", "Drain and road", "Approach road", "Bituminous road", "Village street road", "Culvert and road"]),
    "community hall": (15e5, .14, ["Community hall", "Multipurpose hall", "Community centre", "Panchayat bhawan", "Shed for gatherings", "Hall extension", "Open stage and hall", "Meeting hall"]),
    "drinking water": (8e5, .16, ["Hand pump cluster", "Overhead water tank", "Piped water line", "Borewell with pump", "Water purification unit", "Community tap stand", "Rain water harvesting tank", "Pump house"]),
    "school": (14e5, .12, ["School classroom block", "School boundary wall", "School toilet block", "Library room", "Science lab room", "School kitchen shed", "School gate and path", "Computer room"]),
    "health": (18e5, .08, ["Sub-centre building", "Dispensary room", "Ambulance shed", "Health centre ward", "Delivery room", "Medicine store", "Health centre boundary", "Waiting shed"]),
    "sanitation": (6e5, .12, ["Community toilet", "Drainage line", "Waste collection shed", "Sewer line", "Bathing ghat", "Soak pit block", "Public urinal block", "Covered drain"]),
    "other": (10e5, .10, ["Solar street lights", "Bus shelter", "Playground fencing", "Cremation shed", "Market shed", "Pond renovation", "Public park", "Cattle trough"]),
}
PLACES = ["Temple", "Market", "Panchayat Office", "Primary School", "Pond", "Chowk", "Colony", "Bus Stand", "Dispensary", "Playground",
          "Water Tank", "Cooperative Society", "Post Office", "Weekly Haat", "Ration Shop", "Anganwadi", "Fair Ground", "Canal", "Bridge", "Railway Crossing"]
NAMES = [a + b for a in ["Ba", "Ka", "Ma", "Ra", "Sa", "Ta", "Da", "Na", "Pa", "La"] for b in ["lu", "ri", "no", "ve", "ta", "mi", "sha", "ko"]]
VERBS = ["Construction of", "Improvement of", "Providing", "Repair of"]
PLANTED_AGENCY_IDX = (5, 21)


def generate(n: int = 20000, seed: int = 42, as_of: str = "2026-10-29") -> pd.DataFrame:
    """Generate ``n`` synthetic works (about 3-4% carry a planted anomaly label)."""
    rng = np.random.default_rng(seed)
    asof = pd.Timestamp(as_of)
    n_dup = int(n * 0.007)
    m = n - n_dup
    day = pd.Timedelta(days=1)

    d = rng.integers(0, 40, m)
    state = np.array([f"State S{i // 10 + 1}" for i in d])
    dclass = np.array([CLASSES[i % 5] for i in d])
    const_k = d * 2 + rng.integers(0, 2, m)
    block = rng.integers(1, 6, m)
    village = rng.integers(1, 13, m)
    loc = np.array([f"D{a + 1:02d}-B{b}-V{c:02d}" for a, b, c in zip(d, block, village)])
    # agencies: 8 per state, two planted "pattern" agencies with low weight so they own ~110 works each
    agency_idx = np.zeros(m, dtype=int)
    for s in range(4):
        idx = np.where(d // 10 == s)[0]
        ids = np.arange(s * 8, s * 8 + 8)
        w = np.array([0.157 if a in PLANTED_AGENCY_IDX else 1.0 for a in ids])
        agency_idx[idx] = rng.choice(ids, len(idx), p=w / w.sum())
    pa = np.isin(agency_idx, PLANTED_AGENCY_IDX)

    cats = list(CATEGORIES)
    cat = rng.choice(cats, m, p=[CATEGORIES[c][1] for c in cats])
    rec = pd.Timestamp("2023-04-01") + pd.to_timedelta(rng.integers(0, 822, m), unit="D")
    ptype = np.where(pa, "agency_pattern", "none").astype(object)

    # delay planting, then sanction dates
    elig = np.where((~pa) & (rec <= pd.Timestamp("2025-03-01")))[0]
    dly = rng.choice(elig, int(n * 0.005), replace=False)
    ptype[dly] = "delay"
    sdelay = np.maximum(5, np.round(rng.lognormal(np.log(28), .45, m))).astype(int)
    sdelay[pa] = rng.integers(60, 140, pa.sum())
    sdelay[dly] = rng.integers(110, 300, len(dly))
    san = rec + pd.to_timedelta(sdelay, unit="D")

    # stalled and cost planting
    el2 = np.where((ptype == "none") & (san <= asof - 760 * day))[0]
    stl = rng.choice(el2, int(n * 0.0065), replace=False)
    ptype[stl] = "stalled"
    el3 = np.where(ptype == "none")[0]
    cst = rng.choice(el3, int(n * 0.0075), replace=False)
    ptype[cst] = "cost_outlier"

    # amounts
    med = np.array([CATEGORIES[c][0] for c in cat])
    mult = np.array([CLASS_MULT[c] for c in dclass])
    year = np.array(san.year)
    amt = med * mult * 1.04 ** (year - 2023) * rng.lognormal(0, .35, m)
    amt[pa] *= 1.5
    amt[cst] *= rng.uniform(4, 8, len(cst))
    amt = np.maximum(50000, np.round(amt, -3))

    # status model
    dur = np.maximum(60, np.round(rng.lognormal(np.log(620), .45, m))).astype(int)
    age = (asof - san).days.to_numpy()
    done_possible = (san + pd.to_timedelta(dur, unit="D")) < asof
    u = rng.random(m)
    p_done = np.where(pa, 0.25, 0.97)
    completed = done_possible & (u < p_done)
    not_started = (~completed) & (age < 120) & (rng.random(m) < 0.25) & (~pa)
    status = np.where(completed, "completed", np.where(not_started, "not_started", "in_progress")).astype(object)
    status[stl] = np.where(rng.random(len(stl)) < 0.2, "not_started", "in_progress")
    completed = status == "completed"
    not_started = status == "not_started"
    inprog = status == "in_progress"

    pct = np.zeros(m)
    pct[completed] = 100
    el_frac = np.clip(age / dur, 0, 1)
    pct[inprog] = np.clip(100 * el_frac[inprog] * rng.uniform(.7, 1.1, inprog.sum()), 5, 95)
    pct[stl] = np.where(inprog[stl], rng.uniform(10, 60, len(stl)), 0)

    rel = np.zeros(m)
    rel[completed] = amt[completed] * rng.uniform(.92, 1.0, completed.sum())
    rel[inprog] = np.minimum(amt[inprog] * .95, amt[inprog] * pct[inprog] / 100 * rng.uniform(.9, 1.3, inprog.sum()))
    ns_funded = not_started & (rng.random(m) < .3)
    rel[ns_funded] = amt[ns_funded] * rng.uniform(.15, .3, ns_funded.sum())
    rel[stl] = amt[stl] * rng.uniform(.2, .6, len(stl))
    exp = rel * rng.uniform(.97, 1.0, m)
    exp[inprog] = rel[inprog] * rng.uniform(.8, 1.0, inprog.sum())
    exp[pa] = rel[pa] * rng.uniform(.4, .7, pa.sum())
    rel, exp = np.round(rel, -2), np.round(exp, -2)

    comp_date = pd.Series(san + pd.to_timedelta(dur, unit="D")).where(completed)
    lp = pd.Series(pd.NaT, index=range(m), dtype="datetime64[ns]")
    nat_prog = np.minimum(np.array(asof - day), np.maximum((san + 15 * day).to_numpy(), (asof - pd.to_timedelta(rng.integers(5, 150, m), unit="D")).to_numpy()))
    lp[inprog] = pd.Series(nat_prog)[inprog]
    stale = san + pd.to_timedelta(rng.integers(30, 200, m), unit="D")
    pa_ip = pa & inprog
    lp[pa_ip] = pd.Series(stale)[pa_ip].clip(upper=asof - day)
    s_ip = np.zeros(m, bool); s_ip[stl] = inprog[stl]
    lp[s_ip] = pd.Series(san + pd.to_timedelta(rng.integers(20, 150, m), unit="D"))[s_ip]
    drop = np.zeros(m, bool); drop[stl] = rng.random(len(stl)) < .3
    lp[drop] = pd.NaT
    lp[completed] = comp_date[completed]

    item = np.array([CATEGORIES[c][2][i] for c, i in zip(cat, rng.integers(0, 8, m))])
    lm = np.array([f"{PLACES[a]} {NAMES[b]}" for a, b in zip(rng.integers(0, 20, m), rng.integers(0, 80, m))])
    verb = rng.choice(VERBS, m)
    df = pd.DataFrame({
        "mp_ref": [f"MP-{k:03d}" for k in const_k], "constituency": [f"Constituency {k:03d}" for k in const_k],
        "state": state, "district": [f"District {i + 1:02d}" for i in d], "district_class": dclass, "block_or_village": loc,
        "work_title": [f"{a} at {b}" for a, b in zip(item, lm)],
        "work_description": [f"{v} {a} at {b}" for v, a, b in zip(verb, item, lm)],
        "work_category": cat, "implementing_agency": [f"Agency {a + 1:02d}" for a in agency_idx],
        "recommended_date": rec, "sanctioned_date": san, "sanctioned_amount": amt, "released_amount": rel, "expenditure_amount": exp,
        "status": status, "last_progress_date": lp.to_numpy(), "completion_date": comp_date.to_numpy(), "completion_percent": np.round(pct, 1),
        "_planted_type": ptype})

    # planted near-duplicates: copy an earlier work, re-sanction 10-80 days later with a lightly altered description
    cand = df[(df._planted_type == "none") & (df.sanctioned_date <= asof - 400 * day)].sample(n_dup, random_state=seed)
    dup = cand.copy()
    shift = pd.to_timedelta(rng.integers(10, 80, n_dup), unit="D")
    dup["recommended_date"] += shift
    dup["sanctioned_date"] += shift
    dup["sanctioned_amount"] = np.round(dup.sanctioned_amount * rng.uniform(.95, 1.05, n_dup), -3)
    mode = rng.integers(0, 3, n_dup)
    desc = dup.work_description.to_numpy().astype(object)
    for i in range(n_dup):
        t = desc[i]
        desc[i] = [t.upper(), " ".join(reversed(t.split())), t + " (new)"][mode[i]]
    dup["work_description"] = desc
    dup["completion_date"] = dup.completion_date + shift
    over = dup.completion_date > asof
    dup.loc[over, ["status", "completion_date", "completion_percent"]] = ["in_progress", pd.NaT, 70.0]
    dup.loc[over, "last_progress_date"] = asof - 20 * day
    dup["_planted_type"] = "duplicate"
    df = pd.concat([df, dup], ignore_index=True).sort_values("recommended_date", kind="stable").reset_index(drop=True)
    df.insert(0, "work_id", [f"W{i + 1:06d}" for i in range(len(df))])
    return df
