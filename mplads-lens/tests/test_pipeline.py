import io
import numpy as np
import pandas as pd
from pipeline import anonymise, clean, ingest, score, synthetic


def test_synthetic_deterministic_and_planted_share(works, cfg):
    again = synthetic.generate(seed=cfg["seed"], as_of=cfg["as_of"])
    assert again.work_id.tolist() == works.work_id.tolist() and len(works) == 20000
    share = (works._planted_type != "none").mean()
    assert 0.025 <= share <= 0.05
    assert works.state.nunique() == 4 and works.district.nunique() == 40
    assert works.recommended_date.min() >= pd.Timestamp("2023-04-01")


def test_anonymisation_stable_and_unique(works):
    m1, m2 = anonymise.build_maps(works), anonymise.build_maps(works.sample(frac=1, random_state=1))
    assert m1 == m2
    for c, m in m1.items():
        assert len(set(m.values())) == len(m)
    assert all(v.startswith("Constituency C-") for v in m1["constituency"].values())
    out = anonymise.apply(works.head(5), m1)
    assert not out.constituency.str.contains("Constituency 0").any() and out.mp_ref.str.startswith("Member M-").all()


def test_ingest_column_mapping_and_formats():
    raw = pd.DataFrame({"Sector": ["Road", "Hand pump", "School"], "St": ["A", "A", "B"], "Dist": ["D1", "D2", "D3"], "Amt": ["₹ 12,34,567.50", "5 lakh", "bad"],
                        "Sanctioned On": ["15-03-2024", "01-12-2023", "x"], "Status": ["Completed", "Ongoing", "Not started"]})
    d, rep = ingest.apply_map(raw, {"work_category": "Sector", "state": "St", "district": "Dist", "sanctioned_amount": "Amt", "sanctioned_date": "Sanctioned On", "status": "Status"})
    assert len(d) == 2 and rep["dropped"]["sanctioned amount missing or not positive"] == 1
    assert d.sanctioned_amount.iloc[0] == 1234567.5 and d.sanctioned_amount.iloc[1] == 5e5
    assert d.sanctioned_date.iloc[0] == pd.Timestamp("2024-03-15")
    c = clean.clean(d)
    assert c.work_category.tolist() == ["road", "drinking water"] and c.status.tolist() == ["completed", "in_progress"]


def test_ingest_requires_columns():
    d, rep = ingest.apply_map(pd.DataFrame({"a": [1]}), {})
    assert d.empty and "sanctioned_amount" in rep["missing_required"]


def test_unit_conversion():
    d = pd.DataFrame({"work_category": ["road"], "state": ["A"], "district": ["D"], "sanctioned_amount": [12.0], "released_amount": [1.0], "expenditure_amount": [1.0], "status": ["completed"],
                      "district_class": [None], "block_or_village": [None], "implementing_agency": ["x"], "work_title": ["t"], "work_description": [None],
                      "recommended_date": [pd.Timestamp("2024-01-01")], "sanctioned_date": [pd.Timestamp("2024-02-01")], "completion_date": [pd.NaT], "last_progress_date": [pd.NaT]})
    assert clean.clean(d, "lakh").sanctioned_amount.iloc[0] == 12e5
    assert clean.clean(d, "crore").sanctioned_amount.iloc[0] == 12e7


def test_scoring_bounds_and_weights(works, dets, cfg):
    s, r = score.wide(dets, works.work_id)
    out = score.combine(s, r, cfg["weights"], cfg)
    assert out.score.between(0, 100).all() and len(out) == len(works)
    assert out.flagged.mean() <= 0.08 and set(out.severity) <= {"High", "Medium", "Low"}
    alt = score.combine(s, r, {**cfg["weights"], "cost": 0.0}, cfg)
    assert not np.allclose(out.score, alt.score)           # weights change the ranking
    assert alt.loc[alt.top_detector == "cost"].empty        # zero weight removes the detector
    assert (score.combine(s, r, {k: 0 for k in cfg["weights"]}, cfg).score == 0).all()
