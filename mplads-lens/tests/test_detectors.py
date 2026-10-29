import numpy as np
import pandas as pd
import pytest
from conftest import recall_precision
from detectors import cost, duplicate, stalled
from pipeline import peer_groups


@pytest.mark.parametrize("planted,name", [("cost_outlier", "cost"), ("duplicate", "duplicate"), ("stalled", "stalled"), ("agency_pattern", "agency"), ("delay", "delay")])
def test_recall_at_least_80_percent(works, dets, planted, name):
    rec, _ = recall_precision(works, dets, planted, name)
    assert rec >= 0.8, f"{name} recall {rec:.2f}"


@pytest.mark.parametrize("name", ["cost", "duplicate", "stalled", "agency", "delay"])
def test_each_detector_flags_under_8_percent(works, dets, name):
    assert (dets.detector == name).sum() / len(works) <= 0.08


def test_detector_output_schema(dets):
    assert list(dets.columns) == ["work_id", "detector", "score", "reason", "evidence"]
    assert dets.score.between(0, 100).all() and (dets.reason.str.len() > 5).all()


def test_detectors_do_not_read_planted_column(works, cfg):
    a = cost.run(works, cfg)
    b = cost.run(works.drop(columns=["_planted_type"]), cfg)
    assert a.work_id.tolist() == b.work_id.tolist()


def test_cost_reason_format(dets):
    r = dets[dets.detector == "cost"].reason.iloc[0]
    assert r.startswith("Costs ") and "similar works" in r


def test_peer_fallback_levels():
    rows = []
    for i in range(30):  # big group: category x class x year
        rows.append(("road", "plain", 2024, "S1"))
    for i in range(3):   # tiny class group but state group of 10 total together with the 7 below
        rows.append(("hall", "hill", 2024, "S1"))
    for i in range(7):
        rows.append(("hall", "plain", 2023, "S1"))
    for i in range(3):   # nothing big enough except category overall
        rows.append(("hall", "desert", 2022, "S2"))
    df = pd.DataFrame(rows, columns=["work_category", "district_class", "sanction_year", "state"])
    pg = peer_groups.assign(df, 8)
    assert (pg.pg_level[:30] == 0).all()
    assert (pg.pg_level[30:40] == 1).all()   # 10 halls in S1 -> category x state
    assert (pg.pg_level[40:] == 2).all()     # S2 only has 3 -> category only


def test_duplicate_normalise_ignores_numbers_and_case():
    assert duplicate.normalise("Construction of CC Road, ward 12") == duplicate.normalise("cc road WARD 7")


def test_stalled_rule(cfg):
    w = pd.DataFrame({"status": ["in_progress", "completed", "in_progress"], "sanctioned_date": pd.to_datetime(["2023-01-01"] * 3), "recommended_date": pd.to_datetime(["2022-12-01"] * 3),
                      "released_amount": [10.0, 10.0, 0.0], "last_progress_date": [pd.NaT, pd.NaT, pd.NaT]})
    m, _, _ = stalled.mask(w, cfg)
    assert m.tolist() == [True, False, False]
