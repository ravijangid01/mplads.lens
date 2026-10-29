"""Cleaning: standardise text, fix units, drop exact duplicate rows, add derived columns and a data-quality score."""
from __future__ import annotations
import re
import numpy as np
import pandas as pd

CATEGORIES = ["road", "community hall", "drinking water", "school", "health", "sanitation", "other"]
_CAT_RULES = [("road", r"road|street|culvert|bridge|path"), ("community hall", r"hall|bhawan|bhavan|community|centre"),
              ("drinking water", r"water|hand ?pump|borewell|tap"), ("school", r"school|class|education|library"),
              ("health", r"health|hospital|dispensary|ambulance|clinic"), ("sanitation", r"sanit|toilet|drain|sewer|waste")]
_STATUS = {"completed": "completed", "complete": "completed", "done": "completed", "finished": "completed",
           "in progress": "in_progress", "in_progress": "in_progress", "ongoing": "in_progress", "on-going": "in_progress", "on going": "in_progress",
           "not started": "not_started", "not_started": "not_started", "yet to start": "not_started", "sanctioned": "not_started"}
KEY_FIELDS = ["work_category", "district", "implementing_agency", "recommended_date", "sanctioned_date", "sanctioned_amount",
              "released_amount", "expenditure_amount", "status", "work_description", "last_progress_date"]
UNIT = {"rupees": 1.0, "rupee": 1.0, "lakh": 1e5, "crore": 1e7}


def norm_category(x: object) -> str:
    """Map free-text sector names to the seven canonical categories."""
    s = str(x).strip().lower()
    if s in CATEGORIES:
        return s
    for cat, rx in _CAT_RULES:
        if re.search(rx, s):
            return cat
    return "other"


def norm_agency(x: object) -> str:
    """Trim, collapse whitespace and title-case agency names."""
    return re.sub(r"\s+", " ", str(x)).strip().title() if pd.notna(x) else "Unknown"


def block_of(loc: str) -> str:
    """Block part of a location label such as D07-B3-V05 (falls back to the label itself)."""
    m = re.match(r"^(.*)-V\d+$", str(loc))
    return m.group(1) if m else str(loc)


def clean(df: pd.DataFrame, amount_unit: str = "rupees") -> pd.DataFrame:
    """Return a cleaned canonical frame with derived columns."""
    d = df.copy()
    d["work_category"] = d["work_category"].map(norm_category)
    d["implementing_agency"] = d["implementing_agency"].map(norm_agency)
    d["status"] = d["status"].astype(str).str.strip().str.lower().map(lambda s: _STATUS.get(s, "in_progress"))
    f = UNIT.get(amount_unit, 1.0)
    for c in ["sanctioned_amount", "released_amount", "expenditure_amount"]:
        d[c] = pd.to_numeric(d[c], errors="coerce") * f
    d["district_class"] = d["district_class"].fillna("plain").astype(str).str.lower()
    d["block_or_village"] = d["block_or_village"].fillna(d["district"]).astype(str)
    d["work_description"] = d["work_description"].fillna(d["work_title"]).fillna("")
    d["work_title"] = d["work_title"].fillna(d["work_description"])
    d = d.drop_duplicates().reset_index(drop=True)
    d["block"] = d["block_or_village"].map(block_of)
    base = d["sanctioned_date"].fillna(d["recommended_date"])
    d["sanction_year"] = base.dt.year.fillna(0).astype(int)
    d["sanction_delay_days"] = (d["sanctioned_date"] - d["recommended_date"]).dt.days
    d["completion_days"] = (d["completion_date"] - d["sanctioned_date"]).dt.days
    present = pd.concat([d[c].notna() & (d[c].astype(str).str.strip() != "") for c in KEY_FIELDS if c in d], axis=1)
    d["quality_score"] = (present.mean(axis=1) * 100).round(1)
    return d
