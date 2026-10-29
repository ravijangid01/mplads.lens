"""Load user CSV/Excel files through a column map; parse Indian numbers and dd-mm-yyyy dates; report dropped rows."""
from __future__ import annotations
import io
import re
import numpy as np
import pandas as pd
import yaml

CANONICAL = ["work_id", "mp_ref", "constituency", "state", "district", "district_class", "block_or_village", "work_title",
             "work_description", "work_category", "implementing_agency", "recommended_date", "sanctioned_date", "sanctioned_amount",
             "released_amount", "expenditure_amount", "status", "last_progress_date", "completion_date", "completion_percent"]
REQUIRED = ["work_category", "state", "district", "sanctioned_amount", "status"]
DATES = ["recommended_date", "sanctioned_date", "last_progress_date", "completion_date"]
AMOUNTS = ["sanctioned_amount", "released_amount", "expenditure_amount"]
ALIASES = {"work_id": ["work id", "workid", "id", "sr. no."], "state": ["state"], "district": ["district"], "work_category": ["sector", "category", "work category", "type"],
           "work_title": ["work name", "title", "work title"], "work_description": ["description", "work description"], "implementing_agency": ["agency", "implementing agency"],
           "recommended_date": ["recommended on", "recommended date"], "sanctioned_date": ["sanctioned on", "sanction date", "sanctioned date"],
           "sanctioned_amount": ["sanctioned amount", "sanctioned amount (₹)", "amount sanctioned"], "released_amount": ["amount released", "released amount"],
           "expenditure_amount": ["expenditure", "expenditure amount"], "status": ["status", "work status"], "last_progress_date": ["last updated", "last progress date"],
           "completion_date": ["completed on", "completion date"], "completion_percent": ["physical progress %", "completion percent", "progress"],
           "constituency": ["constituency"], "mp_ref": ["mp", "member", "mp ref"], "block_or_village": ["village", "block", "location"], "district_class": ["district class"]}


def read_table(file, name: str = "") -> pd.DataFrame:
    """Read a CSV or Excel upload (file path, bytes buffer or Streamlit upload)."""
    n = (name or getattr(file, "name", "")).lower()
    if n.endswith((".xlsx", ".xls")):
        return pd.read_excel(file, dtype=object)
    return pd.read_csv(file, dtype=object, encoding_errors="replace")


def parse_indian_number(x: object) -> float:
    """'₹ 12,34,567.50' -> 1234567.5 ; text with lakh/crore suffix is scaled; unparsable -> NaN."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().lower().replace(",", "").replace("₹", "").replace("rs.", "").replace("rs", "").strip()
    mult = 1.0
    if s.endswith("crore") or s.endswith("cr"):
        mult, s = 1e7, re.sub(r"(crore|cr)$", "", s).strip()
    elif s.endswith("lakh") or s.endswith("lac"):
        mult, s = 1e5, re.sub(r"(lakh|lac)$", "", s).strip()
    try:
        return float(s) * mult
    except ValueError:
        return np.nan


def parse_date(series: pd.Series) -> pd.Series:
    """Parse dd-mm-yyyy (and ISO) dates; invalid values become NaT."""
    s = series.astype(object)
    iso = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    dmy = pd.to_datetime(s, dayfirst=True, errors="coerce", format="mixed")
    return iso.fillna(dmy)


def suggest_map(columns: list[str]) -> dict[str, str]:
    """Guess canonical -> file column using aliases."""
    low = {c.strip().lower(): c for c in columns}
    out = {}
    for canon, al in ALIASES.items():
        for a in [canon.replace("_", " "), canon] + al:
            if a in low:
                out[canon] = low[a]
                break
    return out


def load_map_yaml(text: str) -> tuple[dict[str, str], str]:
    """Parse a column-map YAML string -> (mapping, amount_unit)."""
    y = yaml.safe_load(text) or {}
    return dict(y.get("columns", {})), str(y.get("amount_unit", "rupees"))


def apply_map(raw: pd.DataFrame, mapping: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    """Rename/parse into the canonical schema. Returns (frame, report) where report lists missing columns and dropped rows with reasons."""
    report: dict = {"rows_in": len(raw), "dropped": {}, "missing_required": [], "missing_optional": []}
    miss = [c for c in REQUIRED if c not in mapping or mapping[c] not in raw.columns]
    report["missing_required"] = miss
    if miss:
        return pd.DataFrame(columns=CANONICAL), report
    d = pd.DataFrame(index=raw.index)
    for c in CANONICAL:
        src = mapping.get(c)
        if src in raw.columns:
            d[c] = raw[src]
        else:
            d[c] = np.nan
            if c not in ("work_id",):
                report["missing_optional"].append(c)
    for c in AMOUNTS:
        d[c] = d[c].map(parse_indian_number)
    d["completion_percent"] = d["completion_percent"].map(parse_indian_number)
    for c in DATES:
        d[c] = parse_date(d[c])
    if d["work_id"].isna().all():
        d["work_id"] = [f"U{i + 1:06d}" for i in range(len(d))]
    bad_amt = d["sanctioned_amount"].isna() | (d["sanctioned_amount"] <= 0)
    bad_cat = d["work_category"].isna()
    for reason, mask in [("sanctioned amount missing or not positive", bad_amt), ("category missing", bad_cat & ~bad_amt)]:
        report["dropped"][reason] = int(mask.sum())
    d = d[~(bad_amt | bad_cat)].copy()
    for c in ["released_amount", "expenditure_amount"]:
        d[c] = d[c].fillna(0.0)
    report["rows_out"] = len(d)
    return d.reset_index(drop=True), report
