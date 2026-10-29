"""Stable anonymisation: constituencies, members, agencies, districts and locations get hash-based labels."""
from __future__ import annotations
import hashlib
import pandas as pd

PREFIX = {"constituency": ("Constituency C-", 1000), "mp_ref": ("Member M-", 1000), "implementing_agency": ("Agency A-", 1000),
          "district": ("District D-", 1000), "block_or_village": ("Location L-", 100000)}


def _h(kind: str, value: str, mod: int) -> int:
    return int(hashlib.sha256(f"mplads-lens|{kind}|{value}".encode()).hexdigest()[:10], 16) % mod


def build_map(values, kind: str) -> dict[str, str]:
    """Deterministic, collision-free mapping for the given set of values (same input -> same labels)."""
    prefix, mod = PREFIX[kind]
    width = len(str(mod - 1))
    used: set[int] = set()
    out: dict[str, str] = {}
    for v in sorted({str(x) for x in values if pd.notna(x)}, key=lambda v: (_h(kind, v, mod), v)):
        i = _h(kind, v, mod)
        while i in used:
            i = (i + 1) % mod
        used.add(i)
        out[v] = f"{prefix}{i:0{width}d}"
    return out


def build_maps(works: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Maps for every anonymised column present in ``works``."""
    return {c: build_map(works[c], c) for c in PREFIX if c in works}


def apply(df: pd.DataFrame, maps: dict[str, dict[str, str]], on: bool = True) -> pd.DataFrame:
    """Return a copy with identifying columns replaced by anonymous labels when ``on``."""
    if not on:
        return df
    d = df.copy()
    for c, m in maps.items():
        if c in d:
            d[c] = d[c].astype(str).map(m).fillna(d[c].astype(str))
    return d
