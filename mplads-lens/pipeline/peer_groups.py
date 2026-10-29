"""Peer-group assignment with fallback: category x class x year -> category x state -> category."""
from __future__ import annotations
import numpy as np
import pandas as pd

LEVELS = [["work_category", "district_class", "sanction_year"], ["work_category", "state"], ["work_category"]]
LEVEL_NAMES = ["category x district class x sanction year", "category x state (fallback 1)", "category only (fallback 2)"]


def _key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    out = df[cols[0]].astype(str)
    for c in cols[1:]:
        out = out + "|" + df[c].astype(str)
    return out


def assign(df: pd.DataFrame, min_size: int = 8) -> pd.DataFrame:
    """Give every row the finest peer level with at least ``min_size`` works. Columns: pg_level, pg_key, pg_n."""
    level = pd.Series(-1, index=df.index)
    key = pd.Series("", index=df.index, dtype=object)
    size = pd.Series(0, index=df.index)
    for i, cols in enumerate(LEVELS):
        k = _key(df, cols)
        n = k.map(k.value_counts())
        take = (level < 0) & (n >= min_size)
        level[take], key[take], size[take] = i, k[take], n[take]
    rest = level < 0
    if rest.any():
        k = _key(df, LEVELS[-1])
        level[rest], key[rest], size[rest] = len(LEVELS) - 1, k[rest], k.map(k.value_counts())[rest]
    return pd.DataFrame({"pg_level": level, "pg_key": key, "pg_n": size})


def label(level: int, key: str) -> str:
    """Human-readable group description, e.g. 'community hall, hill districts, 2023'."""
    p = key.split("|")
    if level == 0:
        return f"{p[0]}, {p[1]} districts, {p[2]}"
    if level == 1:
        return f"{p[0]}, {p[1]}"
    return f"{p[0]}, all states"


def members(works: pd.DataFrame, level: int, key: str) -> pd.DataFrame:
    """All works belonging to the given peer group."""
    return works[_key(works, LEVELS[level]) == key]
