"""Local review log (sqlite)."""
from __future__ import annotations
import sqlite3
from datetime import datetime
from .config import DATA_DIR

DB = DATA_DIR / "review_log.db"


def _con() -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.execute("CREATE TABLE IF NOT EXISTS review_log (id INTEGER PRIMARY KEY AUTOINCREMENT, work_id TEXT, note TEXT, reviewed_at TEXT)")
    return con


def add(work_id: str, note: str) -> None:
    with _con() as c:
        c.execute("INSERT INTO review_log (work_id, note, reviewed_at) VALUES (?,?,?)", (work_id, note.strip(), datetime.now().isoformat(timespec="seconds")))


def reviewed_ids() -> set[str]:
    with _con() as c:
        return {r[0] for r in c.execute("SELECT DISTINCT work_id FROM review_log")}


def history(work_id: str) -> list[tuple[str, str]]:
    with _con() as c:
        return c.execute("SELECT reviewed_at, note FROM review_log WHERE work_id=? ORDER BY id DESC", (work_id,)).fetchall()
