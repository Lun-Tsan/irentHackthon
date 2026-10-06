"""SQLite store (stdlib only). One connection, one lock, JSON in TEXT columns."""
import json
import sqlite3
import threading
import time
import uuid

from .config import DATA, FIXTURES

DB_PATH = DATA / "demo.sqlite"
_lock = threading.RLock()
_conn = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS vehicles(
  id TEXT PRIMARY KEY, plate TEXT, model TEXT, state TEXT, odometer INTEGER, location TEXT);
CREATE TABLE IF NOT EXISTS rentals(
  id TEXT PRIMARY KEY, vehicle_id TEXT, user TEXT, prev_rental_id TEXT, next_rental_id TEXT,
  status TEXT, points INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS sessions(
  id TEXT PRIMARY KEY, rental_id TEXT, phase TEXT, scenario TEXT DEFAULT 'auto',
  created_at TEXT, completed_at TEXT, summary TEXT);
CREATE TABLE IF NOT EXISTS captures(
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, slot TEXT, accepted INTEGER, created_at TEXT,
  path TEXT, issues TEXT, checks TEXT, unverified INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS photos(
  id TEXT PRIMARY KEY, session_id TEXT, slot TEXT, path TEXT, quality TEXT,
  unverified INTEGER DEFAULT 0, attempt INTEGER, created_at TEXT);
CREATE TABLE IF NOT EXISTS damage_records(
  id TEXT PRIMARY KEY, vehicle_id TEXT, rental_id TEXT, slot TEXT, panel TEXT, score REAL,
  band TEXT, state TEXT, reason TEXT, baseline_photo TEXT, current_photo TEXT, heatmap TEXT,
  needs_repair INTEGER DEFAULT 0, appeal_text TEXT, created_at TEXT, decided_at TEXT);
CREATE TABLE IF NOT EXISTS workorders(
  id TEXT PRIMARY KEY, vehicle_id TEXT, rental_id TEXT, type TEXT, status TEXT, sla_hours REAL,
  detail TEXT, evidence TEXT, created_at TEXT, completed_at TEXT, acceptance TEXT);
CREATE TABLE IF NOT EXISTS notifications(
  id TEXT PRIMARY KEY, rental_id TEXT, kind TEXT, title TEXT, message TEXT, record_id TEXT,
  images TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, entity TEXT, entity_id TEXT, event TEXT,
  from_state TEXT, to_state TEXT, detail TEXT);
"""


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def conn() -> sqlite3.Connection:
    global _conn
    with _lock:
        if _conn is None:
            DATA.mkdir(parents=True, exist_ok=True)
            _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            _conn.row_factory = sqlite3.Row
            _conn.executescript(SCHEMA)
            # columns added after the first demo databases were created (ALTER is idempotent via try/except)
            for col in ("path TEXT", "issues TEXT", "checks TEXT", "unverified INTEGER DEFAULT 0"):
                try:
                    _conn.execute(f"ALTER TABLE captures ADD COLUMN {col}")
                except sqlite3.OperationalError:
                    pass
            _conn.commit()
        return _conn


def fetchall(sql: str, params=()) -> list:
    with _lock:
        return [dict(r) for r in conn().execute(sql, params).fetchall()]


def fetchone(sql: str, params=()):
    with _lock:
        r = conn().execute(sql, params).fetchone()
        return dict(r) if r else None


def execute(sql: str, params=()):
    with _lock:
        conn().execute(sql, params)
        conn().commit()


def insert(table: str, row: dict) -> str:
    cols = ",".join(row.keys())
    qs = ",".join("?" * len(row))
    execute(f"INSERT INTO {table} ({cols}) VALUES ({qs})", tuple(row.values()))
    return row.get("id")


def update(table: str, id_: str, fields: dict):
    sets = ",".join(f"{k}=?" for k in fields)
    execute(f"UPDATE {table} SET {sets} WHERE id=?", (*fields.values(), id_))


def log_event(entity: str, entity_id: str, event: str, from_state=None, to_state=None, detail=None):
    insert("events", {
        "ts": now(), "entity": entity, "entity_id": entity_id, "event": event,
        "from_state": from_state, "to_state": to_state,
        "detail": json.dumps(detail, ensure_ascii=False) if detail is not None else None,
    })


TABLES = ["vehicles", "rentals", "sessions", "captures", "photos", "damage_records", "workorders", "notifications", "events"]


def seed_if_empty():
    if fetchone("SELECT id FROM vehicles LIMIT 1"):
        return
    with open(FIXTURES / "rentals.json", encoding="utf-8") as f:
        fx = json.load(f)
    # INSERT OR IGNORE: the http and https server processes may both seed a fresh database at startup
    for v in fx["vehicles"]:
        cols = ",".join(v.keys())
        execute(f"INSERT OR IGNORE INTO vehicles ({cols}) VALUES ({','.join('?' * len(v))})", tuple(v.values()))
    for r in fx["rentals"]:
        row = {"points": 0, **r}
        cols = ",".join(row.keys())
        execute(f"INSERT OR IGNORE INTO rentals ({cols}) VALUES ({','.join('?' * len(row))})", tuple(row.values()))
    log_event("system", "seed", "seeded", detail={"vehicles": len(fx["vehicles"]), "rentals": len(fx["rentals"])})


def reset():
    """Wipe everything and re-seed. Used between demo takes.
    Deletes rows instead of the file: another server process (http + https run side by side)
    keeps the database open, and Windows refuses to unlink an open SQLite file."""
    with _lock:
        for t in TABLES:
            execute(f"DELETE FROM {t}")
        seed_if_empty()


def telematics(vehicle_id: str) -> dict:
    with open(FIXTURES / "rentals.json", encoding="utf-8") as f:
        fx = json.load(f)
    return fx.get("telematics", {}).get(vehicle_id, {})
