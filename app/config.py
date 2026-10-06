"""Load config.yaml and resolve project paths."""
import json
import os
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
CFG_PATH = ROOT / "config.yaml"
DATA = ROOT / "data"
UPLOADS = DATA / "uploads"
HEATMAPS = DATA / "heatmaps"
FIXTURES = ROOT / "fixtures"
STATIC = ROOT / "app" / "static"

for _d in (UPLOADS, HEATMAPS):
    _d.mkdir(parents=True, exist_ok=True)


def load_cfg() -> dict:
    with open(CFG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = load_cfg()


def slot_map() -> dict:
    return {s["id"]: s for s in CFG["slots"]}


def load_overrides() -> dict:
    """fixtures/overrides.json: sha256 of the uploaded bytes -> forced results.
    Re-read on every call so it can be edited while the server runs."""
    p = FIXTURES / "overrides.json"
    if not p.exists():
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def lan_ip() -> str:
    env = os.environ.get("IRENT_LAN_IP")
    if env:
        return env
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"
