"""FastAPI wiring. Run: python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"""
import atexit
import io
import json
import os
from contextlib import asynccontextmanager

# After uvicorn has shut down cleanly, the process sometimes never finishes exiting on Windows: once
# OpenCV/scikit-image have run in worker threads, ExitProcess can deadlock in a DLL detach (os._exit
# hangs too). TerminateProcess skips DLL teardown, so the atexit hook kills the process with it.
def _hard_exit():
    import sys
    for s in (sys.stdout, sys.stderr):
        try:
            if s:
                s.flush()
        except OSError:
            pass
    if os.name == "nt":
        import signal
        os.kill(os.getpid(), signal.SIGTERM)
    os._exit(0)


if os.environ.get("IRENT_HARD_EXIT", "1") == "1":
    atexit.register(_hard_exit)

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import fsm, pipeline, store
from .config import CFG, DATA, STATIC, lan_ip
from .vision import interior, plate


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.conn()
    store.seed_if_empty()
    yield


app = FastAPI(title="iRent 車況管家 demo", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
app.mount("/files", StaticFiles(directory=str(DATA)), name="files")


# ---------- pages ----------

@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/app")
def user_app():
    return FileResponse(STATIC / "app.html")


@app.get("/admin")
def admin():
    return FileResponse(STATIC / "admin.html")


@app.get("/live")
def live_page():
    return FileResponse(STATIC / "live.html")


@app.get("/api/live")
def live():
    """The most recent session with every capture attempt (for the projector wall), plus its summary once completed."""
    sess = store.fetchone("SELECT * FROM sessions ORDER BY created_at DESC, rowid DESC LIMIT 1")
    if not sess:
        return {"session": None}
    rental = store.fetchone("SELECT * FROM rentals WHERE id=?", (sess["rental_id"],))
    vehicle = store.fetchone("SELECT * FROM vehicles WHERE id=?", (rental["vehicle_id"],)) if rental else None
    caps = store.fetchall("SELECT * FROM captures WHERE session_id=? ORDER BY id DESC", (sess["id"],))
    for c in caps:
        c["url"] = ("/files/uploads/" + "/".join(c["path"].replace("\\", "/").split("/")[-2:])) if c.get("path") else None
        c["issues"] = json.loads(c["issues"]) if c.get("issues") else []
        c["checks"] = json.loads(c["checks"]) if c.get("checks") else {}
    latest = {}
    for c in reversed(caps):  # oldest → newest, so the newest attempt per slot wins
        latest[c["slot"]] = c
    return {"session": {**sess, "summary": json.loads(sess["summary"]) if sess.get("summary") else None},
            "rental": rental, "vehicle": vehicle, "vehicle_state_label": fsm.label(fsm.VEHICLE, vehicle["state"]) if vehicle else None,
            "slots": CFG["slots"], "latest": latest, "attempts": caps[:30]}


# ---------- config / health ----------

@app.get("/api/config")
def get_config():
    ip = lan_ip()
    http_port = os.environ.get("IRENT_HTTP_PORT", CFG["server"]["http_port"])
    https_port = os.environ.get("IRENT_HTTPS_PORT", CFG["server"]["https_port"])
    return {
        "slots": CFG["slots"], "thresholds": CFG["thresholds"],
        "capture": CFG.get("capture", {"orientation": "any"}),
        "plate_ocr": plate.enabled(CFG), "interior_head": interior.available(),
        "urls": {
            "app_https": f"https://{ip}:{https_port}/app",
            "app_http": f"http://{ip}:{http_port}/app",
            "admin": f"http://{ip}:{http_port}/admin",
        },
        "labels": {"vehicle": fsm.VEHICLE["states"], "damage": fsm.DAMAGE["states"], "workorder": fsm.WORKORDER_STATES,
                   "band": pipeline.BAND_LABEL, "interior": interior.LEVEL_LABEL},
    }


@app.get("/api/qr")
def qr(url: str):
    import qrcode
    img = qrcode.make(url)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.post("/api/reset")
def reset():
    store.reset()
    return {"ok": True}


# ---------- rentals / sessions ----------

@app.get("/api/rentals")
def rentals():
    rows = store.fetchall("SELECT r.*, v.plate, v.model, v.state AS vehicle_state FROM rentals r JOIN vehicles v ON v.id=r.vehicle_id")
    return rows


class StartSession(BaseModel):
    rental_id: str
    phase: str  # pickup | return
    scenario: str = "auto"  # auto | clean | damage | dirty  (劇本模式：結果照劇本走，錄影不出包)


@app.post("/api/session/start")
def start_session(body: StartSession):
    if body.phase not in ("pickup", "return"):
        raise HTTPException(400, "phase must be pickup|return")
    if body.scenario not in pipeline.SCENARIOS:
        raise HTTPException(400, f"scenario must be one of {sorted(pipeline.SCENARIOS)}")
    rental = store.fetchone("SELECT * FROM rentals WHERE id=?", (body.rental_id,))
    if not rental:
        raise HTTPException(404, "no such rental")
    vehicle = store.fetchone("SELECT * FROM vehicles WHERE id=?", (rental["vehicle_id"],))
    sid = store.insert("sessions", {"id": store.new_id("S"), "rental_id": body.rental_id, "phase": body.phase,
                                    "scenario": body.scenario, "created_at": store.now()})
    store.log_event("session", sid, "started", detail={"rental": body.rental_id, "phase": body.phase, "scenario": body.scenario})
    return {"session_id": sid, "rental": rental, "vehicle": vehicle, "slots": CFG["slots"], "scenario": body.scenario,
            "telematics": store.telematics(vehicle["id"])}


@app.get("/api/session/{sid}")
def get_session(sid: str):
    sess = store.fetchone("SELECT * FROM sessions WHERE id=?", (sid,))
    if not sess:
        raise HTTPException(404)
    photos = store.fetchall("SELECT slot, attempt, unverified, quality, path FROM photos WHERE session_id=? ORDER BY attempt", (sid,))
    for p in photos:
        p["url"] = "/files/uploads/" + "/".join(p.pop("path").replace("\\", "/").split("/")[-2:])
        p["quality"] = json.loads(p["quality"]) if p["quality"] else None
    if sess.get("summary"):
        sess["summary"] = json.loads(sess["summary"])
    return {**sess, "photos": photos}


@app.post("/api/capture/{sid}/{slot}")
async def capture(sid: str, slot: str, image: UploadFile = File(...), force: str = Form("0")):
    data = await image.read()
    try:
        return pipeline.assess_capture(sid, slot, data, force=force in ("1", "true", "yes"))
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.post("/api/session/{sid}/complete")
def complete(sid: str):
    sess = store.fetchone("SELECT * FROM sessions WHERE id=?", (sid,))
    if not sess:
        raise HTTPException(404)
    if sess.get("completed_at"):
        return json.loads(sess["summary"])
    return pipeline.complete_session(sid)


# ---------- user side ----------

@app.get("/api/notifications/{rental_id}")
def notifications(rental_id: str):
    rows = store.fetchall("SELECT * FROM notifications WHERE rental_id=? ORDER BY created_at DESC, id DESC", (rental_id,))
    for r in rows:
        r["images"] = json.loads(r["images"] or "[]")
        if r.get("record_id"):
            rec = store.fetchone("SELECT state, appeal_text FROM damage_records WHERE id=?", (r["record_id"],))
            r["record_state"] = rec["state"] if rec else None
    rental = store.fetchone("SELECT points FROM rentals WHERE id=?", (rental_id,))
    return {"points": rental["points"] if rental else 0, "items": rows}


class Appeal(BaseModel):
    text: str = ""


@app.post("/api/appeal/{record_id}")
def do_appeal(record_id: str, body: Appeal):
    try:
        return pipeline.appeal(record_id, body.text)
    except fsm.InvalidTransition as e:
        raise HTTPException(409, str(e))


# ---------- admin side ----------

@app.get("/api/admin/overview")
def overview():
    vehicles = store.fetchall("SELECT * FROM vehicles")
    for v in vehicles:
        v["state_label"] = fsm.label(fsm.VEHICLE, v["state"])
    queue = store.fetchall("SELECT * FROM damage_records WHERE state IN ('candidate','appealing') ORDER BY band='human' DESC, score DESC")
    decided = store.fetchall("SELECT * FROM damage_records WHERE state NOT IN ('candidate','appealing') ORDER BY decided_at DESC LIMIT 20")
    for r in queue + decided:
        r["state_label"] = fsm.label(fsm.DAMAGE, r["state"])
        r["band_label"] = pipeline.BAND_LABEL.get(r["band"], r["band"])
    workorders = store.fetchall("SELECT * FROM workorders ORDER BY created_at DESC")
    for w in workorders:
        w["detail"] = json.loads(w["detail"] or "{}")
        w["evidence"] = json.loads(w["evidence"] or "[]")
        w["acceptance"] = json.loads(w["acceptance"]) if w.get("acceptance") else None
        w["status_label"] = fsm.WORKORDER_STATES.get(w["status"], w["status"])
    rentals_ = store.fetchall("SELECT * FROM rentals")
    events = store.fetchall("SELECT * FROM events ORDER BY id DESC LIMIT 40")
    sessions = store.fetchall("SELECT id, rental_id, phase, created_at, completed_at FROM sessions ORDER BY created_at DESC LIMIT 10")
    return {"vehicles": vehicles, "queue": queue, "decided": decided, "workorders": workorders,
            "rentals": rentals_, "events": events, "sessions": sessions}


class Decision(BaseModel):
    decision: str  # new | old | false
    needs_repair: bool = False


@app.post("/api/admin/decide/{record_id}")
def decide(record_id: str, body: Decision):
    try:
        return pipeline.decide(record_id, body.decision, body.needs_repair)
    except (fsm.InvalidTransition, ValueError) as e:
        raise HTTPException(409, str(e))
    except KeyError as e:
        raise HTTPException(404, str(e))


class AppealOutcome(BaseModel):
    outcome: str  # uphold | overturn


@app.post("/api/admin/appeal/{record_id}")
def admin_appeal(record_id: str, body: AppealOutcome):
    try:
        return pipeline.resolve_appeal(record_id, body.outcome)
    except (fsm.InvalidTransition, ValueError) as e:
        raise HTTPException(409, str(e))


@app.post("/api/workorder/{wo_id}/complete")
async def complete_workorder(wo_id: str, image: UploadFile = File(...)):
    data = await image.read()
    try:
        return pipeline.complete_workorder(wo_id, data)
    except KeyError as e:
        raise HTTPException(404, str(e))
