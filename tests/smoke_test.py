"""End-to-end smoke test with synthetic photos. Run:  .venv/Scripts/python.exe tests/smoke_test.py"""
import io
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
os.environ.setdefault("IRENT_LAN_IP", "127.0.0.1")

from PIL import Image, ImageDraw, ImageFilter, ImageFont  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

FONT = "C:/Windows/Fonts/arialbd.ttf"
DEFAULT_PORTRAIT = False  # set from /api/config in run(): iRent's app shoots portrait


def car_photo(sticker=False, blur=False, portrait=None, seed=0):
    if portrait is None:
        portrait = DEFAULT_PORTRAIT
    w, h = (720, 1280) if portrait else (1280, 720)
    img = Image.new("RGB", (w, h), (150 + seed, 160, 170))
    d = ImageDraw.Draw(img)
    for i in range(0, w, 40):  # some texture so ORB has features
        d.line([(i, 0), (i + 20, h)], fill=(130, 140, 150), width=2)
    d.rounded_rectangle([w * 0.15, h * 0.35, w * 0.85, h * 0.8], radius=40, fill=(40, 60, 120), outline=(20, 20, 20), width=6)
    d.rectangle([w * 0.3, h * 0.38, w * 0.7, h * 0.52], fill=(90, 120, 160))
    d.ellipse([w * 0.22, h * 0.72, w * 0.32, h * 0.9], fill=(20, 20, 20))
    d.ellipse([w * 0.68, h * 0.72, w * 0.78, h * 0.9], fill=(20, 20, 20))
    d.rectangle([w * 0.42, h * 0.66, w * 0.58, h * 0.73], fill=(245, 245, 245), outline=(0, 0, 0), width=3)
    try:
        f = ImageFont.truetype(FONT, int(h * 0.05))
    except OSError:
        f = ImageFont.load_default()
    d.text((w * 0.435, h * 0.665), "ABC-1234", fill=(0, 0, 0), font=f)
    if sticker:
        d.rectangle([w * 0.55, h * 0.56, w * 0.62, h * 0.62], fill=(250, 250, 250))
        d.line([(w * 0.56, h * 0.57), (w * 0.61, h * 0.61)], fill=(0, 0, 0), width=6)
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(10))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def cabin_photo(seed=0, portrait=None):
    if portrait is None:
        portrait = DEFAULT_PORTRAIT
    w, h = (720, 1280) if portrait else (1280, 720)
    img = Image.new("RGB", (w, h), (120 + seed, 110, 100))
    d = ImageDraw.Draw(img)
    for i in range(0, w, 60):
        d.rectangle([i, h * 0.4, i + 30, h * 0.97], fill=(90, 80, 70))
    for i in range(0, w, 17):  # seat-fabric texture so the photo isn't flatter than a real cabin
        d.line([(i, 0), (i + 9, h)], fill=(70 + (i * 7) % 60, 60, 55), width=1)
    for j in range(0, h, 23):
        d.line([(0, j), (w, j + 5)], fill=(150, 140, 130), width=1)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def upload(c, sid, slot, data, force=False):
    return c.post(f"/api/capture/{sid}/{slot}", files={"image": ("x.jpg", data, "image/jpeg")}, data={"force": "1" if force else "0"})


def run():
    c = TestClient(app)
    assert c.post("/api/reset").json()["ok"]
    cfg = c.get("/api/config").json()
    print("plate_ocr:", cfg["plate_ocr"], "interior_head:", cfg["interior_head"])

    # --- 環節一：quality gates are real even in scripted mode
    s = c.post("/api/session/start", json={"rental_id": "R-B", "phase": "pickup", "scenario": "clean"}).json()
    sid = s["session_id"]
    want = cfg["capture"]["orientation"]
    global DEFAULT_PORTRAIT
    DEFAULT_PORTRAIT = want != "landscape"  # every later photo is shot the way the config wants
    wrong = car_photo(portrait=(want == "landscape"))  # the orientation the config rejects
    r = upload(c, sid, "ext_fl", wrong).json()
    if want == "any":
        assert r["accepted"], r
    else:
        assert not r["accepted"] and any(i["code"] in ("portrait", "landscape") for i in r["issues"]), r
    r = upload(c, sid, "ext_fl", car_photo(blur=True)).json()
    assert not r["accepted"] and any(i["code"] == "blur" for i in r["issues"]), r
    assert r["can_force"] or want == "any", "after 2 failures the soft-block escape should appear"
    r = upload(c, sid, "ext_fl", car_photo()).json()
    assert r["accepted"] and not r["unverified"], r
    assert r["checks"]["plate"]["match"], r["checks"]
    for slot in ("ext_rl", "int_rear", "int_dash"):
        assert upload(c, sid, slot, cabin_photo() if slot.startswith("int") else car_photo(seed=3)).json()["accepted"]
    summ = c.post(f"/api/session/{sid}/complete").json()
    assert summ["vehicle_state"] == "rented", summ
    print("pickup OK ->", summ["vehicle_state"], summ["notes"])

    # --- 環節二：scripted damage return → one panel to human, one to review
    s = c.post("/api/session/start", json={"rental_id": "R-B", "phase": "return", "scenario": "damage"}).json()
    sid2 = s["session_id"]
    assert upload(c, sid2, "ext_fl", car_photo(sticker=True)).json()["accepted"]
    assert upload(c, sid2, "ext_rl", car_photo(seed=3)).json()["accepted"]
    assert upload(c, sid2, "int_rear", cabin_photo()).json()["accepted"]
    assert upload(c, sid2, "int_dash", cabin_photo(seed=2)).json()["accepted"]
    summ = c.post(f"/api/session/{sid2}/complete").json()
    bands = sorted(r["band"] for e in summ["exterior"] for r in e["records"])
    assert bands == ["human", "review"], summ["exterior"]
    assert summ["vehicle_state"] == "pending_review", summ
    assert all(i["level"] == "clean" for i in summ["interior"]), summ["interior"]
    print("return(damage) OK -> bands", bands)

    # --- 判得公平：human decides, user notified, appeal, overturn
    ov = c.get("/api/admin/overview").json()
    human = [q for q in ov["queue"] if q["band"] == "human"][0]
    review = [q for q in ov["queue"] if q["band"] == "review"][0]
    rec = c.post(f"/api/admin/decide/{human['id']}", json={"decision": "new", "needs_repair": True}).json()
    assert rec["state"] == "confirmed"
    c.post(f"/api/admin/decide/{review['id']}", json={"decision": "false"})
    notes = c.get("/api/notifications/R-B").json()["items"]
    assert any(n["kind"] == "damage_confirmed" and len(n["images"]) == 3 for n in notes), notes
    assert c.post(f"/api/appeal/{human['id']}", json={"text": "反光"}).json()["state"] == "appealing"
    assert c.post(f"/api/admin/appeal/{human['id']}", json={"outcome": "overturn"}).json()["state"] == "closed"
    assert c.post(f"/api/appeal/{human['id']}", json={"text": "again"}).status_code == 409, "closed records can't be appealed"
    ov = c.get("/api/admin/overview").json()
    v = ov["vehicles"][0]
    assert v["state"] == "needs_service", v  # repair work order keeps it off the road
    wo = [w for w in ov["workorders"] if w["type"] == "repair"][0]
    done = c.post(f"/api/workorder/{wo['id']}/complete", files={"image": ("x.jpg", car_photo(), "image/jpeg")}).json()
    assert done["accepted"] and done["vehicle_state"] == "available", done
    print("decide/appeal/repair OK ->", done["vehicle_state"])

    # --- 髒車劇本：cleaning work order, next renter reassigned, acceptance
    c.post("/api/reset")
    s = c.post("/api/session/start", json={"rental_id": "R-B", "phase": "pickup", "scenario": "clean"}).json()
    for slot in ("ext_fl", "int_rear"):
        upload(c, s["session_id"], slot, car_photo() if slot.startswith("ext") else cabin_photo())
    c.post(f"/api/session/{s['session_id']}/complete")
    s = c.post("/api/session/start", json={"rental_id": "R-B", "phase": "return", "scenario": "dirty"}).json()
    upload(c, s["session_id"], "ext_fl", car_photo())
    upload(c, s["session_id"], "int_rear", cabin_photo())
    summ = c.post(f"/api/session/{s['session_id']}/complete").json()
    assert summ["vehicle_state"] == "needs_service" and summ["workorders"], summ
    assert any(n["kind"] == "reassigned" for n in c.get("/api/notifications/R-C").json()["items"])
    done = c.post(f"/api/workorder/{summ['workorders'][0]}/complete", files={"image": ("x.jpg", cabin_photo(), "image/jpeg")}).json()
    assert done["accepted"] and done["vehicle_state"] == "available", done
    print("dirty scenario OK ->", done["vehicle_state"])

    # --- 取車免責：previous return was dirty → pickup registers pre-existing condition
    c.post("/api/reset")
    s = c.post("/api/session/start", json={"rental_id": "R-A", "phase": "return", "scenario": "clean"}).json()
    upload(c, s["session_id"], "ext_fl", car_photo())
    c.post(f"/api/session/{s['session_id']}/complete")
    s = c.post("/api/session/start", json={"rental_id": "R-B", "phase": "pickup", "scenario": "dirty"}).json()
    upload(c, s["session_id"], "ext_fl", car_photo(seed=1))
    upload(c, s["session_id"], "int_rear", cabin_photo())
    summ = c.post(f"/api/session/{s['session_id']}/complete").json()
    notes = c.get("/api/notifications/R-B").json()
    assert any(n["kind"] == "pickup_protect" for n in notes["items"]) and notes["points"] >= 200, notes
    print("pickup protect OK -> points", notes["points"])
    print("ALL OK")


def hard_exit(code: int):
    """Windows: after OpenCV/scikit-image ran in worker threads, both normal exit and os._exit can hang inside
    ExitProcess (a DLL detach deadlock). TerminateProcess skips DLL teardown, so kill ourselves with it."""
    for s in (sys.stdout, sys.stderr):
        try:
            if s:
                s.flush()
        except OSError:
            pass
    if os.name == "nt":
        import signal
        os.kill(os.getpid(), signal.SIGTERM)  # → TerminateProcess
    os._exit(code)


if __name__ == "__main__":
    try:
        run()
    except Exception:
        import traceback
        traceback.print_exc()
        hard_exit(1)
    hard_exit(0)
