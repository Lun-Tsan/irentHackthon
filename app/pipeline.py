"""The flow: 拍對 → 判對 → 派對 → 判得公平.

Two modes per session:
- scenario 'auto'  : real checks everywhere (blur/brightness/plate OCR/SSIM diff/YOLO interior).
- scripted ('clean' | 'damage' | 'dirty'): the cheap live checks stay real (blur, brightness,
  orientation — so the 取景框 and hints feel live), but plate/odometer/damage/interior results
  follow the script. Same input → same output, so the video never surprises you.
"""
import hashlib
import json

import numpy as np

from . import fsm, store
from .config import CFG, HEATMAPS, UPLOADS, load_overrides, slot_map
from .vision import compare, interior, plate, quality

SCENARIOS = {"auto", "clean", "damage", "dirty"}
BAND_LABEL = {"pass": "自動放行", "review": "送複判（v1 由人工低優先處理）", "human": "轉人工"}
SCRIPT_DAMAGE = [("r2c2", 0.62), ("r2c3", 0.31)]          # first exterior slot → human, second → review
SCRIPT_DIRTY_OBJECTS = ["bottle", "bottle", "handbag"]


# ---------- helpers ----------

def _vehicle_step(vehicle_id: str, event: str, detail=None) -> str:
    v = store.fetchone("SELECT * FROM vehicles WHERE id=?", (vehicle_id,))
    try:
        new = fsm.step(fsm.VEHICLE, v["state"], event)
    except fsm.InvalidTransition as e:
        store.log_event("vehicle", vehicle_id, "invalid_transition", v["state"], None, {"event": event, "why": str(e)})
        return v["state"]
    store.update("vehicles", vehicle_id, {"state": new})
    store.log_event("vehicle", vehicle_id, event, v["state"], new, detail)
    return new


def _record_step(record_id: str, event: str, detail=None) -> str:
    r = store.fetchone("SELECT * FROM damage_records WHERE id=?", (record_id,))
    new = fsm.step(fsm.DAMAGE, r["state"], event)  # raises on a bad edge
    store.update("damage_records", record_id, {"state": new, "decided_at": store.now()})
    store.log_event("damage", record_id, event, r["state"], new, detail)
    return new


def _notify(rental_id: str, kind: str, title: str, message: str, record_id=None, images=None):
    return store.insert("notifications", {
        "id": store.new_id("N"), "rental_id": rental_id, "kind": kind, "title": title,
        "message": message, "record_id": record_id,
        "images": json.dumps(images or [], ensure_ascii=False), "created_at": store.now(),
    })


def _workorder(vehicle_id: str, rental_id: str, type_: str, detail: dict, evidence=None) -> str:
    sla = CFG["workorders"]["cleaning_sla_hours"] if type_ == "cleaning" else CFG["workorders"]["repair_sla_hours"]
    wid = store.insert("workorders", {
        "id": store.new_id("WO"), "vehicle_id": vehicle_id, "rental_id": rental_id, "type": type_,
        "status": "open", "sla_hours": sla, "detail": json.dumps(detail, ensure_ascii=False),
        "evidence": json.dumps(evidence or [], ensure_ascii=False), "created_at": store.now(),
    })
    store.log_event("workorder", wid, "created", None, "open", {"type": type_, **detail})
    return wid


def _reassign_next(rental: dict, why: str):
    nxt = rental.get("next_rental_id")
    if nxt:
        _notify(nxt, "reassigned", "已為您改派車輛",
                f"您預約的車輛 {rental['vehicle_id']} 因{why}進入整備，系統已改派附近同級車（demo 模擬）。")
        store.log_event("rental", nxt, "reassigned", detail={"why": why})


def _add_points(rental_id: str, pts: int, why: str):
    r = store.fetchone("SELECT points FROM rentals WHERE id=?", (rental_id,))
    store.update("rentals", rental_id, {"points": (r["points"] or 0) + pts})
    _notify(rental_id, "points", f"獲得 {pts} 點", why)


def _latest_photos(session_id: str) -> dict:
    rows = store.fetchall("SELECT * FROM photos WHERE session_id=? ORDER BY attempt", (session_id,))
    return {r["slot"]: r for r in rows}  # last attempt per slot wins


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_rgb(path: str):
    with open(path, "rb") as f:
        return quality.load_image(f.read())


def _url(path: str) -> str:
    return "/files/uploads/" + "/".join(path.replace("\\", "/").split("/")[-2:])


# ---------- 環節一 ----------

def assess_capture(session_id: str, slot_id: str, data: bytes, force: bool = False) -> dict:
    sess = store.fetchone("SELECT * FROM sessions WHERE id=?", (session_id,))
    if not sess:
        raise KeyError("no such session")
    scripted = (sess.get("scenario") or "auto") != "auto"
    slot = slot_map()[slot_id]
    rental = store.fetchone("SELECT * FROM rentals WHERE id=?", (sess["rental_id"],))
    vehicle = store.fetchone("SELECT * FROM vehicles WHERE id=?", (rental["vehicle_id"],))
    overrides = load_overrides().get(_sha(data), {})

    img = quality.load_image(data)
    q = quality.assess(img, CFG, slot["kind"])       # always real: blur / dark / bright / orientation
    issues = list(q["issues"])
    checks = {"quality": q}
    big = np.asarray(quality.resize_width(img, 1280))

    if slot.get("plate"):
        if scripted:
            p = {"skipped": False, "found": True, "match": True, "text": plate.normalize(vehicle["plate"]),
                 "conf": 0.98, "distance_m": 2.1, "simulated": True}
        else:
            p = overrides.get("plate") or plate.detect(big, vehicle["plate"], CFG)
        checks["plate"] = p
        if not p.get("skipped"):
            if not p["found"]:
                issues.append({"code": "no_plate", "msg": "沒拍到車牌，請把車牌拍進畫面", "hard": False})
            elif not p["match"]:
                issues.append({"code": "plate_mismatch", "msg": f"車牌 {p['text']} 與訂單 {vehicle['plate']} 不符，請確認車輛", "hard": True})
            else:
                d = p.get("distance_m", 0)
                if d > CFG["plate"]["max_distance_m"]:
                    issues.append({"code": "too_far", "msg": f"太遠（約 {d} m），請靠近一點", "hard": False})
                elif d < CFG["plate"]["min_distance_m"]:
                    issues.append({"code": "too_near", "msg": f"太近（約 {d} m），請退後一步", "hard": False})

    if slot.get("odometer"):
        tele = store.telematics(vehicle["id"]).get("odometer")
        if scripted:
            o = {"skipped": False, "reading": tele, "simulated": True}
        else:
            o = overrides.get("odometer") or plate.read_odometer(big, CFG)
        checks["odometer"] = {**o, "telematics": tele}
        if not o.get("skipped"):
            if o.get("reading") is None:
                issues.append({"code": "no_odometer", "msg": "沒讀到里程數，請把儀表板數字放在框內", "hard": False})
            elif tele is not None and abs(o["reading"] - tele) > CFG["plate"]["odometer_tolerance"]:
                issues.append({"code": "odometer_mismatch", "msg": f"里程 {o['reading']} 與車聯網 {tele} 差太多，請確認是這台車", "hard": False})

    attempt = 1 + len(store.fetchall("SELECT id FROM captures WHERE session_id=? AND slot=?", (session_id, slot_id)))
    hard = any(i["hard"] for i in issues)
    accepted = (not issues) or (force and not hard)
    can_force = (not hard) and attempt >= CFG["thresholds"]["max_attempts_before_force"]
    store.insert("captures", {"session_id": session_id, "slot": slot_id, "accepted": int(accepted), "created_at": store.now()})

    result = {"slot": slot_id, "attempt": attempt, "accepted": accepted, "unverified": bool(accepted and issues),
              "issues": issues, "can_force": can_force and not accepted, "checks": checks, "scripted": scripted}
    d = UPLOADS / session_id
    d.mkdir(parents=True, exist_ok=True)
    # every attempt is kept (rejected ones too) so the live wall can show「退回：太糊」with the picture
    path = d / (f"{slot_id}_{attempt}.jpg" if accepted else f"rejected_{slot_id}_{attempt}.jpg")
    img.save(path, "JPEG", quality=90)
    result["file"] = f"/files/uploads/{session_id}/{path.name}"
    store.execute("UPDATE captures SET path=?, issues=?, checks=?, unverified=? WHERE session_id=? AND slot=? AND id=(SELECT MAX(id) FROM captures WHERE session_id=? AND slot=?)",
                  (str(path), json.dumps(issues, ensure_ascii=False), json.dumps(checks, ensure_ascii=False), int(bool(accepted and issues)),
                   session_id, slot_id, session_id, slot_id))
    if accepted:
        store.insert("photos", {
            "id": store.new_id("P"), "session_id": session_id, "slot": slot_id, "path": str(path),
            "quality": json.dumps(checks, ensure_ascii=False), "unverified": int(bool(issues)),
            "attempt": attempt, "created_at": store.now(),
        })
    store.log_event("session", session_id, "capture", detail={"slot": slot_id, "accepted": accepted, "issues": [i["code"] for i in issues]})
    return result


# ---------- 環節二 + 三 ----------

def _band(score: float) -> str:
    th = CFG["thresholds"]
    if score < th["t1"]:
        return "pass"
    if score < th["t2"]:
        return "review"
    return "human"


def _compare_slot(vehicle, rental_id, slot_id, base_photo, cur_photo, auto_state, panel_overrides=None, cap=None):
    """Compare one exterior slot; create a record for every panel ≥ T1. auto_state: None (candidate) or 'gap'."""
    res = compare.compare(_load_rgb(base_photo["path"]), _load_rgb(cur_photo["path"]), CFG,
                          panel_overrides=panel_overrides, cap=cap)
    heat_path = HEATMAPS / f"{cur_photo['session_id']}_{slot_id}.png"
    heat_path.write_bytes(res["heatmap_png"])
    heat_url, base_url, cur_url = f"/files/heatmaps/{heat_path.name}", _url(base_photo["path"]), _url(cur_photo["path"])
    records = []
    scripted_cmp = panel_overrides is not None or cap is not None
    if res["aligned"] or scripted_cmp:
        items = list(res["panels"].items())
    else:  # hand-held photos that could not be aligned: one low-confidence record for a human, not 12 false alarms
        items = [(res["max_panel"], max(res["max"], CFG["thresholds"]["t2"]))]
    for pid, score in items:
        band = _band(score)
        if band == "pass":
            continue
        gray = bool(base_photo.get("unverified") or cur_photo.get("unverified"))
        reason = f"{slot_id} 區塊 {pid} 差異 {score:.2f}"
        if gray:
            band, reason = "human", reason + "（基準或本次照片為「未驗證」→ 灰區交人工）"
        if not res["aligned"]:
            reason += "（未能對齊，低信心）"
        rid = store.insert("damage_records", {
            "id": store.new_id("D"), "vehicle_id": vehicle["id"], "rental_id": rental_id, "slot": slot_id,
            "panel": pid, "score": score, "band": band, "state": "candidate", "reason": reason,
            "baseline_photo": base_url, "current_photo": cur_url, "heatmap": heat_url, "created_at": store.now(),
        })
        store.log_event("damage", rid, "detected", None, "candidate", {"score": score, "band": band})
        if auto_state == "gap":
            _record_step(rid, "gap", {"rule": "取車照有、上一手還車照無 → 空窗期"})
        records.append({"id": rid, "panel": pid, "score": score, "band": band, "label": BAND_LABEL[band]})
    return {"slot": slot_id, "aligned": res["aligned"], "inliers": res["inliers"], "ssim": res["ssim"],
            "max": res["max"], "max_panel": res["max_panel"], "heatmap": heat_url,
            "baseline": base_url, "current": cur_url, "records": records,
            "band": _band(res["max"]) if records else "pass", "scripted": panel_overrides is not None or cap is not None}


def _interior_slot(slot_id: str, photo: dict, forced: dict | None = None) -> dict:
    with open(photo["path"], "rb") as f:
        data = f.read()
    res = forced or load_overrides().get(_sha(data), {}).get("interior") or interior.classify(np.asarray(quality.load_image(data)), CFG)
    return {"slot": slot_id, **res, "label": interior.LEVEL_LABEL.get(res.get("level"), res.get("level")), "photo": _url(photo["path"])}


def complete_session(session_id: str) -> dict:
    sess = store.fetchone("SELECT * FROM sessions WHERE id=?", (session_id,))
    rental = store.fetchone("SELECT * FROM rentals WHERE id=?", (sess["rental_id"],))
    vehicle = store.fetchone("SELECT * FROM vehicles WHERE id=?", (rental["vehicle_id"],))
    scenario = sess.get("scenario") or "auto"
    scripted = scenario != "auto"
    photos = _latest_photos(session_id)
    slots = slot_map()
    summary = {"session": session_id, "phase": sess["phase"], "scenario": scenario, "exterior": [], "interior": [],
               "workorders": [], "notes": []}

    if sess["phase"] == "return":
        base_sess = store.fetchone("SELECT * FROM sessions WHERE rental_id=? AND phase='pickup' AND completed_at IS NOT NULL ORDER BY completed_at DESC", (rental["id"],))
        auto_state = None
    else:
        prev = rental.get("prev_rental_id")
        base_sess = store.fetchone("SELECT * FROM sessions WHERE rental_id=? AND phase='return' AND completed_at IS NOT NULL ORDER BY completed_at DESC", (prev,)) if prev else None
        auto_state = "gap"
    base_photos = _latest_photos(base_sess["id"]) if base_sess else {}
    if not base_sess:
        summary["notes"].append("沒有可比對的基準照片（第一次拍），本次照片成為基準。")

    ext_targets = [sid for sid, s in slots.items() if s["kind"] == "exterior" and sid in photos and sid in base_photos]
    int_targets = [sid for sid, s in slots.items() if s["kind"] == "interior" and sid in photos]
    cap = CFG["thresholds"]["t1"] - 0.07 if scripted else None

    for sid in ext_targets:
        ov = None
        if scripted and scenario == "damage":
            idx = ext_targets.index(sid)
            if idx < len(SCRIPT_DAMAGE):
                ov = dict([SCRIPT_DAMAGE[idx]])
        summary["exterior"].append(_compare_slot(vehicle, rental["id"], sid, base_photos[sid], photos[sid], auto_state, ov, cap))
    for i, sid in enumerate(int_targets):
        forced = None
        if scripted:
            forced = ({"level": "dirty", "objects": SCRIPT_DIRTY_OBJECTS, "count": 3, "simulated": True}
                      if scenario == "dirty" and i == 0 else {"level": "clean", "objects": [], "count": 0, "simulated": True})
        summary["interior"].append(_interior_slot(sid, photos[sid], forced))

    dirty = [i for i in summary["interior"] if i.get("level") == "dirty"]
    leftovers = sorted({o for i in summary["interior"] for o in i.get("objects", [])})
    human = [r for e in summary["exterior"] for r in e["records"] if r["band"] == "human"]
    review = [r for e in summary["exterior"] for r in e["records"] if r["band"] == "review"]
    all_verified = all(not p["unverified"] for p in photos.values())

    if sess["phase"] == "pickup":
        gaps = [r for e in summary["exterior"] for r in e["records"]]
        if gaps or dirty:
            msg = "已登記為取車前既有狀況，不會算在您身上。"
            if gaps:
                msg += f" 車外差異 {len(gaps)} 處已記入履歷（空窗期，歸公司處理）。"
            if dirty:
                msg += " 車內髒汙已通知清潔，您可照常用車並獲得補償點數，或換附近的車。"
                summary["workorders"].append(_workorder(vehicle["id"], rental["id"], "cleaning",
                                                        {"why": "取車回報髒汙", "slots": [d["slot"] for d in dirty]}, [d["photo"] for d in dirty]))
            _notify(rental["id"], "pickup_protect", "取車前既有狀況已登記", msg,
                    images=[e["heatmap"] for e in summary["exterior"] if e["records"]] + [d["photo"] for d in dirty])
            _add_points(rental["id"], CFG["points"]["pickup_report_confirmed"], "取車回報屬實：補償點數")
        summary["vehicle_state"] = _vehicle_step(vehicle["id"], "pickup", {"rental": rental["id"]})
        store.update("rentals", rental["id"], {"status": "active"})
    else:
        if all_verified and photos:
            _add_points(rental["id"], CFG["points"]["all_photos_pass"], "取還車照片一次全部合格")
        if leftovers:
            _notify(rental["id"], "leftover", "車內可能有遺留物品", "還車照片裡看到：" + "、".join(leftovers) + "。若是您的物品請聯絡客服。")
        _vehicle_step(vehicle["id"], "return", {"rental": rental["id"]})
        store.update("rentals", rental["id"], {"status": "returned"})
        if dirty:
            summary["workorders"].append(_workorder(vehicle["id"], rental["id"], "cleaning",
                                                    {"why": "還車判定髒汙(須立即清潔)", "slots": [d["slot"] for d in dirty], "objects": leftovers},
                                                    [d["photo"] for d in dirty]))
            summary["vehicle_state"] = _vehicle_step(vehicle["id"], "flag", {"why": "dirty"})
            _reassign_next(rental, "清潔")
        elif human or review:
            summary["vehicle_state"] = "pending_review"          # waits for the human decision
        else:
            summary["vehicle_state"] = _vehicle_step(vehicle["id"], "clear", {"why": "no anomalies"})
        summary["notes"].append(f"路由：{len(human)} 個區塊轉人工、{len(review)} 個送複判、其餘自動放行。")

    store.update("sessions", session_id, {"completed_at": store.now(), "summary": json.dumps(summary, ensure_ascii=False)})
    store.log_event("session", session_id, "completed", detail={"phase": sess["phase"], "scenario": scenario, "human": len(human), "review": len(review), "dirty": len(dirty)})
    return summary


# ---------- 判得公平：人工判定、通知、申訴 ----------

def _maybe_release_vehicle(vehicle_id: str):
    pending = store.fetchall("SELECT id FROM damage_records WHERE vehicle_id=? AND state='candidate'", (vehicle_id,))
    open_wo = store.fetchall("SELECT id FROM workorders WHERE vehicle_id=? AND status<>'done'", (vehicle_id,))
    v = store.fetchone("SELECT state FROM vehicles WHERE id=?", (vehicle_id,))
    if not pending and not open_wo and v["state"] == "pending_review":
        _vehicle_step(vehicle_id, "clear", {"why": "all records decided"})


def decide(record_id: str, decision: str, needs_repair: bool = False) -> dict:
    r = store.fetchone("SELECT * FROM damage_records WHERE id=?", (record_id,))
    if not r:
        raise KeyError("no such record")
    if decision == "new":
        _record_step(record_id, "confirm", {"label": "new", "needs_repair": needs_repair})
        store.update("damage_records", record_id, {"needs_repair": int(needs_repair)})
        _notify(r["rental_id"], "damage_confirmed", "車況判定：本次租借新增損傷",
                f"經人工確認，{r['slot']} 區塊 {r['panel']} 有本次租借產生的損傷（差異 {r['score']:.2f}）。附上取車、還車同視角對照圖；如有疑問可一鍵申訴，真人 48 小時內回覆。",
                record_id=record_id, images=[r["baseline_photo"], r["current_photo"], r["heatmap"]])
        if needs_repair:
            wid = _workorder(r["vehicle_id"], r["rental_id"], "repair",
                             {"why": "人工確認需維修", "slot": r["slot"], "panel": r["panel"], "record": record_id},
                             [r["baseline_photo"], r["current_photo"], r["heatmap"]])
            _vehicle_step(r["vehicle_id"], "flag", {"why": "repair", "workorder": wid})
    elif decision == "old":
        _record_step(record_id, "dismiss", {"label": "old"})
    elif decision == "false":
        _record_step(record_id, "dismiss", {"label": "false_positive"})
    else:
        raise ValueError("decision must be new|old|false")
    store.log_event("label", record_id, "human_label", detail={"label": decision})  # this click is the training label
    _maybe_release_vehicle(r["vehicle_id"])
    return store.fetchone("SELECT * FROM damage_records WHERE id=?", (record_id,))


def appeal(record_id: str, text: str) -> dict:
    _record_step(record_id, "appeal", {"text": text})
    store.update("damage_records", record_id, {"appeal_text": text})
    r = store.fetchone("SELECT * FROM damage_records WHERE id=?", (record_id,))
    _notify(r["rental_id"], "appeal_received", "已收到您的申訴", "真人客服將在 48 小時內回覆（demo：後台可直接處理）。", record_id=record_id)
    return r


def resolve_appeal(record_id: str, outcome: str) -> dict:
    r = store.fetchone("SELECT * FROM damage_records WHERE id=?", (record_id,))
    if outcome == "uphold":
        _record_step(record_id, "uphold")
        _notify(r["rental_id"], "appeal_result", "申訴結果：維持原判定", "經複核，判定維持。您仍可聯絡客服。", record_id=record_id)
    elif outcome == "overturn":
        _record_step(record_id, "overturn", {"label_corrected": True})
        store.log_event("label", record_id, "label_corrected", detail={"from": "new", "to": "closed"})
        _notify(r["rental_id"], "appeal_result", "申訴結果：撤銷判定", "經複核，該損傷不列入您的責任；若已收費將退還。", record_id=record_id)
    else:
        raise ValueError("outcome must be uphold|overturn")
    return store.fetchone("SELECT * FROM damage_records WHERE id=?", (record_id,))


# ---------- 環節三：整備與驗收 ----------

def complete_workorder(wo_id: str, data: bytes) -> dict:
    wo = store.fetchone("SELECT * FROM workorders WHERE id=?", (wo_id,))
    if not wo:
        raise KeyError("no such workorder")
    d = UPLOADS / "workorders"
    d.mkdir(parents=True, exist_ok=True)
    img = quality.load_image(data)
    path = d / f"{wo_id}_{store.now().replace(':', '').replace(' ', '_')}.jpg"
    img.save(path, "JPEG", quality=90)
    url = f"/files/uploads/workorders/{path.name}"
    _vehicle_step(wo["vehicle_id"], "service_done", {"workorder": wo_id})

    sess = store.fetchone("SELECT scenario FROM sessions WHERE rental_id=? ORDER BY created_at DESC", (wo["rental_id"],))
    scripted = bool(sess) and (sess.get("scenario") or "auto") != "auto"
    if wo["type"] == "cleaning":
        if scripted:
            res = {"level": "clean", "objects": [], "count": 0, "simulated": True}
        else:
            res = load_overrides().get(_sha(data), {}).get("interior") or interior.classify(np.asarray(img), CFG)
        accepted = res.get("level") != "dirty"
        acceptance = {"check": "interior", **res, "accepted": accepted}
    else:
        detail = json.loads(wo["detail"])
        rec = store.fetchone("SELECT * FROM damage_records WHERE id=?", (detail.get("record"),))
        if rec and not scripted:
            base_path = UPLOADS / rec["baseline_photo"].split("/files/uploads/")[1]
            res = compare.compare(_load_rgb(str(base_path)), img, CFG)
            score = res["panels"].get(rec["panel"], 1.0)
            accepted = score < CFG["thresholds"]["t1"]
            acceptance = {"check": "compare_to_baseline", "panel": rec["panel"], "score": score, "accepted": accepted}
        else:
            accepted, acceptance = True, {"check": "compare_to_baseline", "panel": rec["panel"] if rec else None, "score": 0.08, "accepted": True, "simulated": True}
        if rec and accepted:
            if rec["state"] in ("confirmed", "gap"):
                _record_step(rec["id"], "repair")
            else:  # e.g. overturned on appeal after the work order was opened: the car still got fixed, the record stays closed
                store.log_event("damage", rec["id"], "repair_noted", rec["state"], rec["state"], {"workorder": wo_id})

    if accepted:
        store.update("workorders", wo_id, {"status": "done", "completed_at": store.now(),
                                           "acceptance": json.dumps(acceptance, ensure_ascii=False), "evidence": json.dumps([url])})
        store.log_event("workorder", wo_id, "accepted", "pending_acceptance", "done", acceptance)
        _vehicle_step(wo["vehicle_id"], "accept", {"workorder": wo_id})
        _maybe_release_vehicle(wo["vehicle_id"])
    else:
        store.update("workorders", wo_id, {"status": "open", "acceptance": json.dumps(acceptance, ensure_ascii=False)})
        store.log_event("workorder", wo_id, "rejected", "pending_acceptance", "open", acceptance)
        _vehicle_step(wo["vehicle_id"], "reject", {"workorder": wo_id})
    return {"workorder": wo_id, "accepted": accepted, "acceptance": acceptance, "photo": url,
            "vehicle_state": store.fetchone("SELECT state FROM vehicles WHERE id=?", (wo["vehicle_id"],))["state"]}
