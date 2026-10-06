"""車牌與里程 OCR（預訓練 EasyOCR，固定權重 → 同樣輸入同樣輸出）。

車牌是唯一的車外文字錨點：一次證明「拍到車牌」「是這台車」「解析度夠」，
並用車牌像素寬推回拍攝距離（太遠／太近）。
"""
import re

import numpy as np

_reader = None
_CONFUSABLE = str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6"})
PLATE_SHAPES = [
    re.compile(r"^[A-Z]{3}\d{4}$"),  # ABC-1234
    re.compile(r"^[A-Z]{2}\d{4}$"),  # AB-1234
    re.compile(r"^\d{4}[A-Z]{2}$"),  # 1234-AB
    re.compile(r"^[A-Z]{3}\d{3}$"),  # ABC-123 (機車／舊式)
    re.compile(r"^\d{3}[A-Z]{3}$"),
    re.compile(r"^[A-Z]\d{5}$"),
]


def available() -> bool:
    try:
        import easyocr  # noqa: F401
        return True
    except ImportError:
        return False


def enabled(cfg: dict) -> bool:
    mode = str(cfg["plate"].get("enabled", "auto")).lower()
    if mode == "off":
        return False
    if mode == "on":
        return True
    return available()


def reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


def normalize(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def canon(s: str) -> str:
    return normalize(s).translate(_CONFUSABLE)


def _edit_distance(a: str, b: str) -> int:
    if len(a) != len(b):
        return 99
    return sum(1 for x, y in zip(a, b) if x != y)


def looks_like_plate(n: str) -> bool:
    return any(p.match(n) for p in PLATE_SHAPES)


def detect(img_rgb: np.ndarray, expected: str, cfg: dict) -> dict:
    """img_rgb: numpy RGB (~1280 wide). Returns found/match/distance."""
    if not enabled(cfg):
        return {"skipped": True, "found": False, "match": None, "note": "車牌 OCR 未啟用"}
    bgr = img_rgb[:, :, ::-1].copy()
    results = reader().readtext(bgr, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-", detail=1)
    exp = canon(expected)
    best = None
    for bbox, text, conf in results:
        n = normalize(text)
        if len(n) < 5 or len(n) > 8:
            continue
        c = n.translate(_CONFUSABLE)
        match = c == exp or (_edit_distance(c, exp) <= 1)
        plausible = looks_like_plate(n) or match
        if not plausible:
            continue
        xs = [p[0] for p in bbox]
        width_px = max(xs) - min(xs)
        cand = {"text": n, "conf": float(conf), "match": bool(match), "width_px": float(width_px)}
        if best is None or (cand["match"], cand["conf"]) > (best["match"], best["conf"]):
            best = cand
    if best is None:
        return {"skipped": False, "found": False, "match": False, "text": None}
    pc = cfg["plate"]
    focal = pc["focal_px_at_640"] * img_rgb.shape[1] / 640.0
    distance = pc["real_width_m"] * focal / max(best["width_px"], 1.0)
    best.update({"skipped": False, "found": True, "distance_m": round(float(distance), 2)})
    return best


def read_odometer(img_rgb: np.ndarray, cfg: dict) -> dict:
    if not enabled(cfg):
        return {"skipped": True, "reading": None}
    bgr = img_rgb[:, :, ::-1].copy()
    results = reader().readtext(bgr, allowlist="0123456789", detail=1)
    digits = [re.sub(r"\D", "", t) for _, t, _ in results]
    digits = [d for d in digits if 4 <= len(d) <= 7]
    if not digits:
        return {"skipped": False, "reading": None}
    digits.sort(key=len, reverse=True)
    return {"skipped": False, "reading": int(digits[0])}
