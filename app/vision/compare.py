"""環節二 · 新車損（v1 = 結構差異偵測，傳統影像處理，deterministic）。

取車照 = 基準，還車照對齊到基準後算 SSIM 差異圖，切成區塊各出一個分數。
這不是車損分類：反光、水痕也會觸發，所以簡報要老實寫「v1 是差異偵測」，
那正是第二層 VLM 和訓練資料要解的問題。

劇本模式會把某些區塊的分數蓋掉（panel_overrides），熱區圖照蓋掉後的分數畫。
"""
import cv2
import numpy as np
from PIL import Image  # noqa: F401  (type of ref_img / cur_img)
from skimage.exposure import match_histograms
from skimage.metrics import structural_similarity

from .quality import resize_width, to_gray


def _align(ref: np.ndarray, cur: np.ndarray, min_matches: int):
    orb = cv2.ORB_create(nfeatures=2000)
    k1, d1 = orb.detectAndCompute(ref, None)
    k2, d2 = orb.detectAndCompute(cur, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return None, 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(d1, d2), key=lambda m: m.distance)[:300]
    if len(matches) < min_matches:
        return None, len(matches)
    src = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    inliers = int(mask.sum()) if mask is not None else 0
    if H is None or inliers < min_matches:
        return None, inliers
    return H, inliers


def compare(ref_img: Image.Image, cur_img: Image.Image, cfg: dict,
            panel_overrides: dict | None = None, cap: float | None = None) -> dict:
    """panel_overrides: {'r2c2': 0.62} forces those scores; cap: clamp every other panel to ≤ cap."""
    cc = cfg["compare"]
    t1 = cfg["thresholds"]["t1"]
    ref_s = resize_width(ref_img, 640)
    cur_s = resize_width(cur_img, 640)          # keep the aspect ratio; the homography handles the rest
    g1, g2 = to_gray(ref_s), to_gray(cur_s)
    h, w = g1.shape
    H, inliers = _align(g1, g2, cc["min_matches"])
    if H is not None:
        valid = cv2.warpPerspective(np.full(g2.shape, 255, np.uint8), H, (w, h)) > 0
        valid = cv2.erode(valid.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        if valid.mean() < 0.5:                  # degenerate homography: treat as unaligned
            H = None
        else:
            g2 = cv2.warpPerspective(g2, H, (w, h))
    if H is None:
        g2 = cv2.resize(g2, (w, h), interpolation=cv2.INTER_AREA)
        valid = np.ones((h, w), bool)
    g2 = match_histograms(g2, g1).astype(np.uint8)
    g1b = cv2.GaussianBlur(g1, (5, 5), 0)
    g2b = cv2.GaussianBlur(g2, (5, 5), 0)
    ssim, smap = structural_similarity(g1b, g2b, full=True, data_range=255)
    diff = np.clip((1.0 - smap) / 2.0, 0, 1)
    diff[~valid] = 0

    rows, cols = cc["grid_rows"], cc["grid_cols"]
    panels = {}
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * h // rows, (r + 1) * h // rows
            x0, x1 = c * w // cols, (c + 1) * w // cols
            region, vmask = diff[y0:y1, x0:x1], valid[y0:y1, x0:x1]
            vals = region[vmask]
            score = float(np.percentile(vals, cc["percentile"])) if vals.size > 50 else 0.0
            panels[f"r{r + 1}c{c + 1}"] = round(score, 3)
    real_panels = dict(panels)
    if cap is not None:
        panels = {k: round(min(v, cap), 3) for k, v in panels.items()}
    for k, v in (panel_overrides or {}).items():
        if k in panels:
            panels[k] = round(float(v), 3)
    max_panel = max(panels, key=panels.get)

    # heatmap: real diff underneath, scripted panels get a synthetic hot patch so the picture matches the score
    vis = diff.copy()
    for k, v in (panel_overrides or {}).items():
        if k in panels and v >= t1:
            r, c = int(k[1]) - 1, int(k[3]) - 1
            y0, y1 = r * h // rows, (r + 1) * h // rows
            x0, x1 = c * w // cols, (c + 1) * w // cols
            cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
            cv2.ellipse(vis, (cx, cy), ((x1 - x0) // 3, (y1 - y0) // 4), 20, 0, 360, float(v), -1)
            vis = cv2.GaussianBlur(vis, (31, 31), 0)
    base = np.asarray(ref_s)[:, :, ::-1].copy()
    heat = cv2.applyColorMap((np.clip(vis, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(base, 0.55, heat, 0.45, 0)
    for r in range(1, rows):
        cv2.line(overlay, (0, r * h // rows), (w, r * h // rows), (255, 255, 255), 1)
    for c in range(1, cols):
        cv2.line(overlay, (c * w // cols, 0), (c * w // cols, h), (255, 255, 255), 1)
    for pid, s in panels.items():
        r, c = int(pid[1]) - 1, int(pid[3]) - 1
        x0, y0 = c * w // cols, r * h // rows
        if s >= t1:
            cv2.rectangle(overlay, (x0 + 2, y0 + 2), ((c + 1) * w // cols - 2, (r + 1) * h // rows - 2), (0, 0, 255), 3)
        cv2.putText(overlay, f"{s:.2f}", (x0 + 6, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    if H is None:
        cv2.putText(overlay, "UNALIGNED", (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    ok, png = cv2.imencode(".png", overlay)

    return {
        "aligned": H is not None,
        "inliers": int(inliers),
        "ssim": round(float(ssim), 3),
        "panels": panels,
        "real_panels": real_panels,
        "max_panel": max_panel,
        "max": panels[max_panel],
        "heatmap_png": png.tobytes() if ok else b"",
    }
