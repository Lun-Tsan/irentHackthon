"""車內整潔三級（預訓練 YOLO COCO 權重＋規則，deterministic、可解釋）。

「偵測到 2 個瓶子、1 個袋子 → 髒汙(須立即清潔)」。順便就是遺留物提醒。
沒裝 ultralytics 時回 unknown（當普通處理），或用 fixtures/overrides.json 強制結果。
"""
import numpy as np

from ..config import ROOT

TRASH_LIKE = {
    "bottle", "cup", "wine glass", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "backpack", "handbag",
    "suitcase", "umbrella", "cell phone", "book", "remote", "scissors", "toothbrush", "hair drier",
    "laptop", "mouse", "keyboard", "sports ball", "frisbee", "skateboard", "tie", "vase", "teddy bear",
}
_model = None


def available() -> bool:
    try:
        import ultralytics  # noqa: F401
        return True
    except ImportError:
        return False


def model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        weights = ROOT / "models" / "yolov8n.pt"
        weights.parent.mkdir(exist_ok=True)
        _model = YOLO(str(weights) if weights.exists() else "yolov8n.pt")
    return _model


def classify(img_rgb: np.ndarray, cfg: dict) -> dict:
    th = cfg["thresholds"]
    if not available():
        return {"level": "unknown", "objects": [], "count": 0, "note": "整潔頭未啟用（未安裝 ultralytics）"}
    res = model().predict(img_rgb, imgsz=640, conf=0.25, verbose=False)[0]
    names = res.names
    objs = [names[int(c)] for c in res.boxes.cls.tolist()] if res.boxes is not None else []
    trash = [o for o in objs if o in TRASH_LIKE]
    n = len(trash)
    if n >= th["dirty_objects_min"]:
        level = "dirty"
    elif n >= th["normal_objects_min"]:
        level = "normal"
    else:
        level = "clean"
    return {"level": level, "objects": trash, "all_objects": objs, "count": n}


LEVEL_LABEL = {"dirty": "髒汙（須立即清潔）", "normal": "普通", "clean": "乾淨", "unknown": "未判定"}
