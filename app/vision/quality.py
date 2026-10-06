"""環節一 · 品質檢查（純規則，deterministic）。"""
import io

import cv2
import numpy as np
from PIL import Image, ImageOps

try:  # iPhone HEIC support if pillow-heif is installed
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass


def load_image(data: bytes) -> Image.Image:
    """Decode bytes, apply EXIF orientation (phones!), return RGB."""
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def resize_width(img: Image.Image, width: int) -> Image.Image:
    if img.width <= width:
        return img
    h = round(img.height * width / img.width)
    return img.resize((width, h), Image.LANCZOS)


def to_gray(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)


def blur_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness(gray: np.ndarray) -> float:
    return float(gray.mean() / 255.0)


def assess(img: Image.Image, cfg: dict, kind: str = "exterior") -> dict:
    """Returns {'blur','brightness','orientation','issues':[{code,msg,hard}]}"""
    th = cfg["thresholds"]
    small = resize_width(img, 640)
    gray = to_gray(small)
    b = blur_score(gray)
    br = brightness(gray)
    orient = "landscape" if img.width >= img.height else "portrait"
    issues = []
    blur_min = th.get("blur_min_interior", th["blur_min"]) if kind == "interior" else th["blur_min"]
    if b < blur_min:
        issues.append({"code": "blur", "msg": "太糊了，請拿穩再拍一次", "hard": False})
    if br < th["brightness_min"]:
        issues.append({"code": "dark", "msg": "太暗，請開閃光燈或車內燈", "hard": False})
    if br > th["brightness_max"]:
        issues.append({"code": "bright", "msg": "太亮／反光，請換個角度", "hard": False})
    want = cfg.get("capture", {}).get("orientation", "any")
    if want == "landscape" and orient == "portrait":
        issues.append({"code": "portrait", "msg": "請轉成橫向拍攝", "hard": False})
    if want == "portrait" and orient == "landscape":
        issues.append({"code": "landscape", "msg": "請直拿手機拍攝", "hard": False})
    return {"blur": round(b, 1), "brightness": round(br, 3), "orientation": orient, "issues": issues}
