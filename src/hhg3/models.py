"""On-demand model cache.

YuNet (detection) and SFace (recognition) are small ONNX models shipped by the
OpenCV Zoo. They are fetched once into `models/` (gitignored) rather than
vendored, and pinned by URL so a run is reproducible.
"""

from __future__ import annotations

from pathlib import Path

import requests

from hhg3.config import ROOT
from hhg3.logging_utils import info

CACHE = ROOT / "models"

ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODELS = {
    "yunet": ZOO + "/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "sface": ZOO + "/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


def ensure(name: str) -> Path:
    """Return the local path to model `name`, downloading it if missing."""
    if name not in MODELS:
        raise KeyError("unknown model: " + name)
    url = MODELS[name]
    dest = CACHE / url.rsplit("/", 1)[-1]
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    info("downloading %s model -> %s" % (name, dest))
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(1 << 16):
                fh.write(chunk)
    tmp.replace(dest)
    return dest
