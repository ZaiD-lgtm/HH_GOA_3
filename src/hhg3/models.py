"""On-demand model cache.

Weights are fetched once into `models/` (gitignored) rather than vendored, and
pinned by URL so a run is reproducible. Some are published only inside a zip,
so an entry may name the member to extract.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import requests

from hhg3.config import ROOT
from hhg3.logging_utils import info

CACHE = ROOT / "models"

ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
INSIGHTFACE = "https://github.com/deepinsight/insightface/releases/download/v0.7"

MODELS: dict[str, dict[str, str]] = {
    "yunet": {
        "url": ZOO + "/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "file": "face_detection_yunet_2023mar.onnx",
    },
    "sface": {
        "url": ZOO + "/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "file": "face_recognition_sface_2021dec.onnx",
    },
    # ArcFace R50 trained on WebFace600K - the recognition model from the
    # official buffalo_l pack. 512-d. Only published inside the pack's zip.
    "arcface": {
        "url": INSIGHTFACE + "/buffalo_l.zip",
        "file": "w600k_r50.onnx",
        "member": "w600k_r50.onnx",
        "note": "275 MB download, one time",
    },
}


def _download(url: str, dest: Path) -> Path:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(1 << 18):
                fh.write(chunk)
    tmp.replace(dest)
    return dest


def ensure(name: str) -> Path:
    """Return the local path to model `name`, downloading it if missing."""
    if name not in MODELS:
        raise KeyError("unknown model: " + name)
    spec = MODELS[name]
    dest = CACHE / spec["file"]
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    CACHE.mkdir(parents=True, exist_ok=True)
    suffix = " (%s)" % spec["note"] if spec.get("note") else ""
    info("downloading %s model%s -> %s" % (name, suffix, dest))

    if "member" not in spec:
        return _download(spec["url"], dest)

    archive = CACHE / spec["url"].rsplit("/", 1)[-1]
    if not archive.exists():
        _download(spec["url"], archive)
    with zipfile.ZipFile(archive) as zf:
        member = next((n for n in zf.namelist() if n.endswith(spec["member"])), None)
        if member is None:
            raise RuntimeError("%s not found inside %s" % (spec["member"], archive.name))
        with zf.open(member) as src, open(dest, "wb") as out:
            out.write(src.read())
    info("extracted %s from %s" % (spec["file"], archive.name))
    return dest
