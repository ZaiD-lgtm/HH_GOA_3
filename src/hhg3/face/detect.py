"""Face detection backends.

`get_detector("auto")` picks the best backend that is actually importable, so a
bare `pip install -e .` gets YuNet (a real DNN detector with landmarks, ~340 KB)
and installing the `face` extra upgrades it to InsightFace/SCRFD.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from hhg3.logging_utils import info, warn


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]  # x, y, w, h
    score: float
    landmarks: np.ndarray | None = None  # 5x2: right eye, left eye, nose, mouth corners
    raw: np.ndarray | None = None  # detector-native row, used for alignment


class Detector(Protocol):
    name: str

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Return detections, largest/most confident first."""


def _sort(dets: list[Detection]) -> list[Detection]:
    return sorted(dets, key=lambda d: d.bbox[2] * d.bbox[3] * max(d.score, 1e-3), reverse=True)


class YuNetDetector:
    """OpenCV YuNet. Ships with opencv-python >= 4.5.4, no extra runtime needed."""

    name = "opencv-yunet"

    def __init__(self, score_threshold: float = 0.7) -> None:
        import cv2

        from hhg3 import models

        self._cv2 = cv2
        self._model = cv2.FaceDetectorYN.create(
            str(models.ensure("yunet")),
            "",
            (320, 320),
            score_threshold,
            0.3,
            5000,
        )

    def detect(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]
        self._model.setInputSize((w, h))
        _, faces = self._model.detect(image)
        if faces is None:
            return []
        out: list[Detection] = []
        for row in faces:
            row = np.asarray(row, dtype=np.float32)
            x, y, bw, bh = [int(v) for v in row[:4]]
            out.append(
                Detection(
                    bbox=(max(x, 0), max(y, 0), max(bw, 1), max(bh, 1)),
                    score=float(row[14]),
                    landmarks=row[4:14].reshape(5, 2),
                    raw=row,
                )
            )
        return _sort(out)


class InsightFaceDetector:
    """SCRFD via insightface; pairs with InsightFaceEmbedder for ArcFace vectors."""

    name = "insightface-scrfd"

    def __init__(self, model_name: str = "buffalo_l") -> None:
        from hhg3.face._insight import get_app

        self._app = get_app(model_name)

    def detect(self, image: np.ndarray) -> list[Detection]:
        out: list[Detection] = []
        for f in self._app.get(image):
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            kps = getattr(f, "kps", None)
            out.append(
                Detection(
                    bbox=(x1, y1, max(x2 - x1, 1), max(y2 - y1, 1)),
                    score=float(getattr(f, "det_score", 1.0)),
                    landmarks=np.asarray(kps, dtype=np.float32) if kps is not None else None,
                )
            )
        return _sort(out)


_ORDER = ["insightface", "yunet"]
_BUILDERS = {"insightface": InsightFaceDetector, "yunet": YuNetDetector}


def get_detector(name: str = "auto") -> Detector:
    if name != "auto":
        if name not in _BUILDERS:
            raise ValueError("unknown detector: " + name)
        return _BUILDERS[name]()
    for key in _ORDER:
        try:
            det = _BUILDERS[key]()
            info("detector: " + det.name)
            return det
        except Exception as exc:
            warn("detector %r unavailable (%s: %s)" % (key, type(exc).__name__, exc))
    raise RuntimeError("no face detector available - install opencv-python>=4.5.4 at minimum")
