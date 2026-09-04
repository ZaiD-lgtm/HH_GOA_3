"""Face embedding backends.

Three are identity-grade: ArcFace R50 via onnxruntime (512-d, the default),
InsightFace's own runtime (512-d), and OpenCV SFace (128-d).
Each carries its own `default_threshold`, because a cosine score only means
something relative to the model that produced it - 0.5 is a near-certain match
for SFace and an unremarkable one for a raw pixel descriptor.

`FallbackEmbedder` exists only so the plumbing can be exercised with no models
present. It is not face recognition and the pipeline says so.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from hhg3.face.detect import Detection
from hhg3.logging_utils import info, warn


class Embedder(Protocol):
    name: str
    identity_grade: bool
    default_threshold: float

    def embed(self, image: np.ndarray, det: Detection) -> np.ndarray:
        """Return an L2-normalised embedding for the face at `det` in `image`."""


def _l2(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _yunet_row(det: Detection) -> np.ndarray:
    """The 15-float row SFace's alignCrop expects: bbox, 5 landmarks, score."""
    if det.raw is not None:
        return np.asarray(det.raw, dtype=np.float32).reshape(1, -1)
    if det.landmarks is None:
        raise ValueError("SFace needs landmarks; this detector did not provide any")
    x, y, w, h = det.bbox
    pts = np.asarray(det.landmarks, dtype=np.float32).reshape(-1)
    return np.concatenate([[x, y, w, h], pts, [det.score]]).astype(np.float32).reshape(1, -1)


class InsightFaceEmbedder:
    name = "insightface-arcface-r100"
    identity_grade = True
    default_threshold = 0.45  # ArcFace/buffalo_l same-identity cosine

    def __init__(self, model_name: str = "buffalo_l") -> None:
        from hhg3.face._insight import get_app

        self._app = get_app(model_name)

    def embed(self, image: np.ndarray, det: Detection) -> np.ndarray:
        faces = self._app.get(image)
        if not faces:
            raise ValueError("insightface found no face to embed")
        x, y, w, h = det.bbox
        cx, cy = x + w / 2.0, y + h / 2.0

        def dist(f):
            fx1, fy1, fx2, fy2 = f.bbox
            return (cx - (fx1 + fx2) / 2.0) ** 2 + (cy - (fy1 + fy2) / 2.0) ** 2

        return _l2(min(faces, key=dist).normed_embedding)


class SFaceEmbedder:
    """OpenCV SFace. Needs landmarks for alignment, so pair it with YuNet."""

    name = "opencv-sface"
    identity_grade = True
    default_threshold = 0.363  # OpenCV's documented same-identity cosine threshold

    def __init__(self) -> None:
        import cv2

        from hhg3 import models

        self._cv2 = cv2
        self._model = cv2.FaceRecognizerSF.create(str(models.ensure("sface")), "")

    def embed(self, image: np.ndarray, det: Detection) -> np.ndarray:
        aligned = self._model.alignCrop(image, _yunet_row(det))
        return _l2(self._model.feature(aligned))


class ArcFaceOnnxEmbedder:
    """ArcFace R50 (WebFace600K), 512-d, run directly through onnxruntime.

    Deliberately avoids the `insightface` package: only its published ONNX
    weights are used, so there is no build-toolchain dependency on Windows.
    Alignment is the standard 5-point similarity transform onto ArcFace's
    canonical template, which is what makes the vector depend on the face
    rather than on the surrounding photo.
    """

    name = "arcface-r50-w600k"
    identity_grade = True
    default_threshold = 0.45  # cosine, same-identity for buffalo_l recognition
    input_size = (112, 112)

    # Canonical landmark positions for a 112x112 aligned face. Order matches
    # YuNet's output: right eye, left eye, nose, right mouth, left mouth
    # (person's right = image left, i.e. the smaller x).
    TEMPLATE = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )

    def __init__(self) -> None:
        import cv2
        import onnxruntime as ort

        from hhg3 import models

        self._cv2 = cv2
        self._sess = ort.InferenceSession(
            str(models.ensure("arcface")), providers=["CPUExecutionProvider"]
        )
        self._input = self._sess.get_inputs()[0].name

    def _align(self, image: np.ndarray, det: Detection) -> np.ndarray:
        cv2 = self._cv2
        if det.landmarks is None:
            raise ValueError("ArcFace needs 5-point landmarks; use the YuNet detector")
        kps = np.asarray(det.landmarks, dtype=np.float32).reshape(5, 2)
        matrix, _ = cv2.estimateAffinePartial2D(kps, self.TEMPLATE, method=cv2.LMEDS)
        if matrix is None:
            raise ValueError("could not solve the alignment transform")
        return cv2.warpAffine(image, matrix, self.input_size, borderValue=0.0)

    def embed(self, image: np.ndarray, det: Detection) -> np.ndarray:
        aligned = self._align(image, det)
        blob = self._cv2.dnn.blobFromImage(
            aligned, 1.0 / 127.5, self.input_size, (127.5, 127.5, 127.5), swapRB=True
        )
        out = self._sess.run(None, {self._input: blob.astype(np.float32)})[0]
        return _l2(out)


class FallbackEmbedder:
    """Downsampled, illumination-normalised pixel descriptor. Plumbing only."""

    name = "fallback-pixel-64"
    identity_grade = False
    default_threshold = 0.90

    def __init__(self, size: int = 64) -> None:
        import cv2

        self._cv2 = cv2
        self._size = size

    def embed(self, image: np.ndarray, det: Detection) -> np.ndarray:
        cv2 = self._cv2
        x, y, w, h = det.bbox
        crop = image[max(y, 0) : y + h, max(x, 0) : x + w]
        if crop.size == 0:
            raise ValueError("empty crop")
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (self._size, self._size), interpolation=cv2.INTER_AREA)
        g = cv2.equalizeHist(g).astype(np.float32)
        g = (g - g.mean()) / (g.std() + 1e-6)
        return _l2(g)


_ORDER = ["arcface", "insightface", "sface", "fallback"]
_BUILDERS = {
    "arcface": ArcFaceOnnxEmbedder,
    "insightface": InsightFaceEmbedder,
    "sface": SFaceEmbedder,
    "fallback": FallbackEmbedder,
}


def get_embedder(name: str = "auto") -> Embedder:
    if name != "auto":
        if name not in _BUILDERS:
            raise ValueError("unknown embedder: " + name)
        return _BUILDERS[name]()
    for key in _ORDER:
        try:
            emb = _BUILDERS[key]()
            info("embedder: %s (identity_grade=%s)" % (emb.name, emb.identity_grade))
            if not emb.identity_grade:
                warn("no identity-grade embedder - similarity scores are NOT face recognition")
            return emb
        except Exception as exc:
            warn("embedder %r unavailable (%s: %s)" % (key, type(exc).__name__, exc))
    raise RuntimeError("no embedder available")
