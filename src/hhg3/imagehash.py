"""Perceptual hashing, used to confirm candidates when there is no face.

Face matching cannot help on an object photo, so the whole-image path asks a
narrower question instead: is the candidate the *same picture* as the probe?
A DCT perceptual hash answers that while tolerating rescaling and re-encoding,
which is exactly what happens to an image as it travels around the web.

This is a weaker claim than face recognition and the record labels it as such
(`method: "image-phash"`).
"""

from __future__ import annotations

import numpy as np

HASH_SIZE = 8
SCALE = 4  # DCT is taken on HASH_SIZE * SCALE, then the low-frequency corner kept


def phash(image_bgr: np.ndarray) -> np.ndarray:
    """64-bit DCT perceptual hash, returned as a boolean array."""
    import cv2

    size = HASH_SIZE * SCALE
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(gray)[:HASH_SIZE, :HASH_SIZE].ravel()
    # Drop the DC term before taking the median: it carries overall brightness,
    # not structure, and would otherwise skew every bit.
    median = float(np.median(low[1:]))
    return low > median


def agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of matching bits, 0..1. Identical images give 1.0."""
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if a.shape != b.shape:
        raise ValueError("hash length mismatch: %s vs %s" % (a.shape, b.shape))
    return float(np.mean(a == b))
