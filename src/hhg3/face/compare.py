"""Similarity + decision helpers."""

from __future__ import annotations

import numpy as np


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.shape != b.shape:
        raise ValueError("embedding shape mismatch: %s vs %s" % (a.shape, b.shape))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def best_match(probe, others: list) -> tuple[int, float]:
    """Index and score of the closest embedding; (-1, 0.0) when `others` is empty."""
    if not others:
        return -1, 0.0
    scores = [cosine(probe, o) for o in others]
    idx = int(np.argmax(scores))
    return idx, float(scores[idx])
