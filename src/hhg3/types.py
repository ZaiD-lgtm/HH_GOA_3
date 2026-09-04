"""Data carried between pipeline stages.

These dataclasses are the contract between stages: swapping a face backend, a
search provider or a chain backend must not change these shapes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FaceProbe:
    """What we are searching for.

    `kind` is "face" when a face was detected - the normal path, where matching
    is done on face embeddings. It is "image" when none was found, in which case
    there is no bbox and no embedding, and candidates are confirmed by whole-image
    similarity instead. Keeping both in one type means the evidence bundle and
    the chain record have the same shape either way.
    """

    source_path: str
    source_sha256: str
    crop_path: str | None
    crop_sha256: str | None
    embedding: list[float]
    detector: str
    embedder: str
    kind: str = "face"  # face | image
    bbox: tuple[int, int, int, int] | None = None  # x, y, w, h in source pixels
    det_score: float = 0.0

    @property
    def has_face(self) -> bool:
        return self.kind == "face"

    def public(self) -> dict[str, Any]:
        """Probe view that goes into the on-chain record (no raw embedding)."""
        return {
            "kind": self.kind,
            "source_sha256": self.source_sha256,
            "crop_sha256": self.crop_sha256,
            "bbox": list(self.bbox) if self.bbox else None,
            "detector": self.detector,
            "embedder": self.embedder,
        }


@dataclass
class Candidate:
    """One hit returned by a reverse-image / web search provider."""

    provider: str
    page_url: str
    image_url: str | None = None
    title: str | None = None
    domain: str = ""
    platform: str | None = None  # "instagram", "x", ... None => not social
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_social(self) -> bool:
        return self.platform is not None


@dataclass
class Match:
    """A candidate confirmed to show the probe.

    `method` records how it was confirmed, because the two scores are not on the
    same scale: "face-cosine" is a cosine over face embeddings, "image-phash" is
    a perceptual-hash agreement over the whole image.
    """

    candidate: Candidate
    similarity: float
    threshold: float
    candidate_image_path: str
    candidate_image_sha256: str
    matched_bbox: tuple[int, int, int, int] | None = None
    method: str = "face-cosine"

    def public(self) -> dict[str, Any]:
        return {
            "provider": self.candidate.provider,
            "page_url": self.candidate.page_url,
            "image_url": self.candidate.image_url,
            "platform": self.candidate.platform,
            "title": self.candidate.title,
            "image_sha256": self.candidate_image_sha256,
            "method": self.method,
            "similarity": round(self.similarity, 6),
            "threshold": self.threshold,
            "matched_bbox": list(self.matched_bbox) if self.matched_bbox else None,
        }


@dataclass
class Receipt:
    """Proof that a record hash was written to a chain."""

    backend: str
    network: str
    record_hash: str
    tx_hash: str
    block_number: int | None = None
    timestamp: int | None = None
    explorer_url: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
