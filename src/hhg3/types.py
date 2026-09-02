"""Data carried between pipeline stages.

These dataclasses are the contract between stages: swapping a face backend, a
search provider or a chain backend must not change these shapes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FaceProbe:
    """The face we are searching for."""

    source_path: str
    source_sha256: str
    bbox: tuple[int, int, int, int]  # x, y, w, h in source pixels
    crop_path: str
    crop_sha256: str
    embedding: list[float]
    detector: str
    embedder: str
    det_score: float = 0.0

    def public(self) -> dict[str, Any]:
        """Probe view that goes into the on-chain record (no raw embedding)."""
        return {
            "source_sha256": self.source_sha256,
            "crop_sha256": self.crop_sha256,
            "bbox": list(self.bbox),
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
    """A candidate whose image actually contains the probe face."""

    candidate: Candidate
    similarity: float
    threshold: float
    candidate_image_path: str
    candidate_image_sha256: str
    matched_bbox: tuple[int, int, int, int]

    def public(self) -> dict[str, Any]:
        return {
            "provider": self.candidate.provider,
            "page_url": self.candidate.page_url,
            "image_url": self.candidate.image_url,
            "platform": self.candidate.platform,
            "title": self.candidate.title,
            "image_sha256": self.candidate_image_sha256,
            "similarity": round(self.similarity, 6),
            "threshold": self.threshold,
            "matched_bbox": list(self.matched_bbox),
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
