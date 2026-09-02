"""Blockchain anchoring interface.

Anchoring writes a single 32-byte record hash (plus small metadata) to a chain.
Re-verification recomputes that hash from the local evidence bundle and compares
it to what the chain returns - so tampering with the evidence is detectable.
"""

from __future__ import annotations

from typing import Any, Protocol

from hhg3.config import Config
from hhg3.types import Receipt


class AnchorBackend(Protocol):
    name: str

    def available(self) -> bool:
        """True when the backend can actually write (RPC reachable, key present)."""

    def anchor(self, record_hash: str, metadata: dict[str, Any], cfg: Config) -> Receipt:
        """Write `record_hash` to the chain and return a receipt."""

    def fetch(self, receipt: Receipt, cfg: Config) -> dict[str, Any] | None:
        """Read the record back from the chain. None when it is not found."""
