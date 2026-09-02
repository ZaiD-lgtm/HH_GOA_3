"""Search provider interface.

A provider takes the probe face crop and returns *candidate web pages*. It must
not decide identity - that is `verify.match`'s job, which re-runs face
detection + embedding on whatever image the candidate page actually serves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from hhg3.config import Config
from hhg3.types import Candidate


class SearchProvider(Protocol):
    name: str
    genuine: bool  # False => fixture/mock, does not satisfy the task requirement

    def available(self) -> bool:
        """True when credentials / network prerequisites are satisfied."""

    def search(self, crop_path: Path, cfg: Config) -> list[Candidate]:
        """Reverse-image search `crop_path`, newest/most relevant first."""


class ProviderUnavailable(RuntimeError):
    pass
