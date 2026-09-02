"""Fixture provider for offline plumbing tests.

`genuine = False`: the pipeline refuses to write an on-chain record from mock
results unless `--allow-mock` is passed, because the task requires a real search.
"""

from __future__ import annotations

import json
from pathlib import Path

from hhg3.config import Config
from hhg3.logging_utils import warn
from hhg3.search.socialfilter import domain_of, platform_of
from hhg3.types import Candidate

FIXTURES = Path(__file__).resolve().parents[3] / "samples" / "fixtures.json"


class MockProvider:
    name = "mock-fixtures"
    genuine = False

    def available(self) -> bool:
        return FIXTURES.exists()

    def search(self, crop_path: Path, cfg: Config) -> list[Candidate]:
        warn("using MOCK search results - not a genuine search step")
        rows = json.loads(FIXTURES.read_text(encoding="utf-8"))
        return [
            Candidate(
                provider=self.name,
                page_url=row["page_url"],
                image_url=row.get("image_url"),
                title=row.get("title"),
                domain=domain_of(row["page_url"]),
                platform=platform_of(row["page_url"]),
                raw=row,
            )
            for row in rows[: cfg.max_candidates]
        ]
