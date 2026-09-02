"""Google Lens reverse-image search through SerpAPI.

Cheapest reliable route to *genuine* reverse image search: no scraping, stable
JSON, and it surfaces the social-media pages that host a matching photo.
"""

from __future__ import annotations

from pathlib import Path

import requests

from hhg3.config import Config
from hhg3.logging_utils import info
from hhg3.search.imagehost import publish
from hhg3.search.socialfilter import domain_of, platform_of
from hhg3.types import Candidate

ENDPOINT = "https://serpapi.com/search.json"


class SerpApiLensProvider:
    name = "serpapi-google-lens"
    genuine = True

    def available(self) -> bool:
        return bool(Config().serpapi_key)

    def search(self, crop_path: Path, cfg: Config) -> list[Candidate]:
        if not cfg.serpapi_key:
            raise RuntimeError("SERPAPI_KEY not set")
        image_url = publish(crop_path, cfg)
        params = {
            "engine": "google_lens",
            "url": image_url,
            "api_key": cfg.serpapi_key,
            "hl": "en",
        }
        resp = requests.get(ENDPOINT, params=params, timeout=cfg.http_timeout)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError("serpapi: " + str(data["error"]))

        rows = []
        rows += data.get("visual_matches") or []
        rows += data.get("image_results") or []

        out: list[Candidate] = []
        for row in rows[: cfg.max_candidates]:
            page = row.get("link") or row.get("source_link")
            if not page:
                continue
            out.append(
                Candidate(
                    provider=self.name,
                    page_url=page,
                    image_url=row.get("original") or row.get("thumbnail"),
                    title=row.get("title") or row.get("source"),
                    domain=domain_of(page),
                    platform=platform_of(page),
                    raw={"probe_image_url": image_url, "row": row},
                )
            )
        info("serpapi returned %d candidates" % len(out))
        return out
