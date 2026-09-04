"""Google Lens reverse-image search through SerpAPI.

Cheapest reliable route to *genuine* reverse image search: no scraping, stable
JSON, and it surfaces the social-media pages that host a matching photo.

Two things the API requires that are easy to get wrong:
  * `type` is mandatory. Without it Lens answers with an `ai_overview` block and
    no matches at all. `all` returns visual_matches plus organic_results.
  * `url` must be an image Google can actually fetch. catbox works; tmpfiles
    does not (see search/imagehost.py).
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
SEARCH_TYPE = "all"  # all | visual_matches | exact_matches | products | about_this_image


class SerpApiLensProvider:
    name = "serpapi-google-lens"
    genuine = True

    def available(self) -> bool:
        return bool(Config().serpapi_key)

    def search(self, image_path: Path, cfg: Config) -> list[Candidate]:
        if not cfg.serpapi_key:
            raise RuntimeError("SERPAPI_KEY not set")
        image_url = publish(image_path, cfg)
        params = {
            "engine": "google_lens",
            "type": SEARCH_TYPE,
            "url": image_url,
            "api_key": cfg.serpapi_key,
            "hl": "en",
        }
        resp = requests.get(ENDPOINT, params=params, timeout=cfg.http_timeout)
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            raise RuntimeError("serpapi: " + str(data["error"]))

        rows = list(data.get("visual_matches") or [])
        rows += list(data.get("organic_results") or [])

        out: list[Candidate] = []
        seen: set[str] = set()
        for row in rows:
            page = row.get("link") or row.get("source_link")
            if not page or page in seen:
                continue
            seen.add(page)
            out.append(
                Candidate(
                    provider=self.name,
                    page_url=page,
                    image_url=row.get("image") or row.get("original") or row.get("thumbnail"),
                    title=row.get("title") or row.get("source"),
                    domain=domain_of(page),
                    platform=platform_of(page),
                    raw={"probe_image_url": image_url, "row": row},
                )
            )
            if len(out) >= cfg.max_candidates:
                break
        info("serpapi returned %d candidates" % len(out))
        return out
