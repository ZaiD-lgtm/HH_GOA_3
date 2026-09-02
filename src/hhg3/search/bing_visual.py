"""Bing Visual Search - accepts a direct multipart upload, so no image host needed."""

from __future__ import annotations

from pathlib import Path

import requests

from hhg3.config import Config
from hhg3.logging_utils import info
from hhg3.search.socialfilter import domain_of, platform_of
from hhg3.types import Candidate

_WANTED = {"PagesIncluding", "VisualSearch", "SimilarImages"}


class BingVisualSearchProvider:
    name = "bing-visual-search"
    genuine = True

    def available(self) -> bool:
        return bool(Config().bing_key)

    def search(self, crop_path: Path, cfg: Config) -> list[Candidate]:
        if not cfg.bing_key:
            raise RuntimeError("BING_VISUAL_SEARCH_KEY not set")
        with open(crop_path, "rb") as fh:
            resp = requests.post(
                cfg.bing_endpoint,
                headers={"Ocp-Apim-Subscription-Key": cfg.bing_key},
                files={"image": (crop_path.name, fh)},
                timeout=cfg.http_timeout,
            )
        resp.raise_for_status()
        data = resp.json()

        out: list[Candidate] = []
        for tag in data.get("tags", []):
            for action in tag.get("actions", []):
                if action.get("actionType") not in _WANTED:
                    continue
                for row in (action.get("data") or {}).get("value", []):
                    page = row.get("hostPageUrl") or row.get("webSearchUrl")
                    if not page:
                        continue
                    out.append(
                        Candidate(
                            provider=self.name,
                            page_url=page,
                            image_url=row.get("contentUrl") or row.get("thumbnailUrl"),
                            title=row.get("name"),
                            domain=domain_of(page),
                            platform=platform_of(page),
                            raw={"row": row},
                        )
                    )
        info("bing returned %d candidates" % len(out))
        return out[: cfg.max_candidates]
