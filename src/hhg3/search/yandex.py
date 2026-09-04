"""Yandex reverse image search, scripted.

No API key, but it is HTML scraping against an anti-bot endpoint: treat it as a
best-effort fallback, not the primary provider. The parser is deliberately
tolerant - it walks the JSON blob Yandex embeds in the results page.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import requests

from hhg3.config import Config
from hhg3.logging_utils import info, warn
from hhg3.search.imagehost import publish
from hhg3.search.socialfilter import domain_of, platform_of
from hhg3.types import Candidate

SEARCH_URL = "https://yandex.com/images/search"
_STATE_RE = re.compile(r'data-state="([^"]+)"')


class YandexProvider:
    name = "yandex-images"
    genuine = True

    def available(self) -> bool:
        return True  # no credentials, but may be rate-limited/captcha'd

    def search(self, crop_path: Path, cfg: Config) -> list[Candidate]:
        image_url = publish(crop_path, cfg)
        resp = requests.get(
            SEARCH_URL,
            params={"rpt": "imageview", "url": image_url, "cbir_page": "sites"},
            headers={"User-Agent": cfg.user_agent, "Accept-Language": "en-US,en;q=0.9"},
            timeout=cfg.http_timeout,
        )
        resp.raise_for_status()
        if "captcha" in resp.url or "SmartCaptcha" in resp.text:
            raise RuntimeError("yandex served a captcha - use serpapi/gcv instead")

        out: list[Candidate] = []
        for blob in _STATE_RE.findall(resp.text):
            try:
                state = json.loads(blob.replace("&quot;", '"'))
            except Exception:
                continue
            for row in _walk_sites(state):
                page = row.get("url")
                if not page:
                    continue
                out.append(
                    Candidate(
                        provider=self.name,
                        page_url=page,
                        image_url=(row.get("originalImage") or {}).get("url"),
                        title=row.get("title"),
                        domain=domain_of(page),
                        platform=platform_of(page),
                        raw={"probe_image_url": image_url, "row": row},
                    )
                )
        if not out:
            warn("yandex parse produced no candidates (markup may have changed)")
        info("yandex returned %d candidates" % len(out))
        return out[: cfg.max_candidates]


def _walk_sites(node) -> list[dict]:
    """Yandex nests the useful list under an unstable path; find it structurally."""
    found: list[dict] = []
    if isinstance(node, dict):
        if isinstance(node.get("sites"), list):
            found += [s for s in node["sites"] if isinstance(s, dict)]
        for value in node.values():
            found += _walk_sites(value)
    elif isinstance(node, list):
        for value in node:
            found += _walk_sites(value)
    return found
