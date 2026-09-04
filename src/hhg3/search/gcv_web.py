"""Google Cloud Vision Web Detection.

Replaces the retired Bing Visual Search API. Two properties make it the better
fit here: the image is POSTed as base64, so the probe crop never has to be
published to a public image host, and `pagesWithMatchingImages` is already the
shape `Candidate` wants - a page URL plus the matching image on it.

Free for the first 1,000 units/month; needs a GCP project with billing enabled.
"""

from __future__ import annotations

import base64
from pathlib import Path

import requests

from hhg3.config import Config
from hhg3.logging_utils import info
from hhg3.search.socialfilter import domain_of, platform_of
from hhg3.types import Candidate

ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"


def _first_image(page: dict) -> str | None:
    for key in ("fullMatchingImages", "partialMatchingImages"):
        for entry in page.get(key) or []:
            if entry.get("url"):
                return entry["url"]
    return None


class GoogleVisionWebProvider:
    name = "gcv-web-detection"
    genuine = True

    def available(self) -> bool:
        return bool(Config().gcv_key)

    def search(self, crop_path: Path, cfg: Config) -> list[Candidate]:
        if not cfg.gcv_key:
            raise RuntimeError("GOOGLE_VISION_API_KEY not set")
        payload = {
            "requests": [
                {
                    "image": {"content": base64.b64encode(Path(crop_path).read_bytes()).decode()},
                    "features": [{"type": "WEB_DETECTION", "maxResults": cfg.max_candidates}],
                }
            ]
        }
        resp = requests.post(
            ENDPOINT, params={"key": cfg.gcv_key}, json=payload, timeout=cfg.http_timeout
        )
        resp.raise_for_status()
        body = resp.json()["responses"][0]
        if "error" in body:
            raise RuntimeError("cloud vision: " + str(body["error"].get("message", body["error"])))

        detection = body.get("webDetection") or {}
        pages = detection.get("pagesWithMatchingImages") or []

        out: list[Candidate] = []
        for page in pages[: cfg.max_candidates]:
            url = page.get("url")
            if not url:
                continue
            out.append(
                Candidate(
                    provider=self.name,
                    page_url=url,
                    image_url=_first_image(page),
                    title=page.get("pageTitle"),
                    domain=domain_of(url),
                    platform=platform_of(url),
                    raw={"page": page},
                )
            )
        info(
            "cloud vision returned %d pages (%d web entities)"
            % (len(out), len(detection.get("webEntities") or []))
        )
        return out
