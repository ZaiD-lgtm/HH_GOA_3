"""Publish the probe crop at a public URL.

Google Lens (via SerpAPI) and Yandex take an image *URL*, not an upload, so the
crop has to be reachable for the duration of the search - and Google must be
able to fetch it. catbox works (verified: Lens resolves catbox URLs) but
intermittently returns an empty body, so uploads are retried with backoff.

tmpfiles.org was tried and rejected: its /dl/ links answer with text/html, and
Lens returns zero matches for them.

The Cloud Vision provider needs none of this; prefer it when the probe is
someone else's photo.
"""

from __future__ import annotations

import time
from pathlib import Path

import requests

from hhg3.config import Config
from hhg3.logging_utils import info, warn

ATTEMPTS = 3
BACKOFF = 1.5  # seconds, multiplied by attempt number


def _catbox(path: Path, cfg: Config) -> str:
    with open(path, "rb") as fh:
        resp = requests.post(
            "https://catbox.moe/user/api.php",
            data={"reqtype": "fileupload"},
            files={"fileToUpload": (path.name, fh, "image/jpeg")},
            timeout=cfg.http_timeout,
            headers={"User-Agent": cfg.user_agent},
        )
    resp.raise_for_status()
    url = resp.text.strip()
    if not url.startswith("http"):
        raise RuntimeError("catbox returned %r" % url[:200])
    return url


def _imgbb(path: Path, cfg: Config) -> str:
    if not cfg.imgbb_key:
        raise RuntimeError("IMGBB_KEY not set")
    import base64

    payload = base64.b64encode(path.read_bytes()).decode()
    resp = requests.post(
        "https://api.imgbb.com/1/upload",
        data={"key": cfg.imgbb_key, "image": payload, "expiration": 600},
        timeout=cfg.http_timeout,
    )
    resp.raise_for_status()
    return resp.json()["data"]["url"]


_HOSTS = {"catbox": _catbox, "imgbb": _imgbb}


def publish(path: Path, cfg: Config) -> str:
    """Upload `path` and return its public URL. Raises when every host fails."""
    order = [cfg.image_host] + [h for h in _HOSTS if h != cfg.image_host]
    errors = []
    for name in order:
        if name not in _HOSTS:
            continue
        for attempt in range(1, ATTEMPTS + 1):
            try:
                url = _HOSTS[name](path, cfg)
                info("probe crop published via %s: %s" % (name, url))
                return url
            except Exception as exc:
                errors.append("%s#%d: %s" % (name, attempt, exc))
                warn("image host %r attempt %d failed: %s" % (name, attempt, exc))
                if attempt < ATTEMPTS:
                    time.sleep(BACKOFF * attempt)
    raise RuntimeError("could not publish probe crop -> " + " | ".join(errors))
