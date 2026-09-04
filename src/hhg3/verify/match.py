"""Turn search *candidates* into a confirmed *match*.

This is the step that makes the pipeline face identification rather than reverse
image lookup: for each candidate we download the image the page actually serves,
re-run detection + embedding on it, and score every face against the probe.
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import requests

from hhg3.config import Config
from hhg3.face.compare import cosine
from hhg3.hashing import sha256_file
from hhg3.imagehash import agreement, phash
from hhg3.logging_utils import info, ok, warn
from hhg3.types import Candidate, FaceProbe, Match

# Search engines hand us Meta's crawler endpoints (lookaside.fbsbx.com,
# lookaside.instagram.com) rather than CDN files. Measured behaviour:
#   facebook  - serves text/html to a browser UA, the real JPEG to a crawler UA
#   instagram - serves HTML to every UA, but its <head> carries an og:image
#               pointing at the actual cdninstagram file
# So a fetch falls back: browser UA -> crawler UA -> og:image on the HTML.
CRAWLER_UA = "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
MAX_BYTES = 25 * 1024 * 1024
HTML_SNIFF = 1 << 19  # 512 KB reaches <head> on even the worst offenders

_OG_IMAGE_RE = re.compile(
    rb"""<meta[^>]+(?:property|name)\s*=\s*["'](?:og:image|twitter:image)["'][^>]*"""
    rb"""content\s*=\s*["']([^"']+)["']""",
    re.IGNORECASE,
)


def _request(url: str, cfg: Config, user_agent: str):
    return requests.get(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "image/avif,image/webp,image/*,*/*;q=0.8",
        },
        timeout=cfg.http_timeout,
        stream=True,
    )


def _save_image(resp, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(dest, "wb") as fh:
        for chunk in resp.iter_content(1 << 16):
            written += len(chunk)
            if written > MAX_BYTES:
                raise ValueError("image exceeds 25 MB")
            fh.write(chunk)
    return dest


def _og_image_url(resp, page_url: str) -> str | None:
    """Pull og:image / twitter:image out of an HTML response's head."""
    buf = b""
    for chunk in resp.iter_content(1 << 16):
        buf += chunk
        if len(buf) >= HTML_SNIFF:
            break
    found = _OG_IMAGE_RE.search(buf)
    if not found:
        return None
    return urljoin(page_url, html.unescape(found.group(1).decode("utf-8", "replace")))


def fetch_image(url: str, dest: Path, cfg: Config) -> Path:
    """Download the image at `url` to `dest`, working around crawler gateways."""
    last_resp, last_ctype = None, ""
    for user_agent in (cfg.user_agent, CRAWLER_UA):
        resp = _request(url, cfg, user_agent)
        resp.raise_for_status()
        last_ctype = resp.headers.get("Content-Type", "")
        if "image" in last_ctype:
            return _save_image(resp, dest)
        last_resp = resp

    if last_resp is not None and "html" in last_ctype:
        target = _og_image_url(last_resp, url)
        if target:
            info("following og:image -> " + target[:90])
            resp = _request(target, cfg, CRAWLER_UA)
            resp.raise_for_status()
            if "image" in resp.headers.get("Content-Type", ""):
                return _save_image(resp, dest)

    raise ValueError("not an image (Content-Type: %s)" % last_ctype)


def _decode(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode " + str(path))
    return img


def _prepare(cand: Candidate, idx: int, cfg: Config, work_dir: Path, row: dict):
    """Common candidate gate: social filter, image URL, download, decode."""
    if cfg.social_only and not cand.is_social:
        row["status"] = "not-social"
        return None
    if not cand.image_url:
        row["status"] = "no-image-url"
        return None
    dest = work_dir / ("cand_%02d.jpg" % idx)
    fetch_image(cand.image_url, dest, cfg)
    return dest, _decode(dest)


def _score_by_face(probe, candidates, detector, embedder, cfg, work_dir):
    probe_vec = np.asarray(probe.embedding, dtype=np.float32)
    trace: list[dict] = []
    best: Match | None = None

    for idx, cand in enumerate(candidates):
        row = {"index": idx, "page_url": cand.page_url, "image_url": cand.image_url,
               "platform": cand.platform, "status": "skipped", "similarity": None}
        try:
            prepared = _prepare(cand, idx, cfg, work_dir, row)
            if prepared is None:
                trace.append(row)
                continue
            dest, image = prepared

            dets = detector.detect(image)
            if not dets:
                row["status"] = "no-face-in-candidate"
                trace.append(row)
                continue

            scores = []
            for det in dets[:5]:
                try:
                    scores.append((cosine(probe_vec, embedder.embed(image, det)), det))
                except Exception as exc:
                    warn("embed failed on candidate %d: %s" % (idx, exc))
            if not scores:
                row["status"] = "embed-failed"
                trace.append(row)
                continue

            score, det = max(scores, key=lambda pair: pair[0])
            row.update(status="scored", similarity=round(float(score), 6), faces_found=len(dets))
            info("candidate %d %s -> similarity %.4f" % (idx, cand.domain, score))

            if score >= cfg.match_threshold and (best is None or score > best.similarity):
                best = Match(candidate=cand, similarity=float(score), threshold=cfg.match_threshold,
                             candidate_image_path=str(dest), candidate_image_sha256=sha256_file(dest),
                             matched_bbox=det.bbox, method="face-cosine")
        except Exception as exc:
            row["status"] = "error"
            row["error"] = "%s: %s" % (type(exc).__name__, exc)
            warn("candidate %d failed: %s" % (idx, exc))
        trace.append(row)
    return best, trace


def _score_by_image(probe, candidates, cfg, work_dir):
    """No face to compare, so ask whether it is the same picture instead."""
    probe_hash = phash(_decode(Path(probe.source_path)))
    threshold = cfg.image_match_threshold
    trace: list[dict] = []
    best: Match | None = None

    for idx, cand in enumerate(candidates):
        row = {"index": idx, "page_url": cand.page_url, "image_url": cand.image_url,
               "platform": cand.platform, "status": "skipped", "similarity": None}
        try:
            prepared = _prepare(cand, idx, cfg, work_dir, row)
            if prepared is None:
                trace.append(row)
                continue
            dest, image = prepared

            score = agreement(probe_hash, phash(image))
            row.update(status="scored", similarity=round(score, 6))
            info("candidate %d %s -> image agreement %.4f" % (idx, cand.domain, score))

            if score >= threshold and (best is None or score > best.similarity):
                best = Match(candidate=cand, similarity=score, threshold=threshold,
                             candidate_image_path=str(dest), candidate_image_sha256=sha256_file(dest),
                             matched_bbox=None, method="image-phash")
        except Exception as exc:
            row["status"] = "error"
            row["error"] = "%s: %s" % (type(exc).__name__, exc)
            warn("candidate %d failed: %s" % (idx, exc))
        trace.append(row)
    return best, trace


def confirm_candidates(
    probe: FaceProbe,
    candidates: list[Candidate],
    detector,
    embedder,
    cfg: Config,
    work_dir: Path,
) -> tuple[Match | None, list[dict]]:
    """Score candidates against the probe.

    Face probes are confirmed by embedding cosine; a probe with no face falls
    back to perceptual-hash agreement over the whole image. The two scores live
    on different scales and carry different thresholds, so `Match.method`
    records which one produced the result.
    """
    if probe.has_face:
        best, trace = _score_by_face(probe, candidates, detector, embedder, cfg, work_dir)
        threshold, label = cfg.match_threshold, "similarity"
    else:
        best, trace = _score_by_image(probe, candidates, cfg, work_dir)
        threshold, label = cfg.image_match_threshold, "image agreement"

    if best:
        ok("match: %s (%s %.4f >= %.2f)" % (best.candidate.page_url, label, best.similarity, threshold))
    else:
        warn("no candidate cleared the %.2f %s threshold" % (threshold, label))
    return best, trace
