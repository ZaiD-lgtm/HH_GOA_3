"""Turn search *candidates* into a confirmed *match*.

This is the step that makes the pipeline face identification rather than reverse
image lookup: for each candidate we download the image the page actually serves,
re-run detection + embedding on it, and score every face against the probe.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import requests

from hhg3.config import Config
from hhg3.face.compare import cosine
from hhg3.hashing import sha256_file
from hhg3.logging_utils import info, ok, warn
from hhg3.types import Candidate, FaceProbe, Match


def fetch_image(url: str, dest: Path, cfg: Config) -> Path:
    """Download `url` to `dest`. Raises on non-image or oversized responses."""
    resp = requests.get(
        url,
        headers={"User-Agent": cfg.user_agent, "Accept": "image/*,*/*"},
        timeout=cfg.http_timeout,
        stream=True,
    )
    resp.raise_for_status()
    ctype = resp.headers.get("Content-Type", "")
    if "image" not in ctype:
        raise ValueError("not an image (Content-Type: %s)" % ctype)
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(dest, "wb") as fh:
        for chunk in resp.iter_content(1 << 16):
            written += len(chunk)
            if written > 25 * 1024 * 1024:
                raise ValueError("image exceeds 25 MB")
            fh.write(chunk)
    return dest


def _decode(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode " + str(path))
    return img


def confirm_candidates(
    probe: FaceProbe,
    candidates: list[Candidate],
    detector,
    embedder,
    cfg: Config,
    work_dir: Path,
) -> tuple[Match | None, list[dict]]:
    """Score candidates against the probe.

    Returns the best match above `cfg.match_threshold` (or None) plus a trace of
    every candidate examined, which goes into the evidence bundle.
    """
    probe_vec = np.asarray(probe.embedding, dtype=np.float32)
    trace: list[dict] = []
    best: Match | None = None

    for idx, cand in enumerate(candidates):
        row = {
            "index": idx,
            "page_url": cand.page_url,
            "image_url": cand.image_url,
            "platform": cand.platform,
            "status": "skipped",
            "similarity": None,
        }
        if cfg.social_only and not cand.is_social:
            row["status"] = "not-social"
            trace.append(row)
            continue
        if not cand.image_url:
            row["status"] = "no-image-url"
            trace.append(row)
            continue

        dest = work_dir / ("cand_%02d.jpg" % idx)
        try:
            fetch_image(cand.image_url, dest, cfg)
            image = _decode(dest)
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
            row["status"] = "scored"
            row["similarity"] = round(float(score), 6)
            row["faces_found"] = len(dets)
            info("candidate %d %s -> similarity %.4f" % (idx, cand.domain, score))

            if score >= cfg.match_threshold and (best is None or score > best.similarity):
                best = Match(
                    candidate=cand,
                    similarity=float(score),
                    threshold=cfg.match_threshold,
                    candidate_image_path=str(dest),
                    candidate_image_sha256=sha256_file(dest),
                    matched_bbox=det.bbox,
                )
        except Exception as exc:
            row["status"] = "error"
            row["error"] = "%s: %s" % (type(exc).__name__, exc)
            warn("candidate %d failed: %s" % (idx, exc))
        trace.append(row)

    if best:
        ok("match: %s (similarity %.4f >= %.2f)" % (best.candidate.page_url, best.similarity, cfg.match_threshold))
    else:
        warn("no candidate cleared the %.2f similarity threshold" % cfg.match_threshold)
    return best, trace
