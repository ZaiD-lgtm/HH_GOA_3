"""Evidence bundle: the local artefact whose hash goes on-chain.

`build_record` produces the *canonical record* - the minimal, stable dict that is
hashed. Everything else in the bundle (search trace, file paths, timings) is
context for a human and is deliberately excluded from the hash, so re-verifying
does not depend on ephemeral details.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hhg3 import __version__
from hhg3.hashing import sha256_json
from hhg3.types import FaceProbe, Match, Receipt

BUNDLE_NAME = "bundle.json"
RECORD_NAME = "record.json"


def build_record(run_id: str, created_at: str, probe: FaceProbe, match: Match, provider: str) -> dict[str, Any]:
    """The exact object that gets hashed and anchored."""
    return {
        "schema": "hhg3/face-match-record/v1",
        "version": __version__,
        "run_id": run_id,
        "created_at": created_at,
        "search_provider": provider,
        "probe": probe.public(),
        "match": match.public(),
    }


def build_bundle(
    run_id: str,
    probe: FaceProbe,
    match: Match,
    provider: str,
    trace: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record = build_record(run_id, created_at, probe, match, provider)
    return {
        "record": record,
        "record_hash": sha256_json(record),
        "receipt": None,
        "context": {
            "probe_source_path": probe.source_path,
            "probe_crop_path": probe.crop_path,
            "match_image_path": match.candidate_image_path,
            "candidates": candidates,
            "trace": trace,
        },
    }


def save_bundle(run_dir: Path, bundle: dict[str, Any]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / RECORD_NAME).write_text(
        json.dumps(bundle["record"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    path = run_dir / BUNDLE_NAME
    path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_bundle(run_dir: Path) -> dict[str, Any]:
    path = Path(run_dir) / BUNDLE_NAME
    if not path.exists():
        raise FileNotFoundError("no evidence bundle at " + str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def attach_receipt(run_dir: Path, bundle: dict[str, Any], receipt: Receipt) -> dict[str, Any]:
    bundle["receipt"] = receipt.to_dict()
    save_bundle(Path(run_dir), bundle)
    return bundle


def recompute_hash(bundle: dict[str, Any]) -> str:
    """Re-derive the record hash from the stored record - the tamper check."""
    return sha256_json(bundle["record"])
