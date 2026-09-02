"""Stage orchestration: face scan -> search -> match -> anchor -> re-verify."""

from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hhg3 import evidence
from hhg3.chain.registry import get_chain
from hhg3.config import Config
from hhg3.face.detect import get_detector
from hhg3.face.embed import get_embedder
from hhg3.hashing import sha256_file
from hhg3.logging_utils import info, ok, stage, warn
from hhg3.search.registry import get_provider
from hhg3.types import FaceProbe, Receipt
from hhg3.verify.match import confirm_candidates


class PipelineError(RuntimeError):
    pass


def new_run_dir(cfg: Config) -> tuple[str, Path]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    path = Path(cfg.runs_dir) / run_id
    path.mkdir(parents=True, exist_ok=True)
    return run_id, path


# --- stage 1: face -----------------------------------------------------
def scan_face(image_path: Path, detector, embedder, run_dir: Path) -> FaceProbe:
    import cv2

    stage("1/4 face scan: " + str(image_path))
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise PipelineError("could not read image: " + str(image_path))

    dets = detector.detect(image)
    if not dets:
        raise PipelineError("no face detected in " + str(image_path))
    det = dets[0]
    info("faces detected: %d, using bbox %s" % (len(dets), det.bbox))

    x, y, w, h = det.bbox
    pad = int(0.25 * max(w, h))
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, image.shape[1]), min(y + h + pad, image.shape[0])
    crop_path = run_dir / "probe_crop.jpg"
    cv2.imwrite(str(crop_path), image[y0:y1, x0:x1])

    vector = embedder.embed(image, det)
    ok("probe embedded (%d-d, %s)" % (len(vector), embedder.name))
    return FaceProbe(
        source_path=str(image_path),
        source_sha256=sha256_file(image_path),
        bbox=det.bbox,
        crop_path=str(crop_path),
        crop_sha256=sha256_file(crop_path),
        embedding=[float(v) for v in vector],
        detector=detector.name,
        embedder=embedder.name,
        det_score=det.score,
    )


# --- stage 2 + 3: search and confirm -----------------------------------
def find_match(probe: FaceProbe, provider, detector, embedder, cfg: Config, run_dir: Path):
    stage("2/4 reverse-image search via " + provider.name)
    candidates = provider.search(Path(probe.crop_path), cfg)
    if not candidates:
        raise PipelineError("search returned no candidates")
    social = [c for c in candidates if c.is_social]
    info("candidates: %d total, %d on social platforms" % (len(candidates), len(social)))

    stage("3/4 confirming candidates against the probe face")
    match, trace = confirm_candidates(probe, candidates, detector, embedder, cfg, run_dir / "candidates")
    if match is None:
        raise PipelineError(
            "no candidate matched above threshold %.2f - try --threshold, another "
            "--provider, or a clearer input image" % cfg.match_threshold
        )
    return match, trace, [asdict(c) for c in candidates]


# --- stage 4: anchor ---------------------------------------------------
def anchor(bundle: dict[str, Any], cfg: Config, run_dir: Path) -> Receipt:
    backend = get_chain(cfg.chain)
    stage("4/4 anchoring on chain backend " + backend.name)
    record = bundle["record"]
    metadata = {
        "s": record["schema"],
        "r": record["run_id"],
        "p": record["match"]["platform"],
        "u": record["match"]["page_url"],
    }
    receipt = backend.anchor(bundle["record_hash"], metadata, cfg)
    evidence.attach_receipt(run_dir, bundle, receipt)
    if receipt.explorer_url:
        ok("explorer: " + receipt.explorer_url)
    return receipt


def run(image_path: Path, cfg: Config, allow_mock: bool = False) -> dict[str, Any]:
    detector = get_detector(cfg.detector)
    embedder = get_embedder(cfg.embedder)
    if cfg.match_threshold is None:
        cfg.match_threshold = embedder.default_threshold
        info("threshold: %.3f (default for %s)" % (cfg.match_threshold, embedder.name))
    provider = get_provider(cfg.search_provider)
    if not provider.genuine and not allow_mock:
        raise PipelineError(
            "provider %r is a fixture, not a real search - pass --allow-mock to "
            "run it anyway (the submitted demo must use a genuine provider)" % provider.name
        )

    run_id, run_dir = new_run_dir(cfg)
    info("run id: " + run_id)

    probe = scan_face(Path(image_path), detector, embedder, run_dir)
    match, trace, candidates = find_match(probe, provider, detector, embedder, cfg, run_dir)
    bundle = evidence.build_bundle(run_id, probe, match, provider.name, trace, candidates)
    evidence.save_bundle(run_dir, bundle)
    info("record hash: " + bundle["record_hash"])

    receipt = anchor(bundle, cfg, run_dir)
    ok("run complete: " + str(run_dir))
    return {"run_id": run_id, "run_dir": str(run_dir), "bundle": bundle, "receipt": receipt.to_dict()}


# --- re-verification ---------------------------------------------------
def verify(run_dir: Path, cfg: Config) -> dict[str, Any]:
    """Recompute the record hash locally and compare it with the chain."""
    bundle = evidence.load_bundle(Path(run_dir))
    report: dict[str, Any] = {"run_dir": str(run_dir), "checks": []}

    def check(name: str, passed: bool, detail: str = "") -> None:
        report["checks"].append({"check": name, "pass": bool(passed), "detail": detail})
        (ok if passed else warn)("%s: %s %s" % (name, "PASS" if passed else "FAIL", detail))

    recomputed = evidence.recompute_hash(bundle)
    check("record hash recomputed from local evidence", recomputed == bundle["record_hash"], recomputed)

    raw = bundle.get("receipt")
    if not raw:
        check("on-chain receipt present", False, "run has not been anchored")
        report["verified"] = False
        return report

    receipt = Receipt(**raw)
    backend = get_chain(receipt.backend)
    onchain = backend.fetch(receipt, cfg)
    check("record found on chain", onchain is not None, receipt.tx_hash)
    if onchain:
        check("on-chain hash equals recomputed hash", onchain["record_hash"] == recomputed, onchain["record_hash"])

    if receipt.backend == "local":
        from hhg3.chain.local import LocalChain

        valid, detail = LocalChain(Path(receipt.extra.get("chain_file", cfg.local_chain_path))).validate()
        check("local chain integrity", valid, detail)

    report["verified"] = all(c["pass"] for c in report["checks"])
    return report
