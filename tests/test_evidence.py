import pytest

from hhg3 import evidence, pipeline
from hhg3.chain.local import LocalChain
from hhg3.config import Config
from hhg3.types import Candidate, FaceProbe, Match


@pytest.fixture
def bundle():
    probe = FaceProbe(
        source_path="samples/probe.jpg",
        source_sha256="aa" * 32,
        bbox=(10, 20, 100, 100),
        crop_path="runs/x/probe_crop.jpg",
        crop_sha256="bb" * 32,
        embedding=[0.1, 0.2, 0.3],
        detector="opencv-haar",
        embedder="fallback-pixel-64",
    )
    match = Match(
        candidate=Candidate(
            provider="serpapi-google-lens",
            page_url="https://x.com/someone/status/123",
            image_url="https://pbs.twimg.com/media/abc.jpg",
            domain="x.com",
            platform="x",
        ),
        similarity=0.72,
        threshold=0.35,
        candidate_image_path="runs/x/candidates/cand_00.jpg",
        candidate_image_sha256="cc" * 32,
        matched_bbox=(5, 5, 90, 90),
    )
    return evidence.build_bundle("run-1", probe, match, "serpapi-google-lens", [], [])


def test_embedding_never_enters_the_record(bundle):
    assert "embedding" not in bundle["record"]["probe"]


def test_recomputed_hash_matches(bundle):
    assert evidence.recompute_hash(bundle) == bundle["record_hash"]


def test_edited_record_breaks_the_hash(bundle):
    bundle["record"]["match"]["page_url"] = "https://x.com/someone/status/999"
    assert evidence.recompute_hash(bundle) != bundle["record_hash"]


def test_verify_roundtrip_against_local_chain(tmp_path, bundle, monkeypatch):
    cfg = Config()
    cfg.local_chain_path = tmp_path / "chain.json"
    run_dir = tmp_path / "run-1"
    evidence.save_bundle(run_dir, bundle)

    chain = LocalChain(cfg.local_chain_path)
    receipt = chain.anchor(bundle["record_hash"], {"r": "run-1"}, cfg)
    evidence.attach_receipt(run_dir, bundle, receipt)

    report = pipeline.verify(run_dir, cfg)
    assert report["verified"], report["checks"]


def test_verify_fails_after_tampering(tmp_path, bundle):
    cfg = Config()
    cfg.local_chain_path = tmp_path / "chain.json"
    run_dir = tmp_path / "run-2"

    chain = LocalChain(cfg.local_chain_path)
    receipt = chain.anchor(bundle["record_hash"], {"r": "run-2"}, cfg)
    bundle["record"]["match"]["page_url"] = "https://x.com/impostor/status/1"
    evidence.attach_receipt(run_dir, bundle, receipt)

    report = pipeline.verify(run_dir, cfg)
    assert not report["verified"]
