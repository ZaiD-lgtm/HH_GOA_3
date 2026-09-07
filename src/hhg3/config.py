"""Runtime configuration, assembled from environment + CLI flags."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional convenience
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass

ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _rpc_url() -> str:
    """RPC_URL, tolerating a bare host:port.

    web3's HTTPProvider hands the URL straight to requests, which rejects
    "127.0.0.1:8545" with InvalidSchema. is_connected() swallows that and
    returns False, so a missing scheme surfaces as "cannot reach RPC" - which
    reads like a dead node rather than the typo it is.
    """
    url = _env("RPC_URL")
    if url and "://" not in url:
        url = "http://" + url
    return url


@dataclass
class Config:
    # --- face ---
    detector: str = "auto"        # auto | insightface | yunet
    embedder: str = "auto"        # auto | arcface | insightface | sface | fallback
    # Project default. Set to None to fall back to the active embedder's own
    # default_threshold instead (SFace 0.363, ArcFace 0.45).
    match_threshold: float | None = 0.30
    # Whole-image path (no face detected): perceptual-hash agreement, 0..1.
    # Not comparable to match_threshold - different scale, different meaning.
    image_match_threshold: float = 0.85

    # --- search ---
    search_provider: str = "auto"  # auto | serpapi | gcv | yandex | mock
    image_host: str = "catbox"     # catbox | imgbb | none  (public URL for the probe crop)
    max_candidates: int = 25
    # What gets sent to the search provider.
    #   auto   - the padded face box when a face is detected, else the whole image
    #   face   - always the padded face box (fails when no face is found)
    #   source - always the whole image
    search_image: str = "auto"
    # Padding around the face box, as a fraction of its longest side. The crop is
    # clamped to the image, so padding is always real picture, never black bars.
    search_pad: float = 0.6
    probe_pad: float = 0.25  # tighter crop kept as evidence, not used for search
    require_face: bool = False  # True => abort rather than fall back to image search
    social_only: bool = True

    serpapi_key: str = field(default_factory=lambda: _env("SERPAPI_KEY"))
    gcv_key: str = field(default_factory=lambda: _env("GOOGLE_VISION_API_KEY"))
    imgbb_key: str = field(default_factory=lambda: _env("IMGBB_KEY"))

    # --- chain ---
    chain: str = "local"           # local | evm
    rpc_url: str = field(default_factory=_rpc_url)
    private_key: str = field(default_factory=lambda: _env("PRIVATE_KEY"))
    contract_address: str = field(default_factory=lambda: _env("CONTRACT_ADDRESS"))
    evm_mode: str = field(default_factory=lambda: _env("EVM_MODE", "calldata"))  # calldata | contract
    chain_name: str = field(default_factory=lambda: _env("CHAIN_NAME", "sepolia"))
    explorer_base: str = field(
        default_factory=lambda: _env("EXPLORER_BASE", "https://sepolia.etherscan.io/tx/")
    )

    # --- io ---
    runs_dir: Path = ROOT / "runs"
    local_chain_path: Path = ROOT / "runs" / "_localchain.json"
    http_timeout: int = 30
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )

    def with_overrides(self, **kw) -> "Config":
        for key, value in kw.items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)
        return self
