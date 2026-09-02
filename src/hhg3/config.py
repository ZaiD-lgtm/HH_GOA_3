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


@dataclass
class Config:
    # --- face ---
    detector: str = "auto"        # auto | insightface | opencv
    embedder: str = "auto"        # auto | insightface | fallback
    # None => use the active embedder's default_threshold (models are not comparable)
    match_threshold: float | None = None

    # --- search ---
    search_provider: str = "auto"  # auto | serpapi | bing | yandex | mock
    image_host: str = "catbox"     # catbox | imgbb | none  (public URL for the probe crop)
    max_candidates: int = 25
    social_only: bool = True

    serpapi_key: str = field(default_factory=lambda: _env("SERPAPI_KEY"))
    bing_key: str = field(default_factory=lambda: _env("BING_VISUAL_SEARCH_KEY"))
    bing_endpoint: str = field(
        default_factory=lambda: _env(
            "BING_VISUAL_SEARCH_ENDPOINT",
            "https://api.bing.microsoft.com/v7.0/images/visualsearch",
        )
    )
    imgbb_key: str = field(default_factory=lambda: _env("IMGBB_KEY"))

    # --- chain ---
    chain: str = "local"           # local | evm
    rpc_url: str = field(default_factory=lambda: _env("RPC_URL"))
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
