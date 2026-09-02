"""A minimal local proof-of-work chain.

Not a substitute for a public chain - it is the offline demo path and the thing
the tests run against. Blocks are hash-linked, so editing an earlier block
invalidates every block after it, which `validate()` reports.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from hhg3.config import Config
from hhg3.hashing import canonical_json, sha256_hex
from hhg3.logging_utils import info, ok
from hhg3.types import Receipt

GENESIS_PREV = "0" * 64
DIFFICULTY = 4  # leading hex zeros


def _block_hash(block: dict[str, Any]) -> str:
    body = {k: v for k, v in block.items() if k != "hash"}
    return sha256_hex(canonical_json(body))


def _mine(block: dict[str, Any], difficulty: int) -> dict[str, Any]:
    prefix = "0" * difficulty
    nonce = 0
    while True:
        block["nonce"] = nonce
        digest = _block_hash(block)
        if digest.startswith(prefix):
            block["hash"] = digest
            return block
        nonce += 1


class LocalChain:
    """Chain file resolution, most specific first: an explicit constructor path,
    then the path recorded in the receipt being verified, then the config."""

    name = "local"

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else None

    def _resolve(self, cfg: Config | None = None) -> Path:
        return self._path or (cfg or Config()).local_chain_path

    @property
    def path(self) -> Path:
        return self._resolve()

    # --- storage -------------------------------------------------------
    def load(self, path: Path | None = None) -> list[dict[str, Any]]:
        target = Path(path) if path else self._resolve()
        if not target.exists():
            return []
        return json.loads(target.read_text(encoding="utf-8"))

    def _save(self, blocks: list[dict[str, Any]], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(blocks, indent=2), encoding="utf-8")

    # --- AnchorBackend -------------------------------------------------
    def available(self) -> bool:
        return True

    def anchor(self, record_hash: str, metadata: dict[str, Any], cfg: Config) -> Receipt:
        path = self._resolve(cfg)
        blocks = self.load(path)
        prev = blocks[-1]["hash"] if blocks else GENESIS_PREV
        block = _mine(
            {
                "index": len(blocks),
                "timestamp": int(time.time()),
                "prev_hash": prev,
                "record_hash": record_hash,
                "metadata": metadata,
                "nonce": 0,
            },
            DIFFICULTY,
        )
        blocks.append(block)
        self._save(blocks, path)
        ok("anchored in local block #%d (%s)" % (block["index"], block["hash"][:16]))
        return Receipt(
            backend=self.name,
            network="local-pow-d%d" % DIFFICULTY,
            record_hash=record_hash,
            tx_hash=block["hash"],
            block_number=block["index"],
            timestamp=block["timestamp"],
            extra={"chain_file": str(path)},
        )

    def fetch(self, receipt: Receipt, cfg: Config) -> dict[str, Any] | None:
        path = Path(receipt.extra.get("chain_file") or self._resolve(cfg))
        for block in self.load(path):
            if block["hash"] == receipt.tx_hash:
                return {
                    "record_hash": block["record_hash"],
                    "metadata": block["metadata"],
                    "block": block,
                }
        return None

    # --- integrity -----------------------------------------------------
    def validate(self, path: Path | None = None) -> tuple[bool, str]:
        prev = GENESIS_PREV
        for i, block in enumerate(self.load(path)):
            if block["index"] != i:
                return False, "block %d has index %s" % (i, block["index"])
            if block["prev_hash"] != prev:
                return False, "block %d prev_hash does not link to block %d" % (i, i - 1)
            if _block_hash(block) != block["hash"]:
                return False, "block %d hash does not match its contents (tampered)" % i
            if not block["hash"].startswith("0" * DIFFICULTY):
                return False, "block %d fails proof-of-work" % i
            prev = block["hash"]
        info("local chain validated")
        return True, "ok"
