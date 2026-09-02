import json

import pytest

from hhg3.chain.local import LocalChain
from hhg3.config import Config
from hhg3.types import Receipt


@pytest.fixture
def chain(tmp_path):
    return LocalChain(tmp_path / "chain.json")


def test_anchor_then_fetch_roundtrip(chain):
    cfg = Config()
    receipt = chain.anchor("ab" * 32, {"u": "https://x.com/a/status/1"}, cfg)
    stored = chain.fetch(receipt, cfg)
    assert stored is not None
    assert stored["record_hash"] == "ab" * 32
    assert stored["metadata"]["u"] == "https://x.com/a/status/1"


def test_blocks_link_and_validate(chain):
    cfg = Config()
    for i in range(3):
        chain.anchor("%064x" % i, {"i": i}, cfg)
    valid, detail = chain.validate()
    assert valid, detail


def test_tampering_with_a_block_is_detected(chain):
    cfg = Config()
    chain.anchor("11" * 32, {"u": "original"}, cfg)
    chain.anchor("22" * 32, {"u": "second"}, cfg)

    blocks = json.loads(chain.path.read_text())
    blocks[0]["record_hash"] = "99" * 32
    chain.path.write_text(json.dumps(blocks))

    valid, detail = chain.validate()
    assert not valid
    assert "block 0" in detail


def test_fetch_returns_none_for_unknown_tx(chain):
    receipt = Receipt(backend="local", network="local", record_hash="00" * 32, tx_hash="deadbeef")
    assert chain.fetch(receipt, Config()) is None
