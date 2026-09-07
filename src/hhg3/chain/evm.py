"""EVM anchoring.

Two modes:
  calldata  - a 0-value self-transaction whose input data carries the record
              hash. No contract deploy, works on any chain, cheapest path.
  contract  - AnchorRegistry.anchor(bytes32,string); gives a queryable mapping
              and an event log. Deploy with scripts/deploy_contract.py.
"""

from __future__ import annotations

import json
import time
from typing import Any

from hhg3.config import Config
from hhg3.hashing import canonical_json
from hhg3.logging_utils import info, ok
from hhg3.types import Receipt

MAGIC = b"HHG3"
REGISTRY_ABI = json.loads(
    """[
  {"inputs":[{"internalType":"bytes32","name":"recordHash","type":"bytes32"},
             {"internalType":"string","name":"uri","type":"string"}],
   "name":"anchor","outputs":[],"stateMutability":"nonpayable","type":"function"},
  {"inputs":[{"internalType":"bytes32","name":"","type":"bytes32"}],
   "name":"records","outputs":[{"internalType":"address","name":"submitter","type":"address"},
                               {"internalType":"uint64","name":"timestamp","type":"uint64"},
                               {"internalType":"string","name":"uri","type":"string"}],
   "stateMutability":"view","type":"function"},
  {"anonymous":false,
   "inputs":[{"indexed":true,"internalType":"bytes32","name":"recordHash","type":"bytes32"},
             {"indexed":true,"internalType":"address","name":"submitter","type":"address"},
             {"indexed":false,"internalType":"uint64","name":"timestamp","type":"uint64"},
             {"indexed":false,"internalType":"string","name":"uri","type":"string"}],
   "name":"Anchored","type":"event"}
]"""
)


class EvmChain:
    name = "evm"

    def _web3(self, cfg: Config):
        from web3 import Web3

        if not cfg.rpc_url:
            raise RuntimeError("RPC_URL not set")
        w3 = Web3(Web3.HTTPProvider(cfg.rpc_url, request_kwargs={"timeout": cfg.http_timeout}))
        if not w3.is_connected():
            raise RuntimeError("cannot reach RPC at " + cfg.rpc_url)
        return w3

    def available(self) -> bool:
        cfg = Config()
        if not (cfg.rpc_url and cfg.private_key):
            return False
        try:
            self._web3(cfg)
            return True
        except Exception:
            return False

    #  write
    def anchor(self, record_hash: str, metadata: dict[str, Any], cfg: Config) -> Receipt:
        w3 = self._web3(cfg)
        acct = w3.eth.account.from_key(cfg.private_key)
        info("anchoring from %s on %s" % (acct.address, cfg.chain_name))

        if cfg.evm_mode == "contract":
            tx = self._contract_tx(w3, acct, record_hash, metadata, cfg)
        else:
            tx = self._calldata_tx(w3, acct, record_hash, metadata, cfg)

        signed = acct.sign_transaction(tx)
        raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
        tx_hash = w3.eth.send_raw_transaction(raw)
        info("tx submitted: " + tx_hash.hex())
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
        if receipt.status != 1:
            raise RuntimeError("transaction reverted: " + tx_hash.hex())
        block = w3.eth.get_block(receipt.blockNumber)
        ok("mined in block %d" % receipt.blockNumber)

        tx_hex = tx_hash.hex()
        if not tx_hex.startswith("0x"):
            tx_hex = "0x" + tx_hex
        return Receipt(
            backend=self.name,
            network=cfg.chain_name,
            record_hash=record_hash,
            tx_hash=tx_hex,
            block_number=int(receipt.blockNumber),
            timestamp=int(block.timestamp),
            explorer_url=(cfg.explorer_base + tx_hex) if cfg.explorer_base else None,
            extra={"mode": cfg.evm_mode, "from": acct.address, "contract": cfg.contract_address or None},
        )

    def _base_tx(self, w3, acct, cfg) -> dict[str, Any]:
        return {
            "chainId": w3.eth.chain_id,
            "nonce": w3.eth.get_transaction_count(acct.address),
            "from": acct.address,
        }

    def _calldata_tx(self, w3, acct, record_hash, metadata, cfg) -> dict[str, Any]:
        payload = MAGIC + bytes.fromhex(record_hash) + canonical_json(metadata)
        tx = self._base_tx(w3, acct, cfg)
        tx.update({"to": acct.address, "value": 0, "data": payload})
        tx["gas"] = int(w3.eth.estimate_gas(tx) * 1.2)
        tx.update(self._fees(w3))
        return tx

    def _contract_tx(self, w3, acct, record_hash, metadata, cfg) -> dict[str, Any]:
        from web3 import Web3

        if not cfg.contract_address:
            raise RuntimeError("CONTRACT_ADDRESS not set (run scripts/deploy_contract.py)")
        contract = w3.eth.contract(
            address=Web3.to_checksum_address(cfg.contract_address), abi=REGISTRY_ABI
        )
        uri = canonical_json(metadata).decode("utf-8")
        tx = contract.functions.anchor(bytes.fromhex(record_hash), uri).build_transaction(
            {**self._base_tx(w3, acct, cfg), **self._fees(w3)}
        )
        return tx

    def _fees(self, w3) -> dict[str, Any]:
        try:
            base = w3.eth.get_block("latest").baseFeePerGas
            tip = w3.eth.max_priority_fee
            return {"maxFeePerGas": base * 2 + tip, "maxPriorityFeePerGas": tip}
        except Exception:
            return {"gasPrice": w3.eth.gas_price}

    # read
    def fetch(self, receipt: Receipt, cfg: Config) -> dict[str, Any] | None:
        w3 = self._web3(cfg)
        tx = w3.eth.get_transaction(receipt.tx_hash)
        chain_receipt = w3.eth.get_transaction_receipt(receipt.tx_hash)
        if chain_receipt.status != 1:
            return None

        if receipt.extra.get("mode") == "contract":
            from web3 import Web3

            contract = w3.eth.contract(
                address=Web3.to_checksum_address(cfg.contract_address or tx["to"]), abi=REGISTRY_ABI
            )
            submitter, ts, uri = contract.functions.records(bytes.fromhex(receipt.record_hash)).call()
            if int(submitter, 16) == 0:
                return None
            return {
                "record_hash": receipt.record_hash,
                "metadata": json.loads(uri) if uri else {},
                "submitter": submitter,
                "timestamp": int(ts),
            }

        data = bytes(tx["input"])
        if not data.startswith(MAGIC):
            return None
        onchain_hash = data[4:36].hex()
        meta_raw = data[36:]
        return {
            "record_hash": onchain_hash,
            "metadata": json.loads(meta_raw) if meta_raw else {},
            "submitter": tx["from"],
            "timestamp": int(w3.eth.get_block(chain_receipt.blockNumber).timestamp),
        }
