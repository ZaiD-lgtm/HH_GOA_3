"""Compile and deploy contracts/AnchorRegistry.sol.

    python scripts/deploy_contract.py

Needs RPC_URL and PRIVATE_KEY in the environment plus `pip install py-solc-x`.
Prints the address to put in CONTRACT_ADDRESS. Only needed for EVM_MODE=contract;
the default calldata mode deploys nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hhg3.config import Config  # noqa: E402

SOURCE = ROOT / "contracts" / "AnchorRegistry.sol"
SOLC_VERSION = "0.8.24"


def compile_contract() -> tuple[list, str]:
    import solcx

    if SOLC_VERSION not in [str(v) for v in solcx.get_installed_solc_versions()]:
        print("installing solc " + SOLC_VERSION)
        solcx.install_solc(SOLC_VERSION)
    compiled = solcx.compile_source(
        SOURCE.read_text(encoding="utf-8"),
        output_values=["abi", "bin"],
        solc_version=SOLC_VERSION,
    )
    _, artifact = next(iter(compiled.items()))
    return artifact["abi"], artifact["bin"]


def main() -> int:
    from web3 import Web3

    cfg = Config()
    if not (cfg.rpc_url and cfg.private_key):
        print("set RPC_URL and PRIVATE_KEY first", file=sys.stderr)
        return 1

    abi, bytecode = compile_contract()
    w3 = Web3(Web3.HTTPProvider(cfg.rpc_url))
    acct = w3.eth.account.from_key(cfg.private_key)
    print("deploying from %s to %s" % (acct.address, cfg.chain_name))

    contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    tx = contract.constructor().build_transaction(
        {
            "from": acct.address,
            "nonce": w3.eth.get_transaction_count(acct.address),
            "chainId": w3.eth.chain_id,
            "gasPrice": w3.eth.gas_price,
        }
    )
    signed = acct.sign_transaction(tx)
    raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
    tx_hash = w3.eth.send_raw_transaction(raw)
    print("tx: " + tx_hash.hex())
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=600)
    print("\nCONTRACT_ADDRESS=" + receipt.contractAddress)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
