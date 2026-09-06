"""Chain backend selection. local vs local ganache EVM"""

from __future__ import annotations

from hhg3.chain.base import AnchorBackend
from hhg3.chain.evm import EvmChain
from hhg3.chain.local import LocalChain

_BUILDERS = {"local": LocalChain, "evm": EvmChain}


def get_chain(name: str = "local") -> AnchorBackend:
    if name == "auto":
        evm = EvmChain()
        return evm if evm.available() else LocalChain()
    if name not in _BUILDERS:
        raise ValueError("unknown chain backend: " + name)
    return _BUILDERS[name]()
