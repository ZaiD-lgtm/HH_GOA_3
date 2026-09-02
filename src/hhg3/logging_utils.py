"""Tiny console logger. Stages announce themselves so the screen recording reads well."""

from __future__ import annotations

import sys
import time

_START = time.time()
_VERBOSE = True


def set_verbose(flag: bool) -> None:
    global _VERBOSE
    _VERBOSE = flag


def _emit(icon: str, msg: str) -> None:
    if _VERBOSE:
        print(f"[{time.time() - _START:6.1f}s] {icon} {msg}", file=sys.stderr, flush=True)


def stage(msg: str) -> None:
    _emit("==>", msg)


def info(msg: str) -> None:
    _emit("  -", msg)


def warn(msg: str) -> None:
    _emit("  !", msg)


def ok(msg: str) -> None:
    _emit("  +", msg)
