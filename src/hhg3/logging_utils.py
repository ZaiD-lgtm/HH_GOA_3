"""Tiny console logger. Stages announce themselves so the screen recording reads well."""

from __future__ import annotations

import sys
import time

from hhg3.render import enable_vt, paint, supports_color

_START = time.time()
_VERBOSE = True
_COLOR: bool | None = None


def set_verbose(flag: bool) -> None:
    global _VERBOSE
    _VERBOSE = flag


def _color() -> bool:
    """Resolved once, after cli.main has reconfigured the streams."""
    global _COLOR
    if _COLOR is None:
        enable_vt()
        _COLOR = supports_color(sys.stderr)
    return _COLOR


def _emit(icon: str, msg: str, code: str = "") -> None:
    if not _VERBOSE:
        return
    stamp = paint("2", "[%6.1fs]" % (time.time() - _START), _color())
    body = "%s %s" % (icon, msg)
    print("%s %s" % (stamp, paint(code, body, _color()) if code else body),
          file=sys.stderr, flush=True)


def stage(msg: str) -> None:
    """A pipeline step. Set off by a blank line so the phases are countable on video."""
    if _VERBOSE:
        print("", file=sys.stderr, flush=True)
    _emit("==>", msg, "1;36")


def info(msg: str) -> None:
    _emit("  -", msg)


def warn(msg: str) -> None:
    _emit("  !", msg, "33")


def ok(msg: str) -> None:
    _emit("  +", msg, "32")
