"""Command line entry point.

    hhg3 doctor                     which backends are actually available
    hhg3 run --image face.jpg       full pipeline, writes runs/<id>/
    hhg3 verify --run runs/<id>     re-verify local evidence against the chain
    hhg3 chain show                 dump the local chain
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hhg3 import __version__, logging_utils
from hhg3.config import Config


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--quiet", action="store_true", help="suppress stage logging")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hhg3", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the full pipeline")
    run.add_argument("--image", required=True, type=Path, help="input face scan")
    run.add_argument("--provider", default=None, help="auto|serpapi|gcv|yandex|mock")
    run.add_argument("--chain", default=None, help="local|evm|auto")
    run.add_argument("--detector", default=None, help="auto|insightface|yunet")
    run.add_argument("--embedder", default=None, help="auto|arcface|insightface|sface|fallback")
    run.add_argument("--threshold", type=float, default=None, help="cosine match threshold")
    run.add_argument("--max-candidates", type=int, default=None)
    run.add_argument("--any-domain", action="store_true", help="do not restrict to social platforms")
    run.add_argument(
        "--search-image", default=None, choices=["auto", "face", "source"],
        help="what to send the search provider: auto (padded face box when a face "
             "is found, else the whole image), face, or source",
    )
    run.add_argument("--search-pad", type=float, default=None,
                     help="padding around the face box as a fraction of its longest side")
    run.add_argument("--require-face", action="store_true",
                     help="abort when no face is detected instead of searching the whole image")
    run.add_argument("--image-threshold", type=float, default=None,
                     help="perceptual-hash agreement needed on the no-face path")
    run.add_argument("--search-crop", action="store_true",
                     help="alias for --search-image face")
    run.add_argument("--allow-mock", action="store_true", help="permit the fixture provider")
    run.add_argument("--json", action="store_true", help="print the result as JSON")
    _common(run)

    ver = sub.add_parser("verify", help="re-verify a run against the chain")
    ver.add_argument("--run", required=True, type=Path, help="path to runs/<id>")
    ver.add_argument("--chain", default=None)
    ver.add_argument("--json", action="store_true")
    _common(ver)

    doc = sub.add_parser("doctor", help="report available backends")
    _common(doc)

    chain = sub.add_parser("chain", help="inspect the local chain")
    chain.add_argument("action", choices=["show", "validate"])
    _common(chain)
    return parser


def _config_from(args) -> Config:
    cfg = Config()
    cfg.with_overrides(
        search_provider=getattr(args, "provider", None),
        chain=getattr(args, "chain", None),
        detector=getattr(args, "detector", None),
        embedder=getattr(args, "embedder", None),
        match_threshold=getattr(args, "threshold", None),
        max_candidates=getattr(args, "max_candidates", None),
        search_image=getattr(args, "search_image", None),
        search_pad=getattr(args, "search_pad", None),
        image_match_threshold=getattr(args, "image_threshold", None),
    )
    if getattr(args, "any_domain", False):
        cfg.social_only = False
    if getattr(args, "require_face", False):
        cfg.require_face = True
    if getattr(args, "search_crop", False):
        cfg.search_image = "face"
    return cfg


def cmd_run(args) -> int:
    from hhg3 import pipeline

    cfg = _config_from(args)
    try:
        result = pipeline.run(args.image, cfg, allow_mock=args.allow_mock)
    except RuntimeError as exc:  # PipelineError, and missing-backend errors
        print("pipeline failed: %s" % exc, file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        record = result["bundle"]["record"]
        receipt = result["receipt"]
        print("")
        print("run dir      : " + result["run_dir"])
        print("matched post : %s (%s)" % (record["match"]["page_url"], record["match"]["platform"]))
        print("similarity   : %.4f (threshold %.2f)" % (record["match"]["similarity"], record["match"]["threshold"]))
        print("record hash  : " + result["bundle"]["record_hash"])
        print("chain        : %s / %s" % (receipt["backend"], receipt["network"]))
        print("tx           : " + receipt["tx_hash"])
        if receipt.get("explorer_url"):
            print("explorer     : " + receipt["explorer_url"])
        print("")
        print("re-verify with:  hhg3 verify --run " + result["run_dir"])
    return 0


def cmd_verify(args) -> int:
    from hhg3 import pipeline

    cfg = _config_from(args)
    try:
        report = pipeline.verify(args.run, cfg)
    except (FileNotFoundError, ValueError) as exc:
        print("cannot verify: %s" % exc, file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("")
        for row in report["checks"]:
            print("  [%s] %s  %s" % ("PASS" if row["pass"] else "FAIL", row["check"], row["detail"]))
        print("")
        print("VERIFIED" if report["verified"] else "NOT VERIFIED")
    return 0 if report["verified"] else 1


def cmd_doctor(args) -> int:
    from hhg3.chain.registry import get_chain
    from hhg3.search.registry import list_providers

    print("hhg3 %s  (python %s)" % (__version__, sys.version.split()[0]))

    print("\nface backends:")
    for kind, getter in (("detector", "hhg3.face.detect.get_detector"), ("embedder", "hhg3.face.embed.get_embedder")):
        module, _, func = getter.rpartition(".")
        import importlib

        try:
            backend = getattr(importlib.import_module(module), func)("auto")
            print("  %-9s %s" % (kind, backend.name))
        except Exception as exc:
            print("  %-9s UNAVAILABLE (%s)" % (kind, exc))

    print("\nsearch providers:")
    for key, provider in list_providers().items():
        flag = "ready" if provider.available() else "not configured"
        genuine = "genuine" if provider.genuine else "FIXTURE"
        print("  %-8s %-24s %-15s %s" % (key, provider.name, flag, genuine))

    print("\nchain backends:")
    for key in ("local", "evm"):
        backend = get_chain(key)
        print("  %-8s %s" % (key, "ready" if backend.available() else "not configured"))
    return 0


def cmd_chain(args) -> int:
    from hhg3.chain.local import LocalChain

    chain = LocalChain()
    if args.action == "show":
        print(json.dumps(chain.load(), indent=2))
        return 0
    valid, detail = chain.validate()
    print("local chain: %s (%s)" % ("VALID" if valid else "INVALID", detail))
    return 0 if valid else 1


_HANDLERS = {"run": cmd_run, "verify": cmd_verify, "doctor": cmd_doctor, "chain": cmd_chain}


def main(argv: list[str] | None = None) -> int:
    # Post titles carry emoji; a cp1252 Windows console would raise on print().
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    args = build_parser().parse_args(argv)
    logging_utils.set_verbose(not getattr(args, "quiet", False))
    return _HANDLERS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
