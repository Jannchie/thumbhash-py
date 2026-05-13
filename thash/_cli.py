"""Command-line interface: encode image files to a ThumbHash."""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

from .__about__ import __version__


def _format_hash(hash_bytes: list[int], fmt: str) -> str:
    if fmt == "base64":
        return base64.b64encode(bytes(hash_bytes)).decode("ascii")
    if fmt == "hex":
        return bytes(hash_bytes).hex()
    if fmt == "bytes":
        return "[" + ", ".join(str(b) for b in hash_bytes) + "]"
    raise ValueError(f"unknown format: {fmt}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thash",
        description="Encode one or more images to a ThumbHash placeholder hash.",
    )
    parser.add_argument(
        "images",
        nargs="+",
        type=Path,
        help="Image file path(s) to encode.",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=("base64", "hex", "bytes"),
        default="base64",
        help="Output encoding for the hash (default: base64).",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=100,
        metavar="N",
        help="Cap on the longer image dimension before encoding (1..100, default 100).",
    )
    parser.add_argument(
        "--backend",
        choices=("numpy", "pure"),
        default=None,
        help="Force an encoder backend (default: numpy when available).",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"thash {__version__}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Import lazily so `--help` / `--version` work even when optional deps are missing.
    from . import encode

    if encode is None:
        print(
            "error: the CLI requires NumPy. Install with `pip install thash[all]`.",
            file=sys.stderr,
        )
        return 1

    show_path = len(args.images) > 1
    exit_code = 0
    for path in args.images:
        try:
            hash_bytes = encode(
                path,
                target_size=args.target_size,
                backend=args.backend,
            )
        except (OSError, ValueError, ImportError) as exc:
            print(f"error: {path}: {exc}", file=sys.stderr)
            exit_code = 1
            continue

        line = _format_hash(hash_bytes, args.format)
        if show_path:
            print(f"{path}\t{line}")
        else:
            print(line)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
