"""Command-line interface: encode image files to a ThumbHash, or render a hash back to a preview."""

from __future__ import annotations

import argparse
import base64
import binascii
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


_HEX_ALPHABET = set("0123456789abcdefABCDEF")


def _decode_hash_string(arg: str) -> bytes | None:
    """Try to interpret `arg` as a ThumbHash string (hex / base64 std / base64 urlsafe).

    Hex is tried first because hex characters are a subset of the base64 alphabet,
    so an unguarded base64 attempt would silently mis-decode a hex string.
    """
    s = arg.strip()
    if not s:
        return None
    if len(s) % 2 == 0 and s and all(c in _HEX_ALPHABET for c in s):
        try:
            data = bytes.fromhex(s)
        except ValueError:
            data = b""
        if len(data) >= 5:
            return data
    # base64 may be missing '=' padding (common for URL-safe form)
    padded = s + "=" * (-len(s) % 4)
    for altchars in (b"+/", b"-_"):
        try:
            data = base64.b64decode(padded, altchars=altchars, validate=True)
        except (binascii.Error, ValueError):
            continue
        if len(data) >= 5:
            return data
    return None


def _classify_input(arg: str) -> tuple[str, object]:
    """Return ('image', Path) or ('hash', bytes). Raises ValueError if neither."""
    p = Path(arg)
    if p.exists():
        if not p.is_file():
            raise ValueError(f"{arg!r}: path exists but is not a file")
        return "image", p
    data = _decode_hash_string(arg)
    if data is not None:
        return "hash", data
    raise ValueError(f"{arg!r}: not an existing file and not a valid base64/hex ThumbHash")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thash",
        description="Encode images to ThumbHash, or render a hash back to a placeholder image.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Image file paths or ThumbHash strings (base64 / hex).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Render each input back to an RGBA preview and save it. "
            "With one input PATH is treated as a file; with multiple inputs PATH is treated "
            "as a directory and filenames are derived from the input."
        ),
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        metavar="N",
        help="Longer edge of the rendered preview when --output is set (default 256).",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=("base64", "hex", "bytes"),
        default="base64",
        help="Output encoding for the printed hash (default base64).",
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


def _resolve_output_path(
    raw: str,
    kind: str,
    value: object,
    output_arg: Path,
    is_dir_mode: bool,
) -> Path:
    if not is_dir_mode:
        return output_arg
    if kind == "image":
        assert isinstance(value, Path)
        return output_arg / f"{value.stem}.thumb.png"
    assert isinstance(value, (bytes, bytearray))
    return output_arg / f"thumb_{bytes(value)[:3].hex()}.png"


def _render_and_save(hash_bytes: list[int], save_path: Path, *, size: int) -> None:
    from . import thumb_hash_to_rgba

    if thumb_hash_to_rgba is None:
        raise ImportError("rendering a preview needs NumPy and Pillow — install with `pip install thash[all]`")
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("saving a PNG preview needs Pillow — install with `pip install thash[all]`") from exc

    # Reconstruct directly at the requested resolution — ThumbHash only encodes ~5x5 / 7x7
    # frequency coefficients, so IDCT at any target size produces a smooth low-frequency
    # image. This avoids both an upsampling pass and an extra aspect-ratio rounding step.
    _, _, rgba = thumb_hash_to_rgba(hash_bytes, base_size=size)
    Image.fromarray(rgba, mode="RGBA").save(save_path)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    from . import encode

    n_inputs = len(args.inputs)
    is_dir_mode = False
    if args.output is not None:
        if n_inputs > 1:
            args.output.mkdir(parents=True, exist_ok=True)
            is_dir_mode = True
        else:
            parent = args.output.parent
            if str(parent) and not parent.exists():
                parent.mkdir(parents=True, exist_ok=True)

    show_path = n_inputs > 1
    exit_code = 0

    for raw in args.inputs:
        try:
            kind, value = _classify_input(raw)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            exit_code = 1
            continue

        try:
            if kind == "image":
                if encode is None:
                    raise ImportError(
                        "encoding image files needs NumPy and Pillow — install with `pip install thash[all]`"
                    )
                hash_bytes = encode(value, target_size=args.target_size, backend=args.backend)
            else:
                hash_bytes = list(value)  # type: ignore[arg-type]
        except (OSError, ValueError, ImportError, TypeError) as exc:
            print(f"error: {raw}: {exc}", file=sys.stderr)
            exit_code = 1
            continue

        line = _format_hash(hash_bytes, args.format)
        if show_path:
            print(f"{raw}\t{line}")
        else:
            print(line)

        if args.output is not None:
            try:
                save_path = _resolve_output_path(raw, kind, value, args.output, is_dir_mode)
                _render_and_save(hash_bytes, save_path, size=args.size)
            except (ImportError, OSError, ValueError) as exc:
                print(f"error: rendering {raw}: {exc}", file=sys.stderr)
                exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
