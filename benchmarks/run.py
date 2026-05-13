"""Benchmark ThumbHash encoder backends across sizes.

Backends compared:
    - pure   : reference Python (no deps)
    - numpy  : matmul DCT (BLAS-accelerated)

Spec sizes (<=100) run both. Large sizes skip pure (would take minutes).

Usage (from repo root):
    python benchmarks/run.py
    python benchmarks/run.py --repeat 7 --inner 5
    python benchmarks/run.py --skip-large
"""
import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, List, Tuple

# Allow running this script directly without an editable install of thash.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from thash import _numpy, _pure


# (label, width, height, with_alpha, allow_pure)
CASES: List[Tuple[str, int, int, bool, bool]] = [
    ("tiny-square",     10,  10, False, True),
    ("small-square",    32,  32, False, True),
    ("medium-square",   64,  64, False, True),
    ("max-square",     100, 100, False, True),
    ("landscape",      100,  56, False, True),
    ("portrait",        56, 100, False, True),
    ("max-square+a",   100, 100, True,  True),
    ("landscape+a",    100,  56, True,  True),
    # Beyond-spec sizes: directly stress the DCT. Pure Python skipped.
    ("HD-720p",       1280, 720, False, False),
    ("FHD-1080p",     1920,1080, False, False),
    ("QHD-1440p",     2560,1440, False, False),
    ("UHD-4K",        3840,2160, False, False),
]


def make_image_uint8(w: int, h: int, with_alpha: bool, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(h, w, 4), dtype=np.uint8)
    if not with_alpha:
        arr[..., 3] = 255
    return np.ascontiguousarray(arr)


def make_flat_list(arr: np.ndarray) -> List[int]:
    return arr.reshape(-1).tolist()


def time_call(fn: Callable, args: tuple, repeat: int, inner: int) -> List[float]:
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn(*args)
        samples.append((time.perf_counter() - t0) / inner)
    return samples


def fmt_seconds(s: float | None) -> str:
    if s is None:
        return "       ----"
    if s >= 1.0:
        return f"{s:8.3f} s "
    if s >= 1e-3:
        return f"{s * 1e3:8.3f} ms"
    return f"{s * 1e6:8.3f} us"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--inner", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--skip-large", action="store_true",
                        help="Only run spec sizes (<=100x100)")
    args = parser.parse_args()

    backends = [("pure", _pure)]
    if _numpy is not None:
        backends.append(("numpy", _numpy))

    print(f"Backends: {[b[0] for b in backends]}")
    print(f"NumPy {np.__version__}")
    print(f"Each cell: median of {args.repeat} batches x {args.inner} call(s)")
    print()

    header = (
        f"{'case':<14} {'size':>10} {'alpha':>5}  "
        + "  ".join(f"{name:>11}" for name, _ in backends)
        + "   match"
    )
    print(header)
    print("-" * len(header))

    speedups_vs_pure = []
    for label, w, h, with_alpha, allow_pure in CASES:
        if args.skip_large and (w > 100 or h > 100):
            continue

        rgba_arr = make_image_uint8(
            w, h, with_alpha, seed=(w * 31 + h * 7 + int(with_alpha)) & 0xFFFF
        )
        rgba_list = make_flat_list(rgba_arr)

        # Correctness: compare every backend to numpy as ground truth (or pure if numpy missing).
        gt_mod = _numpy if _numpy is not None else _pure
        gt = gt_mod._encode(w, h, rgba_arr if gt_mod is not _pure else rgba_list)
        match_flags = {}
        for name, mod in backends:
            if not allow_pure and mod is _pure:
                match_flags[name] = None
                continue
            out = mod._encode(w, h, rgba_list if mod is _pure else rgba_arr)
            match_flags[name] = out == gt

        # Warmup (per-case)
        for _ in range(args.warmup):
            for name, mod in backends:
                if not allow_pure and mod is _pure:
                    continue
                mod._encode(w, h, rgba_list if mod is _pure else rgba_arr)

        # Time each backend
        medians = {}
        for name, mod in backends:
            if not allow_pure and mod is _pure:
                medians[name] = None
                continue
            samples = time_call(
                mod._encode,
                (w, h, rgba_list if mod is _pure else rgba_arr),
                args.repeat,
                args.inner,
            )
            medians[name] = statistics.median(samples)

        if medians.get("pure") and medians.get("numpy"):
            speedups_vs_pure.append(medians["pure"] / medians["numpy"])

        match_str = "".join(
            "." if match_flags[n] is None else ("Y" if match_flags[n] else "N")
            for n, _ in backends
        )
        cells = "  ".join(fmt_seconds(medians[n]) for n, _ in backends)
        print(f"{label:<14} {w:>5}x{h:<4} {with_alpha!s:>5}  {cells}   {match_str}")

    print()
    print("(match column: one char per backend, order matches header; Y=equal, N=differ, .=skipped)")
    if speedups_vs_pure:
        geo = statistics.geometric_mean(speedups_vs_pure)
        med = statistics.median(speedups_vs_pure)
        print(f"NumPy vs pure (spec sizes only): geo-mean {geo:.1f}x, median {med:.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
