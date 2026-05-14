<img src="https://img.shields.io/pypi/v/thash"> <img src="https://img.shields.io/github/license/Jannchie/thumbhash-py">

# thash

A modern Python port of the [ThumbHash](https://github.com/evanw/thumbhash) encoder by [Evan Wallace](https://github.com/evanw). ThumbHash represents an image as ~20 bytes — small enough to inline in HTML, large enough to render a recognizable color/aspect placeholder before the real image loads.

This is an independently published fork of [`thumbhash`](https://pypi.org/project/thumbhash/) by [Justin Forlenza](https://github.com/justinforlenza). Notable changes vs. upstream:

- **Alpha-channel crash fixed** (operator-precedence bug in `rgba_to_thumb_hash` — see [upstream issue #1](https://github.com/justinforlenza/thumbhash-py/issues/1)).
- **NumPy-accelerated backend** with cached cosine basis and float32 DCT (~100–140× faster than the reference implementation, byte-identical output).
- **High-level `encode()` API** that accepts paths, bytes, PIL images, NumPy arrays, and OpenCV BGR arrays — pick the input you already have, no boilerplate.
- **Decoder + CLI** for rendering a hash back to a placeholder image (`thumb_hash_to_rgba`, or `thash photo.jpg -o preview.png`).
- **Configurable `target_size`** so you can trade hash quality for encoding speed.

## Installation

```sh
# Pure-Python fallback only (no deps)
pip install thash

# Recommended runtime (NumPy fast path + Pillow decoding)
pip install thash[all]
```

If you use [uv](https://docs.astral.sh/uv/):

```sh
uv add thash --extra all
```

Requires Python ≥ 3.10.

## Quick start

The high-level API takes pretty much any image-shaped thing:

```python
from thash import encode

# From a file path or URL-fetched bytes
hash_bytes = encode("photo.jpg")
hash_bytes = encode(open("photo.jpg", "rb").read())

# From a PIL image (already in memory, no re-decode)
from PIL import Image
hash_bytes = encode(Image.open("photo.jpg"))

# From a NumPy array (H,W,3) or (H,W,4) — assumed RGB/RGBA
import numpy as np
arr = np.asarray(Image.open("photo.jpg"))
hash_bytes = encode(arr)

# From an OpenCV BGR array
import cv2
bgr = cv2.imread("photo.jpg")
hash_bytes = encode(bgr, color_order="BGR")

# Grayscale / float arrays in [0, 1] also work — they're normalized for you
hash_bytes = encode(arr.astype(np.float32) / 255.0)
```

### Decoding the hash back

```python
from thash import (
    thumb_hash_to_rgba,
    thumb_hash_to_average_rgba,
    thumb_hash_to_approximate_aspect_ratio,
)

# Render the hash to a small RGBA preview (flat bytes, length 4*w*h)
w, h, rgba = thumb_hash_to_rgba(hash_bytes, base_size=256)

from PIL import Image
Image.frombytes("RGBA", (w, h), rgba).save("preview.png")

# Want a numpy array instead?
import numpy as np
arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)

# Cheaper queries that don't reconstruct pixels:
r, g, b, a = thumb_hash_to_average_rgba(hash_bytes)            # values in [0, 1]
aspect = thumb_hash_to_approximate_aspect_ratio(hash_bytes)    # w / h
```

`base_size` is the longer edge of the reconstructed image. ThumbHash only carries ~5×5 / 7×7 frequency coefficients, so the IDCT is run directly at the requested resolution rather than upsampled — values up to a few hundred pixels look smooth without any extra resampling. The aspect ratio comes from the encoded `lx / ly` (e.g. 7:4 for a landscape, 5:7 for a portrait); near-non-integer ratios like 1.6 get quantized to 1.75, this is a spec property, not an implementation choice.

### Command-line

Installing the package exposes a `thash` command (equivalent to `python -m thash`):

```sh
# --- Encoding: print a hash for each input ---
thash photo.jpg                        # base64 hash, one per line
thash --format hex photo.jpg
thash --format bytes photo.jpg
thash photo.jpg cover.png hero.webp    # multi-file: "path<TAB>hash" per line
thash --target-size 64 photo.jpg       # trade quality for encoding speed

# --- Rendering: save a placeholder preview PNG ---
thash photo.jpg -o preview.png                     # encode + decode + save
thash photo.jpg -o preview.png --size 128          # cap the longer edge
thash "2dYJLJSBdoiAiHVoSHZzcBf4iA==" -o p.png      # base64 hash → PNG (no source image needed)
thash d9d6092c94817688808875684876737017f888 -o p.png  # hex hash → PNG
thash a.jpg b.jpg "2dYJ...==" -o out/              # multi input → directory, auto-named
```

The CLI uses the high-level `encode()` / `thumb_hash_to_rgba()` APIs. It needs Pillow for decoding images / writing PNG previews; NumPy is optional (only accelerates the encode / decode). Install with `pip install thash[pillow]` for the CLI or `[all]` for the fast path too. Hash inputs are auto-detected: hex strings (even length, hex alphabet) are tried first, then base64 (standard and URL-safe).

## Tuning speed vs. quality

`target_size` controls the longer dimension of the image after thumbnail (spec max is 100). Smaller = faster, lower fidelity:

| `target_size` | DCT time | Visual quality |
|---|---|---|
| 100 (default) | ~125 μs | Reference / spec-compatible |
| 64            | ~85 μs  | Indistinguishable in practice |
| 50            | ~75 μs  | Fine for any placeholder use |
| 32            | ~65 μs  | Colors correct, details blurred |
| 16            | ~45 μs  | Average color + rough orientation only |

```python
encode("photo.jpg", target_size=50)         # 4× DCT speedup, hash is still spec-valid
encode("photo.jpg", target_size=50, resize=False)  # error if image is already > 50px
```

> **Note**: For very large input images the bottleneck is usually PIL decode + resize, not the DCT. `target_size` only matters once your input is already small (e.g. a tensor in an ML pipeline). For batch processing many photos from disk, parallelize with `concurrent.futures.ProcessPoolExecutor` before reaching for GPU.

## Backends

The package picks the NumPy backend at import time if available, otherwise falls back to a pure-Python reference implementation. You can force one explicitly:

```python
encode(img, backend="numpy")    # default, BLAS-accelerated matmul
encode(img, backend="pure")     # reference Python, no deps
```

Backend availability is reflected by module flags:

```python
from thash import has_numpy, has_pil
```

### Backend comparison (random RGBA inputs, byte-identical output)

```
case                 size alpha         pure        numpy
---------------------------------------------------------
tiny-square       10x10   False     300 μs        41 μs
small-square      32x32   False     2.7 ms        66 μs
medium-square     64x64   False    11.4 ms        86 μs
max-square       100x100  False    26.8 ms       124 μs
landscape        100x56   False    11.7 ms        98 μs
max-square+a     100x100   True    28.2 ms       168 μs
HD-720p         1280x720  False        —          48 ms
FHD-1080p       1920x1080 False        —         208 ms
UHD-4K          3840x2160 False        —         516 ms
```

NumPy is ~100–140× faster than the reference impl on spec-sized inputs (geometric mean ~88×, median ~137×). Three optimizations stack here:

1. **Cosine basis cached** by `(n, k)` — `np.cos` cost amortizes across calls with shared dimensions (common after thumbnail).
2. **P and Q channels combined** into a single batched 3×3 matmul.
3. **float32 DCT** — Bandwidth halved, BLAS `sgemm` faster than `dgemm`; verified byte-identical on 490 random inputs across all spec shapes.

The pure-Python fallback is kept so the package works with zero deps. Run `uv run python benchmarks/run.py` to reproduce.

## Low-level API

The original byte-list API still works for callers who want to manage RGBA themselves:

```python
from thash import rgba_to_thumb_hash, image_to_thumb_hash

# Flat list: [R, G, B, A, R, G, B, A, ...], length = 4 * w * h
hash_bytes = rgba_to_thumb_hash(width, height, flat_rgba_ints)

# Open a file via Pillow, thumbnail to ≤100x100, encode
hash_bytes = image_to_thumb_hash("photo.jpg")
```

`rgba_to_thumb_hash` automatically picks the NumPy backend if available, falling back to pure Python otherwise.

## Development

```sh
git clone https://github.com/Jannchie/thumbhash-py.git
cd thumbhash-py
uv sync --all-extras --all-groups   # full dev env (deps + dev tools + bench)

uv run pytest                       # tests
uv run ruff check thash benchmarks  # lint
uv run python benchmarks/run.py     # benchmark suite
```

## Credits

- Original ThumbHash algorithm: [Evan Wallace](https://github.com/evanw) — [`evanw/thumbhash`](https://github.com/evanw/thumbhash)
- Original Python port: [Justin Forlenza](https://github.com/justinforlenza) — [`justinforlenza/thumbhash-py`](https://github.com/justinforlenza/thumbhash-py)
