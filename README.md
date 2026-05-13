<img src="https://img.shields.io/pypi/v/thash"> <img src="https://img.shields.io/github/license/Jannchie/thumbhash-py">

# thash

A modern Python port of the [ThumbHash](https://github.com/evanw/thumbhash) encoder by [Evan Wallace](https://github.com/evanw). ThumbHash represents an image as ~20 bytes — small enough to inline in HTML, large enough to render a recognizable color/aspect placeholder before the real image loads.

This is an independently published fork of [`thumbhash`](https://pypi.org/project/thumbhash/) by [Justin Forlenza](https://github.com/justinforlenza). Notable changes vs. upstream:

- **Alpha-channel crash fixed** (operator-precedence bug in `rgba_to_thumb_hash` — see [upstream issue #1](https://github.com/justinforlenza/thumbhash-py/issues/1)).
- **NumPy-accelerated backend** (~40–60× faster than the reference implementation, byte-identical output).
- **High-level `encode()` API** that accepts paths, bytes, PIL images, NumPy arrays, and OpenCV BGR arrays — pick the input you already have, no boilerplate.
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
from thash import thumb_hash_to_average_rgba, thumb_hash_to_approximate_aspect_ratio

r, g, b, a = thumb_hash_to_average_rgba(hash_bytes)   # values in [0, 1]
aspect = thumb_hash_to_approximate_aspect_ratio(hash_bytes)  # w / h
```

(For full decoding back to pixels, see [the JS reference impl](https://github.com/evanw/thumbhash) — only encoding is implemented here.)

## Tuning speed vs. quality

`target_size` controls the longer dimension of the image after thumbnail (spec max is 100). Smaller = faster, lower fidelity:

| `target_size` | DCT time | Visual quality |
|---|---|---|
| 100 (default) | ~350 μs | Reference / spec-compatible |
| 64            | ~225 μs | Indistinguishable in practice |
| 50            | ~150 μs | Fine for any placeholder use |
| 32            | ~140 μs | Colors correct, details blurred |
| 16            | ~85 μs  | Average color + rough orientation only |

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
tiny-square       10x10   False     362 μs       240 μs
medium-square     64x64   False    11.5 ms       225 μs
max-square       100x100  False    36.2 ms       337 μs
max-square+a     100x100   True    30.7 ms       352 μs
HD-720p         1280x720  False        —        72 ms
FHD-1080p       1920x1080 False        —       152 ms
UHD-4K          3840x2160 False        —       686 ms
```

NumPy is ~40–60× faster than the reference impl on spec-sized inputs. The pure-Python fallback is kept so the package works with zero deps. See `benchmarks/run.py` for the full matrix — run it yourself with `uv run python benchmarks/run.py`.

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
