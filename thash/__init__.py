"""thash — a compact placeholder hash for images.

Public surface:
    encode(image, ...)                    -- high-level polymorphic API
    image_to_thumb_hash(fp)               -- legacy path-based API (requires pillow)
    rgba_to_thumb_hash(w, h, rgba)        -- low-level encode from flat RGBA
    thumb_hash_to_rgba(hash, ...)         -- decode to a small RGBA preview (returns bytes)
    thumb_hash_to_average_rgba(hash)
    thumb_hash_to_approximate_aspect_ratio(hash)

The package works with no third-party deps (pure-Python fallback). Installing
numpy enables a 100x+ faster encoder/decoder, and installing Pillow lets the
high-level API accept file paths / bytes / PIL images.

Backend handles (for benchmarking or forcing a path):
    rgba_to_thumb_hash_pure / rgba_to_thumb_hash_numpy
        (the numpy variant is ``None`` when numpy isn't installed)
"""

from typing import Callable

from . import _pure
from ._api import encode
from ._decode import thumb_hash_to_rgba

try:
    from . import _numpy
except ImportError:
    _numpy = None  # type: ignore[assignment]

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None  # type: ignore[assignment]
    ImageOps = None  # type: ignore[assignment]

has_numpy = _numpy is not None
has_pil = Image is not None


# Public entry point: prefers the NumPy backend when available.
rgba_to_thumb_hash: Callable[..., list[int]] = (
    _numpy.rgba_to_thumb_hash if _numpy is not None else _pure.rgba_to_thumb_hash
)

# Explicit backend handles (None if the optional dep isn't installed).
rgba_to_thumb_hash_pure: Callable[..., list[int]] = _pure.rgba_to_thumb_hash
rgba_to_thumb_hash_numpy: Callable[..., list[int]] | None = (
    _numpy.rgba_to_thumb_hash if _numpy is not None else None
)


__all__ = [
    "encode",
    "has_numpy",
    "has_pil",
    "image_to_thumb_hash",
    "rgba_to_thumb_hash",
    "rgba_to_thumb_hash_numpy",
    "rgba_to_thumb_hash_pure",
    "thumb_hash_to_approximate_aspect_ratio",
    "thumb_hash_to_average_rgba",
    "thumb_hash_to_rgba",
]


def image_to_thumb_hash(fp) -> list[int]:
    """Open an image file (or bytes/Path) and encode to a ThumbHash."""
    if Image is None or ImageOps is None:
        raise ImportError("Pillow not installed; install with `pip install thash[pillow]`")

    img = Image.open(fp)
    img = img.convert("RGBA")
    img.thumbnail((100, 100))
    img = ImageOps.exif_transpose(img)

    # tobytes() returns flat RGBA bytes (4*w*h ints in [0, 255]); avoids the
    # ImagingCore iterator that Pillow's stubs don't expose.
    rgba = list(img.tobytes())

    return rgba_to_thumb_hash(img.width, img.height, rgba)


def thumb_hash_to_average_rgba(
    thumb_hash: list[int],
) -> tuple[float, float, float, float] | None:
    """Extract the average color from a ThumbHash, or None if the hash is invalid."""
    if len(thumb_hash) < 5:
        return None

    header = thumb_hash[0] | (thumb_hash[1] << 8) | (thumb_hash[2] << 16)
    lum = (header & 63) / 63.0
    p = ((header >> 6) & 63) / 31.5 - 1.0
    q = ((header >> 12) & 63) / 31.5 - 1.0
    has_alpha = (header >> 23) != 0
    a = (thumb_hash[5] & 15) / 15.0 if has_alpha else 1.0
    b = lum - 2.0 / 3.0 * p
    r = (3.0 * lum - b + q) / 2.0
    g = r - q

    return (
        max(0.0, min(1.0, r)),
        max(0.0, min(1.0, g)),
        max(0.0, min(1.0, b)),
        a,
    )


def thumb_hash_to_approximate_aspect_ratio(thumb_hash: list[int]) -> float | None:
    """Approximate aspect ratio (w/h) of the encoded image, or None if invalid."""
    if len(thumb_hash) < 5:
        return None

    has_alpha = (thumb_hash[2] & 0x80) != 0
    l_max = 5 if has_alpha else 7
    l_min = thumb_hash[3] & 7
    is_landscape = (thumb_hash[4] & 0x80) != 0
    lx = l_max if is_landscape else l_min
    ly = l_min if is_landscape else l_max

    return lx / ly
