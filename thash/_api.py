"""High-level polymorphic encode() API.

Accepts a wide range of image inputs (path, bytes, PIL.Image, numpy array,
OpenCV array) and routes to an accelerated backend. Works without numpy:
path/bytes/PIL.Image inputs go through Pillow's ``tobytes()`` straight into
the pure-Python backend; ndarray inputs require numpy because the input
itself does.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Literal

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None  # type: ignore[assignment]
    ImageOps = None  # type: ignore[assignment]

try:
    import numpy as _np
except ImportError:
    _np = None

from . import _pure

try:
    from . import _numpy
except ImportError:
    _numpy = None


_PIL_REQUIRED = "Pillow required for this input. Install with `pip install thash[pillow]`."
_NUMPY_REQUIRED_FOR_ARRAY = (
    "numpy is required to consume ndarray / array-like inputs. "
    "Install with `pip install thash[numpy]`, or pass a path / bytes / PIL.Image instead."
)
_NUMPY_REQUIRED_FOR_BACKEND = "backend='numpy' was requested but numpy is not installed."


def encode(
    image,
    *,
    color_order: str = "auto",
    resize: bool = True,
    target_size: int = 100,
    backend: Literal["pure", "numpy"] | None = None,
) -> list[int]:
    """Encode an image to a ThumbHash.

    Args:
        image: One of:
            - ``str`` / ``Path``: file path (decoded via Pillow).
            - ``bytes`` / ``bytearray``: encoded image bytes (decoded via Pillow).
            - ``PIL.Image.Image``: used directly (EXIF transpose is applied).
            - ``numpy.ndarray``: shape (H, W), (H, W, 1/3/4); dtype uint8 or float.
              Floats are interpreted as 0..1 and scaled to 0..255. (Requires numpy.)
            - Any object implementing ``__array__`` (e.g. torch tensors on CPU).
              (Requires numpy.)
        color_order: ``'RGB' | 'BGR' | 'RGBA' | 'BGRA' | 'auto'``.
            Only consulted for ndarray inputs. ``'auto'`` assumes RGB/RGBA
            (PIL convention). Set to ``'BGR'`` or ``'BGRA'`` for OpenCV arrays.
        resize: If True (default), thumbnail the image so that
            ``max(w, h) <= target_size``. If False, the caller must ensure the
            image already fits within ``target_size``.
        target_size: Cap on the longer image dimension (spec max is 100).
            Smaller values trade hash quality for speed: 50 is roughly 4x faster
            DCT with negligible visual difference; 32 is much faster but loses
            detail. Must be in [1, 100].
        backend: Encoder backend. ``None`` (default) picks ``'numpy'`` when
            available, otherwise ``'pure'``. Available: ``'pure'`` (reference,
            no deps), ``'numpy'`` (BLAS-accelerated).

    Returns:
        list[int]: the ThumbHash byte sequence.
    """
    if not (1 <= target_size <= 100):
        raise ValueError(f"target_size must be in [1, 100]; got {target_size}")

    w, h, rgba = _to_rgba(image, color_order=color_order, resize=resize, target_size=target_size)
    if w > target_size or h > target_size:
        raise ValueError(f"image is {w}x{h}, exceeds target_size={target_size}. Pass resize=True or pre-downscale.")

    if backend is None:
        backend = "numpy" if _numpy is not None else "pure"

    if backend == "numpy":
        if _numpy is None:
            raise ImportError(_NUMPY_REQUIRED_FOR_BACKEND)
        return _numpy._encode(w, h, _to_numpy_arr(rgba, w, h))
    if backend == "pure":
        return _pure._encode(w, h, _to_flat_list(rgba))
    raise ValueError(f"backend {backend!r} not available. Choose from ('pure', 'numpy').")


def _to_numpy_arr(rgba, w: int, h: int):
    """Coerce an RGBA buffer (bytes/ndarray) into an ndarray for the numpy backend."""
    assert _np is not None
    if isinstance(rgba, _np.ndarray):
        return rgba
    return _np.frombuffer(rgba, dtype=_np.uint8).reshape(h, w, 4)


def _to_flat_list(rgba) -> list[int]:
    """Coerce an RGBA buffer (bytes/bytearray/list/ndarray) into a flat list[int]."""
    if _np is not None and isinstance(rgba, _np.ndarray):
        return rgba.reshape(-1).tolist()
    if isinstance(rgba, (bytes, bytearray, memoryview)):
        return list(rgba)
    return list(rgba)


def _to_rgba(image, *, color_order: str, resize: bool, target_size: int):
    """Normalize any supported input to ``(w, h, rgba_buf)``.

    ``rgba_buf`` is either:
        - ``bytes`` (flat RGBA, length 4*w*h) — for PIL-decoded inputs.
        - ``np.ndarray`` of shape ``(h, w, 4)`` uint8 — for ndarray inputs.

    Both shapes are accepted by both backends (numpy backend calls ``np.asarray``
    internally; pure backend gets a flat list via ``_to_flat_list``).
    """
    if isinstance(image, (str, Path)):
        if Image is None:
            raise ImportError(_PIL_REQUIRED)
        return _pil_to_rgba(Image.open(image), resize=resize, target_size=target_size)
    if isinstance(image, (bytes, bytearray, memoryview)):
        if Image is None:
            raise ImportError(_PIL_REQUIRED)
        return _pil_to_rgba(Image.open(BytesIO(bytes(image))), resize=resize, target_size=target_size)

    if Image is not None and isinstance(image, Image.Image):
        return _pil_to_rgba(image, resize=resize, target_size=target_size)

    if _np is not None and (isinstance(image, _np.ndarray) or hasattr(image, "__array__")):
        return _ndarray_to_rgba(_np.asarray(image), color_order=color_order, resize=resize, target_size=target_size)

    # ndarray-like input but numpy missing — give a targeted error
    if hasattr(image, "__array__"):
        raise ImportError(_NUMPY_REQUIRED_FOR_ARRAY)

    raise TypeError(
        f"Unsupported image type: {type(image).__name__}. Expected path, bytes, PIL.Image, or numpy ndarray."
    )


def _pil_to_rgba(pil, *, resize: bool, target_size: int) -> tuple[int, int, bytes]:
    if ImageOps is None:
        raise ImportError(_PIL_REQUIRED)
    pil = ImageOps.exif_transpose(pil)
    if resize and (pil.width > target_size or pil.height > target_size):
        pil = pil.copy()
        pil.thumbnail((target_size, target_size))
    if pil.mode != "RGBA":
        pil = pil.convert("RGBA")
    return pil.width, pil.height, pil.tobytes()


def _ndarray_to_rgba(arr, *, color_order: str, resize: bool, target_size: int):
    assert _np is not None  # guarded by caller
    np = _np
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"Unsupported ndarray shape: {arr.shape}")
    c = arr.shape[2]
    if c not in (1, 3, 4):
        raise ValueError(f"Channels must be 1, 3, or 4; got {c}")

    if arr.dtype.kind == "f":
        arr = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    elif arr.dtype == np.uint16:
        arr = (arr >> 8).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    order = color_order.upper() if color_order != "auto" else ("RGBA" if c == 4 else "RGB")

    if c == 1:
        arr = np.repeat(arr, 3, axis=2)
        order = "RGB"
        c = 3

    if order == "BGR":
        if c != 3:
            raise ValueError(f"color_order='BGR' requires 3 channels; got {c}")
        arr = arr[..., ::-1]
    elif order == "BGRA":
        if c != 4:
            raise ValueError(f"color_order='BGRA' requires 4 channels; got {c}")
        arr = arr[..., [2, 1, 0, 3]]
    elif order not in ("RGB", "RGBA"):
        raise ValueError(f"color_order must be one of RGB/BGR/RGBA/BGRA/auto; got {color_order!r}")

    if arr.shape[2] == 3:
        alpha = np.full((*arr.shape[:2], 1), 255, dtype=np.uint8)
        arr = np.concatenate([arr, alpha], axis=-1)

    arr = np.ascontiguousarray(arr)

    if resize and (arr.shape[1] > target_size or arr.shape[0] > target_size):
        if Image is None:
            raise ImportError(_PIL_REQUIRED)
        pil = Image.fromarray(arr, mode="RGBA")
        pil.thumbnail((target_size, target_size))
        arr = np.asarray(pil, dtype=np.uint8)
        arr = np.ascontiguousarray(arr)

    h, w = arr.shape[:2]
    return w, h, arr
