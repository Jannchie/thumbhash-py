"""High-level polymorphic encode() API.

Accepts a wide range of image inputs (path, bytes, PIL.Image, numpy array,
OpenCV array) and routes to an accelerated backend.
"""
from io import BytesIO
from pathlib import Path
from types import ModuleType

import numpy as np

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None  # type: ignore[assignment]
    ImageOps = None  # type: ignore[assignment]

from . import _numpy, _pure

_BACKENDS: dict[str, ModuleType] = {
    "pure": _pure,
    "numpy": _numpy,
}


def encode(
    image,
    *,
    color_order: str = "auto",
    resize: bool = True,
    target_size: int = 100,
    backend: str | None = None,
) -> list[int]:
    """Encode an image to a ThumbHash.

    Args:
        image: One of:
            - ``str`` / ``Path``: file path (decoded via Pillow).
            - ``bytes`` / ``bytearray``: encoded image bytes (decoded via Pillow).
            - ``PIL.Image.Image``: used directly (EXIF transpose is applied).
            - ``numpy.ndarray``: shape (H, W), (H, W, 1/3/4); dtype uint8 or float.
              Floats are interpreted as 0..1 and scaled to 0..255.
            - Any object implementing ``__array__`` (e.g. torch tensors on CPU).
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
        backend: Encoder backend. ``None`` (default) picks ``'numpy'``.
            Available: ``'pure'`` (reference, no deps), ``'numpy'`` (BLAS-accelerated).

    Returns:
        List[int]: the ThumbHash byte sequence.
    """
    if not (1 <= target_size <= 100):
        raise ValueError(f"target_size must be in [1, 100]; got {target_size}")

    rgba = _to_rgba_uint8(
        image, color_order=color_order, resize=resize, target_size=target_size
    )
    h, w = rgba.shape[:2]
    if w > target_size or h > target_size:
        raise ValueError(
            f"image is {w}x{h}, exceeds target_size={target_size}. "
            "Pass resize=True or pre-downscale."
        )

    mod = _BACKENDS.get(backend or "numpy")
    if mod is None:
        raise ValueError(
            f"backend {backend!r} not available. Choose from {sorted(_BACKENDS)}."
        )
    # Backends accept either a flat sequence or an (h, w, 4) ndarray; numpy
    # backends np.asarray(...).reshape(h, w, 4) so we pass the ndarray directly
    # for zero-copy.
    if mod is _pure:
        # pure-Python backend needs a flat int sequence
        return mod._encode(w, h, rgba.reshape(-1).tolist())
    return mod._encode(w, h, rgba)


_PIL_REQUIRED = "Pillow required for this input. Install with `pip install thash[pillow]`."


def _to_rgba_uint8(image, *, color_order: str, resize: bool, target_size: int) -> np.ndarray:
    """Normalize any supported input to a contiguous (H, W, 4) uint8 RGBA array."""
    # --- Path / bytes ---
    if isinstance(image, (str, Path)):
        if Image is None:
            raise ImportError(_PIL_REQUIRED)
        return _pil_to_rgba(Image.open(image), resize=resize, target_size=target_size)
    if isinstance(image, (bytes, bytearray, memoryview)):
        if Image is None:
            raise ImportError(_PIL_REQUIRED)
        return _pil_to_rgba(
            Image.open(BytesIO(bytes(image))), resize=resize, target_size=target_size
        )

    # --- PIL Image ---
    if Image is not None and isinstance(image, Image.Image):
        return _pil_to_rgba(image, resize=resize, target_size=target_size)

    # --- Anything array-like (numpy, torch on CPU, cv2 output, etc.) ---
    if isinstance(image, np.ndarray) or hasattr(image, "__array__"):
        return _ndarray_to_rgba(
            np.asarray(image), color_order=color_order, resize=resize, target_size=target_size
        )

    raise TypeError(
        f"Unsupported image type: {type(image).__name__}. "
        "Expected path, bytes, PIL.Image, or numpy ndarray."
    )


def _pil_to_rgba(pil, *, resize: bool, target_size: int) -> np.ndarray:
    if ImageOps is None:
        raise ImportError(_PIL_REQUIRED)
    pil = ImageOps.exif_transpose(pil)
    if resize and (pil.width > target_size or pil.height > target_size):
        pil = pil.copy()
        pil.thumbnail((target_size, target_size))
    if pil.mode != "RGBA":
        pil = pil.convert("RGBA")
    arr = np.asarray(pil, dtype=np.uint8)
    # PIL sometimes returns non-contiguous views (e.g. for cropped images).
    return np.ascontiguousarray(arr)


def _ndarray_to_rgba(arr: np.ndarray, *, color_order: str, resize: bool, target_size: int) -> np.ndarray:
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"Unsupported ndarray shape: {arr.shape}")
    c = arr.shape[2]
    if c not in (1, 3, 4):
        raise ValueError(f"Channels must be 1, 3, or 4; got {c}")

    # Normalize dtype to uint8
    if arr.dtype.kind == "f":
        arr = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    elif arr.dtype == np.uint16:
        arr = (arr >> 8).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    # Decide color order
    order = color_order.upper() if color_order != "auto" else ("RGBA" if c == 4 else "RGB")

    # Grayscale → RGB
    if c == 1:
        arr = np.repeat(arr, 3, axis=2)
        order = "RGB"
        c = 3

    # BGR(A) → RGB(A)
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

    # Add alpha if needed
    if arr.shape[2] == 3:
        alpha = np.full((*arr.shape[:2], 1), 255, dtype=np.uint8)
        arr = np.concatenate([arr, alpha], axis=-1)

    arr = np.ascontiguousarray(arr)

    # Resize via PIL if oversized
    if resize and (arr.shape[1] > target_size or arr.shape[0] > target_size):
        if Image is None:
            raise ImportError(_PIL_REQUIRED)
        pil = Image.fromarray(arr, mode="RGBA")
        pil.thumbnail((target_size, target_size))
        arr = np.asarray(pil, dtype=np.uint8)
        arr = np.ascontiguousarray(arr)

    return arr
