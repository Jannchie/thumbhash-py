"""NumPy-accelerated ThumbHash encoder. Requires numpy."""
from functools import cache
from typing import List, Sequence

import numpy as np

# Whether the DCT runs in float32 (faster, smaller AC coefficients within the
# 4-bit quantization budget — still byte-identical to the float64 path on the
# inputs we tested).
_DCT_DTYPE = np.float32
_DCT_DTYPE_CHAR = np.dtype(_DCT_DTYPE).char


@cache
def _cosine_basis(n: int, k: int, dtype_char: str) -> np.ndarray:
    """Return the (k, n) cos matrix used by the DCT-II projection.

    The result is cached across calls; n/k pairs recur whenever images share
    a dimension (very common: thumbnail caps max(w, h) at 100).
    """
    dtype = np.dtype(dtype_char)
    cx_idx = np.arange(k, dtype=dtype)
    x = np.arange(n, dtype=dtype) + dtype.type(0.5)
    return np.cos((np.pi / n) * np.outer(cx_idx, x)).astype(dtype, copy=False)  # (k, n)


@cache
def _triangular_mask(nx: int, ny: int) -> np.ndarray:
    """(ny, nx) bool mask selecting AC entries in cy-outer / cx-inner order."""
    cy_idx = np.arange(ny)
    cx_idx = np.arange(nx)
    cy_grid, cx_grid = np.meshgrid(cy_idx, cx_idx, indexing="ij")
    return cx_grid * ny < nx * (ny - cy_grid)


def _encode_channel(channel_2d: np.ndarray, nx: int, ny: int, w: int, h: int):
    """DCT-II projection onto an (ny, nx) basis, returning (dc, ac_list, scale)."""
    Cx = _cosine_basis(w, nx, _DCT_DTYPE_CHAR)
    Cy = _cosine_basis(h, ny, _DCT_DTYPE_CHAR)
    F = (Cy @ channel_2d @ Cx.T) / (w * h)

    mask = _triangular_mask(nx, ny)
    selected = F[mask]

    dc = float(selected[0])
    ac = selected[1:]
    scale = float(np.abs(ac).max()) if ac.size else 0.0
    if scale:
        ac = 0.5 + 0.5 / scale * ac
    return dc, ac.tolist(), scale


def _encode_pq(p_ch: np.ndarray, q_ch: np.ndarray, w: int, h: int):
    """Combined 3x3 DCT for P and Q channels (they share Cx/Cy)."""
    Cx = _cosine_basis(w, 3, _DCT_DTYPE_CHAR)
    Cy = _cosine_basis(h, 3, _DCT_DTYPE_CHAR)
    stacked = np.stack([p_ch, q_ch])              # (2, h, w)
    F = (Cy @ stacked @ Cx.T) / (w * h)           # (2, 3, 3) — one batched matmul

    mask = _triangular_mask(3, 3)
    out = []
    for i in range(2):
        selected = F[i][mask]
        dc = float(selected[0])
        ac = selected[1:]
        scale = float(np.abs(ac).max()) if ac.size else 0.0
        if scale:
            ac = 0.5 + 0.5 / scale * ac
        out.append((dc, ac.tolist(), scale))
    return out[0], out[1]


def rgba_to_thumb_hash(w: int, h: int, rgba: Sequence[int]) -> List[int]:
    """Encodes an RGBA image to a ThumbHash (NumPy implementation)."""
    if w > 100 or h > 100:
        raise ValueError(f"{w}x{h} doesn't fit in 100x100")
    return _encode(w, h, rgba)


def _encode(w: int, h: int, rgba: Sequence[int]) -> List[int]:
    """NumPy encoder body without the 100x100 spec guard, for benchmarking."""
    arr = np.asarray(rgba, dtype=_DCT_DTYPE).reshape(h, w, 4)
    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]
    alpha = arr[..., 3] * _DCT_DTYPE(1.0 / 255.0)

    a_over_255 = alpha * _DCT_DTYPE(1.0 / 255.0)
    avg_r = float((a_over_255 * r).sum())
    avg_g = float((a_over_255 * g).sum())
    avg_b = float((a_over_255 * b).sum())
    avg_a = float(alpha.sum())

    if avg_a:
        avg_r /= avg_a
        avg_g /= avg_a
        avg_b /= avg_a

    has_alpha = avg_a < w * h

    l_limit = 5 if has_alpha else 7
    lx = max(1, round(l_limit * w / max(w, h)))
    ly = max(1, round(l_limit * h / max(w, h)))

    one_minus_alpha = _DCT_DTYPE(1.0) - alpha
    rr = _DCT_DTYPE(avg_r) * one_minus_alpha + a_over_255 * r
    gg = _DCT_DTYPE(avg_g) * one_minus_alpha + a_over_255 * g
    bb = _DCT_DTYPE(avg_b) * one_minus_alpha + a_over_255 * b
    l_ch = (rr + gg + bb) * _DCT_DTYPE(1.0 / 3.0)
    p_ch = (rr + gg) * _DCT_DTYPE(0.5) - bb
    q_ch = rr - gg

    l_dc, l_ac, l_scale = _encode_channel(l_ch, max(3, lx), max(3, ly), w, h)
    (p_dc, p_ac, p_scale), (q_dc, q_ac, q_scale) = _encode_pq(p_ch, q_ch, w, h)
    if has_alpha:
        a_dc, a_ac, a_scale = _encode_channel(alpha, 5, 5, w, h)
    else:
        a_dc, a_ac, a_scale = 1.0, [], 1.0

    is_landscape = w > h
    header24 = (
        round(63 * l_dc)
        | (round(31.5 + 31.5 * p_dc) << 6)
        | (round(31.5 + 31.5 * q_dc) << 12)
        | (round(31 * l_scale) << 18)
        | (has_alpha << 23)
    )
    header16 = (
        (ly if is_landscape else lx)
        | (round(63 * p_scale) << 3)
        | (round(63 * q_scale) << 9)
        | (is_landscape << 15)
    )
    thumb_hash = [
        header24 & 255,
        (header24 >> 8) & 255,
        header24 >> 16,
        header16 & 255,
        header16 >> 8,
    ]

    is_odd = False

    if has_alpha:
        thumb_hash.append(round(15 * a_dc) | (round(15 * a_scale) << 4))

    for ac in (l_ac, p_ac, q_ac):
        for f in ac:
            u = round(15.0 * f)
            if is_odd:
                thumb_hash[-1] |= u << 4
            else:
                thumb_hash.append(u)
            is_odd = not is_odd

    if has_alpha:
        for f in a_ac:
            u = round(15.0 * f)
            if is_odd:
                thumb_hash[-1] |= u << 4
            else:
                thumb_hash.append(u)
            is_odd = not is_odd

    return thumb_hash
