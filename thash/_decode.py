"""ThumbHash decoder: hash bytes back to a small RGBA preview.

Mirrors the byte layout produced by ``_pure._encode`` / ``_numpy._encode`` and
the reference JS implementation at https://github.com/evanw/thumbhash.

The reconstruction uses an inverse DCT-II with the standard ``alpha(k)`` weights
(``1`` if ``k == 0`` else ``2``) on both axes.

Works with or without numpy. With numpy installed the heavy IDCT runs as
matrix multiplications; otherwise it falls back to a pure-Python loop. Both
paths return ``(w, h, bytes)`` — a flat RGBA buffer of length ``4 * w * h``.
"""

from __future__ import annotations

import math
from typing import Sequence

try:
    import numpy as _np
except ImportError:
    _np = None


def thumb_hash_to_rgba(thumb_hash: Sequence[int], *, base_size: int = 32) -> tuple[int, int, bytes]:
    """Decode a ThumbHash to a small RGBA image.

    Args:
        thumb_hash: The ThumbHash byte sequence (any iterable of ints in 0..255).
        base_size: Longer edge of the reconstructed image. ThumbHash only
            carries ~5x5 / 7x7 frequency coefficients so anything beyond ~64
            looks the same — upscale with a resampler if you need a larger
            preview.

    Returns:
        ``(width, height, rgba)`` where ``rgba`` is a ``bytes`` object of
        length ``4 * width * height`` containing interleaved R, G, B, A
        (uint8) samples.

    Examples:
        With Pillow::

            from PIL import Image
            w, h, rgba = thumb_hash_to_rgba(hash_bytes, base_size=256)
            Image.frombytes("RGBA", (w, h), rgba).save("preview.png")

        With numpy::

            import numpy as np
            w, h, rgba = thumb_hash_to_rgba(hash_bytes)
            arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    """
    h_bytes = list(thumb_hash)
    if len(h_bytes) < 5:
        raise ValueError(f"ThumbHash too short ({len(h_bytes)} bytes; need >= 5)")

    header24 = h_bytes[0] | (h_bytes[1] << 8) | (h_bytes[2] << 16)
    header16 = h_bytes[3] | (h_bytes[4] << 8)

    l_dc = (header24 & 63) / 63.0
    p_dc = ((header24 >> 6) & 63) / 31.5 - 1.0
    q_dc = ((header24 >> 12) & 63) / 31.5 - 1.0
    l_scale = ((header24 >> 18) & 31) / 31.0
    has_alpha = (header24 >> 23) != 0

    p_scale = ((header16 >> 3) & 63) / 63.0
    q_scale = ((header16 >> 9) & 63) / 63.0
    is_landscape = (header16 >> 15) != 0

    l_min = header16 & 7
    l_max = 5 if has_alpha else 7
    lx = max(3, l_max if is_landscape else l_min)
    ly = max(3, l_min if is_landscape else l_max)

    if has_alpha:
        if len(h_bytes) < 6:
            raise ValueError("ThumbHash claims alpha but is missing the alpha header byte")
        a_dc = (h_bytes[5] & 15) / 15.0
        a_scale = ((h_bytes[5] >> 4) & 15) / 15.0
        ac_start = 6
    else:
        a_dc = 1.0
        a_scale = 1.0
        ac_start = 5

    nibble_offset = 0
    n_nibbles = max(0, (len(h_bytes) - ac_start) * 2)

    def take_nibble() -> int:
        nonlocal nibble_offset
        if nibble_offset >= n_nibbles:
            # Hash truncated — pad with the "zero AC" nibble (= 7.5 -> 0.0).
            return 8
        byte = h_bytes[ac_start + (nibble_offset >> 1)]
        nibble = (byte >> 4) & 15 if nibble_offset & 1 else byte & 15
        nibble_offset += 1
        return nibble

    def decode_ac(nx: int, ny: int, scale: float) -> list[float]:
        ac: list[float] = []
        for cy in range(ny):
            cx = 1 if cy == 0 else 0
            while cx * ny < nx * (ny - cy):
                ac.append((take_nibble() / 7.5 - 1.0) * scale)
                cx += 1
        return ac

    l_ac = decode_ac(lx, ly, l_scale)
    p_ac = decode_ac(3, 3, p_scale)
    q_ac = decode_ac(3, 3, q_scale)
    a_ac = decode_ac(5, 5, a_scale) if has_alpha else []

    ratio = lx / ly
    if ratio > 1:
        w_out = int(base_size)
        h_out = max(1, round(base_size / ratio))
    else:
        h_out = int(base_size)
        w_out = max(1, round(base_size * ratio))

    if _np is not None:
        return w_out, h_out, _reconstruct_numpy(
            w_out, h_out, lx, ly, l_dc, l_ac, p_dc, p_ac, q_dc, q_ac, a_dc, a_ac, has_alpha
        )
    return w_out, h_out, _reconstruct_pure(
        w_out, h_out, lx, ly, l_dc, l_ac, p_dc, p_ac, q_dc, q_ac, a_dc, a_ac, has_alpha
    )


def _pack_coeffs_pure(nx: int, ny: int, dc: float, ac: list[float]) -> list[list[float]]:
    coeffs = [[0.0] * nx for _ in range(ny)]
    coeffs[0][0] = dc
    idx = 0
    for cy in range(ny):
        cx = 1 if cy == 0 else 0
        while cx * ny < nx * (ny - cy):
            coeffs[cy][cx] = ac[idx]
            idx += 1
            cx += 1
    return coeffs


def _cos_table_pure(out: int, n: int) -> list[list[float]]:
    """Return ``out``-by-``n`` table of ``alpha(k) * cos(pi/out * (i + 0.5) * k)``."""
    table = [[0.0] * n for _ in range(out)]
    pi_over_out = math.pi / out
    for i in range(out):
        row = table[i]
        center = i + 0.5
        for k in range(n):
            a = 1.0 if k == 0 else 2.0
            row[k] = a * math.cos(pi_over_out * center * k)
    return table


def _idct_pure(w_out: int, h_out: int, nx: int, ny: int, coeffs: list[list[float]]) -> list[float]:
    cos_x = _cos_table_pure(w_out, nx)
    cos_y = _cos_table_pure(h_out, ny)
    # Per-pixel sum_{ky, kx} cos_y[y][ky] * coeffs[ky][kx] * cos_x[x][kx]
    # Refactored: row_y[kx] = sum_{ky} cos_y[y][ky] * coeffs[ky][kx]; then pixel = dot(row_y, cos_x[x]).
    out = [0.0] * (w_out * h_out)
    for y in range(h_out):
        cy = cos_y[y]
        row_y = [0.0] * nx
        for ky in range(ny):
            cyk = cy[ky]
            crow = coeffs[ky]
            for kx in range(nx):
                row_y[kx] += cyk * crow[kx]
        base = y * w_out
        for x in range(w_out):
            cx = cos_x[x]
            s = 0.0
            for kx in range(nx):
                s += row_y[kx] * cx[kx]
            out[base + x] = s
    return out


def _reconstruct_pure(
    w_out: int,
    h_out: int,
    lx: int,
    ly: int,
    l_dc: float,
    l_ac: list[float],
    p_dc: float,
    p_ac: list[float],
    q_dc: float,
    q_ac: list[float],
    a_dc: float,
    a_ac: list[float],
    has_alpha: bool,
) -> bytes:
    L = _idct_pure(w_out, h_out, lx, ly, _pack_coeffs_pure(lx, ly, l_dc, l_ac))
    P = _idct_pure(w_out, h_out, 3, 3, _pack_coeffs_pure(3, 3, p_dc, p_ac))
    Q = _idct_pure(w_out, h_out, 3, 3, _pack_coeffs_pure(3, 3, q_dc, q_ac))
    A = _idct_pure(w_out, h_out, 5, 5, _pack_coeffs_pure(5, 5, a_dc, a_ac)) if has_alpha else None

    n = w_out * h_out
    out = bytearray(n * 4)
    for i in range(n):
        lv = L[i]
        pv = P[i]
        qv = Q[i]
        bv = lv - 2.0 / 3.0 * pv
        rv = (3.0 * lv - bv + qv) / 2.0
        gv = rv - qv
        av = A[i] if A is not None else 1.0
        j = i * 4
        out[j] = max(0, min(255, int(rv * 255.0 + 0.5)))
        out[j + 1] = max(0, min(255, int(gv * 255.0 + 0.5)))
        out[j + 2] = max(0, min(255, int(bv * 255.0 + 0.5)))
        out[j + 3] = max(0, min(255, int(av * 255.0 + 0.5)))
    return bytes(out)


def _reconstruct_numpy(
    w_out: int,
    h_out: int,
    lx: int,
    ly: int,
    l_dc: float,
    l_ac: list[float],
    p_dc: float,
    p_ac: list[float],
    q_dc: float,
    q_ac: list[float],
    a_dc: float,
    a_ac: list[float],
    has_alpha: bool,
) -> bytes:
    assert _np is not None
    np = _np

    def pack(nx: int, ny: int, dc: float, ac: list[float]):
        coeffs = np.zeros((ny, nx), dtype=np.float32)
        coeffs[0, 0] = dc
        idx = 0
        for cy in range(ny):
            cx = 1 if cy == 0 else 0
            while cx * ny < nx * (ny - cy):
                coeffs[cy, cx] = ac[idx]
                idx += 1
                cx += 1
        return coeffs

    def idct(nx: int, ny: int, coeffs):
        x = np.arange(w_out, dtype=np.float32)
        y = np.arange(h_out, dtype=np.float32)
        ax = np.where(np.arange(nx) == 0, 1.0, 2.0).astype(np.float32)
        ay = np.where(np.arange(ny) == 0, 1.0, 2.0).astype(np.float32)
        cos_x = np.cos(np.pi / w_out * (x[:, None] + 0.5) * np.arange(nx, dtype=np.float32)[None, :]) * ax[None, :]
        cos_y = np.cos(np.pi / h_out * (y[:, None] + 0.5) * np.arange(ny, dtype=np.float32)[None, :]) * ay[None, :]
        return cos_y.astype(np.float32) @ coeffs @ cos_x.astype(np.float32).T  # (h_out, w_out)

    L = idct(lx, ly, pack(lx, ly, l_dc, l_ac))
    P = idct(3, 3, pack(3, 3, p_dc, p_ac))
    Q = idct(3, 3, pack(3, 3, q_dc, q_ac))
    A = idct(5, 5, pack(5, 5, a_dc, a_ac)) if has_alpha else np.ones_like(L)

    B = L - 2.0 / 3.0 * P
    R = (3.0 * L - B + Q) / 2.0
    G = R - Q

    rgba = np.stack([R, G, B, A], axis=-1)
    rgba = np.clip(rgba * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgba).tobytes()
