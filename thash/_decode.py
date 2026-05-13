"""ThumbHash decoder: hash bytes back to a small RGBA preview.

Mirrors the byte layout produced by `_pure._encode` / `_numpy._encode` and
the reference JS implementation in https://github.com/evanw/thumbhash.

The reconstruction uses an inverse DCT-II with the standard alpha(k) weights
(1 if k == 0 else 2) on both axes — without those weights the AC band would
be attenuated by 2x / 4x.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def thumb_hash_to_rgba(thumb_hash: Sequence[int], *, base_size: int = 32):
    """Decode a ThumbHash to an RGBA image.

    Returns ``(width, height, rgba)`` where ``rgba`` is a contiguous
    ``(height, width, 4)`` uint8 numpy array.

    ``base_size`` is the longer edge of the reconstructed image. ThumbHash
    only carries ~5x5 / 7x7 frequency coefficients, so anything beyond ~64
    looks the same — upscale with a resampler (Pillow bilinear/lanczos)
    if you need a larger preview.
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
            # Hash truncated — pad with the "zero AC" nibble (= 7.5 → 0.0).
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

    def reconstruct(nx: int, ny: int, dc: float, ac: list[float]) -> np.ndarray:
        coeffs = np.zeros((ny, nx), dtype=np.float32)
        coeffs[0, 0] = dc
        idx = 0
        for cy in range(ny):
            cx = 1 if cy == 0 else 0
            while cx * ny < nx * (ny - cy):
                coeffs[cy, cx] = ac[idx]
                idx += 1
                cx += 1
        x = np.arange(w_out, dtype=np.float32)
        y = np.arange(h_out, dtype=np.float32)
        # alpha(k) = 1 if k == 0 else 2 — folds the DCT-II inverse weights into the bases.
        ax = np.where(np.arange(nx) == 0, 1.0, 2.0).astype(np.float32)
        ay = np.where(np.arange(ny) == 0, 1.0, 2.0).astype(np.float32)
        cos_x = np.cos(np.pi / w_out * (x[:, None] + 0.5) * np.arange(nx, dtype=np.float32)[None, :]) * ax[None, :]
        cos_y = np.cos(np.pi / h_out * (y[:, None] + 0.5) * np.arange(ny, dtype=np.float32)[None, :]) * ay[None, :]
        return cos_y.astype(np.float32) @ coeffs @ cos_x.astype(np.float32).T  # (h_out, w_out)

    L = reconstruct(lx, ly, l_dc, l_ac)
    P = reconstruct(3, 3, p_dc, p_ac)
    Q = reconstruct(3, 3, q_dc, q_ac)
    A = reconstruct(5, 5, a_dc, a_ac) if has_alpha else np.ones_like(L)

    # YPbPr-ish → RGB (inverse of the encoder's r/g/b → l/p/q transform).
    B = L - 2.0 / 3.0 * P
    R = (3.0 * L - B + Q) / 2.0
    G = R - Q

    rgba = np.stack([R, G, B, A], axis=-1)
    rgba = np.clip(rgba * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return w_out, h_out, np.ascontiguousarray(rgba)
