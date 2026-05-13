"""NumPy-accelerated ThumbHash encoder. Requires numpy."""
from typing import List, Sequence

import numpy as np


def _encode_channel(channel_2d: np.ndarray, nx: int, ny: int, w: int, h: int):
    """DCT-II projection onto an (ny, nx) basis, returning (dc, ac_list, scale).

    Mirrors the triangular iteration order used by the reference encoder:
    ``cy`` outer, ``cx`` inner, while ``cx * ny < nx * (ny - cy)``.
    """
    cx_idx = np.arange(nx)
    cy_idx = np.arange(ny)
    x = np.arange(w) + 0.5
    y = np.arange(h) + 0.5

    Cx = np.cos((np.pi / w) * np.outer(cx_idx, x))  # (nx, w)
    Cy = np.cos((np.pi / h) * np.outer(cy_idx, y))  # (ny, h)

    # F[cy, cx] = (1/(w*h)) * sum_{x,y} channel[y,x] * cos(...) * cos(...)
    F = (Cy @ channel_2d @ Cx.T) / (w * h)  # (ny, nx)

    # Triangular mask in (cy, cx) iteration order.
    cy_grid, cx_grid = np.meshgrid(cy_idx, cx_idx, indexing="ij")
    mask = cx_grid * ny < nx * (ny - cy_grid)
    # Row-major flatten preserves the original cy-outer / cx-inner order.
    selected = F[mask]  # 1-D array, first element is (0,0) -> DC

    dc = float(selected[0])
    ac = selected[1:]
    scale = float(np.abs(ac).max()) if ac.size else 0.0
    if scale:
        ac = 0.5 + 0.5 / scale * ac
    return dc, ac.tolist(), scale


def rgba_to_thumb_hash(w: int, h: int, rgba: Sequence[int]) -> List[int]:
    """Encodes an RGBA image to a ThumbHash (NumPy implementation)."""
    if w > 100 or h > 100:
        raise ValueError(f"{w}x{h} doesn't fit in 100x100")
    return _encode(w, h, rgba)


def _encode(w: int, h: int, rgba: Sequence[int]) -> List[int]:
    """NumPy encoder body without the 100x100 spec guard, for benchmarking."""
    arr = np.asarray(rgba, dtype=np.float64).reshape(h, w, 4)
    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]
    alpha = arr[..., 3] / 255.0  # (h, w), in [0, 1]

    # Premultiplied averages (alpha-weighted), then divide by total alpha.
    a_over_255 = alpha / 255.0
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

    one_minus_alpha = 1.0 - alpha
    rr = avg_r * one_minus_alpha + a_over_255 * r
    gg = avg_g * one_minus_alpha + a_over_255 * g
    bb = avg_b * one_minus_alpha + a_over_255 * b
    l_ch = (rr + gg + bb) / 3.0
    p_ch = (rr + gg) / 2.0 - bb
    q_ch = rr - gg

    l_dc, l_ac, l_scale = _encode_channel(l_ch, max(3, lx), max(3, ly), w, h)
    p_dc, p_ac, p_scale = _encode_channel(p_ch, 3, 3, w, h)
    q_dc, q_ac, q_scale = _encode_channel(q_ch, 3, 3, w, h)
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
    thumb_hash = [header24 & 255, (header24 >> 8) & 255, header24 >> 16, header16 & 255, header16 >> 8]

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
