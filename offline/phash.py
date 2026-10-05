"""Perceptual hash for rejecting near-duplicate keyframes inside a shot.

Stands in for the `imagehash` package, which cannot be installed here (pypi is
unreachable from this network). Same algorithm as imagehash.phash: greyscale,
32x32, 2-D DCT, keep the top-left 8x8 low-frequency block excluding the DC term,
threshold at the median.
"""
from __future__ import annotations

import numpy as np
from scipy.fftpack import dct

_SIZE = 32
_LOW = 8


def _to_grey_32(img: np.ndarray) -> np.ndarray:
    """Greyscale and resize to 32x32 using box averaging."""
    a = np.asarray(img)
    if a.ndim == 3:
        # Rec. 601 luma, matching PIL's "L" conversion
        a = a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    a = a.astype(np.float64)

    h, w = a.shape
    # Average-pool onto a 32x32 grid; index arithmetic avoids an image library.
    ys = (np.arange(_SIZE + 1) * h / _SIZE).astype(int)
    xs = (np.arange(_SIZE + 1) * w / _SIZE).astype(int)
    out = np.empty((_SIZE, _SIZE), dtype=np.float64)
    for i in range(_SIZE):
        y0, y1 = ys[i], max(ys[i + 1], ys[i] + 1)
        for j in range(_SIZE):
            x0, x1 = xs[j], max(xs[j + 1], xs[j] + 1)
            out[i, j] = a[y0:y1, x0:x1].mean()
    return out


def phash(img: np.ndarray) -> int:
    """64-bit perceptual hash of an RGB or greyscale array."""
    grey = _to_grey_32(img)
    coeffs = dct(dct(grey, axis=0, norm="ortho"), axis=1, norm="ortho")
    low = coeffs[:_LOW, :_LOW].flatten()
    # Drop the DC term: it only carries overall brightness.
    rest = low[1:]
    bits = rest > np.median(rest)

    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    """Number of differing bits between two hashes."""
    return int(a ^ b).bit_count()
