"""Deterministic edge tools: candidate edges along a scanline and sub-pixel snapping.

An edge is located at the sub-pixel maximum of the colour-gradient magnitude along the scanline
(parabolic fit through the peak sample and its two neighbours). For a blurred step this is the 50 %
crossing, i.e. the geometric boundary. Coordinates use the av2io convention (boundary between pixels
i and i+1 at i + 0.5).
"""
from __future__ import annotations

import numpy as np

from imaging import gradient_profile


def subpixel_peak(values, k):
    """Parabolic vertex offset in (-0.5, 0.5) around sample k (0 when k is at an end)."""
    if k <= 0 or k >= len(values) - 1:
        return 0.0
    a, b, c = values[k - 1], values[k], values[k + 1]
    denominator = a - 2 * b + c
    if denominator >= 0:
        return 0.0
    return float(np.clip(0.5 * (a - c) / denominator, -0.5, 0.5))


def local_maxima(values):
    return [k for k in range(1, len(values) - 1) if values[k] >= values[k - 1] and values[k] > values[k + 1]]


def candidates(rgb, axis, fixed, lo, hi, sign=0, count=4, sigma=0.7, min_strength=2.0, min_separation=2.0):
    """Strongest gradient peaks in the open scan interval (lo, hi).

    sign: +1 keeps edges whose luminance increases with the index, -1 decreasing, 0 either.
    Returns [(coordinate, strength)] sorted by coordinate.
    """
    indices, magnitude, signed = gradient_profile(rgb, axis, fixed, lo - 2, hi + 2, sigma)
    peaks = []
    for k in local_maxima(magnitude):
        if not (lo < indices[k] < hi) or magnitude[k] < min_strength:
            continue
        if sign and np.sign(signed[k]) != sign:
            continue
        # boundary between samples: the derivative sampled at centres peaks at the boundary
        peaks.append((float(indices[k] + subpixel_peak(magnitude, k)), float(magnitude[k])))
    peaks.sort(key=lambda p: -p[1])
    chosen = []
    for p in peaks:
        if all(abs(p[0] - q[0]) >= min_separation for q in chosen):
            chosen.append(p)
        if len(chosen) == count:
            break
    return sorted(chosen)


def snap(rgb, axis, fixed, estimate, window=1.5, sigma=0.7, min_strength=2.0):
    """Gradient peak within +-window of `estimate`; None when no interior local maximum is strong enough."""
    lo, hi = int(np.floor(estimate - window)) - 1, int(np.ceil(estimate + window)) + 1
    indices, magnitude, _ = gradient_profile(rgb, axis, fixed, lo, hi, sigma)
    best = None
    for k in local_maxima(magnitude):
        position = float(indices[k] + subpixel_peak(magnitude, k))
        if abs(position - estimate) <= window and magnitude[k] >= min_strength:
            if best is None or magnitude[k] > best[1]:
                best = (position, float(magnitude[k]))
    return best
