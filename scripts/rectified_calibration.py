"""Strict admission for the current horizontal, left-origin rectified adapter.

This validates supported matrix form, not physical calibration or same-row accuracy.
"""
from pathlib import Path
import numpy as np


def load_rectified_pair(path, width, height):
    return parse_rectified_pair(Path(path).read_text(), width, height)


def parse_rectified_pair(text, width, height):
    """Admit one caller-owned calibration text snapshot without reopening a file."""
    if any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n <= 0
           for n in (width, height)):
        raise ValueError('Image dimensions must be positive integers')
    required = {'S_rect_101': 2, 'S_rect_103': 2, 'P_rect_101': 12, 'P_rect_103': 12}
    values = {}
    for line in text.splitlines():
        if ':' not in line:
            continue
        key, text = line.split(':', 1)
        if key not in required:
            continue
        if key in values:
            raise ValueError(f'Duplicate calibration field: {key}')
        try:
            numbers = np.array([float(token) for token in text.split()], dtype=np.float64)
        except ValueError as exc:
            raise ValueError(f'Invalid numeric calibration field: {key}') from exc
        if numbers.size != required[key] or not np.isfinite(numbers).all():
            raise ValueError(f'Invalid shape or nonfinite calibration field: {key}')
        values[key] = numbers
    if values.keys() != required.keys():
        raise ValueError('Missing rectified projection or image-size field')
    size = values['S_rect_101']
    if (not np.array_equal(size, values['S_rect_103']) or
            (size <= 0).any() or not np.equal(size, np.floor(size)).all()):
        raise ValueError('Rectified source dimensions must match and be positive integers')
    projections = [values[key].reshape(3, 4).copy() for key in ('P_rect_101', 'P_rect_103')]
    for projection in projections:
        if (not np.array_equal(projection[2], [0, 0, 1, 0]) or
                np.any(projection[[0, 1, 1], [1, 0, 3]] != 0)):
            raise ValueError('Unsupported noncanonical horizontal rectified projection')
        if projection[0, 0] <= 0 or projection[1, 1] <= 0:
            raise ValueError('Rectified focal lengths must be finite and positive')
    left, right = projections
    if left[0, 3] != 0:
        raise ValueError('Current adapter requires a left-origin projection')
    if not np.allclose(left[[0, 1, 1], [0, 1, 2]], right[[0, 1, 1], [0, 1, 2]],
                       rtol=1e-9, atol=1e-9):
        raise ValueError('Rectified pair requires matching fx, fy and cy for same-row input')
    if left[0, 3] - right[0, 3] <= 0:
        raise ValueError('Projection must produce a positive stereo baseline')
    scale = np.array([width, height], dtype=np.float64) / size
    for projection in projections:
        projection[:2] *= scale[:, None]
        if (not np.isfinite(projection).all() or projection[0, 0] <= 0 or projection[1, 1] <= 0):
            raise ValueError('Scaled projection must have finite positive focal lengths')
    if projections[0][0, 3] - projections[1][0, 3] <= 0:
        raise ValueError('Scaled projection must retain a positive stereo baseline')
    return left, right, size.copy(), scale
