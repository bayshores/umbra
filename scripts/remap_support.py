"""Rectangle selection from actual remap support; provenance belongs to callers."""
import numpy as np


def supported_rectangle(mask, crop_size, margin=1):
    """Return (x, y, width, height) of the best supported rectangle's interior.

    The outer rectangle must contain only True pixels. Remove ``margin`` from
    every edge, then require the interior to fit ``crop_size=(width, height)``.
    Maximize interior area; ties choose smallest top, left, height, then width.
    Uses O(mask.height * mask.width) time and O(mask.width) auxiliary space.

    ``mask`` must be a nonempty plain 2D boolean NumPy ndarray. Crop size is a
    tuple/list or 1D ndarray of two positive integers; margin is a nonnegative
    integer. Booleans are not accepted as integers. Invalid inputs or missing
    usable support raise ValueError. No pixels are changed or holes filled.
    """
    if type(mask) is not np.ndarray or mask.ndim != 2 or not mask.size or mask.dtype != np.dtype(bool):
        raise ValueError('mask must be a nonempty plain 2D boolean NumPy array')
    if (not isinstance(crop_size, (tuple, list, np.ndarray))
            or isinstance(crop_size, np.ndarray) and crop_size.ndim != 1
            or len(crop_size) != 2):
        raise ValueError('crop_size must contain two positive integers (width, height)')
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0
           for value in crop_size):
        raise ValueError('crop_size must contain two positive integers (width, height)')
    if isinstance(margin, (bool, np.bool_)) or not isinstance(margin, (int, np.integer)) or margin < 0:
        raise ValueError('margin must be a nonnegative integer')
    crop_width, crop_height = map(int, crop_size)
    margin = int(margin)
    heights = np.zeros(mask.shape[1], dtype=np.int64)
    best = None
    best_key = None
    for bottom, row in enumerate(mask):
        heights = np.where(row, heights + 1, 0)
        stack = []
        for right in range(mask.shape[1] + 1):
            height = int(heights[right]) if right < mask.shape[1] else 0
            left = right
            while stack and stack[-1][1] > height:
                left, outer_height = stack.pop()
                width = right - left - 2 * margin
                inner_height = outer_height - 2 * margin
                if width >= crop_width and inner_height >= crop_height:
                    x, y = left + margin, bottom - outer_height + 1 + margin
                    key = (-width * inner_height, y, x, inner_height, width)
                    if best_key is None or key < best_key:
                        best_key = key
                        best = (x, y, width, inner_height)
            if height and (not stack or stack[-1][1] < height):
                stack.append((left, height))
    if best is None:
        raise ValueError('no all-supported rectangle fits crop_size after removing margin')
    return best
