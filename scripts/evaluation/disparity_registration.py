"""Explicit integer-crop registration; numerical support is not physical confidence.

Callers must verify source/producer identities before calling this pure operation.
It performs no inference, file publication, resampling or manifest authentication.
"""
import hashlib
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rectified_calibration import parse_rectified_pair


CONVENTION = 'x_left_minus_x_right_in_input_pixels'


def register(disparity, validity, calibration_bytes, original_shape, crop, convention):
    """Return (full axial-Z float32, uint8 support, explicit derivation).

    Shape is [height,width]; crop is [x,y,width,height] in decoded image pixels.
    Both rectified inputs must have the same integer crop, no resize or padding.
    Supported dtype/geometry is intentionally narrow and never inferred.
    """
    if convention != CONVENTION:
        raise ValueError('Unsupported disparity convention')
    if (not isinstance(original_shape, (list, tuple)) or len(original_shape) != 2
            or any(type(n) is not int or n <= 0 for n in original_shape)):
        raise ValueError('Original shape must be positive integer [height,width]')
    if (not isinstance(crop, (list, tuple)) or len(crop) != 4
            or any(type(n) is not int for n in crop)):
        raise ValueError('Crop must be integer [x,y,width,height]')
    height, width = original_shape
    x, y, crop_width, crop_height = crop
    if (min(x, y) < 0 or min(crop_width, crop_height) <= 0
            or x + crop_width > width or y + crop_height > height):
        raise ValueError('Crop lies outside original grid')
    shape = (crop_height, crop_width)
    if (not isinstance(disparity, np.ndarray) or disparity.dtype != np.float32
            or disparity.shape != shape):
        raise ValueError('Disparity must be float32 with exact crop shape')
    if (not isinstance(validity, np.ndarray) or validity.dtype != np.uint8
            or validity.shape != shape or not np.isin(validity, (0, 1)).all()):
        raise ValueError('Validity must be binary uint8 with exact crop shape')
    if not isinstance(calibration_bytes, bytes):
        raise ValueError('Calibration must be captured bytes')
    left, right, source_size, scale = parse_rectified_pair(
        calibration_bytes.decode('utf-8'), width, height)
    numerator = float(left[0, 3] - right[0, 3])
    offset = float(left[0, 2] - right[0, 2])
    denominator = disparity.astype(np.float64) - offset
    supported = (validity == 1) & np.isfinite(denominator) & (denominator > 0)
    local = np.full(shape, np.nan, dtype=np.float32)
    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        local[supported] = (numerator / denominator[supported]).astype(np.float32)
    supported &= np.isfinite(local) & (local > 0)
    local[~supported] = np.nan
    full = np.full((height, width), np.nan, dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.uint8)
    selection = np.s_[y:y + crop_height, x:x + crop_width]
    full[selection] = local
    mask[selection] = supported
    derivation = dict(
        operation='integer crop embedding after explicit disparity-to-axial-Z conversion',
        originalDimensions=[height, width], converterInputDimensions=list(shape),
        registeredOutputDimensions=[height, width], crop=list(crop), resize=None, pad=None,
        disparityConvention=convention, depthConvention='axialZ',
        coordinateConvention='OpenCV-left-rectified-x-right-y-down-z-forward',
        calibrationSha256=hashlib.sha256(calibration_bytes).hexdigest(),
        sourceRectifiedDimensions=source_size.tolist(), sourceToDecodedScale=scale.tolist(),
        numeratorPxMeters=numerator, principalPointOffsetPixels=offset,
        formula='Z=(tx_left-tx_right)/(d-(cx_left-cx_right))',
        arithmetic='float64 division followed by float32 storage; unrepresentable depth unsupported',
        outsideCropPixels=height * width - crop_height * crop_width,
        numericallySupportedPixels=int(mask.sum()),
        limitations=['Calibration scale and axial-Z interpretation remain provisional.',
                    'Validity is numerical support, not confidence or free-space evidence.',
                    'No producer identity, execution or source-image registration authentication.'])
    return full, mask, derivation
