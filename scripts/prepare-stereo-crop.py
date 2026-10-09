#!/usr/bin/env python3
"""Prepare identical integer stereo/depth crops without resizing or filling.

Depth decoding and axial-Z interpretation are explicit caller assumptions.
This prepares image coordinates, not a model tensor or accuracy measurement.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image, __version__ as pillow_version
import rectified_calibration


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish_new_directory(source, destination):
    # Native exclusive rename prevents even an empty concurrent output being replaced.
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == 'darwin':
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        result = rename(os.fsencode(source), os.fsencode(destination), 4)  # RENAME_EXCL
    elif sys.platform.startswith('linux') and hasattr(libc, 'renameat2'):
        rename = libc.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
                           ctypes.c_uint]
        result = rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    else:
        raise ValueError('Exclusive directory publication requires macOS or Linux rename support')
    if result:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))


def prepare(args):
    x, y, width, height = args.crop
    if (any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or
            n > np.iinfo(np.int32).max for n in args.crop) or
            min(x, y) < 0 or min(width, height) <= 0):
        raise ValueError('Crop requires nonnegative integer offsets and positive integer sizes')
    if not np.isfinite(args.depth_divisor) or args.depth_divisor <= 0:
        raise ValueError('Depth divisor must be finite and positive')
    output = Path(os.path.abspath(args.output))
    if os.path.lexists(output):
        raise ValueError('Existing output retained; choose a new output directory')
    paths = {name: Path(getattr(args, name)).resolve()
             for name in ('left', 'right', 'depth', 'calibration')}
    source_hashes = {name: sha256(path) for name, path in paths.items()}
    images = {}
    for name in ('left', 'right', 'depth'):
        data = paths[name].read_bytes()
        if hashlib.sha256(data).hexdigest() != source_hashes[name]:
            raise ValueError('Source changed during preparation')
        with Image.open(BytesIO(data)) as image:
            image.load()
            if name != 'depth' and image.mode != 'RGB':
                raise ValueError('Stereo source images must decode as RGB')
            if name == 'depth' and (image.format != 'PNG' or image.mode != 'I;16'):
                raise ValueError('Depth source must be a uint16 grayscale PNG')
            images[name] = np.array(image)
    left_rgb, right_rgb, raw = (images[name] for name in ('left', 'right', 'depth'))
    if (left_rgb.shape != right_rgb.shape or raw.ndim != 2 or raw.dtype != np.uint16 or
            raw.shape != left_rgb.shape[:2]):
        raise ValueError('Stereo/depth dimensions or depth encoding do not match')
    source_height, source_width = raw.shape
    if x + width > source_width or y + height > source_height:
        raise ValueError('Crop lies outside the original decoded image')
    calibration_bytes = paths['calibration'].read_bytes()
    if hashlib.sha256(calibration_bytes).hexdigest() != source_hashes['calibration']:
        raise ValueError('Source changed during preparation')
    left, right, size, scale = rectified_calibration.parse_rectified_pair(
        calibration_bytes.decode('utf-8'), source_width, source_height)
    transform = np.array([[1., 0., -x], [0., 1., -y], [0., 0., 1.]])
    with np.errstate(over='raise', divide='raise', invalid='raise', under='raise'):
        cropped_left, cropped_right = transform @ left, transform @ right
        numerator = left[0, 3] - right[0, 3]
        offset = left[0, 2] - right[0, 2]
        baseline = numerator / left[0, 0]
        if not np.isfinite([numerator, offset, baseline]).all() or baseline <= 0:
            raise ValueError('Stereo disparity parameters must be finite with positive baseline')
        selection = np.s_[y:y + height, x:x + width]
        cropped_raw = raw[selection]
        supported = cropped_raw > 0
        expected = np.full(cropped_raw.shape, np.nan, dtype=np.float64)
        depth = cropped_raw[supported].astype(np.float64) / args.depth_divisor
        expected[supported] = offset + numerator / depth
        if not np.isfinite(expected[supported]).all():
            raise ValueError('Decoded supported depth/disparity must remain finite')
        columns = np.nonzero(supported)[1]
        right_u = columns - expected[supported]
    outside = (right_u < 0) | (right_u > width - 1)
    calibration = {
        'source_rectified_size': size.tolist(), 'decoded_source_size': [source_width, source_height],
        'source_to_decoded_scale': scale.tolist(), 'crop_size': [width, height],
        'original': {'left_projection': left.tolist(), 'right_projection': right.tolist()},
        'transformed': {'left_projection': cropped_left.tolist(),
                        'right_projection': cropped_right.tolist()},
        'disparity_numerator_px_m': float(numerator), 'principal_point_offset_px': float(offset),
        'implied_baseline_m': float(baseline), 'provisional': True,
    }
    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'sources': {name: {'path': str(path), 'sha256': source_hashes[name]}
                    for name, path in paths.items()},
        'processing': {'script_sha256': sha256(Path(__file__)),
                       'calibration_helper_sha256': sha256(Path(rectified_calibration.__file__)),
                       'numpy_version': np.__version__, 'pillow_version': pillow_version},
        'crop': {'x': x, 'y': y, 'width': width, 'height': height,
                 'source_to_crop_matrix': transform.tolist(),
                 'mapping': 'u_crop = u_source - x; v_crop = v_source - y',
                 'pixel_centers': 'integer coordinates; inclusive domain 0..width-1, 0..height-1'},
        'depth': {'raw_dtype': 'uint16', 'undefined_raw': 0,
                  'divisor': args.depth_divisor, 'caller_assumption': 'axial Z in meters = raw / divisor',
                  'unsupported_policy': 'raw zero preserved; expected disparity remains NaN'},
        'support': {'source_nonzero_depth': int(np.count_nonzero(raw)),
                    'crop_nonzero_depth': int(supported.sum()),
                    'crop_zero_depth': int((~supported).sum()),
                    'projected_right_u_outside_crop': int(outside.sum()),
                    'projected_right_u_inside_crop': int((~outside).sum()),
                    'projected_right_u_within_one_pixel_of_crop_border': int(np.count_nonzero(
                        (right_u >= 0) & (right_u <= width - 1) &
                        ((right_u < 1) | (right_u > width - 2)))),
                    'outside_samples_retained': True,
                    'interpretation': 'projected location only; not confirmed correspondence or visibility'},
        'expected_disparity_formula': 'cx_left - cx_right + (tx_left - tx_right) / (raw / divisor)',
        'limitations': 'No resize, padding, hole filling, model tensor contract, inference or accuracy claim.',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.stereo-crop-', dir=output.parent) as temporary:
        staged = Path(temporary) / 'result'
        staged.mkdir()
        for name in ('left', 'right', 'depth'):
            Image.fromarray(images[name][selection]).save(staged / f'{name}.png')
        (staged / 'calibration.json').write_text(json.dumps(calibration, indent=2, allow_nan=False) + '\n')
        if args.expected_disparity:
            np.save(staged / 'expected-disparity.npy', expected, allow_pickle=False)
        manifest['outputs'] = {path.name: {'sha256': sha256(path), 'bytes': path.stat().st_size}
                               for path in sorted(staged.iterdir())}
        manifest['output_hash_scope'] = 'all artifacts except this manifest'
        (staged / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        if any(sha256(path) != source_hashes[name] for name, path in paths.items()):
            raise ValueError('Source changed during preparation')
        publish_new_directory(staged, output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('left', 'right', 'depth', 'calibration', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--crop', type=int, nargs=4, metavar=('X', 'Y', 'WIDTH', 'HEIGHT'), required=True)
    parser.add_argument('--depth-divisor', type=float, required=True)
    parser.add_argument('--expected-disparity', action='store_true',
                        help='Save geometric expected disparity; requires the declared axial-Z assumption')
    args = parser.parse_args()
    try:
        manifest = prepare(args)
    except (ValueError, OSError, FloatingPointError) as exc:
        parser.exit(1, f'Crop preparation rejected: {exc}\n')
    print(json.dumps({'output': str(args.output), 'support': manifest['support']}, allow_nan=False))


if __name__ == '__main__':
    main()
