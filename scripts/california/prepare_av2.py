"""Offline native AV2 preparation; numerical support is never visibility gold."""
import argparse
from decimal import Decimal, localcontext
from io import BytesIO
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys
from uuid import UUID

import cv2
import numpy as np
from PIL import Image
import pyarrow.feather as feather
from scipy.spatial.transform import Rotation

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / 'sponsors'))
from remap_support import supported_rectangle
from common import canonical, digest, parse, publish, record, require, snapshot

ROLES = {'left', 'right', 'lidar', 'poses', 'intrinsics', 'extrinsics', 'map', 'groundRaster', 'groundTransform', 'license'}
MAX_FILE = 16_000_000
MAX_TOTAL = 64_000_000
MAX_DIMENSION = 32766
MAX_PIXELS = 16_000_000
LIMITS = ['All sweep rows retained; no depth raster, collision overwrite, filling or ground-based eligibility filter.',
          'Source-mask and mapped-corner support are numerical, not camera visibility, physical accuracy or free space.',
          'Publisher ground is nominal cityZ with float16 precision; missing/aliased samples remain NaN.',
          'Publisher XYZ is already ego-motion-compensated to the sweep reference; no second deskew is applied. Camera/sweep pose registration cannot compensate dynamic objects.',
          'Publisher-reported return timing is relative to the camera timestamp, not clock/exposure precision, dynamic-motion correction or camera-visible identity. Eligibility remains geometric crop support.',
          'Plan hashes/sourceClass identify snapshots; they do not confer dataset-use authorization.',
          'Development preparation only; no site/holdout/runtime admission, inference or sponsor eligibility.']


def integers(value, length, positive=False):
    require(isinstance(value, list) and len(value) == length and all(type(n) is int for n in value), 'Integer dimensions required')
    require(all(n > 0 if positive else n >= 0 for n in value), 'Invalid dimensions')
    return value


def publisher_reported_timing(camera_ts, sweep_ts, offsets, left_eligible, right_eligible):
    """Exact descriptive ages from publisher offsets; no timing admission rule."""
    require(type(camera_ts) is int and camera_ts >= 0 and type(sweep_ts) is int and sweep_ts >= 0,
            'Nonnegative integer camera/sweep timestamps required')
    require(isinstance(offsets, np.ndarray) and offsets.ndim == 1 and offsets.dtype.kind in 'iu',
            'One-dimensional integral per-return timestamp offsets required')
    for mask in (left_eligible, right_eligible):
        require(isinstance(mask, np.ndarray) and mask.dtype.kind == 'b' and mask.shape == offsets.shape,
                'Boolean eligibility masks must match per-return offsets')
    lag = sweep_ts - camera_ts
    bounds = np.iinfo(np.int64)
    if len(offsets):
        low, high = int(offsets.min()), int(offsets.max())
        require(0 <= low <= high <= bounds.max, 'Invalid per-return timestamp offsets')
        require(bounds.min <= lag + low and lag + high <= bounds.max, 'Per-return timestamp age exceeds int64')
    ages = [int(offset) + lag for offset in offsets]
    masks = {'allReturns': np.ones(offsets.shape, dtype=bool), 'leftEligible': left_eligible,
             'rightEligible': right_eligible, 'bothEligible': left_eligible & right_eligible}
    cohorts = {}
    with localcontext() as context:
        context.prec = 40  # int64 ages and exact linear median/p95 need at most 21 digits.
        for name, mask in masks.items():
            ordered = sorted(age for age, selected in zip(ages, mask) if selected)
            stats = {'count': len(ordered), 'minNs': str(ordered[0]) if ordered else None,
                     'maxNs': str(ordered[-1]) if ordered else None, 'medianNs': None, 'p95Ns': None,
                     'beforeCameraCount': sum(age < 0 for age in ordered),
                     'atCameraCount': sum(age == 0 for age in ordered),
                     'afterCameraCount': sum(age > 0 for age in ordered)}
            if ordered:
                for key, quantile in [('medianNs', Decimal('0.5')), ('p95Ns', Decimal('0.95'))]:
                    position = (len(ordered) - 1) * quantile; index = int(position)
                    low = ordered[index]; high = ordered[min(index + 1, len(ordered) - 1)]
                    value = Decimal(low) + (position - index) * (high - low)
                    stats[key] = format(value.normalize(), 'f')
            cohorts[name] = stats
    return {'cameraTimestampNs': str(camera_ts), 'sweepStartTimestampNs': str(sweep_ts),
            'sweepStartMinusCameraNs': str(lag),
            'basis': 'sweepStartTimestampNs + publisher offset_ns - cameraTimestampNs; cohort statistics are relative ages in ns',
            'cohorts': cohorts,
            'limits': 'Publisher-reported timing, not clock/exposure precision, dynamic-motion correction or camera-visible identity. Eligibility is geometric crop support. Source XYZ is already ego-compensated; no second transformation.'}


def capture(plan_path, output):
    plan_path = Path(plan_path).resolve()
    output = Path(output).absolute()
    require(not os.path.lexists(output), 'Existing output retained')
    plan_bytes = snapshot(plan_path)
    plan = parse(plan_bytes)
    require(plan.get('schemaVersion') == 1 and type(plan['schemaVersion']) is int and plan.get('sourceClass') == 'Argoverse2Sensor', 'Native AV2 plan required')
    log = plan.get('logId')
    require(isinstance(log, str) and str(UUID(log)) == log, 'Canonical log UUID required')
    for name in ('cameraTimestampNs', 'lidarTimestampNs', 'crossingId'):
        require(isinstance(plan.get(name), str) and re.fullmatch('[0-9]+', plan[name]), 'Decimal identity required')
    rect_size = integers(plan['rectifiedSize'], 2, True)
    require(max(rect_size) <= MAX_DIMENSION and rect_size[0] * rect_size[1] <= MAX_PIXELS, 'Rectified grid allocation ceiling')
    crop = integers(plan['crop'], 4)
    require(crop[2:] == [736, 320] and crop[0] + crop[2] <= rect_size[0] and crop[1] + crop[3] <= rect_size[1], 'Existing 736x320 crop contract required')
    sources = plan.get('sources')
    require(isinstance(sources, dict) and set(sources) == ROLES, 'Ten explicit source roles required')
    paths = {}; identities = set(); total = 0
    for role, spec in sources.items():
        require(isinstance(spec, dict) and isinstance(spec.get('path'), str) and spec['path'], 'Source path required')
        require(type(spec.get('bytes')) is int and 0 < spec['bytes'] <= MAX_FILE, 'Source byte ceiling')
        require(isinstance(spec.get('sha256'), str) and re.fullmatch('[a-f0-9]{64}', spec['sha256']), 'Explicit SHA256 required')
        path = (plan_path.parent / spec['path']).resolve()
        require(path != output.resolve() and not path.is_relative_to(output.resolve()), 'Output/raw alias')
        stat = path.stat(); identity = (stat.st_dev, stat.st_ino)
        require(identity not in identities and path not in paths.values(), 'Duplicate source alias')
        identities.add(identity); paths[role] = path; total += spec['bytes']
    require(total <= MAX_TOTAL, 'Cumulative source byte ceiling')
    for side in ('left', 'right'):
        require(paths[side].name == plan['cameraTimestampNs'] + '.jpg' and paths[side].parent.name == 'stereo_front_' + side, 'Stereo identity mismatch')
    require(paths['lidar'].name == plan['lidarTimestampNs'] + '.feather' and paths['lidar'].parent.name == 'lidar', 'Sweep identity mismatch')
    for role, name in [('poses', 'city_SE3_egovehicle.feather'), ('intrinsics', 'intrinsics.feather'), ('extrinsics', 'egovehicle_SE3_sensor.feather')]:
        require(paths[role].name == name, 'Metadata identity mismatch')
    require(re.fullmatch('log_map_archive_' + re.escape(log) + r'____[A-Z]{3}_city_[0-9]+(?:_from_[0-9]+_zvals)?\.json', paths['map'].name), 'Map identity mismatch')
    require(re.fullmatch(re.escape(log) + r'_ground_height_surface____[A-Z]{3}\.npy', paths['groundRaster'].name), 'Ground raster identity mismatch')
    require(paths['groundTransform'].name == log + '___img_Sim2_city.json', 'Ground transform identity mismatch')
    require(paths['map'].name.split('____')[1].split('_')[0] == paths['groundRaster'].name.split('____')[1].split('.')[0], 'Map/ground city mismatch')
    captured = {}
    for role, path in paths.items():
        captured[role] = snapshot(path, sources[role]['sha256'], MAX_FILE)
        require(len(captured[role]) == sources[role]['bytes'], 'Source byte identity mismatch')
    return plan, plan_bytes, paths, captured, output


def table(data, columns):
    value = feather.read_table(BytesIO(data), columns=None, use_threads=False)
    require(len(set(value.column_names)) == len(value.column_names) and set(columns) <= set(value.column_names), 'Missing/duplicate Feather columns')
    return value


def rigid(row):
    q = np.asarray([row[k] for k in ('qx', 'qy', 'qz', 'qw')], float)
    t = np.asarray([row[k] for k in ('tx_m', 'ty_m', 'tz_m')], float)
    require(np.isfinite(q).all() and np.isfinite(t).all() and abs(q @ q - 1) <= 1e-9, 'Invalid rigid pose')
    result = np.eye(4); result[:3, :3] = Rotation.from_quat(q).as_matrix(); result[:3, 3] = t
    return result


def inverse(value):
    result = np.eye(4); result[:3, :3] = value[:3, :3].T; result[:3, 3] = -value[:3, :3].T @ value[:3, 3]
    return result


def transform(points, matrix):
    with np.errstate(invalid='ignore', over='ignore'):
        return points @ matrix[:3, :3].T + matrix[:3, 3]


def ground_raster(data):
    require(len(data) <= MAX_FILE, 'Ground raster byte ceiling')
    stream = BytesIO(data); magic = np.lib.format.read_magic(stream)
    require(magic in ((1, 0), (2, 0)), 'Unsupported NPY version')
    reader = np.lib.format.read_array_header_1_0 if magic == (1, 0) else np.lib.format.read_array_header_2_0
    shape, _fortran, dtype = reader(stream, max_header_size=4096)
    require(len(shape) == 2 and all(type(n) is int and n > 0 for n in shape) and shape[0] * shape[1] <= MAX_FILE // 2, 'Ground raster shape ceiling')
    require(dtype.kind == 'f' and dtype.itemsize == 2 and len(data) == stream.tell() + shape[0] * shape[1] * 2, 'Exact float16 NPY payload required')
    raster = np.load(BytesIO(data), allow_pickle=False, max_header_size=4096)
    require(not np.isinf(raster).any(), 'Infinite nominal ground')
    return raster


def ground_sample(points, raster, sim):
    rotation = sim['R']
    require(isinstance(rotation, list) and len(rotation) == 4 and
            all(type(value) in (int, float) and np.isfinite(value) for value in rotation),
            'Publisher Sim2 R requires four finite flat numeric values')
    r = np.asarray(rotation, float).reshape(2, 2)
    t = np.asarray(sim['t'], float); s = sim['s']
    require(r.shape == (2, 2) and t.shape == (2,) and np.isfinite(r).all() and np.isfinite(t).all(), 'Invalid Sim2 dimensions')
    require(type(s) in (int, float) and np.isfinite(s) and s > 0 and np.allclose(r.T @ r, np.eye(2), rtol=0, atol=1e-9) and abs(np.linalg.det(r) - 1) < 1e-9, 'Proper positive Sim2 required')
    with np.errstate(invalid='ignore', over='ignore'):
        image_xy = s * (points[:, :2] @ r.T + t)
    castable = np.isfinite(image_xy).all(1) & (np.abs(image_xy) < np.iinfo(np.int64).max).all(1)
    indices = np.full(image_xy.shape, -1, np.int64); indices[castable] = np.trunc(image_xy[castable]).astype(np.int64)
    inside = castable & (indices[:, 0] >= 0) & (indices[:, 0] < raster.shape[1]) & (indices[:, 1] >= 0) & (indices[:, 1] < raster.shape[0])
    alias = inside & (image_xy < 0).any(1)
    heights = np.full(len(points), np.nan)
    domain = inside & ~alias
    heights[domain] = raster[indices[domain, 1], indices[domain, 0]]
    return {'nominalGroundHeights': heights, 'groundSupport': domain & np.isfinite(heights), 'groundBoundaryAlias': alias,
            'groundIntegerInBounds': inside, 'groundImageXY': image_xy, 'groundIndices': indices}


def png(rgb):
    stream = BytesIO(); Image.fromarray(rgb).save(stream, format='PNG'); return stream.getvalue()


def decode_native_map(data):
    """Preserve publisher NaN as sourceUnknown; consumed corners stay finite."""
    source_unknown_seen = False
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, 'Duplicate native map JSON key')
            value[key] = item
        return value
    def constant(token):
        nonlocal source_unknown_seen
        require(token == 'NaN', 'Infinite native map JSON value')
        source_unknown_seen = True
        return float('nan')
    def number(token):
        value = float(token)
        require(np.isfinite(value), 'Infinite native map JSON number')
        return value
    value = json.loads(data, object_pairs_hook=pairs, parse_constant=constant, parse_float=number)
    require(isinstance(value, dict), 'Native vector map object required')
    return value, source_unknown_seen


def footprint_membership(city_xyz, corners):
    """Convex XY interior or <=1e-9 m from finite segments, not physical uncertainty."""
    polygon = corners[:, :2]
    edges = np.roll(polygon, -1, axis=0) - polygon
    area = np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1) - polygon[:, 1] * np.roll(polygon[:, 0], -1))
    turns = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    require(abs(area) > 1e-9 and (np.all(turns > 0) or np.all(turns < 0)), 'Nonconvex/degenerate crossing footprint')
    finite = np.isfinite(city_xyz[:, :2]).all(1)
    result = np.zeros(len(city_xyz), dtype=bool)
    delta = city_xyz[finite, None, :2] - polygon[None, :, :]
    cross = edges[None, :, 0] * delta[:, :, 1] - edges[None, :, 1] * delta[:, :, 0]
    interior = np.all(cross >= 0, axis=1) | np.all(cross <= 0, axis=1)
    fraction = np.sum(delta * edges[None, :, :], axis=2) / np.sum(edges * edges, axis=1)
    nearest = delta - np.clip(fraction, 0, 1)[:, :, None] * edges[None, :, :]
    boundary = (np.linalg.norm(nearest, axis=2) <= 1e-9).any(1)
    result[finite] = interior | boundary
    return result


def prepare(plan_path, output):
    plan, plan_bytes, paths, raw, output = capture(plan_path, output)
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False)
    pose_columns = ['qx', 'qy', 'qz', 'qw', 'tx_m', 'ty_m', 'tz_m']
    pose_rows = table(raw['poses'], ['timestamp_ns'] + pose_columns).to_pylist()
    poses = {}
    for row in pose_rows:
        timestamp = row['timestamp_ns']
        require(type(timestamp) is int and timestamp not in poses, 'Invalid/duplicate pose timestamp')
        poses[timestamp] = rigid(row)
    camera_ts, lidar_ts = int(plan['cameraTimestampNs']), int(plan['lidarTimestampNs'])
    require(camera_ts in poses and lidar_ts in poses, 'Exact camera/sweep pose missing')
    sensor_tables = {}
    for role, columns in [('intrinsics', ['sensor_name', 'fx_px', 'fy_px', 'cx_px', 'cy_px', 'width_px', 'height_px']), ('extrinsics', ['sensor_name'] + pose_columns)]:
        rows = table(raw[role], columns).to_pylist(); sensors = {}
        for row in rows:
            require(isinstance(row['sensor_name'], str) and row['sensor_name'] and row['sensor_name'] not in sensors, 'Missing/duplicate sensor')
            if role == 'extrinsics':
                rigid(row)
            else:
                require(np.isfinite([row[k] for k in ('fx_px', 'fy_px', 'cx_px', 'cy_px')]).all(), 'Nonfinite intrinsic sensor row')
            sensors[row['sensor_name']] = row
        require(all('stereo_front_' + side in sensors for side in ('left', 'right')), 'Exact stereo sensor missing')
        sensor_tables[role] = sensors
    ks = []; extr = []; images = []; sizes = []
    for side in ('left', 'right'):
        name = 'stereo_front_' + side; row = sensor_tables['intrinsics'][name]
        vals = np.array([row[k] for k in ('fx_px', 'fy_px', 'cx_px', 'cy_px')], float)
        require(np.isfinite(vals).all() and (vals[:2] > 0).all(), 'Nonfinite/invalid camera intrinsics')
        size = [row['width_px'], row['height_px']]; integers(size, 2, True); sizes.append(size)
        require(max(size) <= MAX_DIMENSION and size[0] * size[1] <= MAX_PIXELS, 'Decoded grid allocation ceiling')
        ks.append(np.array([[vals[0], 0, vals[2]], [0, vals[1], vals[3]], [0, 0, 1.]]))
        extr.append(rigid(sensor_tables['extrinsics'][name]))
        try:
            with Image.open(BytesIO(raw[side])) as image:
                require(image.format == 'JPEG' and image.mode == 'RGB' and list(image.size) == size, 'Native RGB image/calibration mismatch')
                image.load(); images.append(np.array(image))
        except Image.DecompressionBombError as error:
            raise ValueError('Decoded image allocation ceiling') from error
    require(sizes[0] == sizes[1], 'Stereo dimensions mismatch')
    relative = inverse(extr[1]) @ extr[0]
    require(np.linalg.norm(relative[:3, 3]) > 0, 'Zero stereo baseline')
    rect_size = tuple(plan['rectifiedSize']); zero_d = np.zeros(5)
    r1, r2, p1, p2, q, _roi1, _roi2 = cv2.stereoRectify(ks[0], zero_d, ks[1], zero_d, tuple(sizes[0]), relative[:3, :3], relative[:3, 3], flags=cv2.CALIB_ZERO_DISPARITY, alpha=0, newImageSize=rect_size)
    require(all(np.isfinite(v).all() for v in (r1, r2, p1, p2, q)), 'Nonfinite rectification')
    maps = []; masks = []
    for k, rotation, projection in zip(ks, (r1, r2), (p1, p2)):
        mx, my = cv2.initUndistortRectifyMap(k, zero_d, rotation, projection, rect_size, cv2.CV_32FC1)
        strict = np.isfinite(mx) & np.isfinite(my) & (mx >= 0) & (my >= 0) & (mx < sizes[0][0] - 1) & (my < sizes[0][1] - 1)
        endpoints = np.isfinite(mx) & np.isfinite(my) & (np.floor(mx) >= 0) & (np.floor(my) >= 0) & (np.ceil(mx) < sizes[0][0]) & (np.ceil(my) < sizes[0][1])
        maps.append((mx, my)); masks.append(strict & endpoints)
    common_mask = masks[0] & masks[1]
    supported = supported_rectangle(common_mask, plan['crop'][2:], margin=1)
    x, y, width, height = plan['crop']; sx, sy, sw, sh = supported
    require(x >= sx and y >= sy and x + width <= sx + sw and y + height <= sy + sh and common_mask[y:y + height, x:x + width].all(), 'Crop outside common source support')
    a = np.array([[1., 0, -x], [0, 1., -y], [0, 0, 1.]])
    vector_map, map_source_unknown = decode_native_map(raw['map'])
    crossing = vector_map['pedestrian_crossings'][plan['crossingId']]
    require(str(crossing['id']) == plan['crossingId'], 'Crossing identity mismatch')
    corners = np.asarray([[p[k] for k in ('x', 'y', 'z')] for p in crossing['edge1'] + list(reversed(crossing['edge2']))], float)
    require(corners.shape == (4, 3) and np.isfinite(corners).all(), 'Four finite mapped crossing corners required')
    city_from_camera = [poses[camera_ts] @ camera_extr for camera_extr in extr]
    corner_data = {}
    for side, city_pose, rotation, projection in zip(('left', 'right'), city_from_camera, (r1, r2), (p1, p2)):
        xyz = transform(corners, inverse(city_pose)) @ rotation.T
        projected = xyz @ projection[:, :3].T
        with np.errstate(invalid='ignore', divide='ignore'):
            uv = projected[:, :2] / projected[:, 2:3] - [x, y]
        require(np.isfinite(xyz).all(), 'Nonfinite mapped corner camera coordinates')
        supported_corners = (xyz[:, 2] > 0) & np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1)
        corner_data[side] = {'rectifiedXYZ': xyz.tolist(),
                             'cropUV': [[float(v) if np.isfinite(v) else None for v in row] for row in uv],
                             'cropSupported': supported_corners.tolist()}
    sweep = table(raw['lidar'], ['x', 'y', 'z', 'offset_ns'])
    ego_xyz = np.column_stack([np.asarray(sweep[k]) for k in ('x', 'y', 'z')]).astype(float)
    offsets = np.asarray(sweep['offset_ns'])
    city_xyz = transform(ego_xyz, poses[lidar_ts])  # Already ego-compensated: no LiDAR extrinsic or deskew.
    arrays = {'sourceIndex': np.arange(len(ego_xyz), dtype=np.int64), 'egoXYZ': ego_xyz, 'cityXYZ': city_xyz, 'offsetNs': offsets}
    for side, city_pose, rotation, projection in zip(('left', 'right'), city_from_camera, (r1, r2), (p1, p2)):
        xyz = transform(city_xyz, inverse(city_pose)) @ rotation.T
        with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
            projected = xyz @ projection[:, :3].T; uv = projected[:, :2] / projected[:, 2:3] - [x, y]
        uv[~np.isfinite(xyz).all(1) | (xyz[:, 2] <= 0)] = np.nan
        eligible = np.isfinite(xyz).all(1) & (xyz[:, 2] > 0) & np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1)
        arrays[side + 'RectifiedXYZ'] = xyz; arrays[side + 'CropUV'] = uv; arrays[side + 'Eligible'] = eligible
    arrays['bothEligible'] = arrays['leftEligible'] & arrays['rightEligible']
    timing = publisher_reported_timing(camera_ts, lidar_ts, offsets, arrays['leftEligible'], arrays['rightEligible'])
    timing['cameraTimestampNs'] = plan['cameraTimestampNs']
    timing['sweepStartTimestampNs'] = plan['lidarTimestampNs']
    arrays['insideFootprint'] = footprint_membership(city_xyz, corners)
    raster = ground_raster(raw['groundRaster']); sim = parse(raw['groundTransform'])
    arrays.update(ground_sample(city_xyz, raster, sim))
    camera_origin = city_from_camera[0][:3, 3]
    queries = np.vstack((corners, camera_origin))
    target_ground = ground_sample(queries, raster, sim)
    targets = [{'role': 'mappedCorner' if i < 4 else 'leftCameraOrigin', 'cornerIndex': i if i < 4 else None,
                'cityXYZ': point.tolist(), 'nominalCityZ': float(target_ground['nominalGroundHeights'][i]) if target_ground['groundSupport'][i] else None,
                'groundSupport': bool(target_ground['groundSupport'][i]), 'groundBoundaryAlias': bool(target_ground['groundBoundaryAlias'][i])}
               for i, point in enumerate(queries)]
    rgb = [cv2.remap(image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[y:y + height, x:x + width]
           for image, (mx, my) in zip(images, maps)]
    crop_spec = {'x': x, 'y': y, 'width': width, 'height': height, 'source_to_crop_matrix': a.tolist(), 'pixel_centers': 'integer inclusive0..width-1,0..height-1'}
    calibration = {'sourceClass': 'Argoverse2Sensor', 'decoded_source_size': list(rect_size), 'crop_size': [width, height],
                   'source_rectified_size': list(rect_size), 'source_to_decoded_scale': [1., 1.],
                   'original': {'left_projection': p1.tolist(), 'right_projection': p2.tolist()},
                   'transformed': {'left_projection': (a @ p1).tolist(), 'right_projection': (a @ p2).tolist()},
                   'disparity_numerator_px_m': float(p1[0, 3] - p2[0, 3]), 'principal_point_offset_px': float(p1[0, 2] - p2[0, 2]),
                   'originalFieldScope': 'New rectified grid before crop; not native raw camera projections', 'provisional': True}
    references = BytesIO(); np.savez_compressed(references, **arrays)
    source_records = {role: record(paths[role], data, output) for role, data in raw.items()}
    code_paths = [Path(__file__), SCRIPTS / 'remap_support.py', SCRIPTS / 'sponsors/common.py', SCRIPTS / 'prepare-stereo-crop.py']
    processing = {'alpha': 0, 'additionalDistortion': [0.] * 5, 'rawSize': sizes[0], 'rectifiedSize': list(rect_size),
                  'KLeft': ks[0].tolist(), 'KRight': ks[1].tolist(), 'R1': r1.tolist(), 'R2': r2.tolist(), 'P1': p1.tolist(), 'P2': p2.tolist(),
                  'rightCameraFromLeftCamera': relative.tolist(), 'commonSupportedRectangleMargin1': list(supported),
                  'cityFromEgoCamera': poses[camera_ts].tolist(), 'cityFromEgoSweep': poses[lidar_ts].tolist(),
                  'egoFromCameraLeft': extr[0].tolist(), 'egoFromCameraRight': extr[1].tolist(), 'leftCameraCityOrigin': camera_origin.tolist(),
                  'lidarMinusCameraNs': str(lidar_ts - camera_ts), 'mappedCrossing': corner_data, 'nominalTargetQueries': targets,
                  'publisherReportedTiming': timing,
                  'mappedCrossingScope': 'All four nominal corners retained; cropSupported is numerical per-view support, not whole-crossing, ground or physical admission. Undefined projections are null.',
                  'ground': {'policy': 's*(R*cityXY+t); SDK truncation [y,x], reject negative continuous-domain aliases; NaNs retained', 'shape': list(raster.shape), 'Sim2': sim},
                  'sourceRows': len(ego_xyz), 'eligibleLeft': int(arrays['leftEligible'].sum()), 'eligibleRight': int(arrays['rightEligible'].sum()),
                  'eligibleBoth': int(arrays['bothEligible'].sum()), 'insideFootprint': int(arrays['insideFootprint'].sum()),
                  'groundFinite': int(arrays['groundSupport'].sum()), 'groundBoundaryAlias': int(arrays['groundBoundaryAlias'].sum()),
                  'eligibilityPolicy': 'finite positive rectified XYZ, inclusive crop pixel centers; no ground filtering',
                  'footprintPolicy': 'Convex cityXY interior OR Euclidean distance to finite segments <=1e-9 m; numerical boundary policy, not physical uncertainty',
                  'mapSourceUnknownSeen': map_source_unknown,
                  'mapUnknownPolicy': 'Publisher NaN preserved as sourceUnknown without substitution; unused lanes not consumed; selected crossing corners require finite XYZ; infinities rejected',
                  'sourceHashes': {k: digest(v) for k, v in raw.items()}, 'codeHashes': {str(p.relative_to(SCRIPTS)): digest(p.read_bytes()) for p in code_paths},
                  'dependencies': {name: version(name) for name in ('numpy', 'Pillow', 'opencv-python', 'pyarrow', 'scipy')}}
    files = {'left.png': png(rgb[0]), 'right.png': png(rgb[1]), 'calibration.json': canonical(calibration),
             'references.npz': references.getvalue(), 'LICENSE-AND-ATTRIBUTION.md': raw['license']}
    manifest = {'schemaVersion': 1, 'sourceClass': 'Argoverse2Sensor', 'dataset': 'Argoverse2Sensor', 'logId': plan['logId'],
                'cameraTimestampNs': plan['cameraTimestampNs'], 'lidarTimestampNs': plan['lidarTimestampNs'], 'crossingId': plan['crossingId'],
                'plan': record(plan_path, plan_bytes, output), 'sources': source_records, 'crop': crop_spec, 'processing': processing,
                'outputs': {name: {'bytes': len(data), 'sha256': digest(data)} for name, data in files.items()},
                'limits': LIMITS, 'readLimits': {'perFileBytes': MAX_FILE, 'totalBytes': MAX_TOTAL, 'capturedOnce': True,
                                                'maxDimension': MAX_DIMENSION, 'maxDecodedPixels': MAX_PIXELS,
                                                'maxRectifiedPixels': MAX_PIXELS, 'maxGroundPixels': MAX_FILE // 2,
                                                'maxGroundNpyHeaderBytes': 4096},
                'referenceDepthUsedAsPrediction': False, 'siteAdmission': False, 'runtimeAdopted': False}
    files['manifest.json'] = canonical(manifest)
    publish(output, files)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--plan', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        result = prepare(args.plan, args.output)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f'AV2 preparation rejected: {error}\n')
    print(canonical({'sourceRows': result['processing']['sourceRows'], 'output': str(args.output), 'modelCalls': 0}).decode())


if __name__ == '__main__':
    main()
