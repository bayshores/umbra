"""Prepare a measured scene. Missing/corrupt inputs or an unreliable plane stop processing."""
import argparse
from contextlib import ExitStack
import hashlib
from io import BytesIO
import json
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import rectified_calibration
import scene_surface

# These are observed source-file identities, not authentication of loaded bytecode.
code_paths = {'preparation': Path(__file__).resolve(),
              'calibration': Path(rectified_calibration.__file__).resolve(),
              'surface': Path(scene_surface.__file__).resolve()}
code_hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in code_paths.items()}

parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=Path, required=True)
parser.add_argument('--frame', required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--road-tolerance', type=float, required=True)
parser.add_argument('--max-depth', type=float, required=True)
parser.add_argument('--mesh-edge-limit', type=float, required=True)
args = parser.parse_args()
if any(not np.isfinite(n) or n <= 0 for n in (args.road_tolerance, args.max_depth, args.mesh_edge_limit)):
    raise ValueError('Processing tolerances must be finite and positive')
root = args.data_root.resolve()
frame = args.frame
if not frame or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in frame):
    raise ValueError('Invalid frame identifier')
parts = frame.split('_')
if len(parts) != 2 or not all(parts):
    raise ValueError('Expected a DrivingStereo sequence_frame identifier')
sequence = parts[0]
paths = {
    'left': root / 'extracted/left-image-half-size' / (frame + '.jpg'),
    'right': root / 'extracted/right-image-half-size' / (frame + '.jpg'),
    'depth': root / 'extracted/depth-map-half-size' / (frame + '.png'),
    'calibration': root / 'calibration' / (sequence + '-full-reference.txt'),
}
for path in (*paths.values(), root / 'LICENSE'):
    if not path.is_file():
        raise ValueError(f'Required source artifact unavailable: {path}')
destination = args.output.resolve() / frame
if destination.exists():
    raise ValueError('Existing scene retained; prepare into a new output directory')
entries = []
for peer_path in sorted(args.output.glob('*/scene.json')):
    if peer_path.parent.name.startswith('.'):
        continue
    try:
        peer = json.loads(peer_path.read_text())
        if (not isinstance(peer, dict) or peer.get('id') != peer_path.parent.name or
                not isinstance(peer.get('label'), str) or not peer['label']):
            raise ValueError('Invalid scene identity or label')
    except (ValueError, OSError) as exc:
        raise ValueError(f'Invalid existing scene catalog metadata: {peer_path}') from exc
    entries.append({'id': peer['id'], 'label': peer['label']})
entries.append({'id': frame, 'label': parts[1]})
entries.sort(key=lambda entry: entry['id'])
catalog_text = json.dumps(entries, indent=2, allow_nan=False) + '\n'
# Decode, hash and publish the same input bytes even if a source file changes mid-run.
source_bytes = {key: path.read_bytes() for key, path in {**paths, 'license': root / 'LICENSE'}.items()}
rgb = np.asarray(Image.open(BytesIO(source_bytes['left'])).convert('RGB'))
right_size = Image.open(BytesIO(source_bytes['right'])).size
raw = np.asarray(Image.open(BytesIO(source_bytes['depth'])))
height, width = raw.shape
if rgb.shape[:2] != raw.shape or right_size != (width, height) or raw.dtype != np.uint16:
    raise ValueError('Stereo/depth dimensions or depth encoding do not match')
projection, _, _, _ = rectified_calibration.parse_rectified_pair(source_bytes['calibration'].decode('utf-8'), width, height)
fx, fy, cx, cy = projection[0, 0], projection[1, 1], projection[0, 2], projection[1, 2]
depth = raw.astype(float) / 256
valid = (raw > 0) & (depth <= args.max_depth)
v, u = np.indices(raw.shape)
camera_points = np.stack(((u-cx)*depth/fx, (v-cy)*depth/fy, depth), axis=-1)

# Fit the dominant near-road plane to measured points in the lower central image.
# This ROI is a declared selection heuristic; no plane is supplied on failure.
roi = valid & (u > width*.3) & (u < width*.7) & (v > height*.6) & (v < height*.95)
road = camera_points[roi]
if len(road) < 100:
    raise ValueError('Insufficient measured road support')
rng = np.random.default_rng(0)
sample = road[rng.choice(len(road), min(4000, len(road)), replace=False)]
best = None
for _ in range(600):
    a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
    normal = np.cross(b-a, c-a)
    norm = np.linalg.norm(normal)
    if norm < 1e-10:
        continue
    normal /= norm
    if normal[1] < 0:
        normal = -normal
    if normal[1] < np.cos(np.deg2rad(20)):
        continue
    offset = -normal @ a
    mask = np.abs(sample @ normal + offset) <= args.road_tolerance
    count = int(mask.sum())
    if best is None or count > best[0]:
        best = (count, mask)
if best is None or best[0] / len(sample) < .7:
    raise ValueError('Measured road-plane support failed the 70% gate')
inliers = sample[best[1]]
center = inliers.mean(axis=0)
_, singular_values, basis = np.linalg.svd(inliers-center, full_matrices=False)
if singular_values[1] <= np.finfo(float).eps * max(inliers.shape) * singular_values[0]:
    raise ValueError('Road-plane support is numerically collinear')
normal = basis[-1]
if normal[1] < 0:
    normal = -normal
if not np.isfinite(normal).all() or normal[1] < np.cos(np.deg2rad(20)):
    raise ValueError('Refitted road-plane tilt exceeds the supported limit')
offset = float(-normal @ center)
residuals = np.abs(road @ normal + offset)
if not np.isfinite(residuals).all() or (residuals <= args.road_tolerance).mean() < .7:
    raise ValueError('Refitted road-plane support failed the 70% gate')
height_estimate = -offset
if not 0 < height_estimate < np.quantile(road[:, 2], .1):
    raise ValueError('Fitted camera-to-road relationship is implausible')
forward = np.array([0., 0., 1.])
forward -= normal * (forward @ normal)
forward /= np.linalg.norm(forward)
right = np.cross(normal, forward)
rotation = np.stack((right, -normal, -forward))
origin = -offset * normal
points, linear, triangles = scene_surface.build_surface(camera_points, valid, rgb, origin, rotation, args.mesh_edge_limit)
args.output.mkdir(parents=True, exist_ok=True)
# Build a complete new scene before exposing it to the catalog; never replace an existing scene.
with tempfile.TemporaryDirectory(prefix='.preparing-', dir=args.output.resolve()) as staging:
    out = Path(staging)
    points.astype('<f4').tofile(out / 'positions.bin')
    linear.astype('<f4').tofile(out / 'colors.bin')
    triangles.tofile(out / 'triangles.bin')
    supported_depth = np.where(valid, raw, 0).astype('<u2')
    supported_depth.tofile(out / 'depth.bin')
    (out / 'recorded.jpg').write_bytes(source_bytes['left'])
    (out / 'right.jpg').write_bytes(source_bytes['right'])
    mask = np.zeros((height, width, 4), dtype=np.uint8)
    mask[~valid] = [17, 23, 24, 185]
    Image.fromarray(mask).save(out / 'support.png')
    metadata = {
        'id': frame, 'label': frame.split('_')[1], 'sourceClass': 'DrivingStereo measured stereo and sparse filtered LiDAR reference',
        'source': 'https://drivingstereo-dataset.github.io/', 'units': 'meters',
        'sourceHashes': {key: hashlib.sha256(payload).hexdigest() for key, payload in source_bytes.items()},
        'processingCodeHashes': code_hashes,
        'camera': {'eye': ((-origin) @ rotation.T).tolist(), 'forward': (np.array([0., 0., 1.]) @ rotation.T).tolist(),
                   'up': (np.array([0., -1., 0.]) @ rotation.T).tolist(), 'fx': float(fx), 'fy': float(fy), 'cx': float(cx), 'cy': float(cy),
                   'width': width, 'height': height, 'near': float(depth[valid].min()/10), 'far': float(args.max_depth)},
        'transform': {'cameraOriginGround': origin.tolist(), 'worldFromCamera': rotation.tolist()},
        'ground': {'normal': [0, 1, 0], 'cameraNormal': normal.tolist(), 'cameraOffset': offset,
                   'medianResidualMeters': float(np.median(residuals)), 'inlierFraction': float((residuals <= args.road_tolerance).mean()),
                   'method': 'RANSAC/SVD estimate from lower central supported reference points', 'selectionROI': [.3, .7, .6, .95]},
        'bounds': {'min': points.min(axis=0).tolist(), 'max': points.max(axis=0).tolist()},
        'processing': {'roadToleranceMeters': args.road_tolerance, 'maxDepthMeters': args.max_depth, 'meshEdgeLimitMeters': args.mesh_edge_limit,
                       'depthDivisor': 256, 'calibrationConvention': 'Full-resolution rectified projection scaled to actual image dimensions',
                       'randomSeed': 0, 'maxPlaneTiltDegrees': 20, 'minimumPlaneInlierFraction': .7},
        'coverage': float(valid.mean()), 'pointCount': len(points), 'triangleCount': len(triangles),
        'buffers': {'positions': 'positions.bin', 'colors': 'colors.bin', 'triangles': 'triangles.bin', 'depth': 'depth.bin'},
        'artifactHashes': {key: hashlib.sha256((out / name).read_bytes()).hexdigest() for key, name in
                          {'positions': 'positions.bin', 'colors': 'colors.bin', 'triangles': 'triangles.bin', 'depth': 'depth.bin'}.items()},
        'limitations': ['Sparse reference; zero, clipped depth and holes remain unsupported.', 'Ground plane is estimated, not surveyed.',
                        'Metric scale uses scaled full-resolution calibration; one-frame correspondence validation is provisional.',
                        'New camera views expose unknown surfaces; no image completion is performed.', 'Actor placements and dimensions are user-defined what-if inputs.'],
    }
    (out / 'scene.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    (out / 'LICENSE').write_bytes(source_bytes['license'])
    # Stage publication files before exposing the new scene.
    with ExitStack() as temporary_files:
        catalog = args.output / 'index.json'
        with tempfile.NamedTemporaryFile(mode='w', dir=args.output, prefix='.catalog-', delete=False) as staged_catalog:
            staged_catalog_path = Path(staged_catalog.name)
            temporary_files.callback(staged_catalog_path.unlink, missing_ok=True)
            staged_catalog.write(catalog_text)
        staged_license_path = None
        shared_license = args.output / 'LICENSE'
        if not shared_license.exists():
            with tempfile.NamedTemporaryFile(mode='wb', dir=args.output, prefix='.license-', delete=False) as staged_license:
                staged_license_path = Path(staged_license.name)
                temporary_files.callback(staged_license_path.unlink, missing_ok=True)
                staged_license.write(source_bytes['license'])
        # Replacing the catalog consumes our staged path, including when interrupted after rename.
        def rollback_scene():
            if staged_catalog_path.exists() and destination.exists():
                destination.rename(out)

        def rollback_license():
            if staged_catalog_path.exists() and not staged_license_path.exists():
                shared_license.unlink(missing_ok=True)

        # Register ownership-guarded rollback before each filesystem mutation.
        try:
            current_code_hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest()
                                   for key, path in code_paths.items()}
        except OSError as exc:
            raise ValueError('Processing code changed during preparation or became unreadable') from exc
        if current_code_hashes != code_hashes:
            raise ValueError('Processing code changed during preparation')
        with ExitStack() as rollback:
            rollback.callback(rollback_scene)
            out.rename(destination)
            if staged_license_path is not None:
                rollback.callback(rollback_license)
                staged_license_path.replace(shared_license)
            staged_catalog_path.replace(catalog)
            rollback.pop_all()
print(json.dumps({k: metadata[k] for k in ('id', 'coverage', 'pointCount', 'triangleCount', 'ground')}, indent=2))
