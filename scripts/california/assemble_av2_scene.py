"""Offline indexed AV2 scene assembly; projected returns never establish ground/free space."""
import argparse
from io import BytesIO
import math
import os
from pathlib import Path
import re
import sys
from uuid import UUID
import zipfile

import numpy as np
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS / 'sponsors'))
from common import canonical, digest, parse, publish, record, require, snapshot

OUTPUTS = {'left.png', 'right.png', 'calibration.json', 'references.npz', 'LICENSE-AND-ATTRIBUTION.md'}
SOURCE_ROLES = {'left', 'right', 'lidar', 'poses', 'intrinsics', 'extrinsics', 'map', 'groundRaster', 'groundTransform', 'license'}
MAX_MANIFEST = 128 * 1024
MAX_FILE = 16 * 1024 * 1024
MAX_TOTAL = 64 * 1024 * 1024
MAX_NPZ_EXPANDED = 64 * 1024 * 1024
MAX_ROWS = 1_000_000
NUMERICAL_ATOL = 1e-9
REFERENCE_FIELDS = {'sourceIndex', 'egoXYZ', 'cityXYZ', 'offsetNs', 'leftRectifiedXYZ', 'rightRectifiedXYZ',
                    'leftCropUV', 'rightCropUV', 'leftEligible', 'rightEligible', 'bothEligible', 'insideFootprint',
                    'nominalGroundHeights', 'groundSupport', 'groundBoundaryAlias', 'groundIntegerInBounds',
                    'groundImageXY', 'groundIndices'}
WORLD_FROM_CITY = np.array([[1., 0, 0], [0, 0, 1.], [0, -1., 0]])


def matrix(value, shape):
    result = np.asarray(value)
    require(result.shape == shape and result.dtype.kind in 'iuf' and np.isfinite(result).all(), 'Finite numeric matrix required')
    return result.astype(np.float64)


def rotation(value):
    result = matrix(value, (3, 3))
    require(np.allclose(result.T @ result, np.eye(3), rtol=0, atol=1e-9) and
            abs(np.linalg.det(result) - 1) <= 1e-9, 'Proper orthonormal rotation required')
    return result


def rigid(value):
    result = matrix(value, (4, 4))
    rotation(result[:3, :3])
    require(np.array_equal(result[3], [0, 0, 0, 1]), 'Rigid homogeneous pose required')
    return result


def matches(a, b):
    return np.allclose(a, b, rtol=0, atol=NUMERICAL_ATOL, equal_nan=True)


def capture(manifest_path, output):
    manifest_path = Path(manifest_path).resolve()
    output = Path(output).absolute()
    require(not os.path.lexists(output), 'Existing output retained')
    raw = snapshot(manifest_path, max_bytes=MAX_MANIFEST)
    manifest = parse(raw)
    require(isinstance(manifest, dict), 'Preparation manifest object required')
    require(type(manifest.get('schemaVersion')) is int and manifest['schemaVersion'] == 1 and
            manifest.get('sourceClass') == manifest.get('dataset') == 'Argoverse2Sensor', 'Native AV2 preparation required')
    require(isinstance(manifest.get('logId'), str) and str(UUID(manifest['logId'])) == manifest['logId'], 'Canonical log UUID required')
    for key in ('cameraTimestampNs', 'lidarTimestampNs', 'crossingId'):
        require(isinstance(manifest.get(key), str) and re.fullmatch('[0-9]+', manifest[key]), 'Decimal identity required')
    outputs = manifest.get('outputs')
    require(isinstance(outputs, dict) and set(outputs) == OUTPUTS, 'Exact five prepared outputs required')
    total = 0; identities = set(); paths = {}
    for name, spec in outputs.items():
        require(isinstance(spec, dict) and type(spec.get('bytes')) is int and 0 < spec['bytes'] <= MAX_FILE, 'Prepared artifact byte ceiling')
        path = (manifest_path.parent / name).resolve()
        require(path.parent == manifest_path.parent and not path.is_relative_to(output.resolve()), 'Prepared artifact path/alias rejected')
        stat = path.stat(); identity = (stat.st_dev, stat.st_ino)
        require(identity not in identities, 'Duplicate prepared artifact alias')
        identities.add(identity); paths[name] = path; total += spec['bytes']
    require(total <= MAX_TOTAL, 'Cumulative prepared byte ceiling')
    captured = {}
    for name, path in paths.items():
        expected = outputs[name].get('sha256')
        require(isinstance(expected, str) and re.fullmatch('[a-f0-9]{64}', expected), 'Prepared artifact SHA256 required')
        data = snapshot(path, expected, MAX_FILE)
        require(len(data) == outputs[name]['bytes'], 'Prepared artifact byte identity mismatch')
        captured[name] = data
    sources = manifest.get('sources')
    require(isinstance(sources, dict) and set(sources) == SOURCE_ROLES, 'Original source identities required')
    require(isinstance(manifest['processing'].get('sourceHashes'), dict) and
            set(manifest['processing']['sourceHashes']) == SOURCE_ROLES, 'Exact original source hash roles required')
    plan = manifest.get('plan')
    require(isinstance(plan, dict) and isinstance(plan.get('path'), str) and plan['path'] and
            type(plan.get('bytes')) is int and 0 < plan['bytes'] <= 2_000_000 and
            isinstance(plan.get('sha256'), str) and re.fullmatch('[a-f0-9]{64}', plan['sha256']), 'Original plan identity required')
    for role, spec in sources.items():
        require(isinstance(spec, dict) and isinstance(spec.get('path'), str) and spec['path'] and
                type(spec.get('bytes')) is int and 0 < spec['bytes'] <= 16_000_000 and
                isinstance(spec.get('sha256'), str) and re.fullmatch('[a-f0-9]{64}', spec['sha256']) and
                manifest['processing']['sourceHashes'].get(role) == spec['sha256'], 'Original source hash identity mismatch')
    return manifest, raw, captured, output


def references(data, count):
    """Bound every NPY header before loading the five arrays consumed here."""
    wanted = {'sourceIndex', 'cityXYZ', 'leftRectifiedXYZ', 'leftCropUV', 'leftEligible'}
    with zipfile.ZipFile(BytesIO(data)) as archive:
        entries = archive.infolist(); names = [item.filename for item in entries]
        require(len(entries) == 18 and len(set(names)) == len(names) and
                {name + '.npy' for name in REFERENCE_FIELDS} == set(names), 'Exact eighteen unique NPZ arrays required')
        require(sum(item.file_size for item in entries) <= MAX_NPZ_EXPANDED, 'Expanded NPZ byte ceiling')
        for item in entries:
            require(item.file_size <= MAX_NPZ_EXPANDED and item.flag_bits & 1 == 0, 'NPZ member ceiling/encryption rejected')
            with archive.open(item) as stream:
                version = np.lib.format.read_magic(stream)
                require(version in ((1, 0), (2, 0)), 'Unsupported NPY version')
                length_offset = stream.tell()
                length_bytes = 2 if version == (1, 0) else 4
                declared = stream.read(length_bytes)
                require(len(declared) == length_bytes and int.from_bytes(declared, 'little') <= 4096, 'NPY header byte ceiling')
                stream.seek(length_offset)
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, _fortran, dtype = reader(stream, max_header_size=4096)
                require(1 <= len(shape) <= 2 and all(type(n) is int and 0 <= n <= MAX_ROWS for n in shape) and
                        math.prod(shape) <= MAX_ROWS * 3 and dtype.kind in 'fiub' and dtype.itemsize <= 8 and
                        item.file_size == stream.tell() + math.prod(shape) * dtype.itemsize, 'NPY shape/type/payload rejected')
        arrays = {key: np.load(BytesIO(archive.read(key + '.npy')), allow_pickle=False, max_header_size=4096) for key in wanted}
    for key, width in [('cityXYZ', 3), ('leftRectifiedXYZ', 3), ('leftCropUV', 2)]:
        require(arrays[key].shape == (count, width) and arrays[key].dtype.kind == 'f' and arrays[key].dtype.itemsize == 8 and
                not np.isinf(arrays[key]).any(), 'Float64 reference shape/domain required')
    indices = arrays['sourceIndex']
    require(indices.shape == (count,) and indices.dtype.kind in 'iu' and
            np.array_equal(indices, np.arange(count, dtype=np.uint32)), 'Original ordered source IDs required')
    require(arrays['leftEligible'].shape == (count,) and arrays['leftEligible'].dtype == np.dtype(bool), 'Original boolean eligibility required')
    return arrays


def assemble(manifest_path, output, scene_id):
    require(isinstance(scene_id, str) and re.fullmatch('[A-Za-z0-9_-]+', scene_id), 'Explicit safe scene identifier required')
    manifest, manifest_raw, files, output = capture(manifest_path, output)
    processing = manifest['processing']; crop = manifest['crop']
    values = [crop[key] for key in ('x', 'y', 'width', 'height')]
    require(all(type(n) is int and n >= 0 for n in values) and values[2:] == [736, 320], 'Prepared crop contract required')
    x, y, width, height = values
    rect_size = processing['rectifiedSize']
    require(isinstance(rect_size, list) and len(rect_size) == 2 and all(type(n) is int and 0 < n <= 32766 for n in rect_size) and
            rect_size[0] * rect_size[1] <= 16_000_000 and x + width <= rect_size[0] and y + height <= rect_size[1], 'Rectified crop domain required')
    a = np.array([[1., 0, -x], [0, 1., -y], [0, 0, 1.]])
    require(np.array_equal(matrix(crop['source_to_crop_matrix'], (3, 3)), a), 'Crop transform mismatch')
    calibration = parse(files['calibration.json'])
    require(isinstance(calibration, dict), 'Prepared calibration object required')
    p1 = matrix(processing['P1'], (3, 4)); cropped = a @ p1
    require(np.array_equal(p1[[0, 1], 3], [0, 0]) and np.array_equal(p1[2], [0, 0, 1, 0]) and
            p1[0, 1] == p1[1, 0] == 0 and p1[0, 0] > 0 and p1[1, 1] > 0, 'Left pinhole projection required')
    require(calibration['sourceClass'] == 'Argoverse2Sensor' and calibration['crop_size'] == [width, height] and
            calibration['decoded_source_size'] == calibration['source_rectified_size'] == rect_size and
            matches(matrix(calibration['original']['left_projection'], (3, 4)), p1) and
            matches(matrix(calibration['transformed']['left_projection'], (3, 4)), cropped), 'Cropped calibration mismatch')
    city_from_camera = rigid(processing['cityFromEgoCamera']) @ rigid(processing['egoFromCameraLeft'])
    origin = matrix(processing['leftCameraCityOrigin'], (3,))
    require(matches(origin, city_from_camera[:3, 3]), 'Left camera city origin mismatch')
    r1 = rotation(processing['R1'])
    world_from_camera = WORLD_FROM_CITY @ city_from_camera[:3, :3] @ r1.T
    count = processing['sourceRows']
    require(type(count) is int and 0 < count <= MAX_ROWS, 'Source row ceiling')
    arrays = references(files['references.npz'], count)
    city, rect, uv, eligible = [arrays[key] for key in ('cityXYZ', 'leftRectifiedXYZ', 'leftCropUV', 'leftEligible')]
    with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
        projected = rect @ p1[:, :3].T
        expected_uv = projected[:, :2] / projected[:, 2:3] - [x, y]
        domain = np.isfinite(rect).all(1) & (rect[:, 2] > 0)
        expected_uv[~domain] = np.nan
        world = (city - origin) @ WORLD_FROM_CITY.T
        camera_world = rect @ world_from_camera.T
    require(matches(world, camera_world), 'Source city/camera/world representation mismatch')
    require(not np.isinf(expected_uv).any(), 'Projection exceeds finite UV representation')
    require(np.array_equal(np.isnan(uv), np.isnan(expected_uv)), 'Source crop UV projection domain mismatch')
    finite_uv = np.isfinite(expected_uv)
    allowance = 256 * np.finfo(np.float64).eps * (np.abs(uv[finite_uv]) + np.abs(expected_uv[finite_uv]) + 1) + 1e-9
    require(np.isfinite(allowance).all() and (np.abs(uv[finite_uv] - expected_uv[finite_uv]) <= allowance).all(), 'Source crop UV projection mismatch')
    expected_eligible = domain & np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1)
    require(np.array_equal(eligible, expected_eligible) and type(processing['eligibleLeft']) is int and
            processing['eligibleLeft'] == int(eligible.sum()), 'Source eligibility mismatch')
    require(eligible.any(), 'No eligible render points; bounds unavailable')
    with np.errstate(over='ignore', invalid='ignore'):
        positions = world[eligible].astype('<f4')
    require(not np.isinf(world).any() and np.isfinite(positions).all(), 'World coordinates exceed numerical representation')
    with Image.open(BytesIO(files['left.png'])) as image:
        require(image.format == 'PNG' and image.mode == 'RGB' and image.size == (width, height), 'Prepared left RGB crop mismatch')
        image.load(); rgb = np.asarray(image)
    cells = np.floor(uv[eligible] + .5).astype(np.int64)
    srgb = rgb[cells[:, 1], cells[:, 0]].astype(np.float64) / 255
    colors = np.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4).astype('<f4')
    minimum = positions.min(0).astype(np.float64); maximum = positions.max(0).astype(np.float64)
    far = 2 * float(np.linalg.norm(np.maximum(np.abs(minimum), np.abs(maximum))))
    require(math.isfinite(far) and far > .01, 'Render clip bounds unavailable')
    render_names = {'positions': 'positions.bin', 'colors': 'colors.bin', 'triangles': 'triangles.bin'}
    reference_names = {'sourceIndices': 'sourceindices.bin', 'referencePositions': 'referencepositions.bin',
                       'referenceCameraPositions': 'referencecamerapositions.bin',
                       'referenceUV': 'referenceuv.bin', 'referenceEligible': 'referenceeligible.bin'}
    result_files = {'positions.bin': positions.tobytes(), 'colors.bin': colors.tobytes(), 'triangles.bin': np.empty(0, dtype='<u4').tobytes(),
                    'sourceindices.bin': arrays['sourceIndex'].astype('<u4').tobytes(), 'referencepositions.bin': world.astype('<f8').tobytes(),
                    'referencecamerapositions.bin': rect.astype('<f8').tobytes(),
                    'referenceuv.bin': uv.astype('<f8').tobytes(), 'referenceeligible.bin': eligible.astype('u1').tobytes(),
                    'recorded.png': files['left.png'], 'LICENSE-AND-ATTRIBUTION.md': files['LICENSE-AND-ATTRIBUTION.md']}
    provenance = dict(preparedManifest=record(manifest_path, manifest_raw, output), preparedManifestSha256=digest(manifest_raw),
                      references=record(Path(manifest_path).resolve().parent / 'references.npz', files['references.npz'], output),
                      referencesSha256=digest(files['references.npz']), originalSourceRecordBase=str(Path(manifest_path).resolve().parent),
                      originalSources=manifest['sources'], originalPlan=manifest.get('plan'),
                      **{key: manifest[key] for key in ('logId', 'cameraTimestampNs', 'lidarTimestampNs', 'crossingId')})
    scene = dict(id=scene_id, sourceClass='Argoverse2Sensor', units='meters', sourceHashes=processing['sourceHashes'],
                 camera=dict(eye=[0., 0., 0.], forward=world_from_camera[:, 2].tolist(), up=(-world_from_camera[:, 1]).tolist(),
                             fx=float(cropped[0, 0]), fy=float(cropped[1, 1]), cx=float(cropped[0, 2]), cy=float(cropped[1, 2]),
                             width=width, height=height, near=.01, far=far),
                 transform=dict(worldFromCity=WORLD_FROM_CITY.tolist(), cityOrigin=origin.tolist(), originRole='leftCamera',
                                cameraOrigin=[0., 0., 0.], worldFromCamera=world_from_camera.tolist()),
                 pointCount=int(eligible.sum()), triangleCount=0, bounds=dict(min=minimum.tolist(), max=maximum.tolist()),
                 recordedImage=dict(file='recorded.png', mimeType='image/png', bytes=len(files['left.png']), sha256=digest(files['left.png'])),
                 buffers=render_names, artifactHashes={key: digest(result_files[name]) for key, name in render_names.items()},
                 measurements=dict(kind='indexedSparse', evidence='geometricallyProjected', groundPolicy='unvalidated',
                                   pixelCellPolicy='nearestIntegerCenterRejectCollisions', binaryEncoding='littleEndian', count=count, buffers=reference_names,
                                   artifactHashes={key: digest(result_files[name]) for key, name in reference_names.items()}, provenance=provenance),
                 processing=dict(coordinateConvention='Camera-centered city orientation; X=cityX, Y=cityZ, Z=-cityY',
                                 cameraConvention='Virtual R1-rectified OpenCV x-right/y-down/z-forward',
                                 colorPolicy='Projected source crop nearest integer center floor(UV+0.5), converted sRGB to linear RGB; render only',
                                 renderSelection='Every leftEligible row in original source order; no collision or terrain filtering',
                                 fullReferencePolicy='All original source rows, Float64 worldXYZ/cameraXYZ/UV including NaNs; original NPZ retained at prepared source',
                                 groundPolicy='unvalidated; nominal terrain is not observed ground',
                                 nearPolicy='0.01 meters rendering clip only', farPolicy='2 times maximum Euclidean corner radius of rendered world bounds; rendering clip only',
                                 numericalConsistency=dict(worldAbsoluteToleranceMeters=NUMERICAL_ATOL,
                                                           uvTolerance='256*Float64epsilon*(abs(savedUV)+abs(reprojectedUV)+1)+1e-9 pixels',
                                                           role='Arithmetic representation check; no physical accuracy bound'),
                                 readLimits=dict(manifestBytes=MAX_MANIFEST, perFileBytes=MAX_FILE, totalBytes=MAX_TOTAL, expandedNpzBytes=MAX_NPZ_EXPANDED,
                                                 maxSourceRows=MAX_ROWS, maxNpyHeaderBytes=4096, capturedOnce=True)),
                 processingCodeHashes={name: digest(path.read_bytes()) for name, path in
                                       [('assembly', Path(__file__)), ('common', SCRIPTS / 'sponsors/common.py'),
                                        ('exclusivePublication', SCRIPTS / 'prepare-stereo-crop.py')]},
                 limitations=['Geometric projection and sampled color do not establish camera-visible source identity.',
                              'Native grounding/movement is unvalidated; nominal terrain is not promoted to observed road.',
                              'Sparse absent support cannot establish Clear or free space; zero triangles and no depth raster.',
                              'Arithmetic checks do not prove calibration, physical accuracy, site/holdout admission or pipeline completion.'])
    result_files['scene.json'] = canonical(scene)
    result_files['catalog-entry.json'] = canonical(dict(id=scene_id, label=scene_id, metadataSha256=digest(result_files['scene.json'])))
    publish(output, result_files)
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True); parser.add_argument('--output', required=True); parser.add_argument('--id', required=True)
    args = parser.parse_args()
    try:
        scene = assemble(args.manifest, args.output, args.id)
    except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile) as error:
        parser.exit(1, f'AV2 scene rejected: {error}\n')
    print(canonical(dict(id=scene['id'], sourceRows=scene['measurements']['count'], renderPoints=scene['pointCount'],
                         output=args.output, modelCalls=0)).decode())


if __name__ == '__main__':
    main()
