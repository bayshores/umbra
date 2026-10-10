"""Attach an original cached NVIDIA prediction as native renderer-only points.

No model/session execution. Historical cache identity and current native measured
identity are checked separately; full raw rasters survive display clipping.
"""
import argparse
from io import BytesIO
import os
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS / 'sponsors'))
import common
import nvidia_stereo

MAX_FILE = 16 * 1024 * 1024
MAX_TOTAL = 64 * 1024 * 1024
SHAPE = (320, 736)
NAMES = dict(positions='estimatedpositions.bin', colors='estimatedcolors.bin', triangles='estimatedtriangles.bin')
SOURCE_ROLES = {'left', 'right', 'lidar', 'poses', 'intrinsics', 'extrinsics', 'map', 'groundRaster', 'groundTransform', 'license'}


class Capture:
    """Bound every input byte and check the captured identities before publication."""
    def __init__(self, *, strict_paths=False):
        self.files = {}
        self.strict_reader = None
        if strict_paths:
            sys.path.insert(0, str(SCRIPTS / 'cloud'))
            from result_binding import read_path
            self.strict_reader = read_path

    def snapshot(self, path, expected=None):
        if self.strict_reader is None:
            return common.snapshot(path, expected, MAX_FILE)
        raw = self.strict_reader(path, MAX_FILE)
        common.require(expected is None or common.digest(raw) == expected, 'Captured SHA256 mismatch')
        return raw

    def read(self, path, expected=None, size=None):
        path = Path(os.path.abspath(path)) if self.strict_reader else Path(path).resolve()
        if path not in self.files:
            data = self.snapshot(path, expected)
            common.require(sum(len(value) for value in self.files.values()) + len(data) <= MAX_TOTAL,
                           'Cumulative input byte ceiling')
            self.files[path] = data
        data = self.files[path]
        if expected is not None:
            common.require(isinstance(expected, str) and re.fullmatch('[a-f0-9]{64}', expected) and
                           common.digest(data) == expected, 'Captured SHA256 mismatch')
        if size is not None:
            common.require(type(size) is int and 0 <= size <= MAX_FILE and len(data) == size,
                           'Artifact byte identity mismatch')
        return data

    def artifact(self, base, spec):
        common.require(isinstance(spec.get('path'), str) and spec['path'], 'Artifact path required')
        common.require(isinstance(spec.get('sha256'), str), 'Artifact SHA256 required')
        return self.read(Path(base) / spec['path'], spec['sha256'], spec['bytes'])

    def unchanged(self):
        for path, data in self.files.items():
            common.require(self.snapshot(path, common.digest(data)) == data, 'Captured input changed')


def bind_measured(scene, prepared, calibration, prepared_path, capture):
    """Authenticate current source identities and camera-centered measured geometry."""
    common.require(scene['sourceClass'] == prepared['sourceClass'] == prepared['dataset'] == 'Argoverse2Sensor' and
                   scene['units'] == 'meters' and scene['measurements']['kind'] == 'indexedSparse', 'Native AV2 scene required')
    common.require(set(prepared['sources']) == set(prepared['processing']['sourceHashes']) == set(scene['sourceHashes']) == SOURCE_ROLES,
                   'Exact native source roles required')
    common.require('reconstruction' not in scene and 'ground' not in scene and
                   scene['measurements']['groundPolicy'] == 'unvalidated', 'Existing attachment/ground retained')
    provenance = scene['measurements']['provenance']
    common.require(provenance['referencesSha256'] == prepared['outputs']['references.npz']['sha256'] and
                   provenance['originalSources'] == prepared['sources'], 'Current measured provenance mismatch')
    for key in ('logId', 'cameraTimestampNs', 'lidarTimestampNs', 'crossingId'):
        common.require(provenance[key] == prepared[key], 'Measured/native observation mismatch')
    for role, spec in prepared['sources'].items():
        raw = capture.artifact(prepared_path.parent, spec)
        common.require(scene['sourceHashes'][role] == common.digest(raw) == prepared['processing']['sourceHashes'][role],
                       'Current source hash mismatch')
    proc = prepared['processing']
    c, transform = scene['camera'], scene['transform']
    left = np.asarray(calibration['transformed']['left_projection'], np.float64)
    common.require([c[k] for k in ('fx', 'fy', 'cx', 'cy')] == [left[0, 0], left[1, 1], left[0, 2], left[1, 2]] and
                   [c['height'], c['width']] == list(SHAPE), 'Measured/native crop camera mismatch')
    rotation = np.asarray(transform['worldFromCamera'], np.float64)
    city_from_camera = np.asarray(proc['cityFromEgoCamera']) @ np.asarray(proc['egoFromCameraLeft'])
    expected_rotation = np.asarray(transform['worldFromCity']) @ city_from_camera[:3, :3] @ np.asarray(proc['R1']).T
    common.require(rotation.shape == (3, 3) and np.isfinite(rotation).all() and
                   np.array_equal(rotation, expected_rotation) and
                   np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9, rtol=0) and
                   abs(np.linalg.det(rotation) - 1) <= 1e-9, 'Native camera rotation mismatch')
    common.require(np.array_equal(transform['worldFromCity'], [[1, 0, 0], [0, 0, 1], [0, -1, 0]]) and
                   np.array_equal(transform['cityOrigin'], city_from_camera[:3, 3]) and
                   transform['originRole'] == 'leftCamera' and transform['cameraOrigin'] == [0, 0, 0] and
                   'cameraOriginGround' not in transform and c['eye'] == [0, 0, 0] and
                   np.array_equal(c['forward'], rotation[:, 2]) and np.array_equal(c['up'], -rotation[:, 1]),
                   'Native camera-centered pose mismatch')


def bind_native(scene, prepared, calibration, old_crop, old_calibration, proof, producer, prepared_path, capture):
    """Extend the dated equivalence receipt to this current native measurement."""
    bind_measured(scene, prepared, calibration, prepared_path, capture)
    common.require(proof['status'] == 'PASS_exact_saved_prediction_input_equivalence_no_inference' and
                   proof['model'] == producer['model'] and proof['oldCacheKey'] == producer['cacheKey'] and
                   proof['historicalOutputSha256'] == producer['prediction']['sha256'] and
                   proof['historicalValiditySha256'] == producer['validity']['sha256'], 'Input equivalence receipt mismatch')
    common.require(proof['newCalibrationSha256'] == prepared['outputs']['calibration.json']['sha256'],
                   'Current calibration differs from proved native calibration')
    expected_id = 'av2:' + prepared['logId'] + ':' + prepared['cameraTimestampNs'] + ':' + prepared['crossingId']
    common.require(producer['observationId'] == expected_id, 'Native/historical observation mismatch')
    for key in ('logId', 'cameraTimestampNs', 'lidarTimestampNs', 'crossingId'):
        if key != 'crossingId':
            common.require(prepared[key] == old_crop['processing'][key], 'Historical source identity mismatch')
    for role, spec in prepared['sources'].items():
        if role not in ('groundRaster', 'groundTransform', 'license'):
            matches = [pin for pin in old_crop['processing']['sourcePins'].values()
                       if (pin['bytes'], pin['sha256']) == (spec['bytes'], spec['sha256'])]
            common.require(len(matches) == 1, 'Native original source differs from historical crop')
    for key in ('x', 'y', 'width', 'height', 'source_to_crop_matrix'):
        common.require(prepared['crop'][key] == old_crop['crop'][key], 'Native crop changed')
    for key in ('decoded_source_size', 'source_rectified_size', 'source_to_decoded_scale', 'crop_size',
                'original', 'transformed', 'disparity_numerator_px_m', 'principal_point_offset_px', 'provisional'):
        common.require(calibration[key] == old_calibration[key], 'Native numeric calibration changed')
    proc, rect = prepared['processing'], old_crop['processing']['rectification']
    for key in ('R1', 'R2', 'P1', 'P2', 'rightCameraFromLeftCamera', 'additionalDistortion', 'alpha'):
        common.require(proc[key] == rect[key], 'Native rectification changed')
    common.require(proc['rectifiedSize'] == rect['outputSize'] and proc['KLeft'] == rect['rawK']['left'] and
                   proc['KRight'] == rect['rawK']['right'] and proc['egoFromCameraLeft'] == rect['egoFromCamera']['left'] and
                   proc['egoFromCameraRight'] == rect['egoFromCamera']['right'], 'Native raw camera geometry changed')


def bind_direct(scene, prepared, calibration, producer, prepared_path, model_input_path, capture):
    """Direct input identity, without a fabricated historical migration receipt."""
    common.require(Path(model_input_path).resolve() == Path(prepared_path).resolve(),
                   'Direct native model input path mismatch')
    common.require(producer['observationId'] == scene['id'], 'Direct native observation alias mismatch')
    bind_measured(scene, prepared, calibration, prepared_path, capture)


def points(disparity, validity, calibration, camera, rotation, rgb):
    """Exact crop-local axial conversion and row-major fixed rendering domain."""
    common.require(disparity.dtype == np.dtype('<f4') and disparity.shape == SHAPE and validity.shape == SHAPE and
                   validity.dtype == np.dtype('u1') and
                   np.array_equal(validity, (np.isfinite(disparity) & (disparity > 0)).astype('u1')), 'Raw numerical mask mismatch')
    common.require(rgb.dtype == np.uint8 and rgb.shape == (*SHAPE, 3), 'Exact left RGB required')
    left, right = (np.asarray(calibration['transformed'][key], np.float64) for key in ('left_projection', 'right_projection'))
    numerator, offset = float(left[0, 3] - right[0, 3]), float(left[0, 2] - right[0, 2])
    common.require(np.isfinite(left).all() and np.isfinite(right).all() and numerator > 0 and
                   numerator == calibration['disparity_numerator_px_m'] and offset == calibration['principal_point_offset_px'],
                   'Invalid crop disparity geometry')
    near, far = camera['near'], camera['far']
    common.require(all(type(value) in (int, float) and np.isfinite(value) for value in (near, far)) and
                   0 < near < far, 'Invalid existing camera clips')
    denominator = disparity.astype(np.float64) - offset
    numeric = (validity == 1) & np.isfinite(denominator) & (denominator > 0)
    depth = np.full(SHAPE, np.nan, dtype='<f4')
    with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
        depth[numeric] = (numerator / denominator[numeric]).astype('<f4')
    numeric &= np.isfinite(depth) & (depth > 0)
    depth[~numeric] = np.nan
    v, u = np.indices(SHAPE)
    right_centers = (u - disparity.astype(np.float64) >= 0) & (u - disparity.astype(np.float64) <= SHAPE[1] - 1)
    clip = numeric & (depth >= near) & (depth <= far)
    display = clip & right_centers
    common.require(0 < int(display.sum()) <= min(disparity.size, 1_000_000), 'No bounded native display points')
    z = depth[display].astype(np.float64)
    xyz = np.column_stack(((u[display] - camera['cx']) * z / camera['fx'],
                           (v[display] - camera['cy']) * z / camera['fy'], z))
    positions = (xyz @ np.asarray(rotation, np.float64).T).astype('<f4')
    srgb = rgb[display].astype(np.float64) / 255
    colors = np.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4).astype('<f4')
    common.require(np.isfinite(positions).all() and np.isfinite(colors).all() and
                   (colors >= 0).all() and (colors <= 1).all(), 'Invalid Float32 predicted geometry')
    finite_disparity = disparity[np.isfinite(disparity)]
    diagnostics = dict(totalPixels=disparity.size, rawValidityPixels=int(validity.sum()), numericDepthPixels=int(numeric.sum()),
        nonfiniteDisparityPixels=int((~np.isfinite(disparity)).sum()),
        nonpositiveFiniteDisparityPixels=int((np.isfinite(disparity) & (disparity <= 0)).sum()),
        nonpositiveDenominatorPixels=int(((validity == 1) & (denominator <= 0)).sum()),
        unrepresentableDepthPixels=int(((validity == 1) & (denominator > 0) & ~numeric).sum()),
        rawDisparityMinimumPixels=float(finite_disparity.min()), rawDisparityMaximumPixels=float(finite_disparity.max()),
        rawDenominatorMinimumPixels=float(denominator[np.isfinite(denominator)].min()),
        rawDenominatorMaximumPixels=float(denominator[np.isfinite(denominator)].max()),
        rawDepthMinimumMeters=float(depth[numeric].min()), rawDepthMaximumMeters=float(depth[numeric].max()),
        withinDeclaredCameraClipPixels=int(clip.sum()), outsideDeclaredCameraClipPixels=int((numeric & ~clip).sum()),
        belowNearPixels=int((numeric & (depth < near)).sum()), aboveFarPixels=int((numeric & (depth > far)).sum()),
        rightIntegerCenterDomainPixels=int((numeric & right_centers).sum()), rightDomainRejectedPixels=int((numeric & ~right_centers).sum()),
        outsideClipAndRightDomainPixels=int((numeric & ~clip & ~right_centers).sum()),
        numericRejectedFromDisplayPixels=int((numeric & ~display).sum()),
        displayedPixels=int(display.sum()), denominatorOffsetPixels=offset, numeratorPixelMeters=numerator)
    return depth, positions, colors, diagnostics


def attach(scene_path, scene_sha256, prepared_path, prepared_sha256, producer_path, producer_sha256,
           plan_path, plan_sha256, input_equivalence_path, input_equivalence_sha256, model_path, output, *, direct_native=False,
           nebius_options=None):
    common.require(type(direct_native) is bool, 'Explicit binding mode required')
    remote = nebius_options is not None
    common.require(not remote or (direct_native and type(nebius_options) is dict),
                   'Nebius consumption requires explicit direct-native binding')
    common.require((input_equivalence_path is None and input_equivalence_sha256 is None) if direct_native else
                   (input_equivalence_path is not None and input_equivalence_sha256 is not None), 'Exclusive input binding required')
    hashes = [scene_sha256, prepared_sha256, producer_sha256, plan_sha256]
    if not direct_native: hashes.append(input_equivalence_sha256)
    for sha in hashes:
        common.require(isinstance(sha, str) and re.fullmatch('[a-f0-9]{64}', sha), 'Expected source SHA256 required')
    paths = [Path(os.path.abspath(path)) if remote else Path(path).resolve()
             for path in (scene_path, prepared_path, producer_path, plan_path)]
    scene_path, prepared_path, producer_path, plan_path = paths
    if not direct_native: paths.append(Path(input_equivalence_path).resolve())
    output = Path(output).absolute()
    common.require(not os.path.lexists(output), 'Existing destination retained')
    capture = Capture(strict_paths=remote)
    raw = [capture.read(path, sha) for path, sha in zip(paths, hashes)]
    scene, prepared, producer, plan = [common.parse(data) for data in raw[:4]]
    proof = None if direct_native else common.parse(raw[4])
    consumer = None
    remote_kwargs = None
    if remote:
        required = {'expected_submission_id', 'review_path', 'expected_review_sha256', 'expected_job_path',
                    'expected_job_sha256', 'runtime_contract_path', 'expected_runtime_contract_sha256',
                    'token_path', 'receipt_path'}
        common.require(required <= set(nebius_options) <= required | {'expected_binding_sha256', 'scratch_root'},
                       'Exact Nebius consumption arguments required')
        common.require(plan_path.name == 'plan.json' and
                       producer_path == plan_path.parent / 'prediction' / 'prediction.json',
                       'Exact retrieved bundle paths required')
        sys.path.insert(0, str(SCRIPTS / 'cloud'))
        import consume_result as consumer
        remote_kwargs = {key: value for key, value in nebius_options.items() if key not in {'token_path', 'receipt_path'}}
        remote_kwargs.update(expected_plan_sha256=plan_sha256, native_manifest_path=prepared_path,
                             expected_native_manifest_sha256=prepared_sha256)
        checked_producer, _, _ = consumer.check_result(plan_path.parent, **remote_kwargs)
        common.require(common.canonical(checked_producer) == common.canonical(producer), 'Captured remote producer changed')
    code_paths = [Path(__file__), SCRIPTS / 'prepare-stereo-crop.py', SCRIPTS / 'inspect-onnx-model.py', *sorted((SCRIPTS / 'sponsors').glob('*.py'))]
    if remote:
        code_paths.extend(SCRIPTS / 'cloud' / name for name in
                          ('consume_result.py', 'job_evidence.py', 'result_binding.py', 'run_stereo_job.py'))
    code_hashes = {str(path.relative_to(SCRIPTS)): common.digest(capture.read(path)) for path in code_paths}
    if not direct_native:
        common.require(proof['oldPlanSha256'] == plan_sha256 and proof['oldProducerSha256'] == producer_sha256,
                       'Historical proof source identity mismatch')
    common.require(producer['model'] == plan['model'] == nvidia_stereo.MODEL and producer['cacheKey'] == plan['cacheKey'],
                   'Plan/producer identity mismatch')
    # Capture the existing cache-validation dependency graph before invoking its pure readers.
    review_path = Path(os.path.abspath(nebius_options['review_path'])) if remote else plan_path.parent / plan['review']['path']
    review = common.parse(capture.read(review_path, plan['review']['sha256'], plan['review']['bytes']))
    inspection = review['inspection']
    common.require(isinstance(inspection['sha256'], str), 'Inspection SHA256 required')
    capture.read(review_path.parent / inspection['path'], inspection['sha256'])
    for records, base in ((plan['sources'], plan_path.parent), (producer['sources'], producer_path.parent)):
        for name, spec in records.items():
            if remote:
                source = prepared_path if name == 'cropManifest' else prepared_path.parent / name
                capture.read(source, spec['sha256'], spec['bytes'])
            else:
                capture.artifact(base, spec)
    for key in ('prediction', 'validity'): capture.artifact(producer_path.parent, producer[key])
    old_crop_path = prepared_path if remote else (producer_path.parent / producer['sources']['cropManifest']['path']).resolve()
    old_crop = common.parse(capture.read(old_crop_path, producer['sources']['cropManifest']['sha256'],
                                        producer['sources']['cropManifest']['bytes']))
    for name in ('left.png', 'right.png', 'calibration.json'):
        spec = old_crop['outputs'][name]
        capture.read(old_crop_path.parent / name, spec['sha256'], spec['bytes'])
    cached = checked_producer if remote else nvidia_stereo.cached(plan_path, producer_path.parent, model_path)
    common.require(common.canonical(cached) == common.canonical(producer), 'Captured original producer changed')
    old_raw, _, old_files, old_calibration = nvidia_stereo.read_crop(old_crop_path)
    prepared_raw, _, prepared_files, calibration = nvidia_stereo.read_crop(prepared_path)
    common.require(old_raw == capture.read(old_crop_path) and prepared_raw == raw[1], 'Crop manifest changed')
    for name, data in old_files.items():
        common.require(data == capture.read(old_crop_path.parent / name), 'Historical crop changed during read')
    for name, data in prepared_files.items():
        common.require(data == capture.read(prepared_path.parent / name, prepared['outputs'][name]['sha256'],
                                           prepared['outputs'][name]['bytes']), 'Prepared crop changed')
    common.require(prepared_files['left.png'] == old_files['left.png'] and prepared_files['right.png'] == old_files['right.png'],
                   'Native/historical RGB bytes differ')
    common.require(producer['execution']['inputTensorSha256'] ==
                   [common.digest(tensor.tobytes()) for tensor in nvidia_stereo.tensors(prepared_files, plan['processing']['normalization'])],
                   'Native/historical normalized tensors differ')
    references = capture.read(prepared_path.parent / 'references.npz', prepared['outputs']['references.npz']['sha256'],
                              prepared['outputs']['references.npz']['bytes'])
    prov = scene['measurements']['provenance']
    common.require(prov['preparedManifestSha256'] == prepared_sha256 and
                   capture.artifact(scene_path.parent, prov['preparedManifest']) == raw[1] and
                   capture.artifact(scene_path.parent, prov['references']) == references,
                   'Current measured manifest/reference identity mismatch')
    if direct_native:
        common.require(old_raw == raw[1], 'Direct model input manifest differs from native preparation')
        bind_direct(scene, prepared, calibration, producer, prepared_path, old_crop_path, capture)
        binding_bytes = common.canonical(dict(kind='directNative', preparedManifestSha256=prepared_sha256,
            producerManifestSha256=producer_sha256, planSha256=plan_sha256, observationId=producer['observationId']))
        binding_provenance = dict(inputBindingKind='directNative', inputBindingResultSha256=common.digest(binding_bytes))
    else:
        bind_native(scene, prepared, calibration, old_crop, old_calibration, proof, producer, prepared_path, capture)
        binding_bytes = raw[4]
        binding_provenance = dict(inputEquivalenceResultSha256=input_equivalence_sha256)
    files = {}
    owners = [(scene, {'positions': scene['pointCount'] * 12, 'colors': scene['pointCount'] * 12, 'triangles': scene['triangleCount'] * 12}),
        (scene['measurements'], {'referencePositions': scene['measurements']['count'] * 24,
         'referenceCameraPositions': scene['measurements']['count'] * 24, 'referenceUV': scene['measurements']['count'] * 16,
         'referenceEligible': scene['measurements']['count'], 'sourceIndices': scene['measurements']['count'] * 4})]
    for owner, sizes in owners:
        common.require(set(owner['buffers']) == set(owner['artifactHashes']) == set(sizes), 'Exact measured buffer roles required')
        for key, size in sizes.items():
            name = owner['buffers'][key]
            common.require(isinstance(name, str) and re.fullmatch('[a-z]+\\.bin', name) and name not in files and name not in NAMES.values(),
                           'Measured buffer alias/name rejected')
            files[name] = capture.read(scene_path.parent / name, owner['artifactHashes'][key], size)
    image = scene['recordedImage']
    common.require(image['file'] == 'recorded.png' and image['mimeType'] == 'image/png' and
                   capture.read(scene_path.parent / image['file'], image['sha256'], image['bytes']) == prepared_files['left.png'],
                   'Current recorded image differs from original left crop')
    files[image['file']] = prepared_files['left.png']
    license_spec = prepared['outputs']['LICENSE-AND-ATTRIBUTION.md']
    license_bytes = capture.read(prepared_path.parent / 'LICENSE-AND-ATTRIBUTION.md', license_spec['sha256'], license_spec['bytes'])
    common.require(capture.read(scene_path.parent / 'LICENSE-AND-ATTRIBUTION.md', scene['sourceHashes']['license']) == license_bytes,
                   'Measured dataset license mismatch')
    files['LICENSE-AND-ATTRIBUTION.md'] = license_bytes
    disparity_bytes = capture.artifact(producer_path.parent, producer['prediction'])
    validity_bytes = capture.artifact(producer_path.parent, producer['validity'])
    disparity = np.frombuffer(disparity_bytes, '<f4').reshape(SHAPE)
    validity = np.frombuffer(validity_bytes, 'u1').reshape(SHAPE)
    with Image.open(BytesIO(prepared_files['left.png'])) as image: rgb = np.array(image)
    depth, positions, colors, diagnostics = points(disparity, validity, calibration, scene['camera'], scene['transform']['worldFromCamera'], rgb)
    additions = {NAMES['positions']: positions.tobytes(), NAMES['colors']: colors.tobytes(), NAMES['triangles']: b''}
    files.update(additions)
    files.update({'rawdisparity.float32le': disparity_bytes, 'rawvalidity.uint8': validity_bytes, 'rawdepth.float32le': depth.tobytes(),
                  'original-producer.json': raw[2], 'original-plan.json': raw[3],
                  ('input-binding.json' if direct_native else 'input-equivalence.json'): binding_bytes})
    scene['reconstruction'] = dict(kind='nativePoints', evidence='predicted', role='rendererOnly', model=producer['model'],
        pointCount=len(positions), triangleCount=0, buffers=NAMES.copy(),
        bounds=dict(min=positions.min(0).tolist(), max=positions.max(0).tolist()),
        artifactHashes={key: common.digest(additions[name]) for key, name in NAMES.items()},
        provenance=dict(baseMeasuredSceneSha256=scene_sha256, preparedManifestSha256=prepared_sha256,
            calibrationSha256=common.digest(prepared_files['calibration.json']), referencesSha256=common.digest(references),
            **binding_provenance, producerManifestSha256=producer_sha256, planSha256=plan_sha256,
            rawPredictionSha256=common.digest(disparity_bytes), rawValiditySha256=common.digest(validity_bytes), processingCodeHashes=code_hashes),
        reuse=dict(newInferenceRuns=0, originalCacheKey=producer['cacheKey'], originalExecution=producer['execution'],
                   originalRecordBase=str(producer_path.parent), originalProcessing=producer['processing']),
        displayDomain=dict(policy='numericDepth_rightInputBounds_existingCameraClips', nearMeters=scene['camera']['near'],
                           farMeters=scene['camera']['far'], rightPixelCenters=[0, scene['camera']['width'] - 1]),
        rawArtifacts={key: dict(path=name, bytes=len(files[name]), sha256=common.digest(files[name]), shape=list(SHAPE), encoding=encoding)
                      for key, name, encoding in [('disparity', 'rawdisparity.float32le', 'float32le'),
                         ('validity', 'rawvalidity.uint8', 'uint8'), ('depth', 'rawdepth.float32le', 'float32le')]},
        diagnostics=diagnostics,
        limitations=['Estimated rendering only; native measured queries, collision states and ground policy remain authoritative.',
                     'Rendering clips and right-input bounds are not confidence, physical accuracy or free-space evidence.',
                     'Full raw disparity, numerical validity and axial depth retain unsupported pixels and numerical tails.'])
    if remote:
        # Finish every local source/render check before opening a credential or contacting the service.
        capture.unchanged()
        accepted, result, service = consumer.collect_result(plan_path.parent,
            token_path=nebius_options['token_path'], receipt_path=nebius_options['receipt_path'], **remote_kwargs)
        common.require(type(service) is dict and type(service.get('schemaVersion')) is int and service['schemaVersion'] == 1 and
                       service.get('claim') == 'job_control_and_application_output_association' and
                       service.get('provider') == 'nebius_job', 'Direct collected service association required')
        common.require(common.canonical(accepted) == common.canonical(producer) and service['producerSha256'] == producer_sha256 and
                       service['planSha256'] == plan_sha256 and service['nativeManifestSha256'] == prepared_sha256,
                       'Collected service result differs from captured renderer inputs')
        service_raw = common.canonical(service)
        receipt_raw = capture.read(nebius_options['receipt_path'], service['jobControlReceiptSha256'])
        result_raw = capture.read(plan_path.parent / 'job-result.json', service['bindingSha256'])
        common.require(common.parse(result_raw) == result, 'Collected output binding changed')
        scene['reconstruction']['reuse']['serviceExecution'] = service
        scene['reconstruction']['provenance']['serviceExecutionSha256'] = common.digest(service_raw)
        scene['reconstruction']['serviceArtifacts'] = {
            key: dict(path=name, bytes=len(data), sha256=common.digest(data)) for key, name, data in
            [('context', 'service-execution.json', service_raw), ('jobControl', 'job-control.json', receipt_raw),
             ('outputBinding', 'job-result.json', result_raw)]}
        files.update({'service-execution.json': service_raw, 'job-control.json': receipt_raw, 'job-result.json': result_raw})
    # Keep descriptor identities intact while making their external references usable from the new directory.
    prov['preparedManifest']['path'] = os.path.relpath(prepared_path, output)
    prov['references']['path'] = os.path.relpath(prepared_path.parent / 'references.npz', output)
    files['scene.json'] = common.canonical(scene)
    files['catalog-entry.json'] = common.canonical(dict(id=scene['id'], label=scene['id'], metadataSha256=common.digest(files['scene.json'])))
    files['attachment-result.json'] = common.canonical(dict(metadataSha256=common.digest(files['scene.json']), diagnostics=diagnostics,
        sourcePins=[dict(path=str(path), bytes=len(data), sha256=common.digest(data)) for path, data in capture.files.items()],
        newInferenceRuns=0, modelBytesRehashed=False, modelSessions=0, cloudRequests=int(remote),
        **({'jobControlRequests': 1, 'modelInferenceRequests': 0} if remote else {}), newSpend=0))
    capture.unchanged()
    common.publish(output, files)
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('scene', 'prepared', 'producer', 'plan'):
        parser.add_argument('--' + key, type=Path, required=True)
        parser.add_argument('--' + key + '-sha256', required=True)
    parser.add_argument('--direct-native', action='store_true', help='Exact current native manifest was the model input')
    parser.add_argument('--input-equivalence', type=Path)
    parser.add_argument('--input-equivalence-sha256')
    parser.add_argument('--nebius-options', type=Path,
                        help='Pinned private request for explicit direct Job collection; never a receipt or authenticated flag')
    parser.add_argument('--nebius-options-sha256')
    for key in ('model', 'output'): parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    try:
        common.require((args.nebius_options is None) == (args.nebius_options_sha256 is None),
                       'Nebius request path and hash must be supplied together')
        options = None if args.nebius_options is None else common.parse(
            common.snapshot(args.nebius_options, args.nebius_options_sha256, MAX_FILE))
        scene = attach(args.scene, args.scene_sha256, args.prepared, args.prepared_sha256, args.producer, args.producer_sha256,
                       args.plan, args.plan_sha256, args.input_equivalence, args.input_equivalence_sha256, args.model, args.output,
                       direct_native=args.direct_native, nebius_options=options)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(1, f'Native cached attachment blocked: {type(exc).__name__}: {str(exc)[:250]}\n')
    print(f'Attached cached native estimate: {scene["reconstruction"]["pointCount"]} points, zero triangles, new inference runs 0')


if __name__ == '__main__':
    main()
