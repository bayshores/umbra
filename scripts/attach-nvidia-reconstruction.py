"""Attach a source-bound cached estimate for rendering; measured queries stay intact."""
import argparse
from io import BytesIO
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / 'evaluation'))
import adapt_nvidia
import common
import disparity_registration
import matched
import nvidia_stereo
from scene_surface import build_surface


def attach(scene_path, prediction_path, prediction_hash, model_path, output):
    source_paths = [Path(__file__), Path(__file__).parent/'scene_surface.py', Path(adapt_nvidia.__file__),
                    Path(disparity_registration.__file__), Path(matched.__file__), Path(nvidia_stereo.__file__),
                    Path(common.__file__), Path(__file__).parent/'rectified_calibration.py',
                    Path(__file__).parent/'prepare-stereo-crop.py']
    code_hashes = {str(p.relative_to(Path(__file__).parent)): common.digest(p.read_bytes()) for p in source_paths}
    output = Path(output).absolute()
    common.require(not output.exists() and not output.is_symlink(), 'Existing destination retained')
    scene_raw = common.snapshot(scene_path)
    scene = common.parse(scene_raw)
    common.require('reconstruction' not in scene, 'Existing reconstruction retained')
    registered_raw = common.snapshot(prediction_path, prediction_hash)
    registered = common.parse(registered_raw)
    derivation = registered['derivation']
    observations_path = prediction_path.parent / derivation['observationManifest']['path']
    observations_raw = matched.artifact(derivation['observationManifest'], prediction_path.parent)
    observations = common.parse(observations_raw)
    candidates = [o for o in observations['observations'] if o['id'] == registered['observationId']]
    common.require(len(candidates) == 1 and registered['observationId'] == scene['id'], 'Scene/observation identity mismatch')
    observation = candidates[0]
    producer_path = prediction_path.parent / derivation['producerManifest']['path']
    producer_raw = matched.artifact(derivation['producerManifest'], prediction_path.parent)
    plan_path = prediction_path.parent / derivation['plan']['path']
    plan_raw = matched.artifact(derivation['plan'], prediction_path.parent)
    # Cache assertions alone do not require or hash the local model. Check bytes here.
    model_bytes = common.snapshot(model_path, nvidia_stereo.MODEL['sha256'], nvidia_stereo.MODEL['bytes'])
    common.require(len(model_bytes) == nvidia_stereo.MODEL['bytes'], 'Pinned model size mismatch')
    del model_bytes
    producer = nvidia_stereo.cached(plan_path, producer_path.parent, model_path)
    common.require(common.canonical(producer) == common.canonical(common.parse(producer_raw)), 'Producer changed')
    common.require(registered['model'] == producer['model'] == nvidia_stereo.MODEL, 'Unsupported reconstruction model')
    common.require(derivation['originalInferenceProcessing'] == producer['processing'] and
                   registered['execution']['originalInferenceExecution'] == producer['execution'], 'Original execution/processing changed')
    crop_path = producer_path.parent / producer['sources']['cropManifest']['path']
    binding = adapt_nvidia.bind_crop(observation, observations_path.parent, crop_path)
    common.require(common.digest(binding['cropBytes']) == producer['sources']['cropManifest']['sha256'], 'Crop identity mismatch')
    for name, data in binding['files'].items():
        common.require(common.digest(data) == producer['sources'][name]['sha256'], 'Crop input differs from producer')
    disparity = matched.raster(producer['prediction'], producer_path.parent)
    numeric_mask = matched.raster(producer['validity'], producer_path.parent)
    depth, validity, _ = disparity_registration.register(disparity, numeric_mask,
        binding['calibrationBytes'], binding['shape'], binding['crop'], producer['processing']['disparityConvention'])
    saved_depth = matched.raster(registered['prediction'], prediction_path.parent)
    saved_mask = matched.raster(registered['validity'], prediction_path.parent)
    common.require(np.array_equal(depth.view('u4'), saved_depth.view('u4')) and
                   np.array_equal(validity, saved_mask), 'Registered prediction differs from original disparity')
    height, width = binding['shape']
    c, transform, processing = scene['camera'], scene['transform'], scene['processing']
    common.require(scene['units'] == 'meters' and [c['height'], c['width']] == [height, width], 'Measured camera dimensions/units mismatch')
    left, _, _, _ = disparity_registration.parse_rectified_pair(binding['calibrationBytes'].decode(), width, height)
    common.require([c['fx'], c['fy'], c['cx'], c['cy']] == [left[0, 0], left[1, 1], left[0, 2], left[1, 2]], 'Measured/model calibration mismatch')
    rotation = np.asarray(transform['worldFromCamera'], dtype=np.float64)
    origin = np.asarray(transform['cameraOriginGround'], dtype=np.float64)
    common.require(rotation.shape == (3, 3) and origin.shape == (3,) and np.isfinite(rotation).all() and
                   np.isfinite(origin).all() and np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5, rtol=0) and
                   abs(np.linalg.det(rotation)-1) <= 1e-5, 'Invalid measured road transform')
    common.require(np.allclose(c['eye'], -rotation @ origin, atol=1e-5, rtol=0) and
                   np.allclose(c['forward'], rotation[:, 2], atol=1e-5, rtol=0) and
                   np.allclose(c['up'], -rotation[:, 1], atol=1e-5, rtol=0), 'Measured camera pose mismatch')
    reference_raw = matched.artifact(observation['reference']['raster'], observations_path.parent)
    reference = np.asarray(Image.open(BytesIO(reference_raw)))
    common.require(reference.dtype == np.uint16 and reference.shape == (height, width) and
                   processing['depthDivisor'] == observation['reference']['raster']['divisor'], 'Measured reference encoding mismatch')
    max_depth, edge_limit = processing['maxDepthMeters'], processing['meshEdgeLimitMeters']
    common.require(all(type(n) in (float, int) and np.isfinite(n) and n > 0 for n in (max_depth, edge_limit)), 'Invalid display domain')
    files = {}
    types = {'positions': ('<f4', scene['pointCount']*3), 'colors': ('<f4', scene['pointCount']*3),
             'triangles': ('<u4', scene['triangleCount']*3), 'depth': ('<u2', width*height)}
    for key, (dtype, count) in types.items():
        name = scene['buffers'][key]
        common.require(re.fullmatch('[a-z]+\\.bin', name) is not None, 'Invalid measured artifact name')
        data = common.snapshot(scene_path.parent / name, scene['artifactHashes'][key], 32_000_000)
        common.require(len(data) == count*np.dtype(dtype).itemsize, 'Measured buffer dimensions mismatch')
        files[name] = data
    expected_measured_depth = np.where((reference > 0) & (reference / processing['depthDivisor'] <= max_depth), reference, 0).astype('<u2')
    common.require(files[scene['buffers']['depth']] == expected_measured_depth.tobytes(), 'Measured support differs from original reference')
    for name, role in [('recorded.jpg', 'left'), ('right.jpg', 'right')]:
        data = common.snapshot(scene_path.parent / name, scene['sourceHashes'][role])
        common.require(data == binding['sourceBytes'][role], 'Measured image differs from original model source')
        files[name] = data
    common.require(scene['sourceHashes']['depth'] == common.digest(reference_raw) and
                   scene['sourceHashes']['calibration'] == common.digest(binding['calibrationBytes']), 'Measured source provenance mismatch')
    files['support.png'] = common.snapshot(scene_path.parent / 'support.png')
    # Carry the pinned dataset license even when a historical scene omitted a per-scene copy.
    files['LICENSE'] = matched.artifact(observation['license']['artifact'], observations_path.parent)
    rgb = np.asarray(Image.open(BytesIO(files['recorded.jpg'])).convert('RGB'))
    y, x = np.indices((height, width))
    measured_depth = reference.astype(float) / processing['depthDivisor']
    measured_camera_points = np.stack(((x-c['cx'])*measured_depth/c['fx'],
                                      (y-c['cy'])*measured_depth/c['fy'], measured_depth), axis=-1)
    measured_points, measured_colors, measured_triangles = build_surface(measured_camera_points,
        expected_measured_depth > 0, rgb, origin, rotation, edge_limit)
    expected_buffers = {'positions': measured_points.astype('<f4').tobytes(),
                        'colors': measured_colors.astype('<f4').tobytes(), 'triangles': measured_triangles.astype('<u4').tobytes()}
    common.require(all(files[scene['buffers'][key]] == data for key, data in expected_buffers.items()),
                   'Measured geometry differs from captured source and declared transform')
    crop_x, crop_y, crop_width, crop_height = binding['crop']
    right_supported = np.zeros((height, width), dtype=bool)
    crop_u = np.indices(disparity.shape)[1]
    right_supported[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width] = (crop_u-disparity >= 0) & (crop_u-disparity <= crop_width-1)
    display_mask = (validity == 1) & (depth <= max_depth) & right_supported
    camera_points = np.stack(((x-c['cx'])*depth.astype(float)/c['fx'],
                             (y-c['cy'])*depth.astype(float)/c['fy'], depth.astype(float)), axis=-1)
    points, colors, triangles = build_surface(camera_points, display_mask, rgb, origin, rotation, edge_limit)
    names = {'positions': 'estimatedpositions.bin', 'colors': 'estimatedcolors.bin', 'triangles': 'estimatedtriangles.bin'}
    additions = {names['positions']: points.astype('<f4').tobytes(), names['colors']: colors.astype('<f4').tobytes(),
                 names['triangles']: triangles.astype('<u4').tobytes()}
    common.require(not set(additions) & set(files), 'Reconstruction would replace measured buffer')
    scene['reconstruction'] = dict(evidence='predicted', role='rendererOnly', model=producer['model'],
        pointCount=len(points), triangleCount=len(triangles), bounds=dict(min=points.min(axis=0).tolist(), max=points.max(axis=0).tolist()),
        buffers=names, artifactHashes={key: common.digest(additions[name]) for key, name in names.items()},
        provenance=dict(registeredPredictionSha256=prediction_hash, producerManifestSha256=common.digest(producer_raw),
            planSha256=common.digest(plan_raw), observationsSha256=common.digest(observations_raw),
            measuredSceneSha256=common.digest(scene_raw), datasetLicenseSha256=common.digest(files['LICENSE']), processingCodeHashes=code_hashes),
        displayDomain=dict(maxDepthMeters=max_depth, meshEdgeLimitMeters=edge_limit,
            crop=binding['crop'], rightPixelCenters=[0, crop_width-1], rawNumericPixels=int(validity.sum()), displayedPixels=int(display_mask.sum())),
        limitations=['Estimated rendering only; measured anchors and visibility inputs are unchanged.',
            'Display-domain rejection is not confidence, correspondence, physical continuity or free-space evidence.',
            'Original raw prediction and full-reference diagnostics retain all numerical outliers.',
            'Source calibration/axial-Z/reference precision and model exposure remain provisional.'])
    files.update(additions)
    files['scene.json'] = common.canonical(scene)
    common.snapshot(scene_path, common.digest(scene_raw))
    common.snapshot(prediction_path, prediction_hash)
    common.require(code_hashes == {str(p.relative_to(Path(__file__).parent)): common.digest(p.read_bytes()) for p in source_paths}, 'Processing source changed')
    common.publish(output, files)
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('scene', 'prediction', 'model', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--prediction-sha256', required=True)
    args = parser.parse_args()
    try:
        scene = attach(args.scene, args.prediction, args.prediction_sha256, args.model, args.output)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(1, f'Reconstruction attachment blocked: {type(exc).__name__}: {str(exc)[:250]}\n')
    print(f'Prepared cached NVIDIA estimate: {scene["reconstruction"]["pointCount"]} points; measured query buffers unchanged')


if __name__ == '__main__':
    main()
