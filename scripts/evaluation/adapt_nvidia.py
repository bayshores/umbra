#!/usr/bin/env python3
"""Bind a validated saved NVIDIA crop to a pinned development observation.

No inference, artifact acquisition, license acceptance or cloud dispatch. Cached
receipts are consistency assertions, not authentication of historical execution.
"""
import argparse
import io
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image

import disparity_registration
import matched

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sponsors'))
import common
import nvidia_stereo


def pin(path, data, **fields):
    return dict(path=str(path), bytes=len(data), sha256=matched.digest(data), **fields)


def bind_crop(observation, parent, crop_path):
    """Verify source identity, actual decoded crop pixels and original calibration."""
    if observation['split'] != 'development':
        raise ValueError('Only development observations can be adapted before a reviewed release lock')
    roles = [r['role'] for r in observation['sources']]
    if len(roles) != 2 or set(roles) != {'left', 'right'}:
        raise ValueError('Exactly one original left and right source required')
    crop_raw, crop, files, calibrated = nvidia_stereo.read_crop(crop_path)
    source_records = {r['role']: r for r in observation['sources']}
    captured = {name: matched.artifact(record, parent) for name, record in source_records.items()}
    calibration_record = observation['calibration']['artifact']
    calibration_bytes = matched.artifact(calibration_record, parent)
    reference = observation['reference']['raster']
    matched.artifact(reference, parent)
    shape = matched.shape_of(reference)
    matched.full_frame(observation['reference']['processing'], shape)
    if reference['encoding'] != 'uint16png' or reference['quantity'] != 'axialDepthMeters':
        raise ValueError('Only explicit uint16 axial-depth source reference supported')
    expected_sources = {name: record['sha256'] for name, record in source_records.items()}
    expected_sources.update(calibration=calibration_record['sha256'], depth=reference['sha256'])
    if set(crop['sources']) != set(expected_sources) or any(
            crop['sources'][name]['sha256'] != sha for name, sha in expected_sources.items()):
        raise ValueError('Crop original source identities differ from observation')
    if crop['depth']['divisor'] != reference['divisor']:
        raise ValueError('Crop/reference depth unit declaration differs')
    spec = crop['crop']
    x, y, width, height = (spec[k] for k in ('x', 'y', 'width', 'height'))
    if calibrated['decoded_source_size'] != list(shape[::-1]):
        raise ValueError('Crop original dimensions differ from reference grid')
    for name, original_bytes in captured.items():
        with Image.open(io.BytesIO(original_bytes)) as original:
            if original.mode != 'RGB' or original.size != tuple(shape[::-1]):
                raise ValueError('Original stereo source must be full-grid RGB')
            expected_pixels = np.array(original)[y:y+height, x:x+width]
        with Image.open(io.BytesIO(files[name+'.png'])) as image:
            if not np.array_equal(np.array(image), expected_pixels):
                raise ValueError('Decoded crop pixels differ from original stereo source')
    left, right, source_size, scale = disparity_registration.parse_rectified_pair(
        calibration_bytes.decode('utf-8'), shape[1], shape[0])
    translation = np.array([[1., 0., -x], [0., 1., -y], [0., 0., 1.]])
    expected_calibration = dict(
        source_rectified_size=source_size.tolist(), decoded_source_size=list(shape[::-1]),
        source_to_decoded_scale=scale.tolist(), crop_size=[width, height],
        original=dict(left_projection=left.tolist(), right_projection=right.tolist()),
        transformed=dict(left_projection=(translation@left).tolist(), right_projection=(translation@right).tolist()),
        disparity_numerator_px_m=float(left[0, 3]-right[0, 3]),
        principal_point_offset_px=float(left[0, 2]-right[0, 2]),
        implied_baseline_m=float((left[0, 3]-right[0, 3])/left[0, 0]), provisional=True)
    if common.canonical(calibrated) != common.canonical(expected_calibration):
        raise ValueError('Crop calibration differs from captured original projection matrices')
    return dict(shape=list(shape), crop=[x, y, width, height], calibrationBytes=calibration_bytes,
                cropBytes=crop_raw, files=files, sourceBytes=captured)


def adapt(observation_path, observation_hash, plan_path, plan_hash, prediction_dir, prediction_hash, model_path, output):
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError('Existing output retained')
    observation_bytes = common.snapshot(observation_path, observation_hash)
    plan_bytes = common.snapshot(plan_path, plan_hash)
    prediction_bytes = common.snapshot(Path(prediction_dir)/'prediction.json', prediction_hash)
    observations = matched.parse_json(observation_bytes)
    if type(observations['schemaVersion']) is not int or observations['schemaVersion'] != 1:
        raise ValueError('Unsupported observation schema')
    plan = common.parse(plan_bytes)
    # Reuse strict model/tensor/input/cache validation; never dispatch or infer.
    producer = nvidia_stereo.cached(plan_path, prediction_dir, model_path)
    if common.canonical(producer) != common.canonical(common.parse(prediction_bytes)):
        raise ValueError('Producer manifest changed during validation')
    candidates = [o for o in observations['observations'] if o['id'] == producer['observationId']]
    if len(candidates) != 1:
        raise ValueError('Producer must match exactly one observation ID')
    observation = candidates[0]
    crop_record = producer['sources']['cropManifest']
    crop_path = Path(prediction_dir)/crop_record['path']
    binding = bind_crop(observation, observation_path.parent, crop_path)
    if matched.digest(binding['cropBytes']) != crop_record['sha256']:
        raise ValueError('Bound crop manifest differs from producer')
    for name, data in binding['files'].items():
        if matched.digest(data) != producer['sources'][name]['sha256']:
            raise ValueError('Bound crop input differs from producer')
    disparity = matched.raster(producer['prediction'], Path(prediction_dir))
    validity = matched.raster(producer['validity'], Path(prediction_dir))
    depth, mask, derivation = disparity_registration.register(disparity, validity,
        binding['calibrationBytes'], binding['shape'], binding['crop'], producer['processing']['disparityConvention'])
    shape = binding['shape']
    registered = dict(schemaVersion=1, observationId=observation['id'], model=producer['model'],
        sources=[{**common.record(observation_path.parent/r['path'], binding['sourceBytes'][r['role']], output),
                  'role':r['role']} for r in observation['sources']],
        processing=dict(originalDimensions=shape, inputDimensions=shape, outputDimensions=shape,
            crop=None, pad=None, resize=None, normalization='No normalization in registered evaluation grid; original inference retained in derivation',
            coordinateConvention=derivation['coordinateConvention'], depthConvention='axialZ'),
        derivation=dict(registration=derivation, originalInferenceProcessing=producer['processing'],
            producerManifest=common.record(Path(prediction_dir)/'prediction.json', prediction_bytes, output),
            observationManifest=common.record(observation_path, observation_bytes, output),
            plan=common.record(plan_path, plan_bytes, output),
            originalInferenceRecordBase='Parent of producerManifest; embedded original processing/execution/source records retain their original base',
            originalInferenceSources=producer['sources']),
        prediction=pin('depth.float32le', depth.astype('<f4').tobytes(), shape=shape, encoding='float32le', quantity='axialDepthMeters'),
        validity=pin('validity.uint8', mask.tobytes(), shape=shape, encoding='uint8'),
        execution=dict(status='succeeded', kind='Reuse validated declared NVIDIA cache; conversion and development evaluation only',
            originalInferenceExecution=producer['execution'], environment=dict(numpy=np.__version__),
            codeHashes={str(p.relative_to(Path(__file__).parents[2])): matched.digest(p.read_bytes()) for p in
                        (Path(__file__), Path(disparity_registration.__file__), Path(matched.__file__),
                         Path(__file__).parents[1]/'rectified_calibration.py')},
            usage=dict(newInferenceRuns=0, cloudRequests=0)),
        limitations=['Producer consistency assertions require independent actual execution review.',
                    'Full-grid processing describes registration, not original inference input.',
                    'Numerical validity is not confidence, correspondence or free-space evidence.'])
    files = {'prediction.json':common.canonical(registered), 'depth.float32le':depth.astype('<f4').tobytes(),
             'validity.uint8':mask.tobytes()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.bound-evaluation-', dir=output.parent) as temporary:
        stage = Path(temporary)
        for name, data in files.items():
            (stage/name).write_bytes(data)
        result, errors = matched.evaluate(observation_path, stage/'prediction.json')
        if result['observationManifestHash'] != observation_hash:
            raise ValueError('Observation changed during evaluation')
        result['rawErrors'] = pin('signed-errors.float64le', errors.astype('<f8').tobytes(), encoding='float64le', count=int(errors.size))
        files.update({'result.json':common.canonical(result), 'signed-errors.float64le':errors.astype('<f8').tobytes()})
    common.snapshot(plan_path, plan_hash)
    common.snapshot(Path(prediction_dir)/'prediction.json', prediction_hash)
    common.publish(output, files)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('observations', 'plan', 'prediction-dir', 'model', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('observations-sha256', 'plan-sha256', 'prediction-sha256'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    try:
        adapt(args.observations, args.observations_sha256, args.plan, args.plan_sha256,
              args.prediction_dir, args.prediction_sha256, args.model, args.output)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(1, f'Prediction binding blocked: {type(exc).__name__}\n')


if __name__ == '__main__':
    main()
