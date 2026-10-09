#!/usr/bin/env python3
"""Offline pinned NVIDIA stereo preparation and explicitly gated CPU execution.

No acquisition, license acceptance or cloud dispatch. A normalization review is a
coordinator-supplied declaration, not proof derived from graph input names.
"""
import argparse
from datetime import datetime, timezone
from io import BytesIO
from importlib.metadata import version
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image, __version__ as pillow_version

from common import canonical, code_hashes, digest, load_script, parse, publish, record, require, snapshot

MODEL = {'provider': 'nvidia', 'id': 'nvidia/c-foundationstereo-s',
         'revision': 'd776ba855977e99cc9f16246f911d701da0bbc7a',
         'artifactPath': 'onnx/deployable_foundationstereo_small_320x736_v2.0.onnx',
         'sha256': 'f3106bd1c8e9c81d5d8ca74f8d21e1e8f111a84abf047f75625fd441253e7ad9',
         'bytes': 576104420}
SHAPE = [1, 3, 320, 736]


def read_crop(path):
    raw = snapshot(path)
    crop = parse(raw)
    files = {}
    for name in ('left.png', 'right.png', 'calibration.json'):
        pin = crop['outputs'][name]
        files[name] = snapshot(Path(path).parent / name, pin['sha256'])
        require(len(files[name]) == pin['bytes'], 'Crop artifact byte mismatch')
    for name in ('left.png', 'right.png'):
        with Image.open(BytesIO(files[name])) as image:
            image.load()
            require(image.mode == 'RGB' and image.size == (736, 320), 'Exact RGB crop required')
    calibration = parse(files['calibration.json'])
    spec = crop['crop']
    require(spec['width'] == 736 and spec['height'] == 320 and
            type(spec['x']) is int and type(spec['y']) is int and min(spec['x'], spec['y']) >= 0, 'Crop contract')
    transform = np.array([[1., 0., -spec['x']], [0., 1., -spec['y']], [0., 0., 1.]])
    require(np.array_equal(spec['source_to_crop_matrix'], transform), 'Crop translation mismatch')
    width, height = calibration['decoded_source_size']
    require(type(width) is int and type(height) is int and width >= spec['x'] + 736 and
            height >= spec['y'] + 320 and calibration['crop_size'] == [736, 320], 'Original/crop dimensions')
    for side in ('left_projection', 'right_projection'):
        original = np.asarray(calibration['original'][side], dtype=np.float64)
        transformed = np.asarray(calibration['transformed'][side], dtype=np.float64)
        require(original.shape == (3, 4) and transformed.shape == (3, 4) and
                np.isfinite(original).all() and np.array_equal(transform @ original, transformed),
                'Calibration must use the same exact crop translation')
    return raw, crop, files, calibration


def review_contract(path):
    raw = snapshot(path)
    review = parse(raw)
    require(review['model'] == MODEL and review['licenseAuthorization'] is True and
            review['normalizationReviewed'] is True and
            isinstance(review['reviewEvidence'], str) and review['reviewEvidence'], 'Artifact/license review absent')
    normalization = review['normalization']
    require(isinstance(normalization, dict) and normalization.get('mode') in
            ('raw0_255', 'unit0_1', 'meanStd'), 'Normalization unverified')
    if normalization['mode'] == 'meanStd':
        require(set(normalization) == {'mode', 'mean', 'std'}, 'Normalization keys')
        mean, std = np.asarray(normalization['mean']), np.asarray(normalization['std'])
        require(mean.shape == (3,) and std.shape == (3,) and np.isfinite(mean).all() and
                np.isfinite(std).all() and (std > 0).all(), 'Invalid RGB mean/std')
    else:
        require(set(normalization) == {'mode'}, 'Normalization keys')
    require(review['outputQuantity'] == 'disparityPixels' and
            review['outputConvention'] == 'x_left_minus_x_right_in_input_pixels', 'Output units not reviewed')
    require(isinstance(review['inputNames'], list) and len(review['inputNames']) == 2 and
            len(set(review['inputNames'])) == 2 and all(isinstance(n, str) and n for n in review['inputNames']) and
            isinstance(review['outputName'], str) and review['outputName'], 'Reviewed tensor names')
    inspection_path = Path(path).parent / review['inspection']['path']
    inspection_raw = snapshot(inspection_path, review['inspection']['sha256'])
    inspection = parse(inspection_raw)
    require(inspection['artifact']['sha256'] == MODEL['sha256'] and
            inspection['artifact']['bytes'] == MODEL['bytes'] and
            inspection['status'] == 'embedded_graph_checked' and not inspection['external_data'], 'Real graph inspection absent')
    return raw, review


def execution_limits(threads):
    require(type(threads) is int and threads in (1, 2, 4), 'CPU threads must be an integer in 1, 2, 4')
    return {'batchSize': 1, 'jobs': 1, 'retries': 0, 'threads': threads, 'timeoutSeconds': 120}


def prepare(crop_path, observation_id, output, review_path=None, threads=1):
    limits = execution_limits(threads)
    require(isinstance(observation_id, str) and observation_id, 'Observation ID required')
    raw, crop, files, calibration = read_crop(crop_path)
    review_raw, review = review_contract(review_path) if review_path else (None, None)
    sources = {'cropManifest': record(crop_path, raw, output)}
    sources.update({name: record(Path(crop_path).parent / name, data, output)
                    for name, data in files.items()})
    processing = {'originalDimensions': calibration['decoded_source_size'][::-1],
                  'inputDimensions': [320, 736], 'outputDimensions': [320, 736],
                  'crop': crop['crop'], 'pad': None, 'resize': None,
                  'normalization': review['normalization'] if review else 'unverified',
                  'coordinateConvention': 'OpenCV_x_right_y_down_z_forward; left crop integer pixel centers',
                  'disparityConvention': 'x_left_minus_x_right_in_input_pixels',
                  'validityPolicy': 'finite positive disparity only; not confidence, correspondence or free-space',
                  'calibration': calibration,
                  'cpuConfiguration': {'intraOpThreads': threads, 'interOpThreads': 1, 'provider': 'CPUExecutionProvider'}}
    review_record = record(review_path, review_raw, output) if review else None
    identity = {'observationId': observation_id, 'sources': {k: v['sha256'] for k, v in sources.items()},
                'model': MODEL, 'processing': processing, 'limits': limits,
                'reviewSha256': digest(review_raw) if review_raw else None, 'codeHashes': code_hashes(),
                'preparationHelperSha256': digest((Path(__file__).parents[1] / 'prepare-stereo-crop.py').read_bytes()),
                'inspectionHelperSha256': digest((Path(__file__).parents[1] / 'inspect-onnx-model.py').read_bytes())}
    plan = {'schemaVersion': 1, 'status': 'prepared_not_executed', 'observationId': observation_id,
            'sources': sources, 'model': MODEL, 'processing': processing, 'review': review_record,
            'identity': identity, 'cacheKey': digest(canonical(identity)),
            'limits': limits,
            'blockers': [] if review else ['license authorization, real artifact inspection and normalization review absent'],
            'limitations': ['CPU compatibility unverified; no model execution or sponsor eligibility evidence.',
                            'Reference depth and expected disparity are never read as model predictions.']}
    publish(output, {'plan.json': canonical(plan)})
    return plan


def validate_plan(path):
    raw = snapshot(path)
    plan = parse(raw)
    require(plan['status'] == 'prepared_not_executed' and plan['model'] == MODEL, 'Plan/model mismatch')
    require(isinstance(plan['limits'], dict) and
            canonical(plan['limits']) == canonical(execution_limits(plan['limits'].get('threads'))), 'Execution bound changed')
    require(canonical(plan['processing']['cpuConfiguration']) == canonical(
            {'intraOpThreads': plan['limits']['threads'], 'interOpThreads': 1, 'provider': 'CPUExecutionProvider'}),
            'CPU configuration changed')
    require(plan['review'] is not None and not plan['blockers'], 'Execution blocked; no reviewed contract')
    review_path = Path(path).parent / plan['review']['path']
    review_raw, review = review_contract(review_path)
    require(digest(review_raw) == plan['review']['sha256'], 'Review changed')
    crop_path = Path(path).parent / plan['sources']['cropManifest']['path']
    crop_raw, crop, files, calibration = read_crop(crop_path)
    actual = {'cropManifest': crop_raw, **files}
    for name, data in actual.items():
        require(digest(data) == plan['sources'][name]['sha256'] and
                len(data) == plan['sources'][name]['bytes'], 'Prepared input changed')
        source_path = Path(path).parent / plan['sources'][name]['path']
        require(snapshot(source_path, plan['sources'][name]['sha256']) == data, 'Input path mismatch')
    require(plan['processing']['normalization'] == review['normalization'] and
            plan['processing']['calibration'] == calibration and plan['processing']['crop'] == crop['crop'] and
            plan['processing']['pad'] is None and plan['processing']['resize'] is None and
            plan['processing']['inputDimensions'] == [320, 736] and
            plan['processing']['outputDimensions'] == [320, 736] and
            plan['processing']['originalDimensions'] == calibration['decoded_source_size'][::-1], 'Processing changed')
    require(plan['processing']['coordinateConvention'] ==
            'OpenCV_x_right_y_down_z_forward; left crop integer pixel centers' and
            plan['processing']['disparityConvention'] == 'x_left_minus_x_right_in_input_pixels' and
            plan['processing']['validityPolicy'] ==
            'finite positive disparity only; not confidence, correspondence or free-space', 'Conventions changed')
    expected = {'observationId': plan['observationId'], 'sources': {k: digest(v) for k, v in actual.items()},
                'model': MODEL, 'processing': plan['processing'], 'limits': plan['limits'], 'reviewSha256': digest(review_raw),
                'codeHashes': code_hashes(),
                'preparationHelperSha256': digest((Path(__file__).parents[1] / 'prepare-stereo-crop.py').read_bytes()),
                'inspectionHelperSha256': digest((Path(__file__).parents[1] / 'inspect-onnx-model.py').read_bytes())}
    require(canonical(plan['identity']) == canonical(expected) and plan['cacheKey'] == digest(canonical(expected)), 'Cache/code identity changed')
    return raw, plan, review, files


def tensors(files, normalization):
    result = []
    for name in ('left.png', 'right.png'):
        with Image.open(BytesIO(files[name])) as image:
            image.load()
            rgb = np.asarray(image, dtype=np.float32)
        if normalization['mode'] != 'raw0_255':
            rgb = rgb / np.float32(255)
        if normalization['mode'] == 'meanStd':
            rgb = (rgb - np.asarray(normalization['mean'], dtype=np.float32)) / np.asarray(normalization['std'], dtype=np.float32)
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32)
        require(list(tensor.shape) == SHAPE and np.isfinite(tensor).all(), 'Invalid model tensor')
        result.append(tensor)
    return result


def runtime_environment():
    """Installed package identity only; does not authenticate loaded binaries."""
    return {'python': sys.version, 'numpy': np.__version__, 'pillow': pillow_version,
            'onnx': version('onnx'), 'onnxruntime': version('onnxruntime')}


def validate_execution(execution, plan, review, files, output, model_path):
    require(isinstance(execution, dict) and set(execution) ==
            {'status', 'provider', 'modelArtifact', 'createdUtc', 'elapsedSeconds', 'codeHashes',
             'environment', 'usage', 'inputTensorSha256', 'cpuConfiguration'}, 'Execution metadata keys changed')
    expected_model = {'path': os.path.relpath(Path(model_path).resolve(), Path(output).resolve()),
                      'sha256': MODEL['sha256'], 'bytes': MODEL['bytes'], 'readOnlyExternal': True}
    require(canonical(execution['modelArtifact']) == canonical(expected_model), 'Execution model artifact changed')
    expected_tensors = [digest(x.tobytes()) for x in tensors(files, review['normalization'])]
    require(execution['inputTensorSha256'] == expected_tensors, 'Execution input tensors changed')
    require(execution['status'] == 'succeeded' and execution['provider'] == 'local_onnxruntime_cpu' and
            execution['codeHashes'] == plan['identity']['codeHashes'], 'Execution identity changed')
    require(canonical(execution['cpuConfiguration']) == canonical(plan['processing']['cpuConfiguration']),
            'Execution CPU configuration changed')
    require(canonical(execution['usage']) == canonical({'cloudRequests': 0, 'jobs': 1, 'retries': 0, 'threads': plan['limits']['threads']}),
            'Execution resource contract changed')
    elapsed = execution['elapsedSeconds']
    require(type(elapsed) in (int, float) and math.isfinite(elapsed) and 0 <= elapsed <= 120,
            'Execution elapsed time outside child bound')
    require(isinstance(execution['createdUtc'], str), 'Execution timestamp required')
    created = datetime.fromisoformat(execution['createdUtc'].replace('Z', '+00:00'))
    require(created.tzinfo is not None and created.utcoffset().total_seconds() == 0 and
            created <= datetime.now(timezone.utc), 'Execution timestamp must be recorded UTC')
    require(canonical(execution['environment']) == canonical(runtime_environment()), 'Execution environment changed')


def execute_child(plan_path, model_path, stage, output):
    raw, plan, review, files = validate_plan(plan_path)
    import onnx
    import onnxruntime as ort
    environment = runtime_environment()
    require(environment['onnx'] == onnx.__version__ and environment['onnxruntime'] == ort.__version__,
            'Loaded runtime version differs from installed metadata')
    require(digest((Path(__file__).parents[1] / 'inspect-onnx-model.py').read_bytes()) ==
            plan['identity']['inspectionHelperSha256'], 'Inspection helper changed before execution')
    inspector = load_script('inspect-onnx-model.py')
    # This same hash-checked byte snapshot feeds parsing and ORT; no mutable path reopen.
    model_bytes = snapshot(model_path, MODEL['sha256'], MODEL['bytes'])
    require(len(model_bytes) == MODEL['bytes'], 'Model size mismatch')
    graph = onnx.load_model_from_string(model_bytes)
    require(not any(m.DESCRIPTOR.name == 'TensorProto' and
                    (m.data_location == onnx.TensorProto.EXTERNAL or m.external_data)
                    for _, m in inspector.protobuf_messages(graph)), 'External graph data blocked')
    onnx.checker.check_model(graph, full_check=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = plan['processing']['cpuConfiguration']['intraOpThreads']
    options.inter_op_num_threads = plan['processing']['cpuConfiguration']['interOpThreads']
    constructing = time.monotonic()
    print(canonical({'phase': 'session_construct_begin', 'monotonicSeconds': constructing}).decode(), flush=True)
    session = ort.InferenceSession(model_bytes, sess_options=options, providers=['CPUExecutionProvider'])
    constructed = time.monotonic()
    print(canonical({'phase': 'session_construct_end', 'monotonicSeconds': constructed,
                     'elapsedSeconds': constructed - constructing}).decode(), flush=True)
    session.disable_fallback()
    providers = session.get_providers()
    require(providers == ['CPUExecutionProvider'], 'CPU-only provider required')
    actual_options = session.get_session_options()
    cpu_configuration = {'intraOpThreads': actual_options.intra_op_num_threads,
                         'interOpThreads': actual_options.inter_op_num_threads, 'provider': providers[0]}
    require(canonical(cpu_configuration) == canonical(plan['processing']['cpuConfiguration']),
            'Actual CPU configuration differs from plan')
    inputs, outputs = session.get_inputs(), session.get_outputs()
    require([x.name for x in inputs] == review['inputNames'] and
            all(x.type == 'tensor(float)' and x.shape == SHAPE for x in inputs), 'Actual input contract mismatch')
    require(len(outputs) == 1 and outputs[0].name == review['outputName'] and
            outputs[0].type == 'tensor(float)' and outputs[0].shape == [1, 1, 320, 736], 'Actual output contract mismatch')
    left, right = tensors(files, review['normalization'])
    started = time.monotonic()
    print(canonical({'phase': 'inference_begin', 'monotonicSeconds': started}).decode(), flush=True)
    results = session.run([review['outputName']], dict(zip(review['inputNames'], (left, right))))
    ended = time.monotonic()
    elapsed = ended - started
    print(canonical({'phase': 'inference_end', 'monotonicSeconds': ended, 'elapsedSeconds': elapsed}).decode(), flush=True)
    require(len(results) == 1 and results[0].dtype == np.float32 and results[0].shape == (1, 1, 320, 736), 'Output raster contract mismatch')
    raster = results[0][0, 0]
    disparity = np.ascontiguousarray(raster, dtype='<f4').tobytes()
    validity = np.ascontiguousarray(np.isfinite(raster) & (raster > 0), dtype=np.uint8).tobytes()
    require(code_hashes() == plan['identity']['codeHashes'], 'Adapter source changed during execution')
    for filename, key in [('prepare-stereo-crop.py', 'preparationHelperSha256'),
                          ('inspect-onnx-model.py', 'inspectionHelperSha256')]:
        require(digest((Path(__file__).parents[1] / filename).read_bytes()) == plan['identity'][key],
                'Helper source changed during execution')
    def raster_record(name, data, encoding):
        return {'path': name, 'sha256': digest(data), 'bytes': len(data), 'shape': [320, 736], 'encoding': encoding}
    sources = {name: record(Path(plan_path).parent / rec['path'], snapshot(Path(plan_path).parent / rec['path'], rec['sha256']), output)
               for name, rec in plan['sources'].items()}
    manifest = {'schemaVersion': 1, 'observationId': plan['observationId'], 'sources': sources,
                'model': MODEL, 'processing': plan['processing'], 'cacheKey': plan['cacheKey'],
                'prediction': {**raster_record('disparity.float32le', disparity, 'float32le'), 'quantity': 'disparityPixels'},
                'validity': raster_record('validity.uint8', validity, 'uint8'),
                'execution': {'status': 'succeeded', 'provider': 'local_onnxruntime_cpu',
                              'modelArtifact': record(model_path, model_bytes, output),
                              'createdUtc': datetime.now(timezone.utc).isoformat(), 'elapsedSeconds': elapsed,
                              'codeHashes': plan['identity']['codeHashes'],
                              'environment': environment, 'cpuConfiguration': cpu_configuration,
                              'usage': {'cloudRequests': 0, 'jobs': 1, 'retries': 0, 'threads': plan['limits']['threads']},
                              'inputTensorSha256': [digest(x.tobytes()) for x in (left, right)]},
                'limitations': ['Positive finite validity is numerical, not confidence/free-space.',
                                'Local execution does not establish Nebius use or selected-track eligibility.']}
    for name, data in {'prediction.json': canonical(manifest), 'disparity.float32le': disparity, 'validity.uint8': validity}.items():
        (Path(stage) / name).write_bytes(data)


def run(plan_path, model_path, output):
    # Parent preflight never reads graph or credentials. Timeout kills the single child.
    validate_plan(plan_path)
    output = Path(output).absolute()
    if output.exists():
        return cached(plan_path, output, model_path)
    require(Path(model_path).is_file(), 'Pinned model artifact absent')
    output.parent.mkdir(parents=True, exist_ok=True)
    require(not output.exists(), 'Existing prediction retained')
    with tempfile.TemporaryDirectory(prefix='.cpu-stereo-', dir=output.parent) as stage:
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '_execute', '--plan', str(Path(plan_path).resolve()),
                               '--model', str(Path(model_path).resolve()), '--stage', stage, '--output', str(output)],
                              check=True, timeout=120, capture_output=True)
        files = {name: (Path(stage) / name).read_bytes() for name in
                 ('prediction.json', 'disparity.float32le', 'validity.uint8')}
        files['execution-log.jsonl'] = child.stdout
        publish(output, files)


def cached(plan_path, output, model_path):
    """Check cache consistency, not proof that its asserted execution happened."""
    _, plan, review, files = validate_plan(plan_path)
    manifest = parse(snapshot(Path(output) / 'prediction.json'))
    require(manifest['schemaVersion'] == 1 and manifest['observationId'] == plan['observationId'] and
            manifest['cacheKey'] == plan['cacheKey'] and manifest['model'] == MODEL and
            canonical(manifest['processing']) == canonical(plan['processing']) and
            manifest['execution']['status'] == 'succeeded' and
            manifest['execution']['provider'] == 'local_onnxruntime_cpu' and
            manifest['execution']['codeHashes'] == plan['identity']['codeHashes'], 'Cached prediction identity changed')
    validate_execution(manifest['execution'], plan, review, files, output, model_path)
    payloads = {}
    for name, filename, encoding, size in [('prediction', 'disparity.float32le', 'float32le', 320 * 736 * 4),
                                            ('validity', 'validity.uint8', 'uint8', 320 * 736)]:
        rec = manifest[name]
        require(rec['path'] == filename and rec['shape'] == [320, 736] and rec['encoding'] == encoding and
                rec['bytes'] == size, 'Cached raster contract mismatch')
        data = snapshot(Path(output) / filename, rec['sha256'])
        require(len(data) == size, 'Cached raster size mismatch')
        payloads[name] = data
    require(manifest['prediction']['quantity'] == 'disparityPixels', 'Cached units mismatch')
    disparity = np.frombuffer(payloads['prediction'], dtype='<f4')
    validity = np.frombuffer(payloads['validity'], dtype=np.uint8)
    require(np.array_equal(validity, (np.isfinite(disparity) & (disparity > 0)).astype(np.uint8)), 'Cached validity mismatch')
    require(set(manifest['sources']) == set(plan['sources']), 'Cached input set mismatch')
    for name, rec in manifest['sources'].items():
        data = snapshot(Path(output) / rec['path'], rec['sha256'])
        require(rec.get('readOnlyExternal') is True and rec['sha256'] == plan['sources'][name]['sha256'] and
                len(data) == rec['bytes'] == plan['sources'][name]['bytes'], 'Cached source mismatch')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    for name in ('crop', 'output'):
        prep.add_argument('--' + name, type=Path, required=True)
    prep.add_argument('--observation-id', required=True)
    prep.add_argument('--review', type=Path)
    prep.add_argument('--threads', type=int, choices=(1, 2, 4), default=1)
    for command in ('run', '_execute'):
        child = sub.add_parser(command)
        for name in ('plan', 'model', 'output'):
            child.add_argument('--' + name, type=Path, required=True)
        if command == '_execute':
            child.add_argument('--stage', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            result = prepare(args.crop, args.observation_id, args.output, args.review, threads=args.threads)
            print(result['status'])
        elif args.command == 'run':
            result = run(args.plan, args.model, args.output)
            print('validated_cached_prediction' if result else 'actual_cpu_prediction_published')
        else:
            execute_child(args.plan, args.model, args.stage, args.output)
    except Exception as exc:
        # Do not echo child output or data. Failure never publishes a prediction manifest.
        print(f'NVIDIA branch blocked/failed: {type(exc).__name__}: {str(exc)[:300]}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
