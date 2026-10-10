"""Bounded offline application-output consistency; no service authentication.

The bundle retains the exact submitted plan at plan.json and the original CPU
producer under prediction/. The caller owns trusted retrieval and image/runtime
checks. Hash agreement cannot establish that an asserted execution happened.
"""
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import stat
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sponsors'))
import nvidia_stereo as nvidia
from job_evidence import canonical, digest, parse_json, require

MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_OUTPUT_BYTES = 16 * 1024 * 1024
OUTPUT_NAMES = ('prediction.json', 'disparity.float32le', 'validity.uint8', 'execution-log.jsonl')
BINDING_FIELDS = {'schemaVersion', 'claim', 'submissionId', 'planSha256', 'modelSha256',
                  'cacheKey', 'producerSha256', 'inputTensorSha256', 'processingCodeHashes',
                  'wrapperSha256', 'outputs'}
SHA = re.compile(r'[0-9a-f]{64}\Z')
SUBMISSION = re.compile(r'[0-9a-f]{32}\Z')


def sha(value):
    require(type(value) is str and SHA.fullmatch(value) is not None)
    return value


def submission(value):
    require(type(value) is str and SUBMISSION.fullmatch(value) is not None)
    return value


def open_directory(path):
    """Walk an absolute path without following directory symlinks."""
    path = Path(os.path.abspath(path))
    descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def read_at(directory, name, limit=MAX_MANIFEST_BYTES):
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    with os.fdopen(descriptor, 'rb') as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_size <= limit)
        raw = stream.read(limit + 1)
        require(len(raw) == info.st_size and len(raw) <= limit)
        return raw


def read_path(path, limit=MAX_MANIFEST_BYTES):
    path = Path(os.path.abspath(path))
    directory = open_directory(path.parent)
    try:
        return read_at(directory, path.name, limit)
    finally:
        os.close(directory)


def hash_mapping(value):
    require(type(value) is dict and bool(value))
    for name, value_sha in value.items():
        require(type(name) is str and name.endswith('.py') and Path(name).name == name)
        sha(value_sha)


def tensor_hashes(value):
    require(type(value) is list and len(value) == 2)
    for value_sha in value:
        sha(value_sha)


def record_identity(rec):
    require(type(rec) is dict and set(rec) == {'path', 'sha256', 'bytes', 'readOnlyExternal'})
    require(type(rec['path']) is str and bool(rec['path']) and '\x00' not in rec['path'])
    sha(rec['sha256'])
    require(type(rec['bytes']) is int and 0 < rec['bytes'] <= MAX_MANIFEST_BYTES
            and rec['readOnlyExternal'] is True)


def validate_plan_identity(plan):
    require(set(plan) == {'schemaVersion', 'status', 'observationId', 'sources', 'model', 'processing',
                          'review', 'identity', 'cacheKey', 'limits', 'blockers', 'limitations'})
    require(type(plan['schemaVersion']) is int and plan['schemaVersion'] == 1
            and plan['status'] == 'prepared_not_executed' and plan['model'] == nvidia.MODEL
            and plan['blockers'] == [] and type(plan['observationId']) is str
            and bool(plan['observationId']))
    limits = plan['limits']
    require(canonical(limits) == canonical(nvidia.execution_limits(limits['threads'])))
    processing = plan['processing']
    require(canonical(processing['cpuConfiguration']) == canonical(
        {'intraOpThreads': limits['threads'], 'interOpThreads': 1, 'provider': 'CPUExecutionProvider'}))
    require(canonical(processing['inputDimensions']) == canonical([320, 736])
            and canonical(processing['outputDimensions']) == canonical([320, 736])
            and processing['pad'] is None and processing['resize'] is None
            and processing['coordinateConvention'] == 'OpenCV_x_right_y_down_z_forward; left crop integer pixel centers'
            and processing['disparityConvention'] == 'x_left_minus_x_right_in_input_pixels'
            and processing['validityPolicy'] == 'finite positive disparity only; not confidence, correspondence or free-space')
    sources = plan['sources']
    require(type(sources) is dict and set(sources) == {'cropManifest', 'left.png', 'right.png', 'calibration.json'})
    for rec in sources.values():
        record_identity(rec)
    record_identity(plan['review'])
    identity = plan['identity']
    hash_mapping(identity['codeHashes'])
    sha(identity['preparationHelperSha256'])
    sha(identity['inspectionHelperSha256'])
    expected = {'observationId': plan['observationId'], 'sources': {k: v['sha256'] for k, v in sources.items()},
                'model': plan['model'], 'processing': processing, 'limits': limits,
                'reviewSha256': plan['review']['sha256'], 'codeHashes': identity['codeHashes'],
                'preparationHelperSha256': identity['preparationHelperSha256'],
                'inspectionHelperSha256': identity['inspectionHelperSha256']}
    require(canonical(identity) == canonical(expected) and sha(plan['cacheKey']) == digest(canonical(expected)))


def validate_producer(producer, plan, payloads):
    require(set(producer) == {'schemaVersion', 'observationId', 'sources', 'model', 'processing',
                              'cacheKey', 'prediction', 'validity', 'execution', 'limitations'})
    require(type(producer['schemaVersion']) is int and producer['schemaVersion'] == 1
            and producer['observationId'] == plan['observationId'] and producer['model'] == nvidia.MODEL
            and producer['cacheKey'] == plan['cacheKey']
            and canonical(producer['processing']) == canonical(plan['processing']))
    execution = producer['execution']
    require(type(execution) is dict and set(execution) ==
            {'status', 'provider', 'modelArtifact', 'createdUtc', 'elapsedSeconds', 'codeHashes',
             'environment', 'usage', 'inputTensorSha256', 'cpuConfiguration'})
    require(execution['status'] == 'succeeded' and execution['provider'] == 'local_onnxruntime_cpu'
            and canonical(execution['codeHashes']) == canonical(plan['identity']['codeHashes'])
            and canonical(execution['cpuConfiguration']) == canonical(plan['processing']['cpuConfiguration'])
            and canonical(execution['usage']) == canonical(
                {'cloudRequests': 0, 'jobs': 1, 'retries': 0, 'threads': plan['limits']['threads']}))
    model = execution['modelArtifact']
    require(type(model) is dict and set(model) == {'path', 'sha256', 'bytes', 'readOnlyExternal'}
            and type(model['path']) is str and bool(model['path']) and '\x00' not in model['path']
            and model['sha256'] == nvidia.MODEL['sha256'] and type(model['bytes']) is int
            and model['bytes'] == nvidia.MODEL['bytes'] and model['readOnlyExternal'] is True)
    tensor_hashes(execution['inputTensorSha256'])
    environment = execution['environment']
    require(type(environment) is dict and set(environment) == {'python', 'numpy', 'pillow', 'onnx', 'onnxruntime'}
            and all(type(value) is str and 0 < len(value) <= 8192 for value in environment.values()))
    elapsed = execution['elapsedSeconds']
    require(type(elapsed) in (int, float) and 0 <= elapsed <= 120)
    require(type(execution['createdUtc']) is str)
    created = datetime.fromisoformat(execution['createdUtc'].replace('Z', '+00:00'))
    require(created.tzinfo is not None and created.utcoffset().total_seconds() == 0
            and created <= datetime.now(timezone.utc))
    require(set(producer['sources']) == set(plan['sources']))
    for name, rec in producer['sources'].items():
        record_identity(rec)
        require(rec['sha256'] == plan['sources'][name]['sha256'] and rec['bytes'] == plan['sources'][name]['bytes'])
    for key, filename, encoding, size in (
            ('prediction', 'disparity.float32le', 'float32le', 320 * 736 * 4),
            ('validity', 'validity.uint8', 'uint8', 320 * 736)):
        rec = producer[key]
        expected_fields = {'path', 'sha256', 'bytes', 'shape', 'encoding'} | ({'quantity'} if key == 'prediction' else set())
        require(type(rec) is dict and set(rec) == expected_fields and rec['path'] == filename
                and canonical(rec['shape']) == canonical([320, 736]) and rec['encoding'] == encoding
                and type(rec['bytes']) is int and rec['bytes'] == size
                and len(payloads[filename]) == size and sha(rec['sha256']) == digest(payloads[filename]))
    require(producer['prediction']['quantity'] == 'disparityPixels')
    disparity = np.frombuffer(payloads['disparity.float32le'], dtype='<f4')
    validity = np.frombuffer(payloads['validity.uint8'], dtype=np.uint8)
    require(np.array_equal(validity, (np.isfinite(disparity) & (disparity > 0)).astype(np.uint8)))


def bundle_payloads(bundle_root):
    """Read exactly four flat outputs, bounded collectively, preserving raw tails."""
    root = open_directory(bundle_root)
    try:
        plan_raw = read_at(root, 'plan.json')
        prediction = os.open('prediction', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root)
        try:
            require(set(os.listdir(prediction)) == set(OUTPUT_NAMES))
            payloads = {}
            for name in OUTPUT_NAMES:
                limit = MAX_MANIFEST_BYTES if name == 'prediction.json' else MAX_OUTPUT_BYTES
                payloads[name] = read_at(prediction, name, limit)
            require(sum(map(len, payloads.values())) <= MAX_OUTPUT_BYTES)
        finally:
            os.close(prediction)
        return plan_raw, payloads
    finally:
        os.close(root)


def make_binding(plan_raw, payloads, submission_id, wrapper_sha256):
    """Construct consistency metadata after the wrapper's original cache checks."""
    submission(submission_id)
    sha(wrapper_sha256)
    plan, producer = parse_json(plan_raw), parse_json(payloads['prediction.json'])
    validate_plan_identity(plan)
    validate_producer(producer, plan, payloads)
    return {'schemaVersion': 1, 'claim': 'application_output_binding_only', 'submissionId': submission_id,
            'planSha256': digest(plan_raw), 'modelSha256': producer['model']['sha256'],
            'cacheKey': producer['cacheKey'], 'producerSha256': digest(payloads['prediction.json']),
            'inputTensorSha256': producer['execution']['inputTensorSha256'],
            'processingCodeHashes': producer['execution']['codeHashes'], 'wrapperSha256': wrapper_sha256,
            'outputs': {name: {'path': 'prediction/' + name, 'bytes': len(raw), 'sha256': digest(raw)}
                        for name, raw in payloads.items()}}


def validate_binding(bundle_root, expected_submission_id, expected_plan_sha256, *, expected_binding_sha256=None):
    """Return original producer and binding dicts; offline consistency only.

    No model read, inference, external-source traversal or provider authentication.
    The caller must separately revalidate submitted inputs/tensors and the approved
    image's runtime environment before any genuine remote result is admitted.
    """
    submission(expected_submission_id)
    sha(expected_plan_sha256)
    raw = read_path(Path(bundle_root) / 'job-result.json')
    if expected_binding_sha256 is not None:
        require(sha(expected_binding_sha256) == digest(raw))
    binding = parse_json(raw)
    require(set(binding) == BINDING_FIELDS and type(binding['schemaVersion']) is int and binding['schemaVersion'] == 1
            and binding['claim'] == 'application_output_binding_only'
            and submission(binding['submissionId']) == expected_submission_id
            and sha(binding['planSha256']) == expected_plan_sha256)
    plan_raw, payloads = bundle_payloads(bundle_root)
    require(digest(plan_raw) == expected_plan_sha256)
    expected = make_binding(plan_raw, payloads, expected_submission_id, sha(binding['wrapperSha256']))
    require(canonical(binding) == canonical(expected))
    return parse_json(payloads['prediction.json']), binding
