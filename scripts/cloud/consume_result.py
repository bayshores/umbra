#!/usr/bin/env python3
"""Check direct-native outputs offline; explicitly collect ordinary Job control.

Only a direct collector call can supply service context. Controlled retrieval of
the bundle remains the caller's responsibility; no JSON flag authenticates it.
No model weights, model libraries, inference, dispatch or billing are accessed.
"""
import copy
import os
from pathlib import Path
import posixpath
import re
import sys
import tempfile

import job_evidence as evidence
import result_binding as binding

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_FIELDS = {'schemaVersion', 'image', 'wrapperSha256', 'environment', 'pythonCommand',
                  'wrapperPath', 'planPath', 'modelPath', 'outputPath'}
ENVIRONMENT_FIELDS = {'python', 'numpy', 'pillow', 'onnx', 'onnxruntime'}
REMOTE_PATH = re.compile(r'/[A-Za-z0-9_./-]+\Z')


def remote_path(value):
    evidence.require(type(value) is str and len(value) <= 4096
                     and REMOTE_PATH.fullmatch(value) is not None
                     and posixpath.normpath(value) == value and not value.startswith('//')
                     and value != '/')
    return value


def runtime_contract(raw):
    contract = evidence.parse_json(raw)
    evidence.require(set(contract) == RUNTIME_FIELDS and type(contract['schemaVersion']) is int
                     and contract['schemaVersion'] == 1 and contract['pythonCommand'] == 'python')
    binding.sha(contract['wrapperSha256'])
    environment = contract['environment']
    evidence.require(type(environment) is dict and set(environment) == ENVIRONMENT_FIELDS
                     and all(type(value) is str and 0 < len(value) <= 8192
                             for value in environment.values()))
    for key in ('wrapperPath', 'planPath', 'modelPath', 'outputPath'):
        remote_path(contract[key])
    evidence.require(len({contract[key] for key in ('wrapperPath', 'planPath', 'modelPath', 'outputPath')}) == 4)
    return contract


def expected_arguments(contract, submission_id, plan_sha256):
    return ' '.join(('-B', contract['wrapperPath'], '--submission-id', submission_id,
                     '--plan-sha256', plan_sha256, '--plan', contract['planPath'],
                     '--model', contract['modelPath'], '--output', contract['outputPath'], '--execute'))


def resolve_reference(parent, value):
    evidence.require(type(value) is str and 0 < len(value) <= 4096
                     and re.fullmatch(r'[A-Za-z0-9_./-]+', value) is not None
                     and not value.startswith('//') and posixpath.normpath(value) == value)
    return posixpath.normpath(posixpath.join(parent, value))


def _recheck(captured):
    for path, (raw, limit) in captured.items():
        evidence.require(binding.read_path(path, limit) == raw)


def _preflight(bundle_root, *, expected_submission_id, expected_plan_sha256,
               native_manifest_path, expected_native_manifest_sha256,
               review_path, expected_review_sha256, expected_job_path, expected_job_sha256,
               runtime_contract_path, expected_runtime_contract_sha256,
               expected_binding_sha256=None, scratch_root=None):
    binding.submission(expected_submission_id)
    binding.sha(expected_plan_sha256)
    captured = {}

    def capture(path, pin=None, limit=binding.MAX_MANIFEST_BYTES):
        path = Path(os.path.abspath(path))
        raw = binding.read_path(path, limit)
        if pin is not None:
            evidence.require(binding.sha(pin) == evidence.digest(raw))
        if path in captured:
            evidence.require(captured[path][0] == raw)
        captured[path] = (raw, limit)
        return raw

    for path in (native_manifest_path, review_path, expected_job_path, runtime_contract_path):
        evidence.require(Path(path).suffix == '.json')
    expected_raw = capture(expected_job_path, expected_job_sha256)
    expected = evidence.expected_spec(expected_raw)
    runtime_raw = capture(runtime_contract_path, expected_runtime_contract_sha256)
    runtime = runtime_contract(runtime_raw)
    evidence.require(runtime['image'] == expected['image']
                     and expected['containerCommand'] == runtime['pythonCommand']
                     and expected['args'] == expected_arguments(runtime, expected_submission_id, expected_plan_sha256))
    wrapper_raw = capture(Path(__file__).with_name('run_stereo_job.py'), runtime['wrapperSha256'])
    capture(evidence.__file__)
    capture(binding.__file__)
    capture(__file__)

    bundle_root = Path(os.path.abspath(bundle_root))
    plan_raw = capture(bundle_root / 'plan.json', expected_plan_sha256)
    binding_raw = capture(bundle_root / 'job-result.json', expected_binding_sha256)
    for name in binding.OUTPUT_NAMES:
        capture(bundle_root / 'prediction' / name,
                limit=binding.MAX_MANIFEST_BYTES if name == 'prediction.json' else binding.MAX_OUTPUT_BYTES)
    producer, result = binding.validate_binding(bundle_root, expected_submission_id,
                                               expected_plan_sha256, expected_binding_sha256=expected_binding_sha256)
    evidence.require(producer == evidence.parse_json(captured[bundle_root / 'prediction' / 'prediction.json'][0])
                     and result == evidence.parse_json(binding_raw)
                     and result['wrapperSha256'] == evidence.digest(wrapper_raw)
                     and evidence.canonical(producer['execution']['environment']) == evidence.canonical(runtime['environment']))
    # Interpret remote model metadata with POSIX semantics, never against this host.
    model_reference = producer['execution']['modelArtifact']['path']
    evidence.require(not model_reference.startswith('/')
                     and resolve_reference(posixpath.join(runtime['outputPath'], 'prediction'), model_reference)
                     == runtime['modelPath'])

    native_manifest_path = Path(os.path.abspath(native_manifest_path))
    review_path = Path(os.path.abspath(review_path))
    native_raw = capture(native_manifest_path, expected_native_manifest_sha256)
    review_raw = capture(review_path, expected_review_sha256)
    review = evidence.parse_json(review_raw)
    inspection = review['inspection']
    evidence.require(type(inspection['path']) is str and bool(inspection['path'])
                     and '\x00' not in inspection['path'])
    inspection_path = review_path.parent / inspection['path']
    evidence.require(inspection_path.suffix == '.json')
    capture(inspection_path, inspection['sha256'])
    local_sources = {'cropManifest': native_manifest_path,
                     **{name: native_manifest_path.parent / name
                        for name in ('left.png', 'right.png', 'calibration.json')}}
    plan = evidence.parse_json(plan_raw)
    for name in plan['sources']:
        evidence.require(resolve_reference(posixpath.dirname(runtime['planPath']), plan['sources'][name]['path'])
                         == resolve_reference(posixpath.join(runtime['outputPath'], 'prediction'),
                                              producer['sources'][name]['path']))
    for name, path in local_sources.items():
        raw = native_raw if name == 'cropManifest' else capture(path, plan['sources'][name]['sha256'])
        evidence.require(len(raw) == plan['sources'][name]['bytes']
                         and evidence.digest(raw) == plan['sources'][name]['sha256'])
    evidence.require(plan['review']['sha256'] == evidence.digest(review_raw)
                     and plan['review']['bytes'] == len(review_raw))
    for name, pin in plan['identity']['codeHashes'].items():
        capture(Path(binding.nvidia.__file__).parent / name, pin)
    for name, key in (('prepare-stereo-crop.py', 'preparationHelperSha256'),
                      ('inspect-onnx-model.py', 'inspectionHelperSha256')):
        capture(PROJECT_ROOT / 'scripts' / name, plan['identity'][key])

    scratch_root = Path(os.path.abspath(scratch_root if scratch_root is not None else PROJECT_ROOT / '.local'))
    descriptor = binding.open_directory(scratch_root)
    os.close(descriptor)
    # Only these reference strings change in the clearly labelled validation copy.
    # The original remote plan and CPU producer remain byte-for-byte captured.
    local_plan = copy.deepcopy(plan)
    for name, path in local_sources.items():
        local_plan['sources'][name]['path'] = str(path)
    local_plan['review']['path'] = str(review_path)
    with tempfile.TemporaryDirectory(prefix='direct-native-validation-only-', dir=scratch_root) as temporary:
        validation_path = Path(temporary) / 'validation-plan-only.json'
        validation_path.write_bytes(evidence.canonical(local_plan))
        _, validated, validated_review, files = binding.nvidia.validate_plan(validation_path)
        evidence.require(validated == local_plan)
        actual_tensors = [evidence.digest(tensor.tobytes())
                          for tensor in binding.nvidia.tensors(files, validated_review['normalization'])]
    evidence.require(actual_tensors == producer['execution']['inputTensorSha256']
                     and actual_tensors == result['inputTensorSha256'])
    _recheck(captured)
    context = {'schemaVersion': 1, 'claim': 'offline_consistency_only',
               'submissionId': expected_submission_id, 'requestedImage': expected['image'],
               'expectedJobSha256': evidence.digest(expected_raw),
               'runtimeContractSha256': evidence.digest(runtime_raw), 'bindingSha256': evidence.digest(binding_raw),
               'planSha256': evidence.digest(plan_raw), 'producerSha256': result['producerSha256'],
               'wrapperSha256': result['wrapperSha256'], 'nativeManifestSha256': evidence.digest(native_raw),
               'reviewSha256': evidence.digest(review_raw), 'inputTensorSha256': actual_tensors,
               'inputSha256': {name: evidence.digest(captured[path][0]) for name, path in local_sources.items()},
               'cpuConfiguration': producer['execution']['cpuConfiguration'],
               'environment': producer['execution']['environment'],
               'limitations': ['Offline consistency cannot authenticate a provider or admit a remote result.',
                               'Requested image and recorded environment do not attest running hardware or binaries.',
                               'Controlled bundle retrieval is required independently.']}
    return producer, result, context, expected, captured


def check_result(bundle_root, **kwargs):
    """Return original producer/binding and consistency-only context; zero service calls."""
    producer, result, context, _, _ = _preflight(bundle_root, **kwargs)
    return producer, result, context


def collect_result(bundle_root, *, token_path, receipt_path, **kwargs):
    """Preflight, invoke the direct collector once, associate, then recheck inputs.

    An association failure retains the collector's genuine control receipt. The
    caller must supply a bundle retrieved under their control for remote use.
    """
    producer, result, context, expected, captured = _preflight(bundle_root, **kwargs)
    scratch_root = Path(os.path.abspath(kwargs.get('scratch_root') or PROJECT_ROOT / '.local'))
    descriptor = binding.open_directory(scratch_root)
    os.close(descriptor)
    expected_path = Path(os.path.abspath(kwargs['expected_job_path']))
    # Freeze the exact preflight bytes, so a changed caller path cannot redirect
    # the collector between preflight and its credential read/ordinary GET.
    with tempfile.TemporaryDirectory(prefix='direct-native-control-only-', dir=scratch_root) as temporary:
        frozen_expected = Path(temporary) / 'frozen-expected-job.json'
        frozen_expected.write_bytes(captured[expected_path][0])
        frozen_expected.chmod(0o600)
        _recheck(captured)
        receipt = evidence.collect(frozen_expected, token_path, receipt_path)
    receipt_raw = binding.read_path(receipt_path)
    captured[Path(os.path.abspath(receipt_path))] = (receipt_raw, binding.MAX_MANIFEST_BYTES)
    evidence.require(evidence.parse_json(receipt_raw) == receipt and set(receipt) ==
                     {'schemaVersion', 'claim', 'modelExecutionVerified', 'outputsBound', 'acquiredAt', 'job', 'source'}
                     and type(receipt['schemaVersion']) is int and receipt['schemaVersion'] == 1
                     and receipt['claim'] == 'job_control_only' and receipt['modelExecutionVerified'] is False
                     and receipt['outputsBound'] is False)
    job = copy.deepcopy(receipt['job'])
    if job['metadata'].get('resourceVersion') is None:
        job['metadata'].pop('resourceVersion', None)
    normalized = evidence.normalized_job(evidence.canonical(job), expected, receipt['acquiredAt'])
    evidence.require(normalized == receipt['job'])
    source = receipt['source']
    request = {'method': 'GET', 'url': 'https://' + evidence.HOST + '/ai/v1/jobs/' + expected['jobId'],
               'headers': {'Accept': 'application/json', 'Accept-Encoding': 'identity'}}
    evidence.require(set(source) == {'kind', 'apiSchemaCommit', 'collectorSha256', 'expectedSha256',
                                    'request', 'requestSha256', 'responseSha256', 'httpStatus'}
                     and source['kind'] == 'nebius_job_get' and source['apiSchemaCommit'] == evidence.API_COMMIT
                     and source['collectorSha256'] == evidence.digest(captured[Path(evidence.__file__)][0])
                     and source['expectedSha256'] == context['expectedJobSha256']
                     and source['request'] == request and source['requestSha256'] == evidence.digest(evidence.canonical(request))
                     and type(source['httpStatus']) is int and source['httpStatus'] == 200)
    binding.sha(source['responseSha256'])
    created = producer['execution']['createdUtc']
    cpu_instant = evidence.timestamp(created[:-6] + 'Z' if created.endswith('+00:00') else created)
    evidence.require(evidence.timestamp(normalized['status']['startedAt']) <= cpu_instant
                     <= evidence.timestamp(normalized['status']['finishedAt']))
    _recheck(captured)
    context.update(claim='job_control_and_application_output_association', provider='nebius_job',
                   jobId=expected['jobId'], projectId=expected['projectId'],
                   jobControlReceiptSha256=evidence.digest(receipt_raw),
                   timestamps={'jobCreatedAt': normalized['metadata']['createdAt'],
                               'startedAt': normalized['status']['startedAt'],
                               'cpuCreatedUtc': producer['execution']['createdUtc'],
                               'finishedAt': normalized['status']['finishedAt'], 'acquiredAt': receipt['acquiredAt']},
                   limitations=['Service context comes only from the direct ordinary Job collector call.',
                                'Controlled bundle retrieval remains a separate caller requirement.',
                                'Requested image and recorded environment do not attest running hardware or binaries.'])
    return producer, result, context


def main(argv=None):
    parser = evidence.RedactedParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--submission-id', required=True)
    parser.add_argument('--plan-sha256', required=True)
    for name in ('native-manifest', 'review', 'expected-job', 'runtime-contract'):
        parser.add_argument('--' + name, required=True, type=Path)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--binding-sha256')
    parser.add_argument('--scratch-root', type=Path)
    parser.add_argument('--collect', action='store_true')
    parser.add_argument('--token-file', type=Path)
    parser.add_argument('--receipt', type=Path)
    try:
        args = parser.parse_args(argv)
        kwargs = dict(expected_submission_id=args.submission_id, expected_plan_sha256=args.plan_sha256,
                      native_manifest_path=args.native_manifest, expected_native_manifest_sha256=args.native_manifest_sha256,
                      review_path=args.review, expected_review_sha256=args.review_sha256,
                      expected_job_path=args.expected_job, expected_job_sha256=args.expected_job_sha256,
                      runtime_contract_path=args.runtime_contract, expected_runtime_contract_sha256=args.runtime_contract_sha256,
                      expected_binding_sha256=args.binding_sha256, scratch_root=args.scratch_root)
        if args.collect:
            evidence.require(args.token_file is not None and args.receipt is not None)
            collect_result(args.bundle, token_path=args.token_file, receipt_path=args.receipt, **kwargs)
        else:
            evidence.require(args.token_file is None and args.receipt is None)
            check_result(args.bundle, **kwargs)
    except (Exception, KeyboardInterrupt):
        print('Result consumption failed or refused; details redacted.', file=sys.stderr)
        return 1
    print('Associated direct Job control; controlled output retrieval remains required.' if args.collect else
          'Checked offline consistency only; no authenticated remote admission.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
