"""Analytic protocol fixtures only: no real model, credentials or service calls."""
import builtins
import copy
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os
from pathlib import Path
import posixpath
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'cloud'))
import consume_result as consumer

binding, evidence = consumer.binding, consumer.evidence
nvidia = binding.nvidia
ID = '0123456789abcdef0123456789abcdef'
EVIDENCE_ROOT = ROOT / '.local' / 'evaluation' / 'nebius-remote-consumption-2026-10-09' / 'worker'


class ConsumeResult(unittest.TestCase):
    def setUp(self):
        EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='analytic-only-', dir=EVIDENCE_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        original_import, original_open, original_os_open = builtins.__import__, Path.open, os.open

        def guarded_import(name, *args, **kwargs):
            if name.split('.')[0] in ('onnx', 'onnxruntime'):
                raise AssertionError('Model libraries forbidden')
            return original_import(name, *args, **kwargs)

        def guarded_open(path, *args, **kwargs):
            if path.suffix == '.onnx':
                raise AssertionError('Model reads forbidden')
            return original_open(path, *args, **kwargs)

        def guarded_os_open(path, *args, **kwargs):
            if str(path).endswith('.onnx'):
                raise AssertionError('Model descriptor reads forbidden')
            return original_os_open(path, *args, **kwargs)

        for guard in (patch.object(builtins, '__import__', side_effect=guarded_import),
                      patch.object(Path, 'open', autospec=True, side_effect=guarded_open),
                      patch.object(os, 'open', side_effect=guarded_os_open),
                      patch.object(socket, 'socket', side_effect=AssertionError('Real network forbidden')),
                      patch.object(subprocess, 'run', side_effect=AssertionError('Child execution forbidden')),
                      patch.object(subprocess, 'Popen', side_effect=AssertionError('Child execution forbidden')),
                      patch.object(evidence, 'fetch_job', side_effect=AssertionError('Real Job fetch forbidden')),
                      patch.object(nvidia, 'execute_child', side_effect=AssertionError('Inference forbidden'))):
            guard.start()
            self.addCleanup(guard.stop)
        self.build_analytic_fixture()

    def build_analytic_fixture(self):
        native = self.root / 'analytic-native'
        native.mkdir()
        rgb = np.zeros((320, 736, 3), dtype=np.uint8)
        rgb[..., 0] = np.arange(736, dtype=np.uint16).astype(np.uint8)
        for name in ('left.png', 'right.png'):
            Image.fromarray(rgb).save(native / name)
        original = np.array([[4., 0., 2., 0.], [0., 4., 2., 0.], [0., 0., 1., 0.]])
        transform = np.array([[1., 0., -3.], [0., 1., -4.], [0., 0., 1.]])
        calibration = {'decoded_source_size': [800, 400], 'crop_size': [736, 320],
                       'original': {k: original.tolist() for k in ('left_projection', 'right_projection')},
                       'transformed': {k: (transform @ original).tolist() for k in ('left_projection', 'right_projection')}}
        (native / 'calibration.json').write_bytes(evidence.canonical(calibration))
        manifest = {'fixtureOnly': True, 'crop': {'x': 3, 'y': 4, 'width': 736, 'height': 320,
                     'source_to_crop_matrix': transform.tolist()},
                    'outputs': {p.name: {'sha256': evidence.digest(p.read_bytes()), 'bytes': p.stat().st_size}
                                for p in native.iterdir()}}
        self.manifest = native / 'manifest.json'
        self.manifest.write_bytes(evidence.canonical(manifest))
        inspection = self.root / 'analytic-inspection.json'
        inspection.write_bytes(evidence.canonical({'fixtureOnly': True, 'artifact': nvidia.MODEL,
                                                  'status': 'embedded_graph_checked', 'external_data': False}))
        self.review = self.root / 'analytic-review.json'
        self.review.write_bytes(evidence.canonical({'fixtureOnly': True, 'model': nvidia.MODEL,
            'licenseAuthorization': True, 'normalizationReviewed': True,
            'reviewEvidence': 'ANALYTIC ONLY, no model inspection or inference.',
            'normalization': {'mode': 'meanStd', 'mean': [0.4, 0.5, 0.6], 'std': [0.1, 0.2, 0.3]},
            'outputQuantity': 'disparityPixels', 'outputConvention': 'x_left_minus_x_right_in_input_pixels',
            'inputNames': ['left', 'right'], 'outputName': 'disparity',
            'inspection': nvidia.record(inspection, inspection.read_bytes(), self.root)}))
        local_prepared = self.root / 'local-prepared'
        self.plan = nvidia.prepare(self.manifest, 'ANALYTIC-NOT-SPONSOR-EVIDENCE', local_prepared, self.review)
        _, _, review, files = nvidia.validate_plan(local_prepared / 'plan.json')
        self.local_plan_raw = (local_prepared / 'plan.json').read_bytes()
        for name, rec in self.plan['sources'].items():
            rec['path'] = '/data/native/' + ('manifest.json' if name == 'cropManifest' else name)
        self.plan['review']['path'] = '/reviews/review.json'
        self.bundle = self.root / 'analytic-bundle'
        prediction = self.bundle / 'prediction'
        prediction.mkdir(parents=True)
        self.plan_raw = evidence.canonical(self.plan)
        (self.bundle / 'plan.json').write_bytes(self.plan_raw)
        environment = {k: 'analytic approved remote ' + k for k in consumer.ENVIRONMENT_FIELDS}
        self.runtime = {'schemaVersion': 1, 'image': 'registry.example/analytic@sha256:' + 'a' * 64,
            'wrapperSha256': evidence.digest(Path(ROOT / 'scripts/cloud/run_stereo_job.py').read_bytes()),
            'environment': environment, 'pythonCommand': 'python', 'wrapperPath': '/app/run_stereo_job.py',
            'planPath': '/work/prepared/plan.json', 'modelPath': '/models/model.onnx', 'outputPath': '/work/output'}
        disparity = np.ones((320, 736), dtype='<f4').tobytes()
        validity = np.ones((320, 736), dtype=np.uint8).tobytes()

        def rec(name, raw, encoding):
            return {'path': name, 'sha256': evidence.digest(raw), 'bytes': len(raw),
                    'shape': [320, 736], 'encoding': encoding}

        self.producer = {'schemaVersion': 1, 'observationId': self.plan['observationId'],
            'model': self.plan['model'], 'processing': self.plan['processing'], 'cacheKey': self.plan['cacheKey'],
            'sources': {name: {**source, 'path': posixpath.relpath(source['path'], '/work/output/prediction')}
                        for name, source in self.plan['sources'].items()},
            'prediction': {**rec('disparity.float32le', disparity, 'float32le'), 'quantity': 'disparityPixels'},
            'validity': rec('validity.uint8', validity, 'uint8'),
            'execution': {'status': 'succeeded', 'provider': 'local_onnxruntime_cpu',
                'modelArtifact': {'path': '../../../models/model.onnx', 'sha256': nvidia.MODEL['sha256'],
                                  'bytes': nvidia.MODEL['bytes'], 'readOnlyExternal': True},
                'createdUtc': '2026-10-09T00:00:02Z', 'elapsedSeconds': 0,
                'codeHashes': self.plan['identity']['codeHashes'], 'environment': environment,
                'cpuConfiguration': self.plan['processing']['cpuConfiguration'],
                'usage': {'cloudRequests': 0, 'jobs': 1, 'retries': 0, 'threads': 1},
                'inputTensorSha256': [evidence.digest(t.tobytes()) for t in nvidia.tensors(files, review['normalization'])]},
            'limitations': ['ANALYTIC ONLY; no service or model execution.']}
        for name, raw in {'prediction.json': evidence.canonical(self.producer), 'disparity.float32le': disparity,
                          'validity.uint8': validity, 'execution-log.jsonl': b'ANALYTIC ONLY\n\xff\x00'}.items():
            (prediction / name).write_bytes(raw)
        self.rebind()
        self.runtime_path, self.expected_path = self.root / 'runtime.json', self.root / 'expected.json'
        self.runtime_path.write_bytes(evidence.canonical(self.runtime))
        self.expected = {'schemaVersion': 1, 'jobId': 'analytic-job', 'projectId': 'analytic-project',
            'image': self.runtime['image'], 'containerCommand': 'python',
            'args': consumer.expected_arguments(self.runtime, ID, evidence.digest(self.plan_raw))}
        self.expected_path.write_bytes(evidence.canonical(self.expected))
        self.kwargs = dict(expected_submission_id=ID, expected_plan_sha256=evidence.digest(self.plan_raw),
            native_manifest_path=self.manifest, expected_native_manifest_sha256=evidence.digest(self.manifest.read_bytes()),
            review_path=self.review, expected_review_sha256=evidence.digest(self.review.read_bytes()),
            expected_job_path=self.expected_path, expected_job_sha256=evidence.digest(self.expected_path.read_bytes()),
            runtime_contract_path=self.runtime_path, expected_runtime_contract_sha256=evidence.digest(self.runtime_path.read_bytes()),
            scratch_root=self.root)
        self.token, self.receipt = self.root / 'analytic-token-NOT-CREDENTIAL', self.root / 'receipt.json'
        self.token.write_text('ANALYTIC_NOT_A_CREDENTIAL')
        self.token.chmod(0o600)
        self.job = {'metadata': {'id': 'analytic-job', 'parentId': 'analytic-project',
                                'createdAt': '2026-10-09T00:00:00Z'},
                    'spec': {k: self.expected[k] for k in ('image', 'containerCommand', 'args')},
                    'status': {'state': 'COMPLETED', 'startedAt': '2026-10-09T00:00:01Z',
                               'finishedAt': '2026-10-09T00:00:03Z'}}

    def rebind(self):
        (self.bundle / 'prediction/prediction.json').write_bytes(evidence.canonical(self.producer))
        raw, payloads = binding.bundle_payloads(self.bundle)
        result = binding.make_binding(raw, payloads, ID, evidence.digest((ROOT / 'scripts/cloud/run_stereo_job.py').read_bytes()))
        (self.bundle / 'job-result.json').write_bytes(evidence.canonical(result) + b'\n')

    def check(self, **overrides):
        return consumer.check_result(self.bundle, **{**self.kwargs, **overrides})

    def collect(self):
        return consumer.collect_result(self.bundle, token_path=self.token, receipt_path=self.receipt, **self.kwargs)

    def test_offline_preserves_bytes_reproduces_tensors_and_never_collects(self):
        originals = {p: p.read_bytes() for p in (self.bundle / 'plan.json', self.bundle / 'prediction/prediction.json',
                                               self.manifest, self.review)}
        with patch.object(evidence, 'collect', side_effect=AssertionError('Offline cannot collect')):
            producer, result, context = self.check(expected_binding_sha256=evidence.digest((self.bundle / 'job-result.json').read_bytes()))
        self.assertEqual(producer, self.producer)
        self.assertEqual(result['inputTensorSha256'], context['inputTensorSha256'])
        self.assertEqual(context['environment'], self.runtime['environment'])
        self.assertEqual(context['claim'], 'offline_consistency_only')
        self.assertNotIn('provider', context)
        self.assertNotIn('authenticated', context)
        self.assertEqual(originals, {p: p.read_bytes() for p in originals})
        self.assertFalse(list(self.root.glob('direct-native-validation-only-*')))
        self.assertEqual((self.root / 'local-prepared/plan.json').read_bytes(), self.local_plan_raw)

    def test_bad_pins_nonce_or_binding_fail_before_credentials(self):
        for override in ({'expected_submission_id': 'f' * 32}, {'expected_plan_sha256': '0' * 64},
                         {'expected_native_manifest_sha256': '0' * 64}, {'expected_review_sha256': '0' * 64},
                         {'expected_job_sha256': '0' * 64}, {'expected_runtime_contract_sha256': '0' * 64},
                         {'expected_binding_sha256': '0' * 64}):
            with self.subTest(override=override), patch.object(evidence, 'read_regular', side_effect=AssertionError('Credential access')), \
                    patch.object(evidence, 'collect') as collect:
                with self.assertRaises(ValueError):
                    consumer.collect_result(self.bundle, token_path=self.token, receipt_path=self.receipt,
                                            **{**self.kwargs, **override})
                collect.assert_not_called()

    def test_rehashed_config_command_image_and_path_mismatches(self):
        original = self.expected_path.read_bytes()
        for key, value in (('containerCommand', 'sh'), ('args', self.expected['args'] + ' --extra'),
                           ('args', self.expected['args'].replace(ID, 'f' * 32)),
                           ('args', self.expected['args'].replace('-B ', '')),
                           ('image', 'registry.example/other@sha256:' + 'b' * 64)):
            candidate = {**self.expected, key: value}
            raw = evidence.canonical(candidate)
            self.expected_path.write_bytes(raw)
            with self.subTest(key=key, value=value), patch.object(evidence, 'collect') as collect:
                with self.assertRaises(ValueError):
                    self.check(expected_job_sha256=evidence.digest(raw))
                collect.assert_not_called()
        self.expected_path.write_bytes(original)

    def test_rehashed_runtime_contract_extra_fields_environment_wrapper_and_unsafe_paths(self):
        for changes in ({'authenticated': True}, {'schemaVersion': True}, {'pythonCommand': '/usr/bin/python'},
                        {'wrapperSha256': '0' * 64}, {'environment': {**self.runtime['environment'], 'python': 'other'}},
                        {'modelPath': '/models/../model.onnx'}, {'outputPath': '/work/output;echo'},
                        {'wrapperPath': '/app/with space.py'}, {'outputPath': '//work/output'}):
            raw = evidence.canonical({**self.runtime, **changes})
            self.runtime_path.write_bytes(raw)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.check(expected_runtime_contract_sha256=evidence.digest(raw))

    def test_rebound_tensor_environment_model_and_source_path_claims_are_rejected(self):
        original = copy.deepcopy(self.producer)
        candidates = []
        for key, value in (('inputTensorSha256', ['0' * 64, '1' * 64]),
                           ('environment', {**self.runtime['environment'], 'numpy': 'other'}),
                           ('modelArtifact', {**original['execution']['modelArtifact'], 'path': '../../other/model.onnx'})):
            candidate = copy.deepcopy(original)
            candidate['execution'][key] = value
            candidates.append(candidate)
        candidate = copy.deepcopy(original)
        candidate['sources']['left.png']['path'] = '../../../other/left.png'
        candidates.append(candidate)
        for candidate in candidates:
            self.producer = candidate
            self.rebind()
            with self.subTest(execution=candidate['execution']), self.assertRaises(ValueError):
                self.check()

    def test_actual_input_bytes_review_and_calibration_mismatch(self):
        for path in (self.manifest.parent / 'left.png', self.manifest.parent / 'calibration.json', self.review):
            original = path.read_bytes()
            path.write_bytes(original + b'\n')
            with self.subTest(path=path.name), self.assertRaises(ValueError):
                self.check()
            path.write_bytes(original)

    def test_rehashed_processing_claims_still_use_unchanged_native_validator(self):
        original_plan, original_producer = copy.deepcopy(self.plan), copy.deepcopy(self.producer)
        for key, value in (('calibration', {**self.plan['processing']['calibration'], 'crop_size': [1, 1]}),
                           ('originalDimensions', [400, 799]),
                           ('normalization', {'mode': 'raw0_255'})):
            self.plan = copy.deepcopy(original_plan)
            self.plan['processing'][key] = value
            self.plan['identity']['processing'] = self.plan['processing']
            self.plan['cacheKey'] = evidence.digest(evidence.canonical(self.plan['identity']))
            plan_raw = evidence.canonical(self.plan)
            (self.bundle / 'plan.json').write_bytes(plan_raw)
            self.producer = copy.deepcopy(original_producer)
            self.producer['processing'] = self.plan['processing']
            self.producer['cacheKey'] = self.plan['cacheKey']
            self.rebind()
            self.expected['args'] = consumer.expected_arguments(self.runtime, ID, evidence.digest(plan_raw))
            expected_raw = evidence.canonical(self.expected)
            self.expected_path.write_bytes(expected_raw)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check(expected_plan_sha256=evidence.digest(plan_raw), expected_job_sha256=evidence.digest(expected_raw))

    def test_successful_direct_mocked_collection_is_separate_and_private(self):
        with patch.object(evidence, 'fetch_job', return_value=evidence.canonical(self.job)) as fetch:
            producer, result, context = self.collect()
        fetch.assert_called_once_with('analytic-job', 'ANALYTIC_NOT_A_CREDENTIAL')
        self.assertEqual(producer, self.producer)
        self.assertEqual(result['producerSha256'], context['producerSha256'])
        self.assertEqual(context['provider'], 'nebius_job')
        self.assertEqual(context['claim'], 'job_control_and_application_output_association')
        self.assertEqual(context['jobControlReceiptSha256'], evidence.digest(self.receipt.read_bytes()))
        self.assertEqual(context['timestamps']['cpuCreatedUtc'], self.producer['execution']['createdUtc'])
        self.assertEqual(self.receipt.stat().st_mode & 0o777, 0o600)
        self.assertNotIn('authenticated', context)
        self.assertEqual(producer['execution']['usage']['cloudRequests'], 0)

    def test_job_interval_boundaries_and_nanoseconds(self):
        for created, started, finished, succeeds in (
                ('2026-10-09T00:00:01Z', '2026-10-09T00:00:01Z', '2026-10-09T00:00:03Z', True),
                ('2026-10-09T00:00:03Z', '2026-10-09T00:00:01Z', '2026-10-09T00:00:03Z', True),
                ('2026-10-09T00:00:01.000000001Z', '2026-10-09T00:00:01.000000001Z', '2026-10-09T00:00:03Z', True),
                ('2026-10-09T00:00:02.999999999Z', '2026-10-09T00:00:01Z', '2026-10-09T00:00:02.999999998Z', False),
                ('2026-10-09T00:00:01Z', '2026-10-09T00:00:01.000000001Z', '2026-10-09T00:00:03Z', False),
                ('2026-10-09T00:00:03Z', '2026-10-09T00:00:01Z', '2026-10-09T00:00:02.999999999Z', False)):
            self.producer['execution']['createdUtc'] = created
            self.rebind()
            self.job['status'].update(startedAt=started, finishedAt=finished)
            with self.subTest(created=created, started=started, finished=finished), patch.object(
                    evidence, 'fetch_job', return_value=evidence.canonical(self.job)):
                if succeeds:
                    self.collect()
                else:
                    with self.assertRaises(ValueError):
                        self.collect()
                    self.assertTrue(self.receipt.exists())
            self.receipt.unlink()

    def test_post_collection_mutations_fail_and_preserve_control_receipt(self):
        for path in (self.manifest, self.review, self.runtime_path, self.expected_path,
                     self.bundle / 'prediction/execution-log.jsonl', self.bundle / 'plan.json'):
            original = path.read_bytes()

            def mutate_after_preflight(_job_id, _token):
                path.write_bytes(original + b'\n')
                return evidence.canonical(self.job)

            with self.subTest(path=path.name), patch.object(evidence, 'fetch_job', side_effect=mutate_after_preflight):
                with self.assertRaises(ValueError):
                    self.collect()
                self.assertTrue(self.receipt.exists())
                receipt = evidence.parse_json(self.receipt.read_bytes())
                self.assertEqual(receipt['claim'], 'job_control_only')
                self.assertFalse(receipt['outputsBound'])
            path.write_bytes(original)
            self.receipt.unlink()

    def test_collected_wrong_control_or_receipt_mutation_cannot_return_context(self):
        with patch.object(evidence, 'fetch_job', return_value=evidence.canonical({**self.job,
                'metadata': {**self.job['metadata'], 'id': 'other-job'}})):
            with self.assertRaises(ValueError):
                self.collect()
        self.assertFalse(self.receipt.exists())
        actual_collect = evidence.collect

        def alter_return(*args):
            receipt = actual_collect(*args)
            receipt['source']['expectedSha256'] = '0' * 64
            return receipt

        with patch.object(evidence, 'fetch_job', return_value=evidence.canonical(self.job)), patch.object(
                evidence, 'collect', side_effect=alter_return):
            with self.assertRaises(ValueError):
                self.collect()
        self.assertTrue(self.receipt.exists())

    def test_original_config_path_mutation_cannot_redirect_frozen_collector(self):
        actual_collect = evidence.collect

        def change_original_before_collector(expected_path, token_path, receipt_path):
            changed = {**self.expected, 'jobId': 'wrong-job'}
            self.expected_path.write_bytes(evidence.canonical(changed))
            self.assertNotEqual(Path(expected_path), self.expected_path)
            self.assertEqual(Path(expected_path).read_bytes(), evidence.canonical(self.expected))
            return actual_collect(expected_path, token_path, receipt_path)

        with patch.object(evidence, 'fetch_job', return_value=evidence.canonical(self.job)) as fetch, \
                patch.object(evidence, 'collect', side_effect=change_original_before_collector):
            with self.assertRaises(ValueError):
                self.collect()
        fetch.assert_called_once_with('analytic-job', 'ANALYTIC_NOT_A_CREDENTIAL')
        self.assertTrue(self.receipt.exists())
        self.assertFalse(list(self.root.glob('direct-native-control-only-*')))

    def test_receipt_no_overwrite_and_preflight_precedes_token_access(self):
        self.receipt.write_bytes(b'prior evidence')
        with patch.object(evidence, 'read_regular', side_effect=AssertionError('Credential access')):
            with self.assertRaises(FileExistsError):
                self.collect()
        self.assertEqual(self.receipt.read_bytes(), b'prior evidence')

    def test_unchanged_local_cached_reader_keeps_original_environment_and_paths(self):
        output = self.root / 'analytic-local-cache'
        output.mkdir()
        local = copy.deepcopy(self.producer)
        local['execution']['environment'] = nvidia.runtime_environment()
        local['execution']['modelArtifact']['path'] = '../UNOPENED-MODEL.onnx'
        local['sources'] = {name: {**source, 'path': os.path.relpath(
            self.manifest if name == 'cropManifest' else self.manifest.parent / name, output)}
                           for name, source in local['sources'].items()}
        (output / 'prediction.json').write_bytes(evidence.canonical(local))
        for name in ('disparity.float32le', 'validity.uint8'):
            (output / name).write_bytes((self.bundle / 'prediction' / name).read_bytes())
        with patch.object(evidence, 'collect', side_effect=AssertionError('Local reader cannot collect')):
            validated = nvidia.cached(self.root / 'local-prepared/plan.json', output,
                                      self.root / 'UNOPENED-MODEL.onnx')
        self.assertEqual(validated, local)
        self.check()
        self.assertEqual(nvidia.cached(self.root / 'local-prepared/plan.json', output,
                                      self.root / 'UNOPENED-MODEL.onnx'), local)

    def test_symlink_native_bundle_and_scratch_are_refused(self):
        for name, target, override in (('native-link', self.manifest.parent, 'native_manifest_path'),
                                       ('scratch-link', self.root, 'scratch_root')):
            link = self.root / name
            link.symlink_to(target, target_is_directory=True)
            value = link / 'manifest.json' if override == 'native_manifest_path' else link
            with self.subTest(override=override), self.assertRaises(OSError):
                self.check(**{override: value})
        link = self.root / 'bundle-link'
        link.symlink_to(self.bundle, target_is_directory=True)
        with self.assertRaises(OSError):
            consumer.check_result(link, **self.kwargs)

    def test_cli_defaults_check_only_requires_explicit_collection_and_redacts(self):
        argv = ['--bundle', str(self.bundle), '--submission-id', ID, '--plan-sha256', self.kwargs['expected_plan_sha256'],
                '--scratch-root', str(self.root)]
        for name, path_key, hash_key in (
                ('native-manifest', 'native_manifest_path', 'expected_native_manifest_sha256'),
                ('review', 'review_path', 'expected_review_sha256'),
                ('expected-job', 'expected_job_path', 'expected_job_sha256'),
                ('runtime-contract', 'runtime_contract_path', 'expected_runtime_contract_sha256')):
            argv.extend(['--' + name, str(self.kwargs[path_key]), '--' + name + '-sha256', self.kwargs[hash_key]])
        with patch.object(evidence, 'collect') as collect, redirect_stdout(StringIO()) as output:
            self.assertEqual(consumer.main(argv), 0)
            self.assertIn('offline consistency only', output.getvalue())
            collect.assert_not_called()
        with redirect_stderr(StringIO()) as error:
            self.assertEqual(consumer.main(argv + ['--collect']), 1)
            self.assertEqual(consumer.main(argv + ['--bad', 'PRIVATE_VALUE']), 1)
        self.assertNotIn('PRIVATE_VALUE', error.getvalue())


if __name__ == '__main__':
    unittest.main()
