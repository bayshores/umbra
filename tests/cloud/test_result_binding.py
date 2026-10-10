"""Explicit analytic fixtures only; never sponsor/model execution evidence."""
import builtins
import copy
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from io import StringIO
import os
from pathlib import Path
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
import result_binding as binding
import run_stereo_job as wrapper

nvidia = binding.nvidia
ID = '0123456789abcdef0123456789abcdef'


class ResultBinding(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.out = self.root / 'analytic-bundle'
        self.model = self.root / 'analytic-NOT-A-MODEL.onnx'
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.split('.')[0] in ('onnx', 'onnxruntime'):
                raise AssertionError('Model library import forbidden; analytic fixtures only')
            return original_import(name, *args, **kwargs)

        original_open = Path.open

        def guarded_open(path, *args, **kwargs):
            if path.suffix == '.onnx':
                raise AssertionError('Model read forbidden; analytic fixtures only')
            return original_open(path, *args, **kwargs)

        for guard in (patch.object(builtins, '__import__', side_effect=guarded_import),
                      patch.object(Path, 'open', autospec=True, side_effect=guarded_open),
                      patch.object(socket, 'socket', side_effect=AssertionError('Live network forbidden')),
                      patch.object(nvidia.subprocess, 'run', side_effect=AssertionError('Real child execution forbidden'))):
            guard.start()
            self.addCleanup(guard.stop)
        self.prepare_analytic_fixture()

    def prepare_analytic_fixture(self):
        crop = self.root / 'analytic-crop'
        crop.mkdir()
        rgb = np.zeros((320, 736, 3), dtype=np.uint8)
        rgb[..., 0] = 255
        for side in ('left', 'right'):
            Image.fromarray(rgb).save(crop / (side + '.png'))
        matrix = [[4., 0., 2., 0.], [0., 4., 2., 0.], [0., 0., 1., 0.]]
        calibration = {'decoded_source_size': [736, 320], 'crop_size': [736, 320],
                       'original': {k: matrix for k in ('left_projection', 'right_projection')},
                       'transformed': {k: matrix for k in ('left_projection', 'right_projection')}}
        (crop / 'calibration.json').write_bytes(binding.canonical(calibration))
        crop_manifest = {'fixtureOnly': True, 'crop': {'x': 0, 'y': 0, 'width': 736, 'height': 320,
                         'source_to_crop_matrix': [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]},
                         'outputs': {p.name: {'sha256': binding.digest(p.read_bytes()), 'bytes': p.stat().st_size}
                                     for p in crop.iterdir()}}
        cp = crop / 'manifest.json'
        cp.write_bytes(binding.canonical(crop_manifest))
        inspection = self.root / 'analytic-inspection-NOT-A-GRAPH.json'
        inspection.write_bytes(binding.canonical({'fixtureOnly': True, 'artifact': nvidia.MODEL,
                                                 'status': 'embedded_graph_checked', 'external_data': False}))
        review = self.root / 'analytic-review-NOT-MODEL-EVIDENCE.json'
        review.write_bytes(binding.canonical({'model': nvidia.MODEL, 'licenseAuthorization': True,
            'normalizationReviewed': True, 'reviewEvidence': 'Analytic protocol fixture, no model execution.',
            'normalization': {'mode': 'raw0_255'}, 'outputQuantity': 'disparityPixels',
            'outputConvention': 'x_left_minus_x_right_in_input_pixels', 'inputNames': ['left', 'right'],
            'outputName': 'disparity', 'inspection': nvidia.record(inspection, inspection.read_bytes(), self.root)}))
        self.plan = nvidia.prepare(cp, 'analytic-fixture-NOT-SPONSOR-EVIDENCE', self.root / 'prepared', review)
        self.pp = self.root / 'prepared' / 'plan.json'
        self.plan_raw = self.pp.read_bytes()
        self.plan_sha = binding.digest(self.plan_raw)

    def analytic_adapter(self, plan_path, model_path, output):
        """Single synthetic adapter call, with real local cache checks afterward."""
        _, plan, review, files = nvidia.validate_plan(plan_path)
        raster = np.ones((320, 736), dtype='<f4')
        raster.flat[:10] = [np.nan, np.inf, -np.inf, -0., -1., 0., 1e30, 1e-30,
                            np.nextafter(np.float32(0), np.float32(1)), np.finfo(np.float32).max]
        disparity = raster.tobytes()
        validity = (np.isfinite(raster) & (raster > 0)).astype(np.uint8).tobytes()
        def rec(name, data, encoding):
            return {'path': name, 'sha256': binding.digest(data), 'bytes': len(data),
                    'shape': [320, 736], 'encoding': encoding}
        manifest = {'schemaVersion': 1, 'observationId': plan['observationId'],
            'model': nvidia.MODEL, 'processing': plan['processing'], 'cacheKey': plan['cacheKey'],
            'sources': {name: nvidia.record(Path(plan_path).parent / value['path'],
                         nvidia.snapshot(Path(plan_path).parent / value['path']), output)
                        for name, value in plan['sources'].items()},
            'prediction': {**rec('disparity.float32le', disparity, 'float32le'), 'quantity': 'disparityPixels'},
            'validity': rec('validity.uint8', validity, 'uint8'),
            'execution': {'status': 'succeeded', 'provider': 'local_onnxruntime_cpu',
                'modelArtifact': {'path': os.path.relpath(model_path, output), 'sha256': nvidia.MODEL['sha256'],
                                  'bytes': nvidia.MODEL['bytes'], 'readOnlyExternal': True},
                'createdUtc': datetime.now(timezone.utc).isoformat(), 'elapsedSeconds': 0,
                'codeHashes': plan['identity']['codeHashes'], 'environment': nvidia.runtime_environment(),
                'cpuConfiguration': plan['processing']['cpuConfiguration'],
                'usage': {'cloudRequests': 0, 'jobs': 1, 'retries': 0, 'threads': 1},
                'inputTensorSha256': [binding.digest(t.tobytes()) for t in nvidia.tensors(files, review['normalization'])]},
            'limitations': ['ANALYTIC TEST FIXTURE ONLY; no model or service execution.']}
        self.producer_raw = binding.canonical(manifest)
        self.disparity_raw = disparity
        Path(output).mkdir()
        for name, raw in {'prediction.json': self.producer_raw, 'disparity.float32le': disparity,
                          'validity.uint8': validity, 'execution-log.jsonl': b'Analytic raw log only\n\xff\x00'}.items():
            (Path(output) / name).write_bytes(raw)

    def build(self):
        with patch.object(nvidia, 'run', side_effect=self.analytic_adapter) as run:
            result = wrapper.run_job(ID, self.plan_sha, self.pp, self.model, self.out, execute=True)
        run.assert_called_once_with(self.pp, self.model, self.out / 'prediction')
        return result

    def validate(self, **kwargs):
        return binding.validate_binding(self.out, ID, self.plan_sha, **kwargs)

    def write_binding(self, value):
        (self.out / 'job-result.json').write_bytes(binding.canonical(value))

    def rebind_mutated_producer(self, producer):
        pp = self.out / 'prediction' / 'prediction.json'
        pp.write_bytes(binding.canonical(producer))
        original = binding.parse_json((self.out / 'job-result.json').read_bytes())
        original['producerSha256'] = binding.digest(pp.read_bytes())
        original['outputs']['prediction.json'].update(bytes=pp.stat().st_size, sha256=original['producerSha256'])
        self.write_binding(original)

    def test_single_analytic_call_preserves_original_cpu_bytes_and_all_tails(self):
        result = self.build()
        producer, readback = self.validate(expected_binding_sha256=binding.digest((self.out / 'job-result.json').read_bytes()))
        self.assertEqual(result, readback)
        self.assertEqual(set(result), binding.BINDING_FIELDS)
        self.assertEqual((self.out / 'plan.json').read_bytes(), self.plan_raw)
        self.assertEqual((self.out / 'prediction' / 'prediction.json').read_bytes(), self.producer_raw)
        self.assertEqual((self.out / 'prediction' / 'disparity.float32le').read_bytes(), self.disparity_raw)
        self.assertEqual(producer['execution']['provider'], 'local_onnxruntime_cpu')
        self.assertEqual(result['wrapperSha256'], binding.digest(Path(wrapper.__file__).read_bytes()))
        self.assertNotIn('authenticated', result)
        self.assertNotIn('modelExecutionVerified', result)
        self.assertEqual(self.pp.read_bytes(), self.plan_raw)

    def test_execute_is_explicit_and_ids_are_strict_before_any_adapter_call(self):
        for identifier, plan_sha, execute in ((ID, self.plan_sha, False), ('A' * 32, self.plan_sha, True),
                (ID + '0', self.plan_sha, True), (ID, 'B' * 64, True), (ID, True, True)):
            with self.subTest(identifier=identifier), patch.object(nvidia, 'run') as run, patch.object(nvidia, 'validate_plan') as plan:
                with self.assertRaises(ValueError):
                    wrapper.run_job(identifier, plan_sha, self.pp, self.model, self.out, execute=execute)
                run.assert_not_called()
                plan.assert_not_called()
                self.assertFalse(self.out.exists())

    def test_existing_file_directory_and_dangling_symlink_refused_before_reads(self):
        for kind in ('file', 'directory', 'symlink'):
            path = self.root / kind
            if kind == 'file':
                path.write_bytes(b'preserved')
            elif kind == 'directory':
                path.mkdir()
            else:
                path.symlink_to(self.root / 'missing')
            with self.subTest(kind=kind), patch.object(wrapper, 'read_path') as read, patch.object(nvidia, 'run') as run:
                with self.assertRaises(FileExistsError):
                    wrapper.run_job(ID, self.plan_sha, self.pp, self.model, path, execute=True)
                read.assert_not_called()
                run.assert_not_called()
        self.assertEqual((self.root / 'file').read_bytes(), b'preserved')

    def test_wrong_plan_hash_refuses_before_adapter(self):
        with patch.object(nvidia, 'run') as run, self.assertRaises(ValueError):
            wrapper.run_job(ID, '0' * 64, self.pp, self.model, self.out, execute=True)
        run.assert_not_called()
        self.assertFalse((self.out / 'job-result.json').exists())

    def test_failed_adapter_keeps_raw_logs_no_retry_no_cache_no_binding(self):
        for error in (subprocess.CalledProcessError(-9, ['analytic-only'], output=b'raw\xffstdout', stderr=b'raw\x00stderr'),
                      subprocess.TimeoutExpired(['analytic-only'], 120, output=b'raw\xffstdout', stderr=b'raw\x00stderr')):
            out = self.root / type(error).__name__
            with patch.object(nvidia, 'run', side_effect=error) as run, patch.object(nvidia, 'cached') as cache:
                with self.assertRaises(type(error)):
                    wrapper.run_job(ID, self.plan_sha, self.pp, self.model, out, execute=True)
                run.assert_called_once()
                cache.assert_not_called()
            self.assertEqual((out / 'adapter-failure.stdout').read_bytes(), b'raw\xffstdout')
            self.assertEqual((out / 'adapter-failure.stderr').read_bytes(), b'raw\x00stderr')
            self.assertFalse((out / 'job-result.json').exists())

    def test_failed_same_process_cache_does_not_retry_or_publish_binding(self):
        with patch.object(nvidia, 'run', side_effect=self.analytic_adapter) as run, patch.object(nvidia, 'cached', side_effect=ValueError('analytic failure')):
            with self.assertRaises(ValueError):
                wrapper.run_job(ID, self.plan_sha, self.pp, self.model, self.out, execute=True)
        run.assert_called_once()
        self.assertEqual((self.out / 'prediction' / 'prediction.json').read_bytes(), self.producer_raw)
        self.assertFalse((self.out / 'job-result.json').exists())

    def test_adapter_race_cache_return_is_refused_without_cache_retry(self):
        with patch.object(nvidia, 'run', return_value={'analytic': 'existing cache race'}) as run, patch.object(nvidia, 'cached') as cache:
            with self.assertRaises(ValueError):
                wrapper.run_job(ID, self.plan_sha, self.pp, self.model, self.out, execute=True)
        run.assert_called_once()
        cache.assert_not_called()
        self.assertFalse((self.out / 'job-result.json').exists())

    def test_plan_changed_during_adapter_is_refused(self):
        def changed(*args):
            self.analytic_adapter(*args)
            self.pp.write_bytes(b'{}')
        with patch.object(nvidia, 'run', side_effect=changed) as run, self.assertRaises((ValueError, KeyError)):
            wrapper.run_job(ID, self.plan_sha, self.pp, self.model, self.out, execute=True)
        run.assert_called_once()
        self.assertFalse((self.out / 'job-result.json').exists())

    def test_plan_changed_after_binding_write_rolls_back_only_own_marker(self):
        original_write = wrapper.write_new
        def changed(directory, name, raw):
            info = original_write(directory, name, raw)
            if name == 'job-result.json':
                self.pp.write_bytes(b'{}')
            return info
        with patch.object(nvidia, 'run', side_effect=self.analytic_adapter) as run, patch.object(wrapper, 'write_new', side_effect=changed):
            with self.assertRaises(ValueError):
                wrapper.run_job(ID, self.plan_sha, self.pp, self.model, self.out, execute=True)
        run.assert_called_once()
        self.assertFalse((self.out / 'job-result.json').exists())
        self.assertEqual((self.out / 'prediction' / 'prediction.json').read_bytes(), self.producer_raw)

    def test_strict_binding_json_fields_nonce_plan_and_pins(self):
        original = self.build()
        cases = [dict(original, extra=True), {k: v for k, v in original.items() if k != 'cacheKey'},
                 dict(original, schemaVersion=True), dict(original, claim='authenticated'),
                 dict(original, submissionId='1' * 32), dict(original, planSha256='0' * 64),
                 dict(original, modelSha256='0' * 64), dict(original, cacheKey='0' * 64),
                 dict(original, producerSha256='0' * 64), dict(original, wrapperSha256='BAD'),
                 dict(original, inputTensorSha256=['0' * 64, '1' * 64]),
                 dict(original, processingCodeHashes={'fake.py': '0' * 64})]
        for changed in cases:
            with self.subTest(changed=set(changed)):
                self.write_binding(changed)
                with self.assertRaises(ValueError):
                    self.validate()
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e999}', b'[]', b'\xff',
                    b' ' * (binding.MAX_MANIFEST_BYTES + 1)):
            (self.out / 'job-result.json').write_bytes(raw)
            with self.assertRaises((ValueError, UnicodeError)):
                self.validate()
        self.write_binding(original)
        with self.assertRaises(ValueError):
            self.validate(expected_binding_sha256='0' * 64)

    def test_output_paths_fields_extra_and_bit_changes_are_rejected(self):
        original = self.build()
        for path in ('../prediction.json', '/prediction.json', 'prediction/../prediction.json',
                     'prediction\\prediction.json', 'prediction/prediction.json/extra'):
            changed = copy.deepcopy(original)
            changed['outputs']['prediction.json']['path'] = path
            self.write_binding(changed)
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.validate()
        for change in ('missing', 'extra', 'boolbytes', 'extraField'):
            changed = copy.deepcopy(original)
            if change == 'missing':
                del changed['outputs']['validity.uint8']
            elif change == 'extra':
                changed['outputs']['extra'] = changed['outputs']['validity.uint8']
            elif change == 'boolbytes':
                changed['outputs']['validity.uint8']['bytes'] = True
            else:
                changed['outputs']['validity.uint8']['extra'] = True
            self.write_binding(changed)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate()
        self.write_binding(original)
        for name in binding.OUTPUT_NAMES:
            path = self.out / 'prediction' / name
            raw = path.read_bytes()
            path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
            with self.subTest(name=name), self.assertRaises((ValueError, UnicodeError)):
                self.validate()
            path.write_bytes(raw)
        (self.out / 'prediction' / 'extra').write_bytes(b'extra')
        with self.assertRaises(ValueError):
            self.validate()

    def test_bundle_plan_mutation_and_missing_output(self):
        self.build()
        (self.out / 'plan.json').write_bytes(self.plan_raw + b' ')
        with self.assertRaises(ValueError):
            self.validate()
        (self.out / 'plan.json').write_bytes(self.plan_raw)
        (self.out / 'prediction' / 'execution-log.jsonl').unlink()
        with self.assertRaises(ValueError):
            self.validate()

    def test_plan_canonical_identity_rejects_cache_code_sources_and_typed_limits(self):
        for parent, key, value in (('identity', 'codeHashes', {'fake.py': '0' * 64}),
                                   ('identity', 'sources', {}), ('limits', 'threads', True),
                                   ('limits', 'retries', 1), ('limits', 'timeoutSeconds', 121)):
            plan = copy.deepcopy(self.plan)
            plan[parent][key] = value
            with self.subTest(parent=parent, key=key), self.assertRaises(ValueError):
                binding.validate_plan_identity(plan)
        plan = copy.deepcopy(self.plan)
        plan['cacheKey'] = '0' * 64
        with self.assertRaises(ValueError):
            binding.validate_plan_identity(plan)

    def test_json_strictness_applies_to_original_producer_and_nested_binding(self):
        original = self.build()
        producer_path = self.out / 'prediction' / 'prediction.json'
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e999}',
                    b' ' * (binding.MAX_MANIFEST_BYTES + 1)):
            producer_path.write_bytes(raw)
            changed = copy.deepcopy(original)
            changed['producerSha256'] = binding.digest(raw)
            changed['outputs']['prediction.json'].update(bytes=len(raw), sha256=binding.digest(raw))
            self.write_binding(changed)
            with self.assertRaises(ValueError):
                self.validate()
        producer_path.write_bytes(self.producer_raw)
        raw = binding.canonical(original).replace(b'"schemaVersion":1', b'"schemaVersion":1,"schemaVersion":1')
        (self.out / 'job-result.json').write_bytes(raw)
        with self.assertRaises(ValueError):
            self.validate()

    def test_output_ceiling_is_collective_and_prediction_directory_cannot_be_symlink(self):
        self.build()
        log = self.out / 'prediction' / 'execution-log.jsonl'
        with log.open('wb') as stream:
            stream.truncate(binding.MAX_OUTPUT_BYTES)
        with self.assertRaises(ValueError):
            self.validate()
        log.write_bytes(b'Analytic raw log only\n\xff\x00')
        prediction = self.out / 'prediction'
        moved = self.root / 'moved-analytic-prediction'
        prediction.rename(moved)
        prediction.symlink_to(moved, target_is_directory=True)
        with self.assertRaises(OSError):
            self.validate()

    def test_rehashed_producer_model_cache_cpu_usage_environment_tensor_mutations(self):
        self.build()
        original = binding.parse_json(self.producer_raw)
        cases = [('model', 'sha256', '0' * 64), ('execution', 'provider', 'nebius'),
                 ('execution', 'environment', {}), ('execution', 'elapsedSeconds', True),
                 ('execution', 'inputTensorSha256', ['invalid']), ('execution', 'codeHashes', {}),
                 ('execution', 'cpuConfiguration', {'intraOpThreads': 4, 'interOpThreads': 1, 'provider': 'CPUExecutionProvider'}),
                 ('execution', 'usage', {'cloudRequests': 1, 'jobs': 1, 'retries': 0, 'threads': 1}),
                 ('prediction', 'shape', [320.0, 736]), ('prediction', 'path', '../disparity.float32le'),
                 ('prediction', 'bytes', 1), ('validity', 'encoding', 'float32le')]
        for parent, key, value in cases:
            producer = copy.deepcopy(original)
            producer[parent][key] = value
            self.rebind_mutated_producer(producer)
            with self.subTest(parent=parent, key=key), self.assertRaises(ValueError):
                self.validate()
        for key, value in (('cacheKey', '0' * 64), ('sources', {}), ('schemaVersion', True)):
            producer = copy.deepcopy(original)
            producer[key] = value
            self.rebind_mutated_producer(producer)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate()

    def test_validity_must_match_actual_full_raster_even_after_rehash(self):
        self.build()
        path = self.out / 'prediction' / 'validity.uint8'
        raw = bytearray(path.read_bytes())
        raw[0] = 1  # NaN is invalid numerically, even with all asserted hashes updated.
        path.write_bytes(raw)
        producer = binding.parse_json(self.producer_raw)
        producer['validity']['sha256'] = binding.digest(raw)
        self.rebind_mutated_producer(producer)
        changed = binding.parse_json((self.out / 'job-result.json').read_bytes())
        changed['outputs']['validity.uint8']['sha256'] = binding.digest(raw)
        self.write_binding(changed)
        with self.assertRaises(ValueError):
            self.validate()

    def test_no_symlink_files_directories_fifo_or_oversized_log(self):
        self.build()
        path = self.out / 'prediction' / 'execution-log.jsonl'
        raw = path.read_bytes()
        external = self.root / 'external-analytic-log'
        external.write_bytes(raw)
        path.unlink()
        path.symlink_to(external)
        with self.assertRaises(OSError):
            self.validate()
        path.unlink()
        os.mkfifo(path)
        with self.assertRaises(ValueError):
            self.validate()
        path.unlink()
        with path.open('wb') as stream:
            stream.truncate(binding.MAX_OUTPUT_BYTES + 1)
        with self.assertRaises(ValueError):
            self.validate()
        path.write_bytes(raw)
        linked = self.root / 'linked-bundle'
        linked.symlink_to(self.out, target_is_directory=True)
        with self.assertRaises(OSError):
            binding.validate_binding(linked, ID, self.plan_sha)

    def test_cli_refuses_implicit_execution_and_redacts_errors(self):
        args = ['--submission-id', ID, '--plan-sha256', self.plan_sha, '--plan', str(self.pp),
                '--model', str(self.model), '--output', str(self.out)]
        for cli in (args, ['--unrecognized', 'analytic-private-error']):
            stderr, stdout = StringIO(), StringIO()
            with redirect_stderr(stderr), redirect_stdout(stdout), patch.object(nvidia, 'run') as run:
                self.assertEqual(wrapper.main(cli), 1)
            run.assert_not_called()
            self.assertNotIn('analytic-private-error', stderr.getvalue())
            self.assertEqual(stdout.getvalue(), '')


if __name__ == '__main__':
    unittest.main()
