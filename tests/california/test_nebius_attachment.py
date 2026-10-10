"""Analytic attachment fixtures only. Never sponsor, cloud or model evidence."""
import builtins
from copy import deepcopy
from io import BytesIO
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

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests/cloud'))
sys.path.insert(0, str(ROOT / 'tests/california'))
import test_consume_result as protocol_fixtures
import test_attach_av2_prediction as native_fixtures

tool = native_fixtures.tool
consumer = protocol_fixtures.consumer
EVIDENCE = ROOT / '.local/evaluation/nebius-remote-consumption-2026-10-09/attachment-worker'


class NebiusAttachmentTests(unittest.TestCase):
    def setUp(self):
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix='analytic-only-', dir=EVIDENCE)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / 'TEST-ONLY.txt').write_text('Analytic fixture only; no model, cloud or sponsor execution.\n')
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
                      patch.object(socket, 'socket', side_effect=AssertionError('Network forbidden')),
                      patch.object(subprocess, 'run', side_effect=AssertionError('Child execution forbidden')),
                      patch.object(subprocess, 'Popen', side_effect=AssertionError('Child execution forbidden')),
                      patch.object(consumer.evidence, 'fetch_job', side_effect=AssertionError('Real Job fetch forbidden')),
                      patch.object(tool.nvidia_stereo, 'execute_child', side_effect=AssertionError('Inference forbidden'))):
            guard.start()
            self.addCleanup(guard.stop)
        self.build_fixture()

    def write(self, path, value):
        raw = value if isinstance(value, bytes) else tool.common.canonical(value)
        path.write_bytes(raw)
        return dict(path=path.name, sha256=tool.common.digest(raw), bytes=len(raw))

    def build_fixture(self):
        # Reuse the analytic protocol builder, then replace its crop with the
        # measured-native helper's full source/pose/calibration contract.
        self.protocol = fixture = protocol_fixtures.ConsumeResult()
        fixture.root = self.root
        fixture.build_analytic_fixture()
        native = native_fixtures.NativePredictionTests()
        native.directory = fixture.manifest.parent
        native.camera = dict(fx=10., fy=20., cx=3., cy=2., height=320, width=736, near=1., far=20.)
        scene, prepared, calibration, *_ = native.fixture_binding()
        prepared['outputs'].update({name: self.write(native.directory / name, raw) for name, raw in (
            ('left.png', (native.directory / 'left.png').read_bytes()),
            ('right.png', (native.directory / 'right.png').read_bytes()),
            ('calibration.json', tool.common.canonical(calibration)),
            ('LICENSE-AND-ATTRIBUTION.md', (native.directory / 'license').read_bytes()))})
        references = BytesIO()
        np.savez(references, positions=np.array([[1., 0., 5.]]), eligible=np.array([1], dtype='u1'))
        prepared['outputs']['references.npz'] = self.write(native.directory / 'references.npz', references.getvalue())
        self.write(fixture.manifest, prepared)
        scene.update(id='ANALYTIC-NOT-SPONSOR-EVIDENCE', pointCount=1, triangleCount=0)
        measured = scene['measurements']
        measured['count'] = 1
        measured['provenance'].update(preparedManifestSha256=tool.common.digest(fixture.manifest.read_bytes()),
            preparedManifest=self.write(fixture.manifest, prepared), references=prepared['outputs']['references.npz'],
            referencesSha256=prepared['outputs']['references.npz']['sha256'])
        for owner, buffers in ((scene, dict(positions=np.array([1., 0., 5.], '<f4').tobytes(),
                                           colors=np.array([.1, .2, .3], '<f4').tobytes(), triangles=b'')),
                               (measured, dict(referencePositions=np.array([1., 0., 5.], '<f8').tobytes(),
                                    referenceCameraPositions=np.array([1., 0., 5.], '<f8').tobytes(),
                                    referenceUV=np.array([5., 2.], '<f8').tobytes(), referenceEligible=b'\x01',
                                    sourceIndices=np.array([0], '<u4').tobytes()))):
            owner['buffers'], owner['artifactHashes'] = {}, {}
            for name, raw in buffers.items():
                record = self.write(native.directory / (name.lower() + '.bin'), raw)
                owner['buffers'][name], owner['artifactHashes'][name] = record['path'], record['sha256']
        recorded = self.write(native.directory / 'recorded.png', (native.directory / 'left.png').read_bytes())
        scene['recordedImage'] = dict(file=recorded['path'], mimeType='image/png',
                                      sha256=recorded['sha256'], bytes=recorded['bytes'])
        self.scene_path = native.directory / 'scene.json'
        self.scene = deepcopy(scene)
        self.write(self.scene_path, scene)
        local_prepared = self.root / 'native-prepared'
        fixture.plan = tool.nvidia_stereo.prepare(fixture.manifest, scene['id'], local_prepared, fixture.review)
        _, _, review, files = tool.nvidia_stereo.validate_plan(local_prepared / 'plan.json')
        fixture.local_plan = local_prepared / 'plan.json'
        for name, record in fixture.plan['sources'].items():
            record['path'] = '/data/native/' + ('manifest.json' if name == 'cropManifest' else name)
        fixture.plan['review']['path'] = '/reviews/review.json'
        fixture.plan_raw = tool.common.canonical(fixture.plan)
        (fixture.bundle / 'plan.json').write_bytes(fixture.plan_raw)
        fixture.producer.update(observationId=scene['id'], processing=fixture.plan['processing'],
            cacheKey=fixture.plan['cacheKey'], sources={name: {**record,
                'path': posixpath.relpath(record['path'], '/work/output/prediction')}
                for name, record in fixture.plan['sources'].items()})
        fixture.producer['execution'].update(codeHashes=fixture.plan['identity']['codeHashes'],
            inputTensorSha256=[tool.common.digest(t.tobytes()) for t in tool.nvidia_stereo.tensors(files, review['normalization'])])
        disparity = np.full(tool.SHAPE, np.nan, '<f4')
        disparity[3, 8], disparity[4, 9], disparity[1, 3] = 2., 1., 1e-5
        disparity[2, 0], disparity[0, 200] = 1., 100.
        for key, raw in (('prediction', disparity.tobytes()),
                         ('validity', (np.isfinite(disparity) & (disparity > 0)).astype('u1').tobytes())):
            record = fixture.producer[key]
            (fixture.bundle / 'prediction' / record['path']).write_bytes(raw)
            record.update(sha256=tool.common.digest(raw), bytes=len(raw))
        fixture.rebind()
        fixture.expected['args'] = consumer.expected_arguments(fixture.runtime, protocol_fixtures.ID,
                                                              tool.common.digest(fixture.plan_raw))
        fixture.expected_path.write_bytes(tool.common.canonical(fixture.expected))
        fixture.job['spec'] = {key: fixture.expected[key] for key in ('image', 'containerCommand', 'args')}
        fixture.kwargs.update(expected_plan_sha256=tool.common.digest(fixture.plan_raw),
            expected_native_manifest_sha256=tool.common.digest(fixture.manifest.read_bytes()),
            expected_job_sha256=tool.common.digest(fixture.expected_path.read_bytes()))
        self.options = {key: value for key, value in fixture.kwargs.items()
                        if key not in {'expected_plan_sha256', 'native_manifest_path', 'expected_native_manifest_sha256'}}
        self.options.update(token_path=fixture.token, receipt_path=fixture.receipt)
        self.output = self.root / 'analytic-attached'

    def attach(self, options=None, direct_native=True, producer_path=None, plan_path=None):
        fixture = self.protocol
        paths = [self.scene_path, fixture.manifest,
                 producer_path or fixture.bundle / 'prediction/prediction.json', plan_path or fixture.bundle / 'plan.json']
        args = [item for path in paths for item in (path, tool.common.digest(path.read_bytes()))]
        return tool.attach(*args, None, None, self.root / 'NEVER-READ-MODEL.onnx', self.output,
                           direct_native=direct_native, nebius_options=self.options if options is None else options)

    def test_full_analytic_flow_preserves_measurements_raw_tails_and_separate_service_artifacts(self):
        fixture = self.protocol
        with patch.object(consumer.evidence, 'fetch_job', return_value=tool.common.canonical(fixture.job)) as fetch, \
                patch.object(tool.nvidia_stereo, 'cached', side_effect=AssertionError('Remote cannot use local cache')):
            attached = self.attach()
        fetch.assert_called_once_with('analytic-job', 'ANALYTIC_NOT_A_CREDENTIAL')
        for key in ('camera', 'transform', 'sourceHashes', 'pointCount', 'triangleCount', 'buffers', 'artifactHashes'):
            self.assertEqual(attached[key], self.scene[key])
        measured = deepcopy(attached['measurements'])
        for name in ('preparedManifest', 'references'):
            measured['provenance'][name]['path'] = self.scene['measurements']['provenance'][name]['path']
        self.assertEqual(measured, self.scene['measurements'])
        for owner in (self.scene, self.scene['measurements']):
            for name in owner['buffers'].values():
                self.assertEqual((self.output / name).read_bytes(), (self.scene_path.parent / name).read_bytes())
        self.assertNotIn('ground', attached)
        reconstruction = attached['reconstruction']
        self.assertEqual((reconstruction['evidence'], reconstruction['role']), ('predicted', 'rendererOnly'))
        self.assertEqual((reconstruction['pointCount'], reconstruction['triangleCount']), (2, 0))
        self.assertEqual(reconstruction['diagnostics']['rawDepthMaximumMeters'], 1_000_000.)
        for key, record in reconstruction['rawArtifacts'].items():
            raw = (self.output / record['path']).read_bytes()
            self.assertEqual((len(raw), tool.common.digest(raw)), (record['bytes'], record['sha256']))
            if key != 'depth':
                source_key = 'prediction' if key == 'disparity' else 'validity'
                self.assertEqual(raw, (fixture.bundle / 'prediction' / fixture.producer[source_key]['path']).read_bytes())
        depth = np.frombuffer((self.output / 'rawdepth.float32le').read_bytes(), '<f4').reshape(tool.SHAPE)
        self.assertEqual(depth[1, 3], 1_000_000.)
        self.assertTrue(np.isnan(depth[0, 0]))
        service = reconstruction['reuse']['serviceExecution']
        self.assertEqual(service['claim'], 'job_control_and_application_output_association')
        self.assertEqual(service['nativeManifestSha256'], tool.common.digest(fixture.manifest.read_bytes()))
        self.assertEqual(service['producerSha256'], tool.common.digest((self.output / 'original-producer.json').read_bytes()))
        self.assertEqual(service['planSha256'], tool.common.digest((self.output / 'original-plan.json').read_bytes()))
        self.assertEqual(reconstruction['reuse']['originalExecution'], fixture.producer['execution'])
        self.assertEqual(reconstruction['reuse']['originalExecution']['usage']['cloudRequests'], 0)
        self.assertEqual(reconstruction['provenance']['serviceExecutionSha256'],
                         tool.common.digest((self.output / 'service-execution.json').read_bytes()))
        for key, record in reconstruction['serviceArtifacts'].items():
            raw = (self.output / record['path']).read_bytes()
            self.assertEqual((len(raw), tool.common.digest(raw)), (record['bytes'], record['sha256']))
        self.assertEqual(tool.common.parse((self.output / 'service-execution.json').read_bytes()), service)
        self.assertEqual(service['jobControlReceiptSha256'], reconstruction['serviceArtifacts']['jobControl']['sha256'])
        self.assertEqual(service['bindingSha256'], reconstruction['serviceArtifacts']['outputBinding']['sha256'])
        self.assertEqual((self.output / 'job-control.json').read_bytes(), fixture.receipt.read_bytes())
        self.assertEqual((self.output / 'job-result.json').read_bytes(), (fixture.bundle / 'job-result.json').read_bytes())
        binding = tool.common.parse((self.output / 'input-binding.json').read_bytes())
        self.assertEqual(binding['observationId'], self.scene['id'])
        result = tool.common.parse((self.output / 'attachment-result.json').read_bytes())
        self.assertEqual(result['newInferenceRuns'], 0)
        self.assertFalse(result['modelBytesRehashed'])
        self.assertEqual((result['cloudRequests'], result['jobControlRequests'], result['modelInferenceRequests']), (1, 1, 0))
        self.assertEqual(tool.common.parse((self.output / 'catalog-entry.json').read_bytes())['metadataSha256'],
                         tool.common.digest((self.output / 'scene.json').read_bytes()))

    def test_local_native_and_render_failures_precede_collection(self):
        cases = [('native identity', lambda: self.scene.__setitem__('id', 'different-scene')),
                 ('ground ownership', lambda: self.scene['measurements'].__setitem__('groundPolicy', 'validated')),
                 ('measured buffers', lambda: self.scene['artifactHashes'].__setitem__('positions', '0' * 64)),
                 ('render camera', lambda: self.scene['camera'].__setitem__('near', True))]
        original = deepcopy(self.scene)
        for label, mutate in cases:
            self.scene = deepcopy(original)
            mutate()
            self.write(self.scene_path, self.scene)
            with self.subTest(case=label), patch.object(consumer, 'collect_result') as collect:
                with self.assertRaises(ValueError): self.attach()
                collect.assert_not_called()
            self.assertFalse(self.output.exists())
            self.assertFalse(self.protocol.receipt.exists())

    def test_exact_remote_mode_arguments_and_bundle_layout_precede_collection(self):
        bad_options = [dict(self.options, authenticated=True),
                       {key: value for key, value in self.options.items() if key != 'receipt_path'}]
        for options in bad_options:
            with self.subTest(options=set(options)), patch.object(consumer, 'collect_result') as collect:
                with self.assertRaisesRegex(ValueError, 'Exact Nebius consumption arguments'): self.attach(options=options)
                collect.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'Nebius consumption requires'): self.attach(direct_native=False)
        copied = self.root / 'misplaced-producer.json'
        copied.write_bytes((self.protocol.bundle / 'prediction/prediction.json').read_bytes())
        with patch.object(consumer, 'check_result') as check:
            with self.assertRaisesRegex(ValueError, 'Exact retrieved bundle paths'): self.attach(producer_path=copied)
            check.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_failed_trusted_association_keeps_control_receipt_without_publication(self):
        fixture = self.protocol
        job = deepcopy(fixture.job)
        job['status']['finishedAt'] = '2026-10-09T00:00:01Z'
        with patch.object(consumer.evidence, 'fetch_job', return_value=tool.common.canonical(job)):
            with self.assertRaises(ValueError): self.attach()
        self.assertFalse(self.output.exists())
        receipt = tool.common.parse(fixture.receipt.read_bytes())
        self.assertEqual(receipt['claim'], 'job_control_only')
        self.assertFalse(receipt['outputsBound'])

    def test_remote_symlink_files_and_bundle_native_parents_fail_before_collection(self):
        fixture = self.protocol
        original_scene, original_manifest = self.scene_path, fixture.manifest
        for name, target in (('scene', original_scene), ('prepared', original_manifest),
                             ('producer', fixture.bundle / 'prediction/prediction.json'),
                             ('plan', fixture.bundle / 'plan.json'), ('review', fixture.review)):
            link = self.root / (name + '-alias.json')
            link.symlink_to(target)
            kwargs = {}
            if name == 'scene': self.scene_path = link
            elif name == 'prepared': fixture.manifest = link
            elif name == 'producer': kwargs['producer_path'] = link
            elif name == 'plan': kwargs['plan_path'] = link
            else: kwargs['options'] = {**self.options, 'review_path': link}
            with self.subTest(file=name), patch.object(consumer, 'collect_result') as collect:
                with self.assertRaises((OSError, ValueError)): self.attach(**kwargs)
                collect.assert_not_called()
            self.scene_path, fixture.manifest = original_scene, original_manifest
        for name, target in (('bundle', fixture.bundle), ('native', fixture.manifest.parent)):
            link = self.root / (name + '-parent-alias')
            link.symlink_to(target, target_is_directory=True)
            kwargs = {}
            if name == 'bundle':
                kwargs.update(plan_path=link / 'plan.json', producer_path=link / 'prediction/prediction.json')
            else:
                fixture.manifest = link / 'manifest.json'
            with self.subTest(parent=name), patch.object(consumer, 'collect_result') as collect:
                with self.assertRaises((OSError, ValueError)): self.attach(**kwargs)
                collect.assert_not_called()
            fixture.manifest = original_manifest
        source = fixture.manifest.parent / 'left'
        retained = fixture.manifest.parent / 'left-original'
        source.rename(retained)
        source.symlink_to(retained)
        try:
            with patch.object(consumer, 'collect_result') as collect:
                with self.assertRaises((OSError, ValueError)): self.attach()
                collect.assert_not_called()
        finally:
            source.unlink()
            retained.rename(source)
        self.assertFalse(self.output.exists())
        self.assertFalse(fixture.receipt.exists())

    def test_existing_local_cache_branch_is_independent_of_remote_consumer(self):
        fixture = self.protocol
        local_cache = self.root / 'analytic-local-cache'
        local_cache.mkdir()
        producer = deepcopy(fixture.producer)
        producer['execution']['environment'] = tool.nvidia_stereo.runtime_environment()
        model = self.root / 'NEVER-READ-MODEL.onnx'
        producer['execution']['modelArtifact']['path'] = os.path.relpath(model, local_cache)
        producer['sources'] = {name: {**record, 'path': os.path.relpath(
            fixture.manifest if name == 'cropManifest' else fixture.manifest.parent / name, local_cache)}
            for name, record in producer['sources'].items()}
        producer_path = local_cache / 'prediction.json'
        self.write(producer_path, producer)
        for key in ('prediction', 'validity'):
            name = producer[key]['path']
            (local_cache / name).write_bytes((fixture.bundle / 'prediction' / name).read_bytes())
        paths = [self.scene_path, fixture.manifest, producer_path, fixture.local_plan]
        args = [item for path in paths for item in (path, tool.common.digest(path.read_bytes()))]
        with patch.object(consumer, 'check_result', side_effect=AssertionError('Local branch cannot check remote')), \
                patch.object(consumer, 'collect_result', side_effect=AssertionError('Local branch cannot collect remote')):
            attached = tool.attach(*args, None, None, model, self.output, direct_native=True)
        self.assertNotIn('serviceArtifacts', attached['reconstruction'])
        self.assertNotIn('serviceExecution', attached['reconstruction']['reuse'])
        result = tool.common.parse((self.output / 'attachment-result.json').read_bytes())
        self.assertEqual(result['cloudRequests'], 0)
        self.assertFalse(fixture.receipt.exists())

    def test_collected_identity_and_postcollection_source_raster_mutations_block_publication(self):
        fixture = self.protocol
        actual_collect = consumer.collect_result
        source_path = fixture.manifest.parent / 'lidar'
        raster_path = fixture.bundle / 'prediction/disparity.float32le'
        for label in ('producer', 'plan', 'native', 'claim', 'provider', 'schema', 'source', 'raster'):
            originals = {path: path.read_bytes() for path in (source_path, raster_path)}

            def tamper_after_collection(*args, **kwargs):
                producer, result, service = actual_collect(*args, **kwargs)
                if label == 'producer':
                    producer = {**producer, 'observationId': 'wrong-scene'}
                elif label in ('plan', 'native'):
                    service = {**service, ('planSha256' if label == 'plan' else 'nativeManifestSha256'): '0' * 64}
                elif label in ('claim', 'provider', 'schema'):
                    service = {**service, {'claim': 'claim', 'provider': 'provider', 'schema': 'schemaVersion'}[label]:
                               {'claim': 'offline_consistency_only', 'provider': 'local_onnxruntime_cpu', 'schema': True}[label]}
                else:
                    path = source_path if label == 'source' else raster_path
                    path.write_bytes(originals[path] + b'changed-after-collection')
                return producer, result, service

            with self.subTest(case=label), patch.object(consumer.evidence, 'fetch_job',
                    return_value=tool.common.canonical(fixture.job)), patch.object(consumer, 'collect_result',
                    side_effect=tamper_after_collection):
                with self.assertRaises(ValueError): self.attach()
            self.assertFalse(self.output.exists())
            self.assertTrue(fixture.receipt.exists())
            fixture.receipt.unlink()
            for path, raw in originals.items(): path.write_bytes(raw)


if __name__ == '__main__': unittest.main()
