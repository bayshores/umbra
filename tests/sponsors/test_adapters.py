"""Analytic protocol fixtures only. No actual sponsor response/model execution."""
import copy
from datetime import datetime, timezone, timedelta
from io import BytesIO, StringIO
from contextlib import redirect_stdout
from types import SimpleNamespace
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'sponsors'))
import common
import nvidia_stereo as nvidia
import token_factory as tf


class Adapters(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        evidence_file = self.root / 'fixture-evidence.txt'
        evidence_file.write_text('Analytic fixture only; required metric reference absent.')
        self.evidence = {'schemaVersion': 1, 'observationId': 'analytic-fixture',
                         'evidence': [{'id': 'fixture-only', 'state': 'Unknown',
                                       'summary': 'Required independent reference absent.',
                                       'artifact': common.record(evidence_file, evidence_file.read_bytes(), self.root)}],
                         'requiredMissing': ['independentMetricReference']}
        self.ep = self.root / 'evidence.json'
        self.ep.write_bytes(common.canonical(self.evidence))

    def tf_fixture(self):
        plan = tf.prepare(self.ep, self.root / 'plan')
        audit = {'observationId': 'analytic-fixture', 'disposition': 'unresolved',
                 'checks': [{'evidenceId': 'fixture-only', 'state': 'Unknown', 'assessment': 'unresolved',
                             'reason': 'Analytic fixture says reference is absent.'}],
                 'requiredMissing': ['independentMetricReference']}
        response = {'id': 'analytic-fixture-not-provider-execution', 'model': tf.MODEL,
                    'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant',
                                                                 'content': common.canonical(audit).decode()}}],
                    'usage': {'prompt_tokens': 100, 'completion_tokens': 90, 'total_tokens': 190}}
        rp = self.root / 'fixture-response.json'
        rp.write_bytes(common.canonical(response))
        receipt = {'source': 'coordinator_dispatch', 'endpoint': tf.ENDPOINT, 'httpStatus': 200,
                   'attempts': 1, 'retries': 0, 'reservationId': 'analytic-fixture-no-spend',
                   'requestSha256': common.digest(common.canonical(plan['request'])),
                   'responseSha256': common.digest(rp.read_bytes()),
                   'billing': {'inputTokenUpperBound': 200, 'inputUsdPerMillion': '0.06',
                               'outputUsdPerMillion': '0.24', 'reservedUsd': '0.0002',
                               'extraChargesUpperBoundUsd': '0'}}
        rc = self.root / 'fixture-transport.json'
        rc.write_bytes(common.canonical(receipt))
        return plan, audit, response, receipt, rp, rc

    def test_preparation_does_not_dispatch_or_publish_audit(self):
        plan = tf.prepare(self.ep, self.root / 'plan')
        self.assertEqual(plan['status'], 'prepared_not_dispatched')
        self.assertEqual(plan['limits']['retries'], 0)
        self.assertNotIn('response_format', plan['request'])
        self.assertIsNone(plan['cost']['inputTokenUpperBound'])
        self.assertFalse((self.root / 'plan' / 'audit.json').exists())

    def test_strict_json(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}', b'{"a":1e999}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                common.parse(raw)

    def test_fixture_import_usage_and_cached_integrity(self):
        _, _, _, _, rp, rc = self.tf_fixture()
        pp, out = self.root / 'plan' / 'plan.json', self.root / 'accepted-fixture'
        result = tf.accept(pp, rp, rc, out)
        self.assertEqual(result['execution']['tokenCostUsd'], '0.0000276')
        self.assertEqual(tf.cached(pp, out), result)
        rp.write_bytes(b'{}')
        with self.assertRaises(ValueError):
            tf.cached(pp, out)

    def test_untrusted_citations_unknown_and_missing_cannot_promote(self):
        _, audit, _, _, _, _ = self.tf_fixture()
        for field, value in [('evidenceId', 'invented'), ('state', 'measured'), ('assessment', 'retained')]:
            altered = copy.deepcopy(audit)
            altered['checks'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                tf.validate_audit(altered, self.evidence)
        for key, value in [('disposition', 'reviewable'), ('requiredMissing', []), ('geometry', {'distance': 10})]:
            altered = copy.deepcopy(audit)
            altered[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                tf.validate_audit(altered, self.evidence)

    def test_transport_usage_and_reservation_fail_closed(self):
        _, _, response, receipt, rp, rc = self.tf_fixture()
        pp = self.root / 'plan' / 'plan.json'
        cases = [('finish', None), ('model', None), ('usage', None), ('boolUsage', None),
                 ('reservation', None), ('responseHash', None), ('retry', None)]
        for name, _ in cases:
            modified_response, modified_receipt = copy.deepcopy(response), copy.deepcopy(receipt)
            if name == 'finish':
                modified_response['choices'][0]['finish_reason'] = 'length'
            elif name == 'model':
                modified_response['model'] = 'other'
            elif name == 'usage':
                modified_response['usage']['total_tokens'] = 1
            elif name == 'boolUsage':
                modified_response['usage']['prompt_tokens'] = True
            elif name == 'reservation':
                modified_receipt['billing']['reservedUsd'] = '0'
            elif name == 'retry':
                modified_receipt['retries'] = 1
            rp.write_bytes(common.canonical(modified_response))
            modified_receipt['responseSha256'] = common.digest(rp.read_bytes()) if name != 'responseHash' else '0' * 64
            rc.write_bytes(common.canonical(modified_receipt))
            out = self.root / ('rejected-' + name)
            with self.subTest(name=name), self.assertRaises(ValueError):
                tf.accept(pp, rp, rc, out)
            self.assertFalse(out.exists())

    def test_input_evidence_change_blocks_reuse(self):
        _, _, _, _, rp, rc = self.tf_fixture()
        (self.root / 'fixture-evidence.txt').write_text('changed')
        with self.assertRaises(ValueError):
            tf.accept(self.root / 'plan' / 'plan.json', rp, rc, self.root / 'no-result')

    def test_existing_destination_preserved(self):
        tf.prepare(self.ep, self.root / 'plan')
        before = (self.root / 'plan' / 'plan.json').read_bytes()
        with self.assertRaises(ValueError):
            tf.prepare(self.ep, self.root / 'plan')
        self.assertEqual((self.root / 'plan' / 'plan.json').read_bytes(), before)

    def crop_fixture(self):
        directory = self.root / 'crop'
        directory.mkdir()
        rgb = np.zeros((320, 736, 3), dtype=np.uint8)
        rgb[..., 0] = 255
        for side in ('left', 'right'):
            Image.fromarray(rgb).save(directory / (side + '.png'))
        original = [[4., 0., 2., 0.], [0., 4., 2., 0.], [0., 0., 1., 0.]]
        transformed = [[4., 0., 1., 0.], [0., 4., 1., 0.], [0., 0., 1., 0.]]
        calibration = {'decoded_source_size': [738, 322], 'crop_size': [736, 320],
                       'original': {side: original for side in ('left_projection', 'right_projection')},
                       'transformed': {side: transformed for side in ('left_projection', 'right_projection')}}
        (directory / 'calibration.json').write_bytes(common.canonical(calibration))
        crop = {'crop': {'x': 1, 'y': 1, 'width': 736, 'height': 320,
                         'source_to_crop_matrix': [[1., 0., -1.], [0., 1., -1.], [0., 0., 1.]]},
                'outputs': {p.name: {'sha256': common.digest(p.read_bytes()), 'bytes': p.stat().st_size}
                            for p in directory.iterdir()}}
        manifest = directory / 'manifest.json'
        manifest.write_bytes(common.canonical(crop))
        return manifest

    def reviewed_nvidia_fixture(self):
        """Analytic review metadata for validation only; no ONNX/ORT calls."""
        cp = self.crop_fixture()
        inspection = self.root / 'analytic-inspection-not-graph.json'
        inspection.write_bytes(common.canonical({'fixtureOnly': True, 'artifact': nvidia.MODEL,
                                                 'status': 'embedded_graph_checked', 'external_data': False}))
        review = self.root / 'analytic-review-not-model.json'
        review.write_bytes(common.canonical({'model': nvidia.MODEL, 'licenseAuthorization': True,
                'normalizationReviewed': True, 'reviewEvidence': 'Analytic test-only declaration; no model execution.',
                'normalization': {'mode': 'raw0_255'}, 'outputQuantity': 'disparityPixels',
                'outputConvention': 'x_left_minus_x_right_in_input_pixels', 'inputNames': ['left', 'right'],
                'outputName': 'disparity', 'inspection': common.record(inspection, inspection.read_bytes(), self.root)}))
        return cp, review

    def test_nvidia_thread_selection_binds_cache_identity_and_preserves_default(self):
        cp, review = self.reviewed_nvidia_fixture()
        plans = [nvidia.prepare(cp, 'analytic-fixture', self.root / ('threads-' + str(threads)), review, threads=threads)
                 for threads in (1, 2, 4)]
        default = nvidia.prepare(cp, 'analytic-fixture', self.root / 'threads-default', review)
        self.assertEqual(default['cacheKey'], plans[0]['cacheKey'])
        self.assertEqual(len({plan['cacheKey'] for plan in plans}), 3)
        self.assertEqual(plans[0]['identity']['sources'], plans[2]['identity']['sources'])
        for plan in plans:
            threads = plan['limits']['threads']
            self.assertEqual(plan['limits'], {'batchSize': 1, 'jobs': 1, 'retries': 0, 'timeoutSeconds': 120, 'threads': threads})
            self.assertEqual(plan['identity']['limits'], plan['limits'])
            self.assertEqual(plan['processing']['cpuConfiguration'],
                             {'intraOpThreads': threads, 'interOpThreads': 1, 'provider': 'CPUExecutionProvider'})
            self.assertEqual(plan['identity']['inspectionHelperSha256'],
                             common.digest((ROOT / 'scripts' / 'inspect-onnx-model.py').read_bytes()))
            nvidia.validate_plan(self.root / ('threads-' + str(threads)) / 'plan.json')
        for threads in (True, False, 0, 3, 8, 1.0, '1', None):
            with self.subTest(threads=threads), self.assertRaises(ValueError):
                nvidia.prepare(cp, 'analytic-fixture', self.root / 'unsupported', review, threads=threads)
        self.assertFalse((self.root / 'unsupported').exists())

    def test_nvidia_plan_rejects_typed_bounds_cpu_and_source_identity_changes(self):
        cp, review = self.reviewed_nvidia_fixture()
        plan = nvidia.prepare(cp, 'analytic-fixture', self.root / 'typed-plan', review, threads=4)
        pp = self.root / 'typed-plan' / 'plan.json'
        mutations = [('limits', 'threads', True), ('limits', 'threads', 3), ('limits', 'threads', 1.0),
                     ('limits', 'batchSize', True), ('limits', 'batchSize', 2), ('limits', 'jobs', 2),
                     ('limits', 'jobs', True), ('limits', 'retries', False), ('limits', 'retries', 1),
                     ('limits', 'timeoutSeconds', 121), ('limits', 'timeoutSeconds', 120.0),
                     ('limits', 'extra', 1), ('cpuConfiguration', 'intraOpThreads', 1),
                     ('cpuConfiguration', 'interOpThreads', True), ('cpuConfiguration', 'interOpThreads', 2),
                     ('cpuConfiguration', 'provider', 'CUDAExecutionProvider'), ('cpuConfiguration', 'extra', 1),
                     ('identity', 'inspectionHelperSha256', '0' * 64), ('identity', 'codeHashes', {}),
                     ('identity', 'limits', {**plan['limits'], 'batchSize': True})]
        for parent, key, value in mutations:
            changed = copy.deepcopy(plan)
            container = changed['processing'][parent] if parent == 'cpuConfiguration' else changed[parent]
            container[key] = value
            pp.write_bytes(common.canonical(changed))
            with self.subTest(parent=parent, key=key, value=value), self.assertRaises(ValueError):
                nvidia.validate_plan(pp)
        pp.write_bytes(common.canonical(plan))
        nvidia.validate_plan(pp)
        original_read = Path.read_bytes
        for helper in ('inspect-onnx-model.py', 'prepare-stereo-crop.py'):
            with patch.object(Path, 'read_bytes', autospec=True,
                    side_effect=lambda path: b'changed test-only source' if path.name == helper else original_read(path)):
                with self.subTest(helper=helper), self.assertRaises(ValueError):
                    nvidia.validate_plan(pp)
        (cp.parent / 'right.png').write_bytes(b'changed test-only crop')
        with self.assertRaises(ValueError):
            nvidia.validate_plan(pp)

    def test_successful_child_stdout_is_saved_without_model_execution(self):
        model = self.root / 'analytic-not-model.onnx'
        model.write_bytes(b'Analytic test-only placeholder; no model reads.')
        out = self.root / 'analytic-child-result'
        stdout = common.canonical({'fixtureOnly': True, 'phase': 'session_construct_begin', 'monotonicSeconds': 1.0}) + b'\n'
        def child_stub(command, **kwargs):
            stage = Path(command[command.index('--stage') + 1])
            for name in ('prediction.json', 'disparity.float32le', 'validity.uint8'):
                (stage / name).write_bytes(b'Analytic test-only child payload; not a prediction.')
            return nvidia.subprocess.CompletedProcess(command, 0, stdout=stdout, stderr=b'')
        with patch.object(nvidia, 'validate_plan'), patch.object(nvidia.subprocess, 'run', side_effect=child_stub) as process:
            nvidia.run(self.root / 'analytic-not-plan.json', model, out)
            self.assertEqual(process.call_args.kwargs['timeout'], 120)
            self.assertTrue(process.call_args.kwargs['check'])
            self.assertTrue(process.call_args.kwargs['capture_output'])
            self.assertEqual(process.call_count, 1)
        self.assertEqual((out / 'execution-log.jsonl').read_bytes(), stdout)

    def test_analytic_child_options_phases_and_source_pins_without_onnx_ort_calls(self):
        cp, review_path = self.reviewed_nvidia_fixture()
        plan = nvidia.prepare(cp, 'analytic-fixture', self.root / 'analytic-child-plan', review_path, threads=4)
        pp = self.root / 'analytic-child-plan' / 'plan.json'
        raw, plan, review, files = nvidia.validate_plan(pp)
        stage = self.root / 'analytic-stage'
        stage.mkdir()
        output = self.root / 'analytic-output-not-runtime'
        model = self.root / 'analytic-absent-model.onnx'
        model_bytes = b'analytic-test-only-not-onnx'
        model_contract = {**nvidia.MODEL, 'bytes': len(model_bytes), 'sha256': common.digest(model_bytes)}
        environment = {'onnx': 'analytic-stub', 'onnxruntime': 'analytic-stub'}
        real_snapshot, real_read = nvidia.snapshot, Path.read_bytes
        for fault in (None, 'actual-options', 'provider', 'helper-before', 'helper-after'):
            ran = []
            def analytic_run(*args):
                ran.append(True)
                raster = np.ones((1, 1, 320, 736), dtype=np.float32)
                raster[0, 0, 0, :3] = [0, -1, np.nan]
                return [raster]
            options = SimpleNamespace()
            session = Mock()
            session.get_providers.return_value = ['CUDAExecutionProvider'] if fault == 'provider' else ['CPUExecutionProvider']
            session.get_session_options.return_value = SimpleNamespace(
                intra_op_num_threads=1 if fault == 'actual-options' else 4, inter_op_num_threads=1)
            session.get_inputs.return_value = [SimpleNamespace(name=name, type='tensor(float)', shape=nvidia.SHAPE)
                                               for name in review['inputNames']]
            session.get_outputs.return_value = [SimpleNamespace(name=review['outputName'], type='tensor(float)', shape=[1, 1, 320, 736])]
            session.run.side_effect = analytic_run
            onnx_stub = SimpleNamespace(__version__='analytic-stub', load_model_from_string=Mock(return_value='analytic-not-graph'),
                                        TensorProto=SimpleNamespace(EXTERNAL=1), checker=SimpleNamespace(check_model=Mock()))
            ort_stub = SimpleNamespace(__version__='analytic-stub', SessionOptions=Mock(return_value=options),
                                       InferenceSession=Mock(return_value=session))
            stdout = StringIO()
            def helper_read(path):
                if path.name == 'inspect-onnx-model.py' and (fault == 'helper-before' or fault == 'helper-after' and ran):
                    return b'analytic-test-only-changed-helper'
                return real_read(path)
            with patch.dict(sys.modules, {'onnx': onnx_stub, 'onnxruntime': ort_stub}), \
                    patch.object(nvidia, 'validate_plan', return_value=(raw, plan, review, files)), \
                    patch.object(nvidia, 'runtime_environment', return_value=environment), \
                    patch.object(nvidia, 'MODEL', model_contract), \
                    patch.object(nvidia, 'load_script', return_value=SimpleNamespace(protobuf_messages=lambda graph: [])), \
                    patch.object(nvidia, 'snapshot', side_effect=lambda path, *args: model_bytes if path == model else real_snapshot(path, *args)), \
                    patch.object(Path, 'read_bytes', autospec=True, side_effect=helper_read), redirect_stdout(stdout):
                if fault:
                    with self.subTest(fault=fault), self.assertRaises(ValueError):
                        nvidia.execute_child(pp, model, stage, output)
                else:
                    nvidia.execute_child(pp, model, stage, output)
                    manifest = common.parse((stage / 'prediction.json').read_bytes())
                    self.assertEqual(manifest['execution']['cpuConfiguration'], plan['processing']['cpuConfiguration'])
                    self.assertEqual(manifest['execution']['usage']['threads'], 4)
                    self.assertEqual(list((stage / 'validity.uint8').read_bytes()[:4]), [0, 0, 0, 1])
            if fault == 'helper-before':
                ort_stub.InferenceSession.assert_not_called()
            else:
                self.assertEqual((options.intra_op_num_threads, options.inter_op_num_threads), (4, 1))
                self.assertEqual(ort_stub.InferenceSession.call_args.kwargs['providers'], ['CPUExecutionProvider'])
                if fault == 'provider':
                    session.get_session_options.assert_not_called()
                else:
                    session.get_session_options.assert_called_once()
            if fault in (None, 'helper-after'):
                phases = [common.parse(line.encode()) for line in stdout.getvalue().splitlines()]
                self.assertEqual([entry['phase'] for entry in phases],
                                 ['session_construct_begin', 'session_construct_end', 'inference_begin', 'inference_end'])
                self.assertEqual(sorted(entry['monotonicSeconds'] for entry in phases), [entry['monotonicSeconds'] for entry in phases])
                self.assertTrue(all(entry.get('elapsedSeconds', 0) >= 0 for entry in phases))
            else:
                session.run.assert_not_called()
            for path in stage.iterdir():
                path.unlink()

    def test_nvidia_unreviewed_preparation_blocks_without_prediction(self):
        cp = self.crop_fixture()
        out = self.root / 'nvidia-plan'
        plan = nvidia.prepare(cp, 'analytic-fixture', out)
        self.assertTrue(plan['blockers'])
        self.assertEqual(plan['processing']['normalization'], 'unverified')
        with patch.object(nvidia.subprocess, 'run') as process:
            with self.assertRaises(ValueError):
                nvidia.run(out / 'plan.json', self.root / 'absent.onnx', self.root / 'no-prediction')
            process.assert_not_called()
        self.assertFalse((self.root / 'no-prediction').exists())

    def test_crop_hash_and_transformed_calibration_reject(self):
        cp = self.crop_fixture()
        (cp.parent / 'left.png').write_bytes(b'wrong')
        with self.assertRaises(ValueError):
            nvidia.prepare(cp, 'analytic-fixture', self.root / 'no-plan')

    def test_normalization_uses_actual_rgb_no_implicit_resize(self):
        cp = self.crop_fixture()
        _, _, files, _ = nvidia.read_crop(cp)
        raw = nvidia.tensors(files, {'mode': 'raw0_255'})[0]
        unit = nvidia.tensors(files, {'mode': 'unit0_1'})[0]
        self.assertEqual(raw.shape, (1, 3, 320, 736))
        self.assertEqual(raw[0, 0, 0, 0], 255)
        self.assertEqual(unit[0, 0, 0, 0], 1)
        mean_std = nvidia.tensors(files, {'mode': 'meanStd', 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]})[0]
        self.assertEqual(mean_std[0, 0, 0, 0], 1)
        self.assertEqual(mean_std[0, 1, 0, 0], -1)

    def test_model_contract_review_required_not_inferred(self):
        review = self.root / 'unverified-review.json'
        review.write_bytes(common.canonical({'model': nvidia.MODEL, 'licenseAuthorization': False,
                                             'normalizationReviewed': False, 'reviewEvidence': ''}))
        with self.assertRaises(ValueError):
            nvidia.review_contract(review)

    def test_calibration_crop_translation_rejects_even_with_matching_hash(self):
        cp = self.crop_fixture()
        path = cp.parent / 'calibration.json'
        calibration = common.parse(path.read_bytes())
        calibration['transformed']['left_projection'][0][2] = 9
        path.write_bytes(common.canonical(calibration))
        crop = common.parse(cp.read_bytes())
        crop['outputs']['calibration.json'] = {'sha256': common.digest(path.read_bytes()), 'bytes': path.stat().st_size}
        cp.write_bytes(common.canonical(crop))
        with self.assertRaises(ValueError):
            nvidia.read_crop(cp)

    def test_bounded_child_timeout_no_partial_prediction(self):
        fake_model = self.root / 'fixture-not-model.onnx'
        fake_model.write_text('not a model; no runtime reads')
        out = self.root / 'timed-out'
        with patch.object(nvidia, 'validate_plan'), patch.object(nvidia.subprocess, 'run',
                side_effect=nvidia.subprocess.TimeoutExpired('analytic stub', 120)) as process:
            with self.assertRaises(nvidia.subprocess.TimeoutExpired):
                nvidia.run(self.root / 'analytic-plan', fake_model, out)
            self.assertEqual(process.call_args.kwargs['timeout'], 120)
        self.assertFalse(out.exists())
        self.assertEqual(list(self.root.glob('.cpu-stereo-*')), [])

    def preflight_resources(self):
        now = datetime.now(timezone.utc)
        ledger = {'services': {'token_factory': {'selected_model': tf.MODEL,
                   'balance_verified': True, 'expiration_verified': True, 'coverage_verified': True,
                   'pricing_verified': True, 'no_paid_spillover_verified': True, 'local_api_credential_present': True,
                   'displayed_price_per_million_tokens': {'input_usd': 0.06, 'output_usd': 0.24}}},
                  'recorded_run_consumption_usd': 0, 'reserved_worst_case_usd': 0,
                  'effective_run_cap_usd': 0.2, 'verified_initial_remaining_free_credit_value_usd': 1,
                  'last_account_check_utc': now.isoformat(), 'demo_judging_reserve_usd': 0.01}
        bound = {'model': tf.MODEL, 'endpoint': tf.ENDPOINT, 'maxPromptTokens': 4096,
                 'contextBoundVerified': True, 'outputBoundIncludesAllBillableGeneration': True,
                 'extraChargesBoundVerified': True, 'extraChargesUpperBoundUsd': '0',
                 'verifiedAtUtc': now.isoformat(), 'expiresAtUtc': (now + timedelta(hours=1)).isoformat()}
        lp, bp = self.root / 'ledger-fixture.json', self.root / 'bound-fixture.json'
        lp.write_bytes(common.canonical(ledger))
        bp.write_bytes(common.canonical(bound))
        return lp, bp, ledger, bound

    def test_preflight_cap_missing_expiry_and_money_bounds(self):
        self.tf_fixture()
        lp, bp, ledger, bound = self.preflight_resources()
        now = datetime.now(timezone.utc)
        for name in ('eligible', 'null-expiry', 'unlocked', 'overspend', 'unauthorized-cap', 'stale-account', 'missing-demo-reserve'):
            edited_ledger, edited_bound = copy.deepcopy(ledger), copy.deepcopy(bound)
            if name == 'null-expiry':
                edited_bound['expiresAtUtc'] = None
            elif name == 'unlocked':
                edited_ledger['effective_run_cap_usd'] = None
            elif name == 'overspend':
                edited_ledger['reserved_worst_case_usd'] = 0.1999
            elif name == 'unauthorized-cap':
                edited_ledger['effective_run_cap_usd'] = 2
            elif name == 'stale-account':
                edited_ledger['last_account_check_utc'] = (now - timedelta(hours=2)).isoformat()
            elif name == 'missing-demo-reserve':
                edited_ledger['demo_judging_reserve_usd'] = None
            lp.write_bytes(common.canonical(edited_ledger))
            bp.write_bytes(common.canonical(edited_bound))
            result = tf.preflight(self.root / 'plan' / 'plan.json', lp, bp)
            with self.subTest(name=name):
                self.assertEqual(result['status'], 'eligible_for_coordinator_reservation' if name == 'eligible' else 'blocked')
                self.assertEqual(result['worstCaseUsd'], '0.00036864')

    def test_complete_prepared_contract_shared_by_preflight_accept_and_cache(self):
        original, _, _, _, rp, rc = self.tf_fixture()
        pp = self.root / 'plan' / 'plan.json'
        lp, bp, _, _ = self.preflight_resources()
        out = self.root / 'accepted-fixture'
        tf.accept(pp, rp, rc, out)
        cases = [('endpoint', 'https://example.invalid/changed'), ('role', 'geometryAuthority'),
                 ('status', 'succeeded'), ('limits', {'requests': 999, 'retries': 999}),
                 ('schemaVersion', True), ('source.bytes', original['source']['bytes'] + 1),
                 ('source.readOnlyExternal', False), ('cost.status', 'ready_to_dispatch'),
                 ('identity.role', 'changed'), ('identity.evidenceSha256', '0' * 64),
                 ('identity.body', {'unrelated': True})]
        for field, value in cases:
            changed = copy.deepcopy(original)
            if '.' in field:
                parent, key = field.split('.')
                changed[parent][key] = value
                if parent == 'identity':
                    changed['cacheKey'] = common.digest(common.canonical(changed['identity']))
            else:
                changed[field] = value
            pp.write_bytes(common.canonical(changed))
            for name, check in [('preflight', lambda: tf.preflight(pp, lp, bp)),
                                ('accept', lambda: tf.accept(pp, rp, rc, self.root / 'rejected')),
                                ('cache', lambda: tf.cached(pp, out))]:
                with self.subTest(field=field, path=name), self.assertRaises(ValueError):
                    check()
        pp.write_bytes(common.canonical(original))
        self.assertEqual(tf.preflight(pp, lp, bp)['status'], 'eligible_for_coordinator_reservation')
        self.assertEqual(tf.cached(pp, out)['execution']['status'], 'validated_imported_response')

    def nvidia_cache_fixture(self, threads=1):
        """Prerequisite-mocked analytic cache fixture; no graph or model execution."""
        cp = self.crop_fixture()
        _, _, files, _ = nvidia.read_crop(cp)
        pp = self.root / 'fixture-plan.json'
        pp.write_text('Prerequisite-mocked fixture; not a reviewed plan.')
        review = {'normalization': {'mode': 'meanStd', 'mean': [0.5, 0.4, 0.3], 'std': [0.2, 0.3, 0.4]}}
        plan = {'schemaVersion': 1, 'observationId': 'analytic-fixture', 'cacheKey': 'fixture-key',
                'model': nvidia.MODEL, 'processing': {'normalization': review['normalization'],
                    'cpuConfiguration': {'intraOpThreads': threads, 'interOpThreads': 1, 'provider': 'CPUExecutionProvider'}},
                'limits': nvidia.execution_limits(threads), 'identity': {'codeHashes': common.code_hashes()}, 'sources': {}}
        model = self.root / 'absent-graph-no-acquisition.onnx'
        out = self.root / 'analytic-cache-not-runtime'
        out.mkdir()
        (out / 'LABEL.txt').write_text('Analytic metadata fixture only. No graph acquisition or model execution.')
        disparity = np.ones((320, 736), dtype='<f4').tobytes()
        validity = np.ones((320, 736), dtype=np.uint8).tobytes()
        def raster(name, data, encoding):
            (out / name).write_bytes(data)
            return {'path': name, 'sha256': common.digest(data), 'bytes': len(data), 'shape': [320, 736], 'encoding': encoding}
        environment = {'python': sys.version, 'numpy': np.__version__, 'pillow': nvidia.pillow_version,
                       'onnx': 'fixture-onnx-version', 'onnxruntime': 'fixture-ort-version'}
        manifest = {'fixtureOnly': True, 'schemaVersion': 1, 'observationId': plan['observationId'],
                    'cacheKey': plan['cacheKey'], 'model': nvidia.MODEL, 'processing': plan['processing'], 'sources': {},
                    'prediction': {**raster('disparity.float32le', disparity, 'float32le'), 'quantity': 'disparityPixels'},
                    'validity': raster('validity.uint8', validity, 'uint8'),
                    'execution': {'status': 'succeeded', 'provider': 'local_onnxruntime_cpu',
                                  'modelArtifact': {'path': os.path.relpath(model.resolve(), out.resolve()),
                                                    'sha256': nvidia.MODEL['sha256'], 'bytes': nvidia.MODEL['bytes'],
                                                    'readOnlyExternal': True},
                                  'createdUtc': datetime.now(timezone.utc).isoformat(), 'elapsedSeconds': 0.1,
                                  'codeHashes': plan['identity']['codeHashes'], 'environment': environment,
                                  'cpuConfiguration': plan['processing']['cpuConfiguration'],
                                  'usage': {'cloudRequests': 0, 'jobs': 1, 'retries': 0, 'threads': threads},
                                  'inputTensorSha256': [common.digest(t.tobytes()) for t in nvidia.tensors(files, review['normalization'])]}}
        (out / 'prediction.json').write_bytes(common.canonical(manifest))
        return pp, out, model, plan, review, files, environment, manifest

    def test_cached_execution_metadata_bound_to_inputs_model_and_resource_contract(self):
        pp, out, model, plan, review, files, environment, original = self.nvidia_cache_fixture()
        mutations = [('modelArtifact.sha256', '0' * 64), ('modelArtifact.bytes', 1),
                     ('modelArtifact.path', '../another-graph.onnx'), ('modelArtifact.readOnlyExternal', False),
                     ('inputTensorSha256', ['0' * 64, '0' * 64]), ('environment', {}),
                     ('environment.onnxruntime', 'altered'), ('environment.numpy', 'altered'),
                     ('elapsedSeconds', -1), ('elapsedSeconds', 121), ('elapsedSeconds', True),
                     ('createdUtc', 'invalid'), ('createdUtc', '2099-01-01T00:00:00Z'),
                     ('usage.cloudRequests', 1), ('usage.jobs', 2), ('usage.retries', 1),
                     ('usage.threads', 2), ('usage.threads', True), ('cpuConfiguration', {}),
                     ('cpuConfiguration.intraOpThreads', True), ('cpuConfiguration.intraOpThreads', 4),
                     ('cpuConfiguration.interOpThreads', True), ('cpuConfiguration.interOpThreads', 2),
                     ('cpuConfiguration.provider', 'CUDAExecutionProvider')]
        with patch.object(nvidia, 'validate_plan', return_value=(b'fixture', plan, review, files)), \
                patch.object(nvidia, 'runtime_environment', return_value=environment):
            self.assertTrue(nvidia.cached(pp, out, model)['fixtureOnly'])
            for field, value in mutations:
                changed = copy.deepcopy(original)
                if '.' in field:
                    parent, key = field.split('.')
                    changed['execution'][parent][key] = value
                else:
                    changed['execution'][field] = value
                (out / 'prediction.json').write_bytes(common.canonical(changed))
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    nvidia.cached(pp, out, model)
            (out / 'prediction.json').write_bytes(common.canonical(original))
            with self.assertRaises(ValueError):
                nvidia.cached(pp, out, self.root / 'different-caller-model.onnx')
            # Valid-looking hashes computed under another normalization cannot
            # replace tensors reconstructed from the actual reviewed RGB inputs.
            changed = copy.deepcopy(original)
            changed['execution']['inputTensorSha256'] = [common.digest(t.tobytes())
                                                        for t in nvidia.tensors(files, {'mode': 'raw0_255'})]
            (out / 'prediction.json').write_bytes(common.canonical(changed))
            with self.assertRaises(ValueError):
                nvidia.cached(pp, out, model)

    def test_multithread_cache_requires_matching_actual_cpu_configuration_and_usage(self):
        pp, out, model, plan, review, files, environment, original = self.nvidia_cache_fixture(threads=4)
        with patch.object(nvidia, 'validate_plan', return_value=(b'fixture', plan, review, files)), \
                patch.object(nvidia, 'runtime_environment', return_value=environment):
            self.assertEqual(nvidia.cached(pp, out, model)['execution']['usage']['threads'], 4)
            for mutation in ('missing-config', 'wrong-config', 'wrong-usage', 'wrong-validity'):
                changed = copy.deepcopy(original)
                if mutation == 'missing-config':
                    del changed['execution']['cpuConfiguration']
                elif mutation == 'wrong-config':
                    changed['execution']['cpuConfiguration']['intraOpThreads'] = 1
                elif mutation == 'wrong-usage':
                    changed['execution']['usage']['threads'] = 1
                else:
                    invalid = np.zeros((320, 736), dtype=np.uint8).tobytes()
                    (out / 'validity.uint8').write_bytes(invalid)
                    changed['validity']['sha256'] = common.digest(invalid)
                (out / 'prediction.json').write_bytes(common.canonical(changed))
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    nvidia.cached(pp, out, model)

    def test_cached_execution_environment_is_current_package_identity(self):
        with patch.object(nvidia, 'version', side_effect=['fixture-onnx', 'fixture-ort']) as package:
            value = nvidia.runtime_environment()
        self.assertEqual([c.args for c in package.call_args_list], [('onnx',), ('onnxruntime',)])
        self.assertEqual(value, {'python': sys.version, 'numpy': np.__version__, 'pillow': nvidia.pillow_version,
                                 'onnx': 'fixture-onnx', 'onnxruntime': 'fixture-ort'})


if __name__ == '__main__':
    unittest.main()
