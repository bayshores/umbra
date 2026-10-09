"""Analytic inventory fixtures only; these are not California dataset evidence."""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('inspect_probe', ROOT / 'scripts/california/inspect_probe.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ProbeChecks(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / '.local/worker-output/california'
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.root = Path(self.temp.name)
        log = '00000000-0000-0000-0000-000000000000'
        self.prefix = f'datasets/av2/sensor/train/{log}/'
        names = ['annotations.feather', 'city_SE3_egovehicle.feather',
                 'calibration/egovehicle_SE3_sensor.feather', 'calibration/intrinsics.feather',
                 'sensors/cameras/stereo_front_left/315970628649927223.jpg',
                 'sensors/cameras/stereo_front_right/315970628649927223.jpg',
                 'sensors/lidar/315970628659812000.feather',
                 f'map/log_map_archive_{log}____PAO_city_test.json']
        self.plan = {'log_id': log, 'split': 'train', 'stereo_timestamp_ns': '315970628649927223',
                     'lidar_timestamp_ns': '315970628659812000', 'approval_received': False,
                     'terms_url': 'https://www.argoverse.org/about.html#terms-of-use',
                     'total_listing_bytes': len(names),
                     'files': [{'path': name, 'listing_bytes': 1} for name in names]}
        self.plan_path = self.root / 'plan.json'
        self.listing_path = self.root / 'listing.xml'
        self.save_plan()
        self.save_listing(names)

    def tearDown(self):
        self.temp.cleanup()

    def save_plan(self):
        self.plan_path.write_text(json.dumps(self.plan))

    def save_listing(self, names, truncated='false', size=1):
        entries = ''.join(f'<Contents><Key>{self.prefix}{name}</Key><Size>{size}</Size></Contents>' for name in names)
        self.listing_path.write_text(f'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><Name>argoverse</Name><Prefix>{self.prefix}</Prefix><IsTruncated>{truncated}</IsTruncated>{entries}</ListBucketResult>')

    def inspect(self, local_root=None):
        return module.inspect(self.plan_path, [self.listing_path], local_root)

    def test_city_inventory_never_admits_site_or_holdout(self):
        report = self.inspect()
        record = report['observations'][0]
        self.assertTrue(record['geographyEvidence']['californiaCityCandidate'])
        self.assertEqual(record['admission']['status'], 'unresolved')
        self.assertIsNone(record['siteGroup'])
        self.assertEqual(record['split'], 'unassigned')
        self.assertEqual(record['reference']['timing']['absoluteDeltaNs'], '9884777')
        self.assertEqual(report['inspection']['verifiedUsableCaliforniaSites'], 0)

    def test_pending_approval_reads_no_payload_even_if_present(self):
        # Permit only plan/listing/code reads; any payload inspection fails the test.
        original = Path.open
        def open_guard(path, *args, **kwargs):
            if path not in (self.plan_path, self.listing_path, ROOT / 'scripts/california/inspect_probe.py'):
                raise AssertionError('Payload opened while approval pending')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'open', open_guard):
            report = self.inspect(self.root / 'not-permitted')
        self.assertTrue(all(item['payloadStatus'] == 'notRead' for item in report['inspection']['inventory']))

    def test_listing_drift_rejects_without_city_claim(self):
        self.save_listing([item['path'] for item in self.plan['files']], size=2)
        report = self.inspect()
        self.assertEqual(report['observations'][0]['admission']['status'], 'rejected')
        self.assertFalse(report['observations'][0]['geographyEvidence']['californiaCityCandidate'])
        self.assertEqual(len(report['inspection']['failures']), 8)

    def test_missing_object_rejects(self):
        self.save_listing([item['path'] for item in self.plan['files']][:-1])
        self.assertEqual(self.inspect()['observations'][0]['admission']['status'], 'rejected')

    def test_truncated_listing_and_numeric_timestamp_reject(self):
        self.save_listing([], truncated='true')
        with self.assertRaisesRegex(ValueError, 'Incomplete listing'):
            self.inspect()
        self.plan['stereo_timestamp_ns'] = 315970628649927223
        self.save_plan()
        with self.assertRaisesRegex(ValueError, 'decimal strings'):
            self.inspect()

    def test_traversal_and_oversize_selection_reject(self):
        self.plan['files'][0]['path'] = '../outside'
        self.save_plan()
        with self.assertRaisesRegex(ValueError, 'traversal'):
            self.inspect()
        self.plan['files'][0]['path'] = 'annotations.feather'
        self.plan['files'][0]['listing_bytes'] = 16_000_000
        self.plan['total_listing_bytes'] = 16_000_007
        self.save_plan()
        with self.assertRaisesRegex(ValueError, 'inspection bound'):
            self.inspect()

    def test_approved_local_inventory_pins_exact_bytes_but_stays_unresolved(self):
        self.plan['approval_received'] = True
        self.save_plan()
        raw = self.root / 'raw'
        for item in self.plan['files']:
            path = raw / item['path']
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'x')
        payload = raw / 'annotations.feather'
        original_pin = module.pin
        def replace_after_read(path, data):
            if path == payload:
                payload.write_bytes(b'y')  # Same-length mutation after capture, before pinning.
            return original_pin(path, data)
        with patch.object(module, 'pin', replace_after_read):
            report = self.inspect(raw)
        self.assertEqual(len(report['observations'][0]['sources']), 10)
        self.assertEqual(report['observations'][0]['admission']['status'], 'unresolved')
        def payload_pin(result):
            return next(source for source in result['observations'][0]['sources'] if source['path'] == str(payload.resolve()))
        self.assertEqual(payload_pin(report)['sha256'], hashlib.sha256(b'x').hexdigest())
        self.assertEqual(payload.read_bytes(), b'y')
        replacement = self.inspect(raw)
        self.assertEqual(payload_pin(replacement)['bytes'], 1)
        self.assertEqual(payload_pin(replacement)['sha256'], hashlib.sha256(b'y').hexdigest())
        self.assertNotEqual(payload_pin(report)['sha256'], payload_pin(replacement)['sha256'])
        (raw / 'annotations.feather').write_bytes(b'corrupt')
        report = self.inspect(raw)
        self.assertEqual(report['observations'][0]['admission']['status'], 'rejected')
        self.assertEqual(len(report['observations'][0]['sources']), 9)

    def test_output_never_overwrites_input(self):
        before = self.plan_path.read_bytes()
        argv = ['inspect_probe', '--plan', str(self.plan_path), '--listing', str(self.listing_path),
                '--output', str(self.plan_path)]
        with patch('sys.argv', argv), self.assertRaisesRegex(ValueError, 'overwrite'):
            module.main()
        self.assertEqual(before, self.plan_path.read_bytes())

    def test_existing_hardlink_output_cannot_truncate_input(self):
        before = self.plan_path.read_bytes()
        alias = self.root / 'output-hardlink.json'
        alias.hardlink_to(self.plan_path)
        argv = ['inspect_probe', '--plan', str(self.plan_path), '--listing', str(self.listing_path),
                '--output', str(alias)]
        with patch('sys.argv', argv), self.assertRaises(FileExistsError):
            module.main()
        self.assertEqual(before, self.plan_path.read_bytes())
        self.assertEqual(before, alias.read_bytes())

    def test_approved_any_or_all_missing_payloads_return_failure(self):
        self.plan['approval_received'] = True
        self.save_plan()
        raw = self.root / 'raw'
        for missing in (1, len(self.plan['files'])):
            with self.subTest(missing=missing):
                for index, item in enumerate(self.plan['files']):
                    path = raw / item['path']
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if index < missing:
                        path.unlink(missing_ok=True)
                    else:
                        path.write_bytes(b'x')
                output = self.root / f'missing-{missing}.json'
                argv = ['inspect_probe', '--plan', str(self.plan_path), '--listing', str(self.listing_path),
                        '--local-root', str(raw), '--output', str(output)]
                with patch('sys.argv', argv):
                    self.assertEqual(module.main(), 1)
                report = json.loads(output.read_bytes())
                self.assertEqual(report['observations'][0]['admission']['status'], 'rejected')
                self.assertEqual(len(report['inspection']['failures']), missing)

    def test_approved_without_input_root_reports_incomplete(self):
        self.plan['approval_received'] = True
        self.save_plan()
        report = self.inspect()
        self.assertEqual(report['observations'][0]['admission']['status'], 'rejected')
        self.assertIn('explicit --local-root', report['inspection']['failures'][0])

    def test_wrong_bucket_and_lookalike_prefix_reject(self):
        before = self.listing_path.read_text()
        for wrong in (before.replace('<Name>argoverse</Name>', '<Name>other</Name>'),
                      before.replace(f'<Prefix>{self.prefix}</Prefix>', f'<Prefix>{self.prefix}lookalike/</Prefix>')):
            with self.subTest(wrong=wrong[:100]):
                self.listing_path.write_text(wrong)
                with self.assertRaisesRegex(ValueError, 'bucket or exact prefix'):
                    self.inspect()

    def test_approved_payload_symlink_escape_rejects(self):
        self.plan['approval_received'] = True
        self.save_plan()
        raw = self.root / 'raw'
        raw.mkdir()
        outside = self.root / 'outside'
        outside.write_bytes(b'x')
        (raw / 'annotations.feather').symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            self.inspect(raw)


if __name__ == '__main__':
    unittest.main()
