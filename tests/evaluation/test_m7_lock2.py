"""Lock-2 split routing and isolation. Frozen split metadata and synthetic bytes only; no AV2 data or network."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = load('lock2_common', 'scripts/evaluation/m7/common.py')
writer = load('lock2_writer', '.local/evaluation/pao-accuracy/lock/write_protocol_lock.py')


def load_shape():
    saved_path = list(sys.path)
    saved = {name: sys.modules.pop(name) for name in ('common', 'prepare_av2') if name in sys.modules}
    try:
        return load('lock2_shape', 'scripts/california/car_shape.py')
    finally:
        for name in ('common', 'prepare_av2'):
            sys.modules.pop(name, None)
        sys.modules.update(saved)
        sys.path[:] = saved_path


shape = load_shape()


class SplitRouting(unittest.TestCase):
    def test_every_runnable_log_uses_its_frozen_split(self):
        for site in common.split_sites():
            if site['role'] not in common.DEVELOPMENT_ROLES + common.HELD_OUT_ROLES:
                continue
            log, split = site['logId'], site['av2Split']
            prefix = f'datasets/av2/sensor/{split}/{log}/'
            with self.subTest(log=log[:8]):
                self.assertEqual(common.av2_split(log), split)
                self.assertEqual(shape.av2_split(log), split)
                self.assertEqual(common.log_prefix(log), prefix)
                self.assertEqual(shape.lidar_prefix(log), prefix + 'sensors/lidar/')
                self.assertEqual(shape.LIDAR_PREFIX.format(log=log), shape.lidar_prefix(log))
                self.assertEqual(shape.LISTING_URL.format(log=log), shape.listing_url(log))
                self.assertEqual(common.listing_url(log, 'sensors/lidar/'), common.BUCKET +
                                 '?list-type=2&delimiter=%2F&max-keys=1000&prefix=' +
                                 (prefix + 'sensors/lidar/').replace('/', '%2F'))
                self.assertEqual(shape.listing_url(log), shape.BUCKET_URL + '?list-type=2&prefix=' +
                                 (prefix + 'sensors/lidar/').replace('/', '%2F') + '&delimiter=%2F&max-keys=1000')

    def test_development_listing_urls_are_byte_identical_to_lock1(self):
        for site in common.split_sites():
            if site['role'] not in common.DEVELOPMENT_ROLES:
                continue
            log = site['logId']
            self.assertEqual(common.listing_url(log, 'sensors/lidar/').encode(),
                             (f'https://argoverse.s3.amazonaws.com/?list-type=2&delimiter=%2F&max-keys=1000&prefix='
                              f'datasets%2Fav2%2Fsensor%2Ftrain%2F{log}%2Fsensors%2Flidar%2F').encode())
            self.assertEqual(shape.listing_url(log).encode(),
                             (f'https://argoverse.s3.amazonaws.com/?list-type=2&prefix=datasets%2Fav2%2Fsensor%2Ftrain%2F{log}'
                              '%2Fsensors%2Flidar%2F&delimiter=%2F&max-keys=1000').encode())

    def test_unknown_splits_missing_logs_and_reserve_are_refused(self):
        site = common.split_sites()[0]
        variants = [{**site, 'av2Split': value} for value in (None, '', 'test', 'TRAIN', 'other')]
        variants += [{**site, 'role': 'reserve', 'av2Split': value} for value in ('train', 'val', 'test')]
        variants += [{**site, 'role': 'other'}, {k: v for k, v in site.items() if k != 'av2Split'}]
        for entry in variants:
            raw = json.dumps({'sites': [entry]}).encode()
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'split.json'
                path.write_bytes(raw)
                with mock.patch.object(common, 'split_sites', return_value=[entry]), \
                        mock.patch.object(shape, 'SPLIT_PATH', path), mock.patch.object(shape, 'SPLIT_SHA256', common.sha(raw)):
                    for fn in (common.log_prefix, shape.lidar_prefix, shape.listing_url):
                        with self.subTest(entry=entry, fn=fn.__name__), self.assertRaisesRegex(SystemExit, 'av2Split|non-runnable'):
                            fn(site['logId'])
                    with mock.patch.object(shape, 'snapshot', side_effect=AssertionError('must refuse before data access')), \
                            self.assertRaises(SystemExit):
                        shape.make_plan(site['logId'], 'absent', '', 'absent', '', 'absent', '')
        for fn in (common.log_prefix, shape.lidar_prefix):
            with self.assertRaisesRegex(SystemExit, 'not a log'):
                fn('00000000-0000-0000-0000-000000000000')
        for entry in common.split_sites():
            if entry['role'] == 'reserve':
                for fn in (common.log_prefix, shape.lidar_prefix):
                    with self.assertRaisesRegex(SystemExit, 'reserve'):
                        fn(entry['logId'])

    def test_split_hash_mismatch_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'split.json'
            path.write_bytes(b'{"sites": []}')
            with mock.patch.object(common, 'SPLIT', path), mock.patch.object(shape, 'SPLIT_PATH', path):
                for fn in (common.log_prefix, shape.lidar_prefix):
                    with self.assertRaisesRegex(SystemExit, 'frozen sha256'):
                        fn('00000000-0000-0000-0000-000000000000')

    def test_synthetic_listings_and_sweep_plans_require_the_correct_split(self):
        for entry in common.split_sites():
            if entry['role'] not in common.DEVELOPMENT_ROLES + common.HELD_OUT_ROLES:
                continue
            log, split = entry['logId'], entry['av2Split']
            for folder_split in ('train', 'val'):
                prefix = f'datasets/av2/sensor/{folder_split}/{log}/sensors/lidar/'
                key = prefix + '1.feather'
                xml = (f'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                       f'<Prefix>{prefix}</Prefix><IsTruncated>false</IsTruncated><KeyCount>1</KeyCount>'
                       f'<Contents><Key>{key}</Key><Size>3</Size><ETag>"' + 'a' * 32 + '"</ETag></Contents></ListBucketResult>').encode()
                plan = {'schema': shape.PLAN_SCHEMA, 'shapeProtocol': {'sha256': shape.SHAPE_PROTOCOL_SHA256},
                        'rule': dict(shape.RULE), 'log': log, 'kept': 1,
                        'inputs': {'listing': {'url': shape.listing_url(log)}},
                        'sweeps': [{'timestampNs': '1', 'set': 'shape', 'key': key}]}
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / 'synthetic.xml'
                    path.write_bytes(xml)
                    if folder_split == split:
                        self.assertEqual(common.read_listing(path, log, 'sensors/lidar/')['1.feather']['key'], key)
                        self.assertEqual(shape.parse_listing(xml, log)[1]['key'], key)
                        self.assertIs(shape.check_plan(plan), plan)
                        plan['inputs']['listing']['url'] = shape.listing_url(log).replace(f'%2F{split}%2F', '%2Ftest%2F')
                        with self.assertRaisesRegex(ValueError, 'listing URL'):
                            shape.check_plan(plan)
                    else:
                        with self.assertRaises(common.SiteStop):
                            common.read_listing(path, log, 'sensors/lidar/')
                        with self.assertRaisesRegex(ValueError, 'another log or sensor'):
                            shape.parse_listing(xml, log)
                        with self.assertRaisesRegex(ValueError, 'another log or split'):
                            shape.check_plan(plan)

    def test_development_download_receipts_are_byte_identical_to_lock1(self):
        payload, clock = b'synthetic sweep', '2026-10-16T00:00:00Z'
        etag = hashlib.md5(payload).hexdigest()
        for site in common.split_sites():
            if site['role'] not in common.DEVELOPMENT_ROLES:
                continue
            log = site['logId']
            key = f'datasets/av2/sensor/train/{log}/sensors/lidar/1.feather'
            plan = {'schema': shape.PLAN_SCHEMA, 'shapeProtocol': {'sha256': shape.SHAPE_PROTOCOL_SHA256},
                    'rule': dict(shape.RULE), 'log': log, 'track': 'synthetic-track', 'kept': 1,
                    'inputs': {'listing': {'url': shape.listing_url(log)}},
                    'sweeps': [{'timestampNs': '1', 'set': 'shape', 'key': key, 'bytes': len(payload), 'etag': etag}]}
            raw = json.dumps(plan).encode()
            with tempfile.TemporaryDirectory() as folder, mock.patch.object(shape, 'utc_now', return_value=clock):
                path = Path(folder)
                (path / shape.PLAN_NAME).write_bytes(raw)
                def fetch(url):
                    self.assertEqual(url, shape.BUCKET_URL + key)
                    return 200, len(payload), etag, payload
                shape.download_sweeps(path, common.sha(raw), fetch=fetch, sleep=lambda _: self.fail('unexpected retry'))
                expected = {'schema': shape.RECEIPT_SCHEMA, 'planSha256': common.sha(raw), 'log': log,
                            'track': 'synthetic-track', 'rule': dict(shape.RULE), 'startedUtc': clock, 'finishedUtc': clock,
                            'sweeps': [{'timestampNs': '1', 'set': 'shape', 'file': 'sweeps/1.feather', 'key': key,
                                        'url': shape.BUCKET_URL + key, 'etag': etag, 'source': 'download', 'attempts': 1,
                                        'status': 200, 'contentLength': len(payload), 'bytes': len(payload),
                                        'sha256': common.sha(payload), 'md5': etag, 'md5MatchesEtag': True, 'sizeMatches': True}],
                            'downloadDropped': [], 'bytesOnDisk': len(payload),
                            'note': 'Public Argoverse 2 files ($0). Success = HTTP 200, listing size, and MD5 = ETag for a single-part ETag.'}
                self.assertEqual((path / shape.RECEIPT_NAME).read_bytes(),
                                 (json.dumps(expected, indent=1, sort_keys=True) + '\n').encode())


class Lock2(unittest.TestCase):
    def test_paths_are_isolated_from_lock1(self):
        folder = ROOT / '.local/evaluation/pao-accuracy/lock-2'
        self.assertEqual(common.LOCK_PATH, folder / 'protocol-lock.json')
        self.assertEqual(common.LOCK_RECORD, folder / 'protocol-lock.record.json')
        self.assertEqual((writer.LOCK, writer.RECORD, writer.CANDIDATE),
                         (common.LOCK_PATH, common.LOCK_RECORD, folder / 'protocol-lock.candidate.json'))
        self.assertEqual(shape.LOCK_PATH, common.LOCK_PATH)
        self.assertEqual(common.ANCHOR_DIR, 'scripts/evaluation/m7/anchors-2')
        self.assertEqual(writer.ANCHOR_REL, common.ANCHOR_REL)
        for relative in (common.ANCHOR_REL, common.ADDENDUM_ANCHOR_REL, common.LOST_SITES_REL,
                         common.LABELS_MANIFEST_REL, common.RUNS_REL):
            self.assertTrue(relative.startswith(common.ANCHOR_DIR + '/'))
        self.assertEqual(common.RUN_MARKERS, ROOT / '.local/evaluation/pao-accuracy/m7/lock-2/run-markers')
        self.assertNotEqual(common.marker_path('build', 'aaaaaaaa', '1').parent,
                            ROOT / '.local/evaluation/pao-accuracy/m7/run-markers')

    def test_preflight_refuses_lock1_before_reading_files_or_fetching(self):
        with mock.patch.object(Path, 'read_bytes', side_effect=AssertionError('must not read lock1')), \
                mock.patch.object(common, 'fetch_origin', side_effect=AssertionError('must not fetch')):
            with self.assertRaisesRegex(SystemExit, 'Lock 1.*superseded'):
                common.preflight(common.SUPERSEDED_LOCK_SHA)
            with mock.patch.object(common, 'LOCK_PATH', ROOT / common.SUPERSEDED_LOCK_REL), \
                    self.assertRaisesRegex(SystemExit, 'Lock 1.*superseded'):
                common.preflight('a' * 64)

    def test_only_lock2_anchors_may_differ_from_pipeline_commit(self):
        for changed, allowed in ((common.ANCHOR_REL, True), (common.RUNS_REL, True),
                                 ('scripts/evaluation/pao-lock-anchor.json', False),
                                 ('scripts/evaluation/m7/anchors/labels-addendum-anchor.json', False),
                                 ('scripts/evaluation/m7/common.py', False)):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'protocol-lock.json'
                record = Path(folder) / 'protocol-lock.record.json'
                candidate = {'status': common.CANDIDATE_STATUS, 'pipeline': {'commit': 'a' * 40}, 'approval': None}
                lock = {**candidate, 'status': 'LOCKED', 'approval': {'by': 'Sean', 'utc': 'synthetic', 'how': 'synthetic',
                         'recordedUtc': 'synthetic', 'candidateSha256': common.sha(common.dump(candidate))}}
                path.write_bytes(common.dump(lock))
                record.write_bytes(common.dump({'file': common.rel(path), 'sha256': common.sha_file(path),
                                               'candidateSha256': lock['approval']['candidateSha256']}))
                def git(*args):
                    return changed + '\n' if args[0] == 'diff' else ''
                with mock.patch.object(common, 'LOCK_PATH', path), mock.patch.object(common, 'LOCK_RECORD', record), \
                        mock.patch.object(common, 'git', side_effect=git), \
                        mock.patch.object(common.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)), \
                        mock.patch.object(common, 'fetch_origin', side_effect=RuntimeError('reached anchor check')):
                    with self.assertRaises(RuntimeError if allowed else SystemExit) as error:
                        common.preflight(common.sha_file(path))
                    self.assertIn('reached anchor check' if allowed else 'Tracked files differ from pipeline.commit', str(error.exception))

    def test_writer_is_blocked_until_coordinator_sets_commit(self):
        with mock.patch.object(writer, 'COMMIT', None), \
                mock.patch.object(writer, 'check_pins', side_effect=AssertionError('must stop before reading pins')), \
                self.assertRaisesRegex(SystemExit, 'COMMIT is a placeholder'):
            writer.gather()

    def test_replacement_builder_protocol_changes_only_the_shape_code_pin(self):
        protocol = json.loads((ROOT / writer.PINS['builderProtocol'][0]).read_bytes())
        source = json.loads((ROOT / writer.PINS['builderSourceProtocol'][0]).read_bytes())
        self.assertEqual(protocol, {**source, 'code': {**source['code'], 'scripts/california/car_shape.py':
                                                    common.sha_file(ROOT / 'scripts/california/car_shape.py')}})
        for pin in ('builderProtocol', 'builderSourceProtocol', 'builderFreeze', 'builderCodeUpdate'):
            path, digest = writer.PINS[pin]
            self.assertEqual(common.sha_file(ROOT / path), digest)
        self.assertIn('scripts/evaluation/m7/build_scene.py', writer.M7_CODE)
        update = json.loads((ROOT / writer.PINS['builderCodeUpdate'][0]).read_bytes())
        self.assertEqual(update['adapter']['sha256'], common.sha_file(ROOT / update['adapter']['path']))

    def test_scientific_rules_sites_and_answers_carry_over(self):
        lock1 = json.loads((ROOT / common.SUPERSEDED_LOCK_REL).read_bytes())
        pins = {key: {'path': path, 'sha256': sha} for key, (path, sha) in {**writer.PINS, **writer.LATER_PINS}.items()}
        lock2 = writer.build_lock(pins, {}, 'a' * 40, {}, {}, 'synthetic')
        for key in ('resultRules', 'clicks', 'tooCloseToCall', 'referenceBlocked', 'metrics', 'sites', 'gate', 'changeRule'):
            self.assertEqual(lock2[key], lock1[key], key)
        self.assertEqual(lock2['labels']['addendum'], lock1['labels']['addendum'])
        for key in ('H measured-shape rows', 'H 50%', 'B all three hard', 'D/no minimum', 'band 0.075 m', 'declared-default', '"None"'):
            self.assertIn(key, lock2['authority'])
        self.assertIn(common.SUPERSEDED_LOCK_SHA, lock2['history'])
        self.assertEqual(lock2['revision'], 6)

    def test_adapter_preserves_builder_rule_and_code_checks(self):
        # A separate interpreter preserves the builder's module-origin checks, without importing either generic common here.
        code = '''import importlib.util, json
from pathlib import Path
root = Path.cwd()
spec = importlib.util.spec_from_file_location('adapter', root / 'scripts/evaluation/m7/build_scene.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
tool = adapter.load_builder()
assert tool.LOCK_PATH_V4 == root / '.local/evaluation/pao-accuracy/lock-2/protocol-lock.json'
protocol = json.loads((root / '.local/evaluation/pao-accuracy/lock-2/builder-protocol-v4.3.json').read_bytes())
assert tool.check_frozen_protocol(protocol, protocol['split']['sha256'], protocol['placementProtocol']['sha256'], 'v4-18.5')[0] == 4.3
protocol['code']['scripts/california/car_shape.py'] = '0' * 64
try:
    tool.check_frozen_protocol(protocol, protocol['split']['sha256'], protocol['placementProtocol']['sha256'], 'v4-18.5')
except ValueError as error:
    assert 'Frozen code hash mismatch' in str(error)
else:
    raise AssertionError('changed code accepted')
'''
        result = subprocess.run([sys.executable, '-I', '-B', '-c', code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
