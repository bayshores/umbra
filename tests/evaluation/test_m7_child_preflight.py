"""Child lock checks use synthetic fixtures only; no AV2 data, real Git, or network."""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def isolated_builder():
    names = ('common', 'prepare_av2', 'car_shape', 'remap_support', 'rectified_calibration')
    saved_path = sys.path[:]
    saved = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    try:
        adapter = load('child_preflight_adapter', 'scripts/evaluation/m7/build_scene.py')
        yield adapter, adapter.load_builder()
    finally:
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
        sys.path[:] = saved_path


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    with isolated_builder() as (adapter, builder):
        shape = builder.car_shape
        m7 = load('child_preflight_common', 'scripts/evaluation/m7/common.py')
        sites = m7.split_sites()  # Frozen metadata only.
        split_raw = m7.SPLIT.read_bytes()
        split = tmp_path / 'split.json'
        split.write_bytes(split_raw)
        anchor_rel, labels_anchor_rel = 'anchors/lock.json', 'anchors/labels.json'
        (tmp_path / 'anchors').mkdir()
        lock_path, record = tmp_path / 'protocol-lock.json', tmp_path / 'record.json'
        addendum = tmp_path / 'labels-addendum.json'
        addendum.write_bytes(b'{"synthetic": true}\n')
        for module, values in (
            (m7, {'ROOT': tmp_path, 'SPLIT': split, 'LOCK_PATH': lock_path, 'LOCK_RECORD': record,
                  'ANCHOR_REL': anchor_rel, 'ANCHOR_DIR': 'anchors', 'ADDENDUM_ANCHOR_REL': labels_anchor_rel,
                  'LABELS_ADDENDUM': addendum, 'RUN_MARKERS': tmp_path / 'run-markers'}),
            (shape, {'LOCK_PATH': lock_path, 'SPLIT_PATH': split}),
            (builder, {'LOCK_PATH_V4': lock_path}),
        ):
            for name, value in values.items():
                monkeypatch.setattr(module, name, value)
        monkeypatch.setattr(shape, 'load_m7_common', lambda: m7)
        monkeypatch.setattr(adapter, 'load_m7_common', lambda: m7)
        real_labels_addendum = m7.labels_addendum
        monkeypatch.setattr(m7, 'labels_addendum', lambda held: real_labels_addendum(held, path=addendum))
        for path in ('score.py', 'driver.py', 'builder.py', 'asset.json'):
            (tmp_path / path).write_bytes(('synthetic ' + path).encode())
        pipeline = {'commit': 'a' * 40, 'scoringCode': {'score.py': m7.sha_file(tmp_path / 'score.py')},
                    'm7Code': {'driver.py': m7.sha_file(tmp_path / 'driver.py')}}
        for key, path in (('shapeProtocol', shape.SHAPE_PROTOCOL_PATH), ('shapeSplitAddendum', shape.SPLIT_ADDENDUM_PATH)):
            target = tmp_path / Path(path).name
            target.write_bytes((ROOT / path).read_bytes())  # Frozen protocol metadata only.
            pipeline[key] = {'path': target.name, 'sha256': m7.sha_file(target)}
        candidate = {'status': m7.CANDIDATE_STATUS, 'approval': None, 'pipeline': pipeline,
                     'sceneBuilder': {'code': {'builder.py': m7.sha_file(tmp_path / 'builder.py')}},
                     'asset': {'path': 'asset.json', 'sha256': m7.sha_file(tmp_path / 'asset.json')},
                     'labels': {'addendum': {'path': addendum.name, 'sha256': m7.sha_file(addendum)}, 'noAddendum': None}}
        lock = {**candidate, 'status': 'LOCKED', 'approval': {'by': 'synthetic approver', 'utc': 'synthetic',
                'how': 'synthetic', 'recordedUtc': 'synthetic', 'candidateSha256': m7.sha(m7.dump(candidate))}}
        lock_path.write_bytes(m7.dump(lock))
        digest = m7.sha_file(lock_path)
        record.write_bytes(m7.dump({'file': m7.rel(lock_path), 'sha256': digest,
                                   'candidateSha256': lock['approval']['candidateSha256']}))
        anchor = tmp_path / anchor_rel
        anchor.write_bytes(m7.dump({'protocolLockSha256': digest}))
        (tmp_path / labels_anchor_rel).write_bytes(m7.dump({'labelsAddendumSha256': m7.sha_file(addendum)}))
        state = {'status': '', 'changed': '', 'ancestor': True, 'unpushed': None}
        calls = []

        def git(*args):
            calls.append(args)
            if args[0] == 'status':
                return state['status'] if len(args) == 2 else ''
            if args[0] == 'diff':
                return state['changed']
            if args[0] == 'rev-list':
                return '1\n'
            raise AssertionError(f'unexpected Git call: {args}')

        def run(args, **kwargs):
            assert args[0] == 'git', 'unexpected child process'
            calls.append(tuple(args))
            if args[1:3] == ['merge-base', '--is-ancestor']:
                return subprocess.CompletedProcess(args, 0 if state['ancestor'] else 1)
            if args[1] == 'ls-files':
                return subprocess.CompletedProcess(args, 0)
            if args[1:3] == ['diff', '--quiet']:
                return subprocess.CompletedProcess(args, int(args[-1] == state['unpushed']))
            raise AssertionError(f'unexpected subprocess: {args}')

        monkeypatch.setattr(m7, 'git', git)
        monkeypatch.setattr(m7.subprocess, 'run', run)
        fetch = mock.Mock(side_effect=AssertionError('child must never fetch'))
        monkeypatch.setattr(m7, 'fetch_origin', fetch)
        monkeypatch.setattr(m7.subprocess, 'Popen', mock.Mock(side_effect=AssertionError('no child process')))
        helpers = {name: getattr(shape, name) for name in ('make_plan', 'download_sweeps', 'approach_split_document',
                                                          'build_car_shape', 'load_receipted', 'fetch_listing',
                                                          'http_fetch', 'fetch_with_rule')}
        for module, names in ((m7, ('curl', 'write_new', 'run_marker')),
                              (shape, ('http_fetch', 'write_exclusive', 'make_plan', 'download_sweeps', 'approach_split_document')),
                              (builder, ('snapshot', 'publish', 'find_sources', 'load_poses', 'load_annotations'))):
            for name in names:
                monkeypatch.setattr(module, name, mock.Mock(side_effect=AssertionError(f'protected I/O reached: {name}')))
        original_snapshot = shape.snapshot

        def snapshot(path, *args, **kwargs):
            assert Path(path) == lock_path, 'only the synthetic lock may be opened'
            return original_snapshot(path, *args, **kwargs)

        monkeypatch.setattr(shape, 'snapshot', snapshot)
        yield SimpleNamespace(m7=m7, shape=shape, adapter=adapter, builder=builder, root=tmp_path,
                              lock=lock, digest=digest, record=record, anchor=anchor, addendum=addendum,
                              sites=sites, state=state, calls=calls, fetch=fetch, helpers=helpers)


def child_argv(f, command, site):
    log = site['logId']
    common = ['--held-out-run', '--protocol-lock-sha256', f.digest]
    metadata = [item for key in ('ground', 'annotations', 'poses') for item in
                ('--' + key, str(f.root / log / (key + '.json')), '--' + key + '-sha256', 'b' * 64)]
    if command == 'plan':
        return ['shape', 'plan', '--log', log, *metadata, *common]
    if command == 'download':
        return ['shape', 'download', '--log', log, '--plan-sha256', 'b' * 64, *common]
    return ['shape', 'approach-split', *metadata, '--name', 'synthetic', *common]


def invoke(f, entry, site):
    if entry in ('plan', 'download', 'approach-split'):
        with mock.patch.object(sys, 'argv', child_argv(f, entry, site)):
            f.shape.main()
    elif entry == 'adapter-build':
        f.builder.build(SimpleNamespace(log_id=site['logId'], held_out_run=True, protocol_lock_sha256=f.digest))
    else:
        names = ('protocol', 'protocol-sha256', 'placement-protocol', 'placement-protocol-sha256', 'split', 'split-lock',
                 'log-dir', 'listing-left', 'listing-right', 'listing-lidar', 'license', 'output')
        argv = ['adapter', *[item for key in names for item in ('--' + key, str(f.root / key))],
                '--log-id', site['logId'], '--crossing-id', str(site['crossingId']),
                '--held-out-run', '--protocol-lock-sha256', f.digest]
        with mock.patch.object(f.adapter, 'load_builder', return_value=f.builder), mock.patch.object(sys, 'argv', argv):
            f.adapter.main()


@pytest.mark.parametrize('entry', ['adapter-main', 'adapter-build', 'plan', 'download', 'approach-split'])
@pytest.mark.parametrize('failure,reason', [('record', 'missing or names another lock'),
                                         ('anchor', 'does not exist'), ('unpushed', 'differs from origin/main')])
def test_children_refuse_before_protected_io(fixture, entry, failure, reason):
    f = fixture
    if failure == 'record':
        f.record.unlink()
    elif failure == 'anchor':
        f.anchor.unlink()
    else:
        f.state['unpushed'] = f.m7.ANCHOR_REL
    sites = {s['logId']: s for s in f.sites if s['role'] in f.m7.HELD_OUT_ROLES}
    assert sites
    for site in sites.values():
        with pytest.raises(SystemExit, match=reason):
            invoke(f, entry, site)
    f.fetch.assert_not_called()


@pytest.mark.parametrize('helper', ['make_plan', 'download_sweeps', 'approach_split_document', 'build_car_shape',
                                  'load_receipted', 'fetch_listing', 'http_fetch', 'fetch_with_rule'])
@pytest.mark.parametrize('failure,reason', [('record', 'missing or names another lock'),
                                         ('anchor', 'does not exist'), ('unpushed', 'differs from origin/main')])
def test_direct_shape_helpers_require_full_preflight(fixture, helper, failure, reason):
    f = fixture
    if failure == 'record':
        f.record.unlink()
    elif failure == 'anchor':
        f.anchor.unlink()
    else:
        f.state['unpushed'] = f.m7.ANCHOR_REL
    for site in {s['logId']: s for s in f.sites if s['role'] in f.m7.HELD_OUT_ROLES}.values():
        log = site['logId']
        folder = f.root / f'shape-v2-{log}'
        auth = {'held_out_run': True, 'lock_sha256': f.digest}
        url = f.shape.BUCKET_URL + f.shape.lidar_prefix(log) + '1.feather'
        args = {
            'make_plan': (log, 'absent', 'b' * 64, f.root / log / 'annotations', 'b' * 64,
                          f.root / log / 'poses', 'b' * 64),
            'download_sweeps': (folder, 'b' * 64),
            'approach_split_document': (b'{}', folder, b'', b''),
            'build_car_shape': (folder, 'b' * 64, {}, [], b'', b'', 'b' * 64),
            'load_receipted': (folder, 'b' * 64, {'log': log}, {}, [], b'', b''),
            'fetch_listing': (log, f.fetch, f.fetch),
            'http_fetch': (url,),
            'fetch_with_rule': (url, f.fetch, f.fetch, f.fetch),
        }[helper]
        with pytest.raises(SystemExit, match=reason):
            f.helpers[helper](*args, **auth)
    f.fetch.assert_not_called()


def test_cached_r5_shape_guard_cannot_bypass_adapter_preflight(fixture, monkeypatch):
    f = fixture
    f.record.unlink()
    old_guard = mock.Mock(return_value={'path': str(f.m7.LOCK_PATH), 'sha256': f.digest})
    monkeypatch.setattr(f.shape, 'check_held_out', old_guard)
    monkeypatch.setattr(f.shape, 'load_m7_common', mock.Mock(side_effect=AssertionError('cached shape loader used')))
    f.builder.check_module_origins()  # The cached module still has the expected source path, as in the review probe.
    site = next(s for s in f.sites if s['role'] in f.m7.HELD_OUT_ROLES)
    with pytest.raises(SystemExit, match='missing or names another lock'):
        invoke(f, 'adapter-build', site)
    old_guard.assert_not_called()
    f.fetch.assert_not_called()


def test_ground_environment_context_does_not_bypass_full_preflight(fixture, monkeypatch):
    f = fixture
    f.record.unlink()
    site = next(s for s in f.sites if s['role'] in f.m7.HELD_OUT_ROLES)
    monkeypatch.setenv(f.shape.GROUND_LOCK_ENV, f.digest)
    with pytest.raises(SystemExit, match='missing or names another lock'):
        f.helpers['build_car_shape'](f.root / f"shape-v2-{site['logId']}", 'b' * 64, {}, [], b'', b'', 'b' * 64)
    f.fetch.assert_not_called()


def test_authorized_http_helper_receives_the_same_lock_context(fixture, monkeypatch):
    f = fixture
    site = next(s for s in f.sites if s['role'] in f.m7.HELD_OUT_ROLES)
    http = f.helpers['http_fetch']
    monkeypatch.setattr(f.shape, 'http_fetch', http)
    response = mock.MagicMock()
    response.__enter__.return_value = response
    response.status, response.headers = 200, {'Content-Length': '3', 'ETag': 'synthetic'}
    response.read.return_value = b'abc'
    network = mock.Mock(return_value=response)
    monkeypatch.setattr(f.shape.urllib.request, 'urlopen', network)
    url = f.shape.BUCKET_URL + f.shape.lidar_prefix(site['logId']) + '1.feather'
    result = f.helpers['fetch_with_rule'](url, lambda data: data == b'abc', http, f.fetch,
                                         held_out_run=True, lock_sha256=f.digest)
    assert result['ok'] and result['attempts'] == 1
    assert network.call_count == 1  # A synthetic HTTP response; no network operation.
    f.fetch.assert_not_called()


def test_no_fetch_preflight_accepts_current_synthetic_marker_and_output(fixture):
    f = fixture
    f.m7.RUN_MARKERS.mkdir()
    (f.m7.RUN_MARKERS / 'synthetic.json').write_bytes(b'{"fixture":true}\n')
    (f.root / 'current-output').mkdir()
    (f.root / 'current-output' / 'synthetic.json').write_bytes(b'{"fixture":true}\n')
    assert f.m7.preflight(f.digest, fetch=False) == f.lock
    site = next(s for s in f.sites if s['role'] in f.m7.HELD_OUT_ROLES)
    assert f.m7.guard(site['logId'], site['crossingId'], False, f.digest, fetch=False) == (site, f.lock)
    f.fetch.assert_not_called()
    assert ('git', 'merge-base', '--is-ancestor', f.lock['pipeline']['commit'], 'HEAD') in f.calls
    assert any(call[:3] == ('git', 'diff', '--quiet') for call in f.calls)


@pytest.mark.parametrize('failure,reason', [
    ('status', 'modified or untracked'), ('ancestor', 'not an ancestor'), ('changed', 'Tracked files differ'),
    ('asset.json', 'sha256 in the lock'), ('score.py', 'sha256 in the lock'),
    ('driver.py', 'sha256 in the lock'), ('builder.py', 'sha256 in the lock'),
    ('protocol.json', 'sha256 in the lock'), ('split-addendum.json', 'sha256 in the lock'),
    ('labels', 'differs from its committed anchor'), ('labels-pin', 'differs from the one the lock pins'),
])
def test_no_fetch_keeps_every_substantive_check(fixture, failure, reason):
    f = fixture
    if failure in ('status', 'ancestor', 'changed'):
        f.state[failure] = {'status': ' M source.py\n', 'ancestor': False, 'changed': 'source.py\n'}[failure]
    elif failure == 'labels':
        f.addendum.write_bytes(b'changed synthetic labels')
        f.lock['labels']['addendum']['sha256'] = f.m7.sha_file(f.addendum)
    elif failure == 'labels-pin':
        f.lock['labels'] = {'addendum': None, 'noAddendum': 'synthetic answer'}
    else:
        (f.root / failure).write_bytes(b'changed synthetic pin')
    if failure.startswith('labels'):
        # Keep candidate/record/lock anchor valid to reach the distinct labels checks.
        candidate = {**f.lock, 'status': f.m7.CANDIDATE_STATUS, 'approval': None}
        f.lock['approval']['candidateSha256'] = f.m7.sha(f.m7.dump(candidate))
        f.m7.LOCK_PATH.write_bytes(f.m7.dump(f.lock))
        f.digest = f.m7.sha_file(f.m7.LOCK_PATH)
        f.record.write_bytes(f.m7.dump({'file': f.m7.rel(f.m7.LOCK_PATH), 'sha256': f.digest,
                                      'candidateSha256': f.lock['approval']['candidateSha256']}))
        f.anchor.write_bytes(f.m7.dump({'protocolLockSha256': f.digest}))
    with pytest.raises(SystemExit, match=reason):
        f.m7.preflight(f.digest, fetch=False)
    f.fetch.assert_not_called()


def test_development_logs_need_neither_lock_nor_m7_loader(fixture):
    f = fixture
    f.m7.LOCK_PATH.unlink()
    logs = {s['logId'] for s in f.sites if s['logId'].startswith(('8c52d911', '453e5558', 'c780d53a'))}
    assert {log[:8] for log in logs} == {'8c52d911', '453e5558', 'c780d53a'}
    with mock.patch.object(f.shape, 'load_m7_common', side_effect=AssertionError('development must not load M7')):
        for log in logs:
            assert f.shape.check_held_out(log, False, None) is None


def test_m7_loader_preserves_sponsor_common_origin():
    code = '''import importlib.util, sys
from pathlib import Path
root = Path.cwd()
spec = importlib.util.spec_from_file_location('child_adapter', root / 'scripts/evaluation/m7/build_scene.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
tool = adapter.load_builder()
sponsor = sys.modules['common']
assert Path(sponsor.__file__).resolve() == root / 'scripts/sponsors/common.py'
m7 = tool.car_shape.load_m7_common()
assert m7.__name__ == '_pao_lock2_m7_common'
assert Path(m7.__file__).resolve() == root / 'scripts/evaluation/m7/common.py'
assert sys.modules['common'] is sponsor
sys.modules['_pao_lock2_builder_common'] = type('OldCommon', (), {'preflight': lambda *_: None})()
fresh = adapter.load_m7_common()
assert fresh is not sys.modules['_pao_lock2_builder_common']
assert fresh is not adapter.load_m7_common()
assert Path(fresh.__file__).resolve() == root / 'scripts/evaluation/m7/common.py'
assert sys.modules['common'] is sponsor
tool.check_module_origins()
'''
    result = subprocess.run([sys.executable, '-I', '-B', '-c', code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
