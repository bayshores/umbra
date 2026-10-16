"""NV-Grounding-DINO job tooling: wrapper, preparation and consumer, all offline.

No model, container, network or cloud call is used. A fake `grounding_dino`, a fake `nvidia-smi` and a fake
`torch` package stand in for the TAO container so the wrapper's real code path (hash checks, pre-flight,
downloads over file://, label parsing, duplicate-box rule, bundle, log block, acknowledgement) runs on the host.
Fake outputs are labelled test-only and refused by the consumer and detect-vehicles unless explicitly allowed.

Run: python3 -m unittest discover -s tests/cloud -p 'test_*.py'
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from io import StringIO
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'cloud'))
import consume_nvdino_result as consumer  # noqa: E402
import prepare_nvdino_job as prepare  # noqa: E402
import run_nvdino_job as job  # noqa: E402

SID = '0123456789abcdef0123456789abcdef'
WORKDIR = ROOT / '.local/evaluation/auto-placement-2026-10-11/iter3-plans/workdir'
CHUNKS_DIR = WORKDIR / 'chunks'
EXPERIMENT = WORKDIR / 'skeptic/experiment.yaml'
LICENSE_DIR = ROOT / '.local/evaluation/nebius-credit-unblock-2026-10-10/inject'
HAVE_SOURCES = CHUNKS_DIR.is_dir() and EXPERIMENT.is_file() and LICENSE_DIR.is_dir()
# The main-thread coordinator is private (.local); its pure phase-0, cancel and retrieval logic is tested offline.
COORDINATOR = ROOT / '.local/evaluation/nvdino-job-2026-10-12/coordinator.py'


def load_coordinator():
    if not COORDINATOR.is_file():
        return None
    spec = importlib.util.spec_from_file_location('nvdino_coordinator', COORDINATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


coord = load_coordinator()
# The unpatched function, for tests that point it at a temporary ledger (setUp patches the module attribute).
REAL_CEILING_EXCEPTION = coord.ceiling_exception if coord else None
CAPS = ['car', 'truck', 'bus', 'person']
# Labelled fixture lines in TAO's format (not model output): a car/truck pair with identical coordinates.
FIXTURE_LABELS = (
    'car 0.00 0 0.00 44.000 176.000 153.000 219.100 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.374\n'
    'truck 0.00 0 0.00 44.000 176.000 153.000 219.100 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.341\n'
    'car 0.00 0 0.00 149.200 181.200 235.700 217.800 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.896\n'
    'person 0.00 0 0.00 453.400 178.700 463.500 199.600 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.516\n')

FAKE_TAO = r'''#!/usr/bin/env python3
"""SELFTEST FAKE grounding_dino: checks its inputs, copies fixture labels. Not NVIDIA software."""
import os, shutil, sys
here = os.path.dirname(os.path.abspath(__file__))
assert sys.argv[1:3] == ['inference', '-e'] and len(sys.argv) == 4, sys.argv
for key, value in (('TELEMETRY_OPT_OUT', 'yes'), ('HF_HUB_OFFLINE', '1'), ('TRANSFORMERS_OFFLINE', '1')):
    assert os.environ.get(key) == value, key
assert 'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD' not in os.environ
spec = open(sys.argv[3]).read()
get = lambda key: spec.split(key + ': ', 1)[1].split('\n')[0].split(',')[0].split('   #')[0].strip()
results, checkpoint = get('results_dir'), get('checkpoint')
bert, images = get('  text_encoder_type'), spec.split('image_dir: [', 1)[1].split(']')[0]
assert os.path.isfile(checkpoint) and len(os.listdir(bert)) == 5 and os.listdir(images) == ['recorded.jpg']
print('fake grounding_dino: inputs present; writing fixture labels', flush=True)
if os.environ.get('FAKE_TAO_FAIL') == '1':
    print('fake failure requested', flush=True)
    sys.exit(1)
os.makedirs(os.path.join(results, 'labels'))
os.makedirs(os.path.join(results, 'images_annotated'))
shutil.copyfile(os.path.join(here, '..', 'fixture-labels.txt'), os.path.join(results, 'labels', 'recorded.txt'))
shutil.copyfile(os.path.join(images, 'recorded.jpg'), os.path.join(results, 'images_annotated', 'recorded.jpg'))
'''
FAKE_SMI = '#!/bin/sh\necho "SELFTEST FAKE GPU, 000.00.00, 1 MiB"\n'
FAKE_TORCH = '''"""SELFTEST FAKE torch: only what the wrapper's pre-flight check touches."""
__version__ = '0.0.0+selftest-fake'
class version:
    cuda = 'fake'
class _T:
    def __init__(self, v): self.v = v
    def __mul__(self, k): return _T([x * k for x in self.v])
    def sum(self): return _T([sum(self.v)])
    def item(self): return float(self.v[0])
def ones(n, device=None): return _T([1.0] * n)
class cuda:
    is_available = staticmethod(lambda: True)
    device_count = staticmethod(lambda: 1)
    get_device_name = staticmethod(lambda i: 'SELFTEST FAKE GPU')
'''
FAKE_UNPICKLER = "def _get_allowed_globals():\n    return {'collections.Counter': object}\n"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_fakes(root, labels=FIXTURE_LABELS):
    """fakebin/{grounding_dino,nvidia-smi}, fakepy/torch, fixture-labels.txt under root."""
    (root / 'fakebin').mkdir(parents=True)
    (root / 'fakepy/torch').mkdir(parents=True)
    for name, text in (('grounding_dino', FAKE_TAO), ('nvidia-smi', FAKE_SMI)):
        (root / 'fakebin' / name).write_text(text)
        (root / 'fakebin' / name).chmod(0o755)
    (root / 'fakepy/torch/__init__.py').write_text(FAKE_TORCH)
    (root / 'fakepy/torch/_weights_only_unpickler.py').write_text(FAKE_UNPICKLER)
    (root / 'fixture-labels.txt').write_text(labels)


def fake_remote(remote_dir, url_root):
    """Tiny fake weight/BERT files with matching test pins, served over file://."""
    remote_dir.mkdir(parents=True)
    files = [('weights', prepare.WEIGHTS['file'], b'SELFTEST FAKE WEIGHTS, not NV-Grounding-DINO\n')]
    files += [('bert', name, ('SELFTEST FAKE %s\n' % name).encode()) for name, _, _ in prepare.BERT]
    out = []
    for group, name, raw in files:
        (remote_dir / name).write_bytes(raw)
        out.append((group, name, raw, url_root + '/' + name))
    return out


def make_test_model(weights_raw):
    model = dict(prepare.MODEL)
    model.update(name='SELFTEST fake detector (not NV-Grounding-DINO)', weightsBytes=len(weights_raw),
                 weightsSha256=sha(weights_raw))
    return model


def build_test_job(container_root, host_root, sid=SID, url_root=None):
    """Test-only prepared files under host_root/prepared + fakes; plan paths live under container_root."""
    c = container_root
    paths = {'work': c + '/workload', 'images': c + '/workload/images', 'weights': c + '/workload/weights',
             'bert': c + '/workload/bert', 'results': c + '/workload/tao-results', 'output': c + '/output/' + sid,
             'ack': c + '/output/.' + sid + '-retrieved'}
    write_fakes(host_root / 'fakes')
    remote = fake_remote(host_root / 'remote', url_root or 'file://' + str(host_root / 'remote'))
    downloads = [{'group': g, 'url': url, 'path': paths[g] + '/' + n, 'bytes': len(r), 'sha256': sha(r)}
                 for g, n, r, url in remote]
    files = {'run_nvdino_job.py': Path(job.__file__).read_bytes(),
             'infer.yaml': prepare.infer_yaml(EXPERIMENT.read_bytes(), paths),
             'NVIDIA-Open-Model-License.txt': (LICENSE_DIR / 'NVIDIA-Open-Model-License.txt').read_bytes(),
             'Notice': (LICENSE_DIR / 'Notice').read_bytes()}
    for name, _, _ in prepare.CHUNKS:
        files[name] = (CHUNKS_DIR / name).read_bytes()
    where = {'run_nvdino_job.py': (c + '/workload/input/run_nvdino_job.py', 'wrapper'),
             'infer.yaml': (c + '/workload/input/infer.yaml', 'spec'),
             'NVIDIA-Open-Model-License.txt': (c + '/model/NVIDIA-Open-Model-License.txt', 'license'),
             'Notice': (c + '/model/Notice', 'notice')}
    injected = []
    for name, raw in files.items():
        if name.startswith('recorded.jpg.part'):
            item = {'path': c + '/workload/input/' + name, 'role': 'chunk', 'index': int(name[-1])}
        else:
            item = {'path': where[name][0], 'role': where[name][1]}
        item.update(bytes=len(raw), sha256=sha(raw))
        injected.append(item)
    plan = prepare.build_plan(sid, injected, paths=paths, downloads=downloads, test_only=True, min_free_bytes=1,
                              model=make_test_model(remote[0][2]))
    plan_raw = prepare.plan_bytes(plan)
    files['plan.json'] = plan_raw
    prepared = host_root / 'prepared'
    prepared.mkdir()
    for name, raw in files.items():
        (prepared / name).write_bytes(raw)
    mounts = {name: (where[name][0] if name in where else c + '/workload/input/' + name) for name in files}
    return plan, plan_raw, prepared, mounts


@unittest.skipUnless(HAVE_SOURCES, 'local NV-DINO job sources (.local) are not present')
class HostRun(unittest.TestCase):
    """The real wrapper end to end on the host, with fakes and file:// downloads."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(os.path.realpath(temp.name))
        self.plan, self.plan_raw, prepared, mounts = build_test_job(str(self.root / 'c'), self.root)
        for name, target in mounts.items():
            Path(target).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(prepared / name, target)
        self.wrapper = mounts['run_nvdino_job.py']
        self.plan_path = mounts['plan.json']
        self.plan_sha = sha(self.plan_raw)
        self.out = Path(self.plan['paths']['output'])
        self.env = {**os.environ, 'TELEMETRY_OPT_OUT': 'yes', 'PYTHONPATH': str(self.root / 'fakes/fakepy'),
                    'PATH': str(self.root / 'fakes/fakebin') + os.pathsep + os.environ['PATH']}
        self.env.pop('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', None)
        self.plan_python = sys.executable

    def run_wrapper(self, *extra, env=None, plan_sha=None, window='0', ack_after=None):
        argv = [sys.executable, '-B', self.wrapper, '--submission-id', SID, '--plan', self.plan_path,
                '--plan-sha256', plan_sha or self.plan_sha, '--output', str(self.out),
                '--retrieval-window-seconds', window, *extra]
        if ack_after is not None:
            def ack():
                deadline = time.monotonic() + 30
                while not (self.out / 'job-result.json').exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
                time.sleep(ack_after)
                Path(self.plan['paths']['ack']).touch()
            thread = threading.Thread(target=ack)
            thread.start()
            self.addCleanup(thread.join)
        return subprocess.run(argv, capture_output=True, text=True, env=env or self.env, timeout=120)

    def test_end_to_end(self):
        result = self.run_wrapper('--execute', window='300', ack_after=0.2)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(job.READY, result.stdout)
        self.assertEqual(sorted(os.listdir(self.out)), sorted(job.OUTPUT_NAMES))
        boxes = json.loads((self.out / 'boxes.json').read_text())
        self.assertTrue(boxes['testOnly'])
        self.assertEqual(boxes['rawLineCount'], 4)
        self.assertEqual([(b['class'], b['score']) for b in boxes['boxes']], [('car', 0.896), ('person', 0.516),
                                                                               ('car', 0.374)])
        self.assertEqual(boxes['boxes'][2]['mergedLabels'], [{'class': 'truck', 'score': 0.341, 'line': 2}])
        result_doc = json.loads((self.out / 'job-result.json').read_text())
        self.assertEqual((result_doc['status'], result_doc['mergedLineCount'], result_doc['planSha256']),
                         ('succeeded', 1, self.plan_sha))
        # Consumer: bundle, log block, both compared; test plans need --allow-test.
        logs = self.root / 'logs.txt'
        logs.write_text(''.join('2026-10-12T00:00:00Z stdout ' + line + '\n' for line in result.stdout.splitlines()))
        with self.assertRaises(job.Refused):
            consumer.consume(self.plan_path, self.plan_sha, SID, bundle=self.out)
        from_bundle = consumer.consume(self.plan_path, self.plan_sha, SID, bundle=self.out, compare_log_block=logs,
                                       allow_test=True)
        from_logs = consumer.consume(self.plan_path, self.plan_sha, SID, log_block=logs, allow_test=True)
        self.assertTrue(from_bundle['source']['logBlockMatchesBundle'])
        self.assertEqual(from_bundle['boxes'], from_logs['boxes'])
        self.assertEqual(len(from_logs['boxes']), 3)
        self.assertNotIn('timing', json.dumps(from_bundle).lower())
        # Tampering is refused.
        tampered = self.root / 'tampered'
        shutil.copytree(self.out, tampered)
        doc = json.loads((tampered / 'boxes.json').read_text())
        doc['boxes'][0]['score'] = 0.99
        (tampered / 'boxes.json').write_text(json.dumps(doc))
        with self.assertRaises(job.Refused):
            consumer.consume(self.plan_path, self.plan_sha, SID, bundle=tampered, allow_test=True)
        extra = self.root / 'extra'
        shutil.copytree(self.out, extra)
        (extra / 'notes.txt').write_text('x')
        with self.assertRaises(job.Refused):
            consumer.consume(self.plan_path, self.plan_sha, SID, bundle=extra, allow_test=True)
        lines = logs.read_text().splitlines()
        bad = [line.replace(line[-4:], 'AAAA') if job.LINE + ' 00002 ' in line else line for line in lines]
        (self.root / 'bad.txt').write_text('\n'.join(bad) + '\n')
        with self.assertRaises(job.Refused):
            consumer.consume(self.plan_path, self.plan_sha, SID, log_block=self.root / 'bad.txt', allow_test=True)
        with self.assertRaises(job.Refused):
            consumer.consume(self.plan_path, '0' * 64, SID, bundle=self.out, allow_test=True)

    def test_refuses_without_execute_and_wrong_plan(self):
        result = self.run_wrapper()
        self.assertEqual(result.returncode, 2)
        self.assertIn(job.FAILED + ' step=arguments', result.stdout)
        result = self.run_wrapper('--execute', plan_sha='0' * 64)
        self.assertEqual(result.returncode, 1)
        self.assertIn('plan sha256 differs', result.stdout)
        self.assertFalse(self.out.exists())

    def test_preflight_fails_before_any_download(self):
        env = dict(self.env)
        env['PYTHONPATH'] = str(self.root / 'nothing-here')  # no torch at all
        result = self.run_wrapper('--execute', env=env)
        self.assertEqual(result.returncode, 1)
        self.assertIn(job.FAILED + ' step=preflight', result.stdout)
        self.assertFalse(Path(self.plan['paths']['weights']).exists())
        shutil.rmtree(self.out)
        env = dict(self.env)
        env.pop('TELEMETRY_OPT_OUT')
        result = self.run_wrapper('--execute', env=env)
        self.assertIn(job.FAILED + ' step=preflight', result.stdout)
        self.assertIn('TELEMETRY_OPT_OUT', result.stdout)

    def test_tao_failure_is_logged_and_leaves_no_result(self):
        result = self.run_wrapper('--execute', env={**self.env, 'FAKE_TAO_FAIL': '1'})
        self.assertEqual(result.returncode, 1)
        self.assertIn(job.FAILED + ' step=tao', result.stdout)
        self.assertIn('fake failure requested', result.stdout)
        self.assertFalse((self.out / 'job-result.json').exists())
        self.assertNotIn(job.BEGIN, result.stdout)


class RetrievalWait(unittest.TestCase):
    def plan(self, root):
        return {'paths': {'ack': str(root / ('.' + SID + '-retrieved'))}}

    def test_ack_and_expiry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, clock = self.plan(root), job.Clock(100)
            threading.Timer(0.1, (root / ('.' + SID + '-retrieved')).touch).start()
            started = time.monotonic()
            with redirect_stdout(StringIO()):
                job.wait_for_ack(plan, 5, clock)
            self.assertLess(time.monotonic() - started, 4)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(TimeoutError), redirect_stdout(StringIO()):
                job.wait_for_ack(self.plan(Path(temp)), 1, job.Clock(100))

    def test_ack_must_be_empty_regular_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / ('.' + SID + '-retrieved')).write_text('x')
            with self.assertRaises(job.Refused), redirect_stdout(StringIO()):
                job.wait_for_ack(self.plan(root), 2, job.Clock(100))


class Labels(unittest.TestCase):
    def test_parse_and_dedupe(self):
        rows = job.parse_labels(FIXTURE_LABELS, CAPS, 0.25)
        self.assertEqual(len(rows), 4)
        boxes = job.dedupe(rows, CAPS)
        self.assertEqual([b['class'] for b in boxes], ['car', 'person', 'car'])
        self.assertEqual(boxes[2]['box'], [44.0, 176.0, 153.0, 219.1])

    def test_tie_keeps_earlier_caption(self):
        text = ('truck 0.00 0 0.00 1.000 2.000 3.000 4.000 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.400\n'
                'car 0.00 0 0.00 1.000 2.000 3.000 4.000 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.400\n')
        boxes = job.dedupe(job.parse_labels(text, CAPS, 0.25), CAPS)
        self.assertEqual((len(boxes), boxes[0]['class'], boxes[0]['mergedLabels'][0]['class']), (1, 'car', 'truck'))

    def test_empty_file_is_zero_boxes(self):
        self.assertEqual(job.parse_labels('', CAPS, 0.25), [])

    def test_strict_lines(self):
        good = 'car 0.00 0 0.00 1.000 2.000 3.000 4.000 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.400\n'
        for bad in (good[:-1], good.replace('car', 'dog'), good.replace('0.400', '0.200'),
                    good.replace('1.000', 'nan'), good.replace('1.000 2.000 3.000', '9.000 2.000 3.000'),
                    good.replace(' 0.00 0 0.00 ', ' 0.00 0 '), good.replace('4.000', '4.0')):
            with self.assertRaises(job.Refused, msg=bad):
                job.parse_labels(bad, CAPS, 0.25)


class LogBlock(unittest.TestCase):
    files = {'boxes.json': b'{"a": 1}\n', 'labels.txt': FIXTURE_LABELS.encode(), 'job-result.json': b'{"b": 2}\n'}

    def test_round_trip_with_prefixes_and_repeats(self):
        lines, h = job.encode_block(SID, self.files)
        self.assertTrue(all(len(line) <= 110 for line in lines[1:-1]))
        noisy = ['stdout F ' + line for line in lines] + ['stdout F ' + lines[2], 'unrelated line']
        files, again = job.decode_block('\n'.join(noisy), SID)
        self.assertEqual((files, again), (self.files, h))

    def test_json_wrapped_logs(self):
        lines, h = job.encode_block(SID, self.files)
        per_line = '\n'.join(json.dumps({'timestamp': '2026-10-12T00:00:00Z', 'message': line}) for line in lines)
        one_doc = json.dumps([{'message': line} for line in lines]).replace('/', '\\/')
        pretty = json.dumps({'items': [{'message': line} for line in lines]}, indent=2)
        for text in (per_line, one_doc, pretty):
            self.assertEqual(job.decode_block(text, SID), (self.files, h))
        cut = [line[:-10] if job.LINE + ' 00001 ' in line else line for line in lines]
        with self.assertRaises(Exception):
            job.decode_block('\n'.join(json.dumps({'message': line}) for line in cut), SID)

    def test_refusals(self):
        lines, _ = job.encode_block(SID, self.files)
        cases = {'missing line': [line for line in lines if ' 00002 ' not in line],
                 'conflict': lines + [lines[1][:-4] + 'AAAA'],
                 'tampered': [lines[1][:-4] + 'AAAA' if i == 1 else line for i, line in enumerate(lines)],
                 'no trailer': lines[:-1]}
        for name, case in cases.items():
            with self.assertRaises(Exception, msg=name):
                job.decode_block('\n'.join(case), SID)
        with self.assertRaises(job.Refused):
            job.decode_block('\n'.join(lines), 'f' * 32)


@unittest.skipUnless(HAVE_SOURCES, 'local NV-DINO job sources (.local) are not present')
class DownloadGuards(unittest.TestCase):
    AV2_URLS = (
        'https://argoverse.s3.amazonaws.com/datasets/av2/sensor/val/example/annotations.feather',
        'https://ARGOVERSE.s3.amazonaws.com/?prefix=x',
        'https://argoverse.s3.us-west-2.amazonaws.com/x',
        'https://s3.amazonaws.com/argoverse/x',
        'https://s3.us-west-2.amazonaws.com/argoverse/x',
        'https://other.example/datasets/av2/sensor/val/x',
        'https://huggingface.co/x?prefix=datasets%2Fav2%2Fsensor',
        'https://other.example/%61rgoverse/x',
        'file:///tmp/datasets/av2/x',
    )

    def real_plan(self):
        injected = [{'path': '/workload/input/run_nvdino_job.py', 'role': 'wrapper', 'bytes': 10, 'sha256': 'a' * 64},
                    {'path': '/workload/input/infer.yaml', 'role': 'spec', 'bytes': 10, 'sha256': 'b' * 64}]
        injected += [{'path': '/workload/input/' + n, 'role': 'chunk', 'index': i, 'bytes': b, 'sha256': h}
                     for i, (n, b, h) in enumerate(prepare.CHUNKS)]
        return prepare.build_plan(SID, injected)

    def test_pinned_model_urls_and_test_files_still_validate(self):
        plan = self.real_plan()
        self.assertIs(job.validate_plan(plan), plan)
        plan['testOnly'] = True
        for item in plan['downloads']:
            item['url'] = 'file:///tmp/model/' + os.path.basename(item['path'])
        self.assertIs(job.validate_plan(plan), plan)
        self.assertLessEqual(Path(job.__file__).stat().st_size, 65536)

    def test_non_test_plans_refuse_unpinned_model_urls(self):
        for group in ('weights', 'bert'):
            for url in ('https://other.example/model', 'https://huggingface.co.attacker.example/model',
                        'https://huggingface.co@attacker.example/model',
                        'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/other/model'):
                plan = self.real_plan()
                next(d for d in plan['downloads'] if d['group'] == group)['url'] = url
                with self.subTest(group=group, url=url), self.assertRaisesRegex(job.Refused, 'model resource'):
                    job.validate_plan(plan)

    def test_plan_refuses_av2_urls_before_injected_reads_or_output(self):
        for url in self.AV2_URLS:
            for test in (False, True):
                plan = self.real_plan(); plan['testOnly'] = test
                plan['downloads'][0]['url'] = url
                raw = job.pretty(plan)
                args = SimpleNamespace(execute=True, submission_id=SID, plan_sha256=sha(raw), plan='synthetic-plan.json',
                                       output=plan['paths']['output'], retrieval_window_seconds=0)
                with self.subTest(url=url, test=test), \
                        mock.patch.object(job, 'read_regular', side_effect=[raw, AssertionError('injected read')]) as read, \
                        mock.patch.object(job.os, 'makedirs', side_effect=AssertionError('folder created')) as mkdir, \
                        mock.patch.object(job.os, 'open', side_effect=AssertionError('file opened')) as opened, \
                        mock.patch.object(job.urllib.request, 'build_opener', side_effect=AssertionError('request')) as request, \
                        self.assertRaisesRegex(job.Refused, 'AV2'):
                    job.run_job(args, None, {})
                read.assert_called_once_with(args.plan, job.MAX_PLAN_BYTES)
                mkdir.assert_not_called(); opened.assert_not_called(); request.assert_not_called()

    def command_plans(self, test):
        """Plans that each swap one child command for another program (lock-2 review r11 F11)."""
        url = self.AV2_URLS[0]
        edits = (lambda p: p['preflight'].update(nvidiaSmi=['curl', url]),
                 lambda p: p['preflight'].update(nvidiaSmi=list(job.NVIDIA_SMI_ARGV) + [url]),
                 lambda p: p['preflight'].update(python='curl'),
                 lambda p: p['preflight'].update(python='/usr/bin/python3'),
                 lambda p: p['tao'].update(argv=['curl', url, '-e', p['spec']['path']]),
                 lambda p: p['tao'].update(argv=['sh', '-c', 'curl ' + url, '-e', p['spec']['path']]),
                 lambda p: p['tao'].update(argv=['grounding_dino', 'inference', url, '-e', p['spec']['path']]))
        for edit in edits:
            plan = self.real_plan(); plan['testOnly'] = test
            edit(plan)
            yield plan

    def test_plan_accepts_only_the_fixed_child_commands(self):
        plan = self.real_plan()
        self.assertEqual(plan['preflight']['nvidiaSmi'], list(job.NVIDIA_SMI_ARGV))
        self.assertEqual(plan['preflight']['python'], job.PREFLIGHT_PYTHON)
        self.assertEqual(plan['tao']['argv'], ['grounding_dino', 'inference', '-e', plan['spec']['path']])
        for test in (False, True):
            for plan in self.command_plans(test):
                with self.subTest(test=test, pre=plan['preflight'], tao=plan['tao']['argv']), \
                        self.assertRaisesRegex(job.Refused, 'preflight|tao argv'):
                    job.validate_plan(plan)

    def test_command_plans_refused_before_any_child_or_file(self):
        for test in (False, True):
            for plan in self.command_plans(test):
                raw = job.pretty(plan)
                args = SimpleNamespace(execute=True, submission_id=SID, plan_sha256=sha(raw), plan='synthetic-plan.json',
                                       output=plan['paths']['output'], retrieval_window_seconds=0)
                with self.subTest(test=test, pre=plan['preflight'], tao=plan['tao']['argv']), \
                        mock.patch.object(job, 'read_regular', side_effect=[raw, AssertionError('injected read')]), \
                        mock.patch.object(job.subprocess, 'Popen', side_effect=AssertionError('child started')) as popen, \
                        mock.patch.object(job.os, 'makedirs', side_effect=AssertionError('folder created')) as mkdir, \
                        mock.patch.object(job.os, 'open', side_effect=AssertionError('file opened')) as opened, \
                        self.assertRaisesRegex(job.Refused, 'preflight|tao argv'):
                    job.run_job(args, None, {})
                popen.assert_not_called(); mkdir.assert_not_called(); opened.assert_not_called()

    def test_child_runner_refuses_other_commands_before_a_process(self):
        url = self.AV2_URLS[0]
        for argv in (['curl', url], ['python3', '-c', 'print(1)'], list(job.NVIDIA_SMI_ARGV) + [url],
                     ['grounding_dino', 'inference', '-e', 'relative/infer.yaml'], ('nvidia-smi',), []):
            with self.subTest(argv=argv), \
                    mock.patch.object(job.subprocess, 'Popen', side_effect=AssertionError('child started')) as popen, \
                    self.assertRaisesRegex(job.Refused, 'child command refused'):
                job.run_child(argv, 5)
            popen.assert_not_called()
        for argv in (list(job.NVIDIA_SMI_ARGV), ['python3', '-B', '-c', job.TORCH_CHECK],
                     ['grounding_dino', 'inference', '-e', '/workload/input/infer.yaml']):
            self.assertTrue(job.allowed_command(argv), argv)

    def test_download_refuses_av2_before_request_or_files(self):
        for url in self.AV2_URLS:
            item = {'url': url, 'path': '/synthetic/model', 'bytes': 3, 'sha256': sha(b'abc')}
            with self.subTest(url=url), \
                    mock.patch.object(job.os, 'makedirs', side_effect=AssertionError('folder created')) as mkdir, \
                    mock.patch.object(job.os, 'open', side_effect=AssertionError('file opened')) as opened, \
                    mock.patch.object(job.urllib.request, 'Request', side_effect=AssertionError('request')) as request, \
                    self.assertRaisesRegex(job.Refused, 'AV2'):
                job.download(item, time.monotonic() + 30)
            mkdir.assert_not_called(); opened.assert_not_called(); request.assert_not_called()

    def test_redirect_refuses_av2_before_constructing_the_next_request(self):
        request = job.urllib.request.Request('https://huggingface.co/model')
        for url in self.AV2_URLS:
            with self.subTest(url=url), \
                    mock.patch.object(job.urllib.request.HTTPRedirectHandler, 'redirect_request',
                                      side_effect=AssertionError('redirect followed')) as redirect, \
                    self.assertRaisesRegex(job.Refused, 'AV2'):
                job.HttpsOnly().redirect_request(request, None, 302, 'Found', {}, url)
            redirect.assert_not_called()
        redirected = job.HttpsOnly().redirect_request(request, None, 302, 'Found', {}, 'https://cdn.example/model')
        self.assertEqual(redirected.full_url, 'https://cdn.example/model')

    def test_final_av2_url_refused_before_body_read_or_files(self):
        item = {'url': 'https://huggingface.co/model', 'path': '/synthetic/model', 'bytes': 3, 'sha256': sha(b'abc')}
        for url in self.AV2_URLS:
            response = mock.MagicMock()
            response.__enter__.return_value = response
            response.geturl.return_value = url
            response.read.side_effect = AssertionError('body read')
            opener = mock.Mock(); opener.open.return_value = response
            with self.subTest(url=url), mock.patch.object(job.urllib.request, 'build_opener', return_value=opener), \
                    mock.patch.object(job.os, 'makedirs', side_effect=AssertionError('folder created')) as mkdir, \
                    mock.patch.object(job.os, 'open', side_effect=AssertionError('file opened')) as opened, \
                    self.assertRaisesRegex(job.Refused, 'AV2'):
                job.download(item, time.monotonic() + 30)
            opener.open.assert_called_once()
            self.assertEqual(opener.open.call_args.args[0].full_url, item['url'])
            response.read.assert_not_called(); mkdir.assert_not_called(); opened.assert_not_called()

    def test_file_download_still_checks_and_writes_the_pin(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            source = root / 'source'; source.write_bytes(b'abc')
            target = root / 'output' / 'model'
            item = {'url': source.as_uri(), 'path': str(target), 'bytes': 3, 'sha256': sha(b'abc')}
            result = job.download(item, time.monotonic() + 30)
            self.assertEqual(target.read_bytes(), b'abc')
            self.assertEqual(result['finalHost'], 'file')


class PlanSpecPrepare(unittest.TestCase):
    def real_plan(self):
        paths = prepare.real_paths(SID)
        spec = prepare.infer_yaml(EXPERIMENT.read_bytes(), paths)
        injected = [{'path': '/workload/input/run_nvdino_job.py', 'role': 'wrapper', 'bytes': 10, 'sha256': 'a' * 64},
                    {'path': '/workload/input/infer.yaml', 'role': 'spec', 'bytes': len(spec), 'sha256': sha(spec)}]
        injected += [{'path': '/workload/input/' + n, 'role': 'chunk', 'index': i, 'bytes': b, 'sha256': h}
                     for i, (n, b, h) in enumerate(prepare.CHUNKS)]
        return prepare.build_plan(SID, injected), spec.decode()

    def test_spec_is_the_corrected_skeptic_spec(self):
        plan, spec = self.real_plan()
        model = job.check_spec(spec, plan)
        self.assertEqual(model, {'pretrained_backbone_path': 'null', 'backbone': 'swin_tiny_224_1k',
                                 'num_feature_levels': '4', 'dec_layers': '6', 'enc_layers': '6',
                                 'num_queries': '900', 'dropout_ratio': '0.0', 'dim_feedforward': '2048',
                                 'log_scale': 'auto', 'class_embed_bias': 'True', 'text_encoder_type': '/workload/bert'})
        self.assertTrue(spec.startswith('results_dir: /workload/tao-results\nmodel:'))
        self.assertIn('\ninference: {num_gpus: 1, checkpoint: /workload/weights/grounding_dino_swin_tiny_commercial_'
                      'trainable.pth, conf_threshold: 0.25, color_map: {car: green, truck: blue, bus: yellow, '
                      'person: red}}\n', spec)
        self.assertIn('\ndataset: {infer_data_sources: {image_dir: [/workload/images], captions: [car, truck, bus, '
                      'person]}, batch_size: 1, workers: 1}\n', spec)
        self.assertNotIn('use_dn', spec)
        for old, new in (('log_scale: auto', 'log_scale: None'), ('class_embed_bias: True', 'class_embed_bias: False'),
                         ('pretrained_backbone_path: null', 'pretrained_backbone_path: /checkpoints/x.pth'),
                         ('  text_encoder_type: /workload/bert', '  text_encoder_type: bert-base-uncased'),
                         ('workers: 1}', 'workers: 1}\nmodel.use_dn: False')):
            with self.assertRaises(job.Refused, msg=new):
                job.check_spec(spec.replace(old, new), plan)

    def test_plan_rules(self):
        plan, _ = self.real_plan()
        self.assertEqual(plan['image'], prepare.IMAGE)
        self.assertGreaterEqual(plan['preflight']['minFreeBytes'], sum(d['bytes'] for d in plan['downloads']))
        self.assertTrue(all(d['url'].startswith('https://') for d in plan['downloads']))
        for change in (lambda p: p['downloads'][1].update(url='file:///x'),
                       lambda p: p['downloads'][1].update(url='http://huggingface.co/x'),
                       lambda p: p['downloads'][0].update(sha256='b' * 64),
                       lambda p: p['limits'].update(retries=1),
                       lambda p: p['tao']['env'].update(TELEMETRY_OPT_OUT='no'),
                       lambda p: p['downloadTimeoutSeconds'].update(weights=901),
                       lambda p: p.update(image='nvcr.io/nvidia/tao/tao-toolkit:7.2.0-pyt')):
            broken = json.loads(json.dumps(plan))
            change(broken)
            with self.assertRaises((job.Refused, KeyError, StopIteration)):
                job.validate_plan(broken, SID)

    def test_prepare_writes_once_and_checks_inject_sizes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            argv_file = root / 'old-argv.json'
            argv_file.write_text(json.dumps({'argv': ['--ssh-key', 'ssh-ed25519 ' + 'A' * 68]}))
            out = root / 'prepared'
            args = ['--output-dir', str(out), '--project-id', 'project-' + 'a' * 20, '--ssh-key-from-argv',
                    str(argv_file), '--chunks-dir', str(CHUNKS_DIR), '--experiment-yaml', str(EXPERIMENT),
                    '--license-dir', str(LICENSE_DIR), '--submission-id', SID]
            pins = WORKDIR / 'nvdino-pins.json'
            if pins.is_file():
                args += ['--pins', str(pins)]
            with redirect_stdout(StringIO()):
                self.assertEqual(prepare.main(args), 0)
            create = json.loads((out / 'create-argv.json').read_text())
            argv = create['argv']
            self.assertEqual(argv[argv.index('--container-command') + 1], prepare.ENTRYPOINT)
            self.assertEqual(argv[-1], '--async')
            for flag, value in (('--image', prepare.IMAGE), ('--platform', 'gpu-l40s-a'),
                                ('--preset', '1gpu-8vcpu-32gb'), ('--disk-size', '200Gi'), ('--timeout', '1h'),
                                ('--restart-policy', 'never'), ('--env', 'TELEMETRY_OPT_OUT=yes'),
                                ('--working-dir', '/workload')):
                self.assertEqual(argv[argv.index(flag) + 1], value, flag)
            self.assertIn('--preemptible', argv)
            self.assertIn('--follows-spot-price', argv)
            injects = [argv[i + 1] for i, a in enumerate(argv) if a == '--inject-file']
            self.assertEqual(len(injects), 8)
            for spec in injects:
                local = spec.split(':')[0]
                self.assertLessEqual(os.path.getsize(local), 65536)
            plan_raw = (out / 'plan.json').read_bytes()
            self.assertEqual(sha(plan_raw), create['planSha256'])
            self.assertIn('--plan-sha256 ' + create['planSha256'], create['args'])
            self.assertNotIn('"', create['args'])
            job.validate_plan(json.loads(plan_raw), SID, '/output/' + SID)
            with self.assertRaises(SystemExit):
                prepare.main(args)
            big = root / 'big'
            big.write_bytes(b'x' * 65537)
            with self.assertRaises(SystemExit):
                prepare.check_inject(big)
            self.assertEqual(json.loads(plan_raw)['retrieval']['windowSeconds'], 300)
            self.assertTrue(create['args'].endswith('--retrieval-window-seconds 300'))

    def prepare_args(self, root, project='project-' + 'a' * 20, *extra):
        argv_file = root / 'old-argv.json'
        argv_file.write_text(json.dumps({'argv': ['--ssh-key', 'ssh-ed25519 ' + 'A' * 68]}))
        return ['--output-dir', str(root / 'prepared'), '--project-id', project, '--ssh-key-from-argv', str(argv_file),
                '--chunks-dir', str(CHUNKS_DIR), '--experiment-yaml', str(EXPERIMENT), '--license-dir',
                str(LICENSE_DIR), '--submission-id', SID, *extra]

    def test_retrieval_window_zero_is_recorded_in_plan_and_argv(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with redirect_stdout(StringIO()):
                self.assertEqual(prepare.main(self.prepare_args(root, 'project-' + 'a' * 20,
                                                                '--retrieval-window', '0')), 0)
            out = root / 'prepared'
            plan = json.loads((out / 'plan.json').read_text())
            create = json.loads((out / 'create-argv.json').read_text())
            manifest = json.loads((out / 'prepared-manifest.json').read_text())
            self.assertEqual(plan['retrieval']['windowSeconds'], 0)
            self.assertEqual(plan['retrieval']['channels'], ['log-block'])
            self.assertEqual(plan['limits']['retrievalWindowSeconds'], 300)  # the wrapper's maximum, unchanged
            self.assertTrue(create['args'].endswith('--execute --retrieval-window-seconds 0'))
            self.assertEqual(create['args'], create['argv'][create['argv'].index('--args') + 1])
            self.assertEqual((create['retrievalWindowSeconds'], manifest['retrievalWindowSeconds']), (0, 0))
            job.validate_plan(plan, SID, '/output/' + SID)

    def test_retrieval_window_rejects_other_values(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(SystemExit), redirect_stderr(StringIO()):
                prepare.main(self.prepare_args(root, 'project-' + 'a' * 20, '--retrieval-window', '60'))
            self.assertFalse((root / 'prepared').exists())
        with self.assertRaises(SystemExit):
            prepare.run_args(SID, 'a' * 64, retrieval_window=60)

    @unittest.skipUnless(coord, 'private coordinator not present')
    def test_coordinator_reads_the_recorded_window(self):
        for window in ('0', '300'):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                with redirect_stdout(StringIO()):
                    prepare.main(self.prepare_args(root, coord.PROJECT, '--retrieval-window', window))
                prepared = coord.Prepared(root / 'prepared')
                self.assertEqual((prepared.window, prepared.window_recorded), (int(window), True))
                self.assertEqual(coord.required_checks(prepared.window)[-1], 'ssh' if window == '300' else 'logs-probe')


def utc_now():
    return datetime.now(timezone.utc)


ESTIMATE = {'hourly_cost': {'general': {'total': {'cost': '0.7492', 'cost_rounded': '0.75'}}}}
PAGE_OK = ('## Price changes\n'
           '* Starting October 8, 2026, preemptible pricing for NVIDIA B300, B200, H200 and H100 GPU platforms '
           'switches to [dynamic spot pricing](/signup-billing/pricing-policy).\n'
           '| Preemptible NVIDIA® L40S Intel. GPU | $0.65 | 1 GPU hour |\n')
PAGE_HTML = ('<ul><li>Starting October 8, preemptible pricing for NVIDIA H100 GPU platforms switches to '
             '<a href="/p">dynamic spot pricing</a>.</li></ul><table><tr><td>NVIDIA&reg; L40S Intel</td></tr></table>'
             '<script>var note = "L40S dynamic";</script>')
PAGE_GENERAL = PAGE_OK + '<Note>\n  Spot prices for preemptible GPU platforms are dynamic: they can change.\n</Note>\n'
PAGE_L40S = PAGE_OK.replace('H200 and H100', 'H200, H100 and L40S')


def quota_items(limits=None, pct='0.00', region='eu-north1', extra=()):
    limits = {'compute.instance.gpu.l40s': '8', 'compute.instance.preemptible.count': '2',
              'vpc.ipv4-address.public.count': '3', **(limits or {})}
    items = []
    for name, limit in limits.items():
        status = {'state': 'STATE_ACTIVE', 'usage_state': 'USAGE_STATE_NOT_USED'}
        if pct is not None:
            status['usage_percentage'] = pct
        items.append({'metadata': {'parent_id': 'tenant-x', 'name': name}, 'spec': {'limit': limit, 'region': region},
                      'status': status})
    return {'items': items + list(extra)}


SG = 'vpcsecuritygroup-test'


def rule(access, direction, protocol='ANY', priority=500, kind='STATELESS', rid='r', **block):
    return {'metadata': {'id': 'vpcsecurityrule-' + rid, 'parent_id': SG},
            'spec': {'access': access, direction: block, 'protocol': protocol, 'priority': priority, 'type': kind}}


DEFAULT_RULES = [rule('ALLOW', 'ingress', rid='in'), rule('ALLOW', 'egress', rid='out')]


@unittest.skipUnless(coord, 'private coordinator not present')
class Phase0Records(unittest.TestCase):
    """phase0-record validation: saved files in, record out; no CLI call is possible from these functions."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(os.path.realpath(temp.name))
        (self.base / 'sources').mkdir()
        self.observed = utc_now().isoformat()
        # Never read the live budget ledger: no ceiling exception unless a test supplies one.
        patcher = mock.patch.object(coord, 'ceiling_exception', return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def save(self, name, content):
        path = self.base / 'sources' / name
        path.write_text(content if isinstance(content, str) else json.dumps(content))
        return path

    def record(self, check, files, scalars=None, observed=None):
        return coord.make_record(check, files, scalars or {}, observed or self.observed, self.base)

    def price(self, page=PAGE_OK, estimate=ESTIMATE, url=None):
        return self.record('price', {'estimate': self.save('est.json', estimate),
                                     'pricingPage': self.save('page.md', page)},
                           {'pricingUrl': url or coord.PRICING_URL})

    def test_price(self):
        good = self.price()
        self.assertTrue(good['passed'], good['reasons'])
        self.assertEqual(good['facts']['hourlyUsd'], '0.7492')
        self.assertEqual(good['inputs']['estimate']['path'], 'sources/est.json')
        self.assertTrue(self.price(PAGE_HTML)['passed'])
        self.assertTrue(self.price(estimate={'argv': ['x'], 'exit_code': 0, 'response': ESTIMATE})['passed'])
        cases = {'0.80': self.price(estimate={'hourly_cost': {'general': {'total': {'cost': '0.80'}}}}),
                 'general note': self.price(PAGE_GENERAL), 'l40s dynamic': self.price(PAGE_L40S),
                 'no dynamic line': self.price('| L40S | $0.65 |\n'), 'bad url': self.price(url='https://x.test/p'),
                 'unreadable estimate': self.price(estimate='not json')}
        for name, result in cases.items():
            self.assertFalse(result['passed'], name)
        self.assertIn('names no platform', ' '.join(cases['general note']['reasons']))
        self.assertTrue(cases['l40s dynamic']['facts']['l40sDynamic'])
        # Only an active recorded exception (fake here) accepts the dynamic-spot risk; other failures still fail.
        with mock.patch.object(coord, 'ceiling_exception', return_value={'approvedBy': 'Sean, test fixture'}):
            accepted = self.price(PAGE_GENERAL)
            still_failed = self.price(url='https://x.test/p')
        self.assertTrue(accepted['passed'], accepted['reasons'])
        self.assertEqual(accepted['facts']['dynamicPricingAcceptedBy'], 'Sean, test fixture')
        self.assertIn('names no platform', ' '.join(accepted['facts']['dynamicPricingRiskAccepted']))
        self.assertFalse(still_failed['passed'])

    def test_ceiling_exception_status(self):
        """A temporary ledger only: an exception whose status is not active (e.g. 'unused') is ignored."""
        ledger = self.base / 'cloud-budget.json'
        exc = {'approvedBy': 'Sean, test fixture', 'reservationPlanUsd': coord.RESERVATION_USD, 'noRetry': True}
        cases = {'no status': (exc, True), 'active': ({**exc, 'status': 'active'}, True),
                 'unused': ({**exc, 'status': 'unused: declined; exception no longer applies'}, False),
                 'retry allowed': ({**exc, 'noRetry': False}, False),
                 'not Sean': ({**exc, 'approvedBy': 'someone'}, False),
                 'small plan': ({**exc, 'reservationPlanUsd': 0.5}, False)}
        for name, (entry, active) in cases.items():
            ledger.write_text(json.dumps({'nvdino_job_ceiling_exception': entry}))
            with mock.patch.object(coord, 'LEDGER', ledger):
                result = REAL_CEILING_EXCEPTION()
            self.assertEqual(result is not None, active, name)
        ledger.write_text('not json')
        with mock.patch.object(coord, 'LEDGER', ledger):
            self.assertIsNone(REAL_CEILING_EXCEPTION())

    def test_quotas(self):
        run = lambda doc: self.record('quotas', {'quotaList': self.save('q.json', doc)})  # noqa: E731
        good = run(quota_items())
        self.assertTrue(good['passed'], good['reasons'])
        self.assertEqual(good['facts']['compute.instance.preemptible.count']['available'], 2)
        self.assertTrue(run({'response': quota_items(pct=None)})['passed'])
        dup = quota_items()['items'][0]
        for doc in (quota_items({'compute.instance.preemptible.count': '0'}), quota_items(pct='100.00'),
                    quota_items(region='eu-west1'), quota_items(extra=[dup]),
                    {'items': [dict(i, status={'state': 'STATE_ACTIVE', 'usage_state': 'USAGE_STATE_USED'})
                               for i in quota_items()['items']]}):
            self.assertFalse(run(doc)['passed'], doc)
        half = quota_items({'compute.instance.preemptible.count': '2'}, pct='50.00')
        self.assertEqual(run(half)['facts']['compute.instance.preemptible.count']['available'], 1)

    def ssh(self, rules, groups=None):
        groups = groups or {'items': [{'metadata': {'id': SG, 'parent_id': 'project-x'}, 'spec': {}}]}
        return self.record('ssh', {'securityGroups': self.save('sg.json', groups),
                                   'securityRules': self.save('rules.json', {'items': rules})})

    def test_ssh(self):
        good = self.ssh(DEFAULT_RULES)
        self.assertTrue(good['passed'], good['reasons'])
        self.assertEqual((good['facts']['ingress']['decision'], good['facts']['egress']['decision']), ('allow', 'allow'))
        stateful = self.ssh([rule('ALLOW', 'ingress', 'TCP', kind='STATEFUL', destination_ports=[22],
                                  source_cidrs=['0.0.0.0/0'])])
        self.assertTrue(stateful['passed'], stateful['reasons'])
        self.assertIn('stateful', stateful['facts']['egress'])
        # Priority 0 means 500, so an allow at 0 still loses to a deny at 400 but wins against one at 600.
        self.assertTrue(self.ssh([rule('ALLOW', 'ingress', priority=0, rid='a'),
                                  rule('DENY', 'ingress', 'TCP', 600, rid='d', destination_ports=[22]),
                                  rule('ALLOW', 'egress', rid='o')])['passed'])
        failing = {
            'deny 22 first': DEFAULT_RULES + [rule('DENY', 'ingress', 'TCP', 100, rid='d', destination_ports=[22])],
            'deny maybe': DEFAULT_RULES + [rule('DENY', 'ingress', 'TCP', 100, rid='d', source_cidrs=['192.0.2.0/24'])],
            'tie deny wins': [rule('ALLOW', 'ingress', rid='a'), rule('DENY', 'ingress', rid='d'),
                              rule('ALLOW', 'egress', rid='o')],
            'allow 0 loses to 400': [rule('ALLOW', 'ingress', priority=0, rid='a'),
                                     rule('DENY', 'ingress', 'TCP', 400, rid='d', destination_ports=[22]),
                                     rule('ALLOW', 'egress', rid='o')],
            'stateless no egress': [rule('ALLOW', 'ingress', 'TCP', destination_ports=[22])],
            'specific source only': [rule('ALLOW', 'ingress', 'TCP', kind='STATEFUL', destination_ports=[22],
                                          source_cidrs=['203.0.113.0/24'])],
            'other port': [rule('ALLOW', 'ingress', 'TCP', kind='STATEFUL', destination_ports=[443])],
            'udp only': [rule('ALLOW', 'ingress', 'UDP', kind='STATEFUL')],
            'empty group': [],
        }
        for name, rules in failing.items():
            self.assertFalse(self.ssh(rules)['passed'], name)
        two = {'items': [{'metadata': {'id': SG}}, {'metadata': {'id': SG + '2'}}]}
        self.assertFalse(self.ssh(DEFAULT_RULES, two)['passed'])
        foreign = [dict(r, metadata={'id': 'x', 'parent_id': 'vpcsecuritygroup-other'}) for r in DEFAULT_RULES]
        self.assertFalse(self.ssh(foreign)['passed'])

    def probe(self, logs, reference, job_id='aijob-e00selftestprobe0'):
        return self.record('logs-probe', {'logs': self.save('logs.txt', logs), 'reference': self.save('ref.txt', reference)},
                           {'jobId': job_id})

    def test_logs_probe(self):
        lines = ['probe line %04d payload' % i for i in range(650)]
        reference = '\n'.join(lines) + '\n'
        per_line = '\n'.join(json.dumps({'timestamp': '2026-10-12T00:00:00Z', 'message': line}) for line in lines)
        one_doc = json.dumps({'items': [{'message': line} for line in lines]})
        prefixed = '\n'.join('2026-10-12T00:00:00Z stdout F ' + line for line in lines)
        for logs in (per_line, one_doc, prefixed):
            result = self.probe(logs, reference)
            self.assertTrue(result['passed'], result['reasons'])
            self.assertEqual(result['facts']['matchedInOrder'], 650)
        cut = '\n'.join(line for i, line in enumerate(lines) if i != 400)
        self.assertIn('not whole', ' '.join(self.probe(cut, reference)['reasons']))
        short = self.probe(prefixed, '\n'.join(lines[:100]))
        self.assertFalse(short['passed'])
        self.assertIn('>= 600', ' '.join(short['reasons']))
        self.assertFalse(self.probe(prefixed, reference, 'job-1')['passed'])
        reordered = '\n'.join(lines[1:] + lines[:1])
        self.assertFalse(self.probe(reordered, reference)['passed'])

    def balance(self, **fields):
        doc = {'observedUtc': self.observed, 'source': 'Nebius Billing page (read-only)', 'balanceUsd': 24.99,
               'billingSettingsChanged': False, **fields}
        return coord.make_record('balance', {'balance': self.save('balance.json', doc)}, {}, None, self.base)

    def test_balance(self):
        good = self.balance()
        self.assertTrue(good['passed'], good['reasons'])
        self.assertEqual(good['observedDate'], utc_now().date().isoformat())
        for fields in ({'balanceUsd': 0.5}, {'balanceUsd': True}, {'balanceUsd': '24.99'}, {'source': ''},
                       {'billingSettingsChanged': None}):
            self.assertFalse(self.balance(**fields)['passed'], fields)
        with self.assertRaises(coord.Phase0Error):
            self.balance(observedUtc='yesterday')
        with self.assertRaises(coord.Phase0Error):
            self.balance(observedUtc='2026-10-12T00:00:00+02:00')

    def test_observed_time_and_source_location(self):
        later = (utc_now() + timedelta(minutes=10)).isoformat()
        result = self.record('quotas', {'quotaList': self.save('q.json', quota_items())}, observed=later)
        self.assertIn('observed time is later than the saved file', ' '.join(result['reasons']))
        with self.assertRaises(coord.Phase0Error):
            self.record('quotas', {'quotaList': self.save('q.json', quota_items())}, observed='2026-10-12 noon')
        with self.assertRaises(coord.Phase0Error):
            coord.make_record('quotas', {'quotaList': self.save('q.json', quota_items())}, {}, None, self.base)
        with tempfile.TemporaryDirectory() as other:
            outside = Path(other) / 'q.json'
            outside.write_text(json.dumps(quota_items()))
            with self.assertRaises(coord.Phase0Error):
                self.record('quotas', {'quotaList': outside})
            link = self.base / 'sources' / 'link.json'
            link.symlink_to(outside)
            with self.assertRaises(coord.Phase0Error):
                self.record('quotas', {'quotaList': link})

    def write_all(self):
        page = PAGE_OK
        records = [self.price(page), self.record('quotas', {'quotaList': self.save('q.json', quota_items())}),
                   self.ssh(DEFAULT_RULES), self.balance(),
                   self.probe('\n'.join('l%04d' % i for i in range(600)), '\n'.join('l%04d' % i for i in range(600)))]
        for record in records:
            self.assertTrue(record['passed'], (record['check'], record['reasons']))
            coord.write_record(record, self.base)

    def test_run_guard(self):
        self.write_all()
        self.assertEqual(coord.phase0_problems(300, self.base), [])
        self.assertEqual(coord.phase0_problems(0, self.base), [])
        self.assertEqual(len((self.base / 'records.jsonl').read_text().splitlines()), 5)
        stale = coord.phase0_problems(300, self.base, today='2099-01-01')
        self.assertEqual(len(stale), 4)
        self.assertTrue(all('not today' in p for p in stale))
        os.remove(self.base / 'record-ssh.json')
        self.assertIn('ssh: missing (record-ssh.json)', coord.phase0_problems(300, self.base))
        self.assertEqual(coord.phase0_problems(0, self.base), [])  # window 0 needs the logs probe, not ssh
        # A changed source or an edited record is refused; so is a record that failed.
        (self.base / 'sources/est.json').write_text(json.dumps(ESTIMATE) + ' ')
        self.assertTrue(any('does not match' in p for p in coord.phase0_problems(0, self.base)))
        self.write_all()
        path = self.base / 'record-quotas.json'
        doc = json.loads(path.read_text())
        doc['facts']['compute.instance.preemptible.count']['available'] = 5
        path.write_text(json.dumps(doc))
        self.assertTrue(any(p.startswith('quotas: record does not match') for p in coord.phase0_problems(300, self.base)))
        coord.write_record(self.price(PAGE_GENERAL), self.base)
        self.assertTrue(any(p.startswith('price: failed') for p in coord.phase0_problems(300, self.base)))


@unittest.skipUnless(coord, 'private coordinator not present')
class CancelRetrieveFinish(unittest.TestCase):
    """Cancel cap and retries, SSH retrieval retries and the explicit logs --tail, with the CLI faked."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)
        self.events, self.sleeps, self.calls = [], [], []
        self.log = lambda event, **fields: self.events.append((event, fields))
        for name, value in (('_sleep', self.sleeps.append), ('job_get', lambda job_id: {'status': {'state': 'RUNNING'}})):
            patcher = mock.patch.object(coord, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def fake_cli(self, codes, outputs=None):
        def cli(*args, timeout=120):
            self.calls.append(args)
            if args[:3] == ('ai', 'job', 'cancel'):
                code = codes.pop(0)
                return code, b'{}' if code == 0 else b'', b'' if code == 0 else b'transient'
            return 0, (outputs or {}).get(args[2], b'[]'), b''
        return mock.patch.object(coord, 'cli', cli)

    def cancel(self, submitted=None, mode='run'):
        return coord.request_cancel('aijob-x', 'test', self.log, mode, self.dir, submitted or utc_now())

    def test_cancel_retries_then_records_success_separately(self):
        with self.fake_cli([1, 0]):
            self.assertEqual(self.cancel(), 'succeeded')
            self.assertEqual(self.cancel(), 'succeeded')  # a later caller sees the success and makes no call
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(json.loads((self.dir / 'cancel-succeeded.json').read_text())['attempt'], 2)
        self.assertTrue((self.dir / 'cancel-requested.json').exists())
        self.assertFalse((self.dir / 'cancel-attempt-3.json').exists())
        self.assertEqual(len(self.sleeps), 1)
        self.assertGreater(self.sleeps[0], 10)

    def test_cancel_cap_is_shared_and_bounded(self):
        with self.fake_cli([1, 1, 1]):
            self.assertEqual(self.cancel(), 'exhausted')
            self.assertEqual(self.cancel(), 'exhausted')
        self.assertEqual(len(self.calls), coord.CANCEL_ATTEMPTS)
        self.assertFalse((self.dir / 'cancel-succeeded.json').exists())

    def test_cancel_retry_stops_after_provider_timeout_or_terminal_job(self):
        with self.fake_cli([1]):
            old = utc_now() - timedelta(seconds=coord.PROVIDER_TIMEOUT + 1)
            self.assertEqual(self.cancel(submitted=old), 'provider-timeout')
        self.assertEqual(len(self.calls), 1)
        for name in os.listdir(self.dir):
            os.remove(self.dir / name)
        self.calls.clear()
        with self.fake_cli([1]), mock.patch.object(coord, 'job_get', lambda job_id: {'status': {'state': 'COMPLETED'}}):
            self.assertEqual(self.cancel(), 'terminal')
        self.assertEqual(len(self.calls), 1)

    def test_cancel_selftest_makes_no_call(self):
        with self.fake_cli([]):
            self.assertEqual(self.cancel(mode='selftest'), 'selftest')
        self.assertEqual(self.calls, [])
        self.assertTrue((self.dir / 'cancel-requested.json').exists())

    def ready(self, outcomes, deadline=None):
        tried = []

        def retrieve(remote, ctx, log, stage):
            tried.append(stage.name)
            (stage / 'remote-listing.json').write_text('[]')
            outcome = outcomes.pop(0)
            if outcome == 'transport':
                raise RuntimeError('remote tar failed')
            if outcome == 'refuse':
                raise coord.Refusal('unexpected remote layout')
            (stage / 'retrieved.tar').write_bytes(b'tar')
            (stage / 'result-bundle').mkdir()
            (stage / 'result-bundle/boxes.json').write_text('{}')
            return stage / 'result-bundle'
        ctx = {'sid': SID, 'dir': self.dir}
        with mock.patch.object(coord, 'retrieve', retrieve), \
                mock.patch.object(coord, 'consumer', lambda ctx, **kw: (0, 'checked', '')), \
                mock.patch.object(coord, 'acknowledge', lambda remote, sid: True):
            return coord.handle_ready(mock.Mock(container='cid'), ctx, self.log, deadline), tried

    def test_retrieval_retries_transport_failures(self):
        ok, tried = self.ready(['transport', 'ok'])
        self.assertTrue(ok)
        self.assertEqual(tried, ['retrieval-attempt-1', 'retrieval-attempt-2'])
        self.assertTrue((self.dir / 'result-bundle/boxes.json').is_file())
        self.assertTrue((self.dir / 'retrieved.tar').is_file() and (self.dir / 'remote-listing.json').is_file())
        self.assertTrue((self.dir / 'retrieval-attempt-1/remote-listing.json').is_file())  # failed attempt kept
        self.assertFalse((self.dir / 'retrieval-attempt-2').exists())
        self.assertEqual(self.sleeps, [coord.RETRIEVE_BACKOFF])

    def test_retrieval_limits(self):
        ok, tried = self.ready(['transport'] * 3)
        self.assertEqual((ok, len(tried)), (False, coord.RETRIEVE_ATTEMPTS))
        self.assertFalse((self.dir / 'result-bundle').exists())
        for name in os.listdir(self.dir):
            shutil.rmtree(self.dir / name)
        ok, tried = self.ready(['refuse', 'ok'])
        self.assertEqual((ok, tried), (False, ['retrieval-attempt-1']))
        shutil.rmtree(self.dir / 'retrieval-attempt-1')
        ok, tried = self.ready(['transport', 'ok'], deadline=time.monotonic())
        self.assertEqual((ok, tried), (False, ['retrieval-attempt-1']))

    def test_finish_reads_logs_with_explicit_tail_and_counts_lines(self):
        ctx = {'sid': SID, 'dir': self.dir, 'plan': self.dir / 'plan.json', 'plan_sha': 'a' * 64}
        with self.fake_cli([], {'logs': b'one\ntwo\nthree\n'}), \
                mock.patch.object(coord, 'job_get', lambda job_id: {'status': {'state': 'COMPLETED', 'instances': []}}):
            coord.finish('aijob-x', ctx, self.log)
        logs_call = next(c for c in self.calls if c[:3] == ('ai', 'job', 'logs'))
        self.assertEqual(logs_call[3:], ('aijob-x', '--since', '3h', '--tail', '100000'))
        event = next(f for e, f in self.events if e == 'job-logs')
        self.assertEqual((event['lines'], event['tail']), (3, 100000))
        self.assertEqual((self.dir / 'job-logs.txt').read_bytes(), b'one\ntwo\nthree\n')


if __name__ == '__main__':
    unittest.main()
