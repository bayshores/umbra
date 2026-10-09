"""Input admission regressions, plus one optional real-data preparation replay.

Run: python3 -B -m unittest discover -s tests -p 'test_scene_preparation.py' -v
Analytic matrices test supported input form only, never metric accuracy.
"""
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from rectified_calibration import load_rectified_pair

SCRATCH = (ROOT / '.local/evaluation/preparation-regressions').resolve()
FRAME = '2018-10-19-09-30-39_2018-10-19-09-40-40-716'
PEER_FRAME = '2018-10-19-09-30-39_2018-10-19-09-31-02-413'
SEQUENCE = FRAME.split('_')[0]
DATA = ROOT / '.local/data/drivingstereo'
REFERENCE = ROOT / 'public/scenes' / FRAME
SOURCE_PATHS = {
    'left': Path('extracted/left-image-half-size') / (FRAME + '.jpg'),
    'right': Path('extracted/right-image-half-size') / (FRAME + '.jpg'),
    'depth': Path('extracted/depth-map-half-size') / (FRAME + '.png'),
    'calibration': Path('calibration') / (SEQUENCE + '-full-reference.txt'),
    'license': Path('LICENSE'),
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json(path):
    def reject_constant(value):
        raise ValueError(f'Nonstandard JSON constant: {value}')
    return json.loads(path.read_text(), parse_constant=reject_constant)


def directory_inventory(output):
    return {'files': {str(p.relative_to(output)): sha256(p) for p in output.rglob('*') if p.is_file()},
            'directories': sorted(str(p.relative_to(output)) for p in output.rglob('*') if p.is_dir())}


def fixture_fields():
    return {
        'S_rect_101': [1758, 800], 'S_rect_103': [1758, 800],
        'P_rect_101': [2002.991, 0, 905.3329, 0, 0, 2002.991, 388.5141, 0, 0, 0, 1, 0],
        'P_rect_103': [2002.991, 0, 895.3329, -1091.469, 0, 2002.991, 388.5141, 0, 0, 0, 1, 0],
    }


def calibration_text(fields):
    return ''.join(f'{key}: {" ".join(map(str, values))}\n' for key, values in fields.items())


def prepare_command(data, output, overrides=()):
    return [str(ROOT / 'scripts/prepare-scene.py'), '--data-root', str(data),
            '--frame', FRAME, '--output', str(output), '--road-tolerance', '0.08',
            '--max-depth', '80', '--mesh-edge-limit', '0.4', *overrides]


def run_preparation(data, output, overrides=(), reject_before_geometry=False):
    command = prepare_command(data, output, overrides)
    if reject_before_geometry:
        # Trap the first reconstruction allocation without replacing input validation.
        wrapper = ('import numpy as np, runpy, sys\n'
                   'def forbidden(*a, **k):\n'
                   '    raise RuntimeError("REGRESSION_REACHED_GEOMETRY")\n'
                   'np.indices = forbidden\n'
                   'sys.argv = sys.argv[1:]\n'
                   f'sys.path.insert(0, {str(ROOT / "scripts")!r})\n'
                   'runpy.run_path(sys.argv[0], run_name="__main__")\n')
        command = [sys.executable, '-B', '-c', wrapper, *command]
    else:
        command = [sys.executable, '-B', *command]
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                          timeout=60, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})


class RectifiedCalibrationTests(unittest.TestCase):
    def setUp(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='analytic-', dir=SCRATCH)
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / 'calibration.txt'

    def assert_rejected(self, fields, suffix=''):
        self.path.write_text(calibration_text(fields) + suffix)
        with self.assertRaises(ValueError):
            load_rectified_pair(self.path, 879, 400)

    def test_malformed_source_sizes(self):
        for key in ('S_rect_101', 'S_rect_103'):
            for values in ([1758], [np.nan, 800], [1758, np.inf], [1759, 800]):
                with self.subTest(key=key, values=values):
                    fields = fixture_fields()
                    fields[key] = values
                    self.assert_rejected(fields)
        # Matching invalid sizes must still fail; mismatch alone cannot mask this guard.
        for values in ([0, 800], [-1758, 800], [1758, 0], [1758.5, 800]):
            with self.subTest(matching_invalid_sizes=values):
                fields = fixture_fields()
                fields['S_rect_101'] = fields['S_rect_103'] = values
                self.assert_rejected(fields)

    def test_missing_duplicate_and_malformed_fields(self):
        for key in fixture_fields():
            with self.subTest(missing=key):
                fields = fixture_fields()
                del fields[key]
                self.assert_rejected(fields)
            with self.subTest(duplicate=key):
                fields = fixture_fields()
                self.assert_rejected(fields, calibration_text({key: fields[key]}))
        for values in ([1], [np.nan] * 12, [np.inf] * 12, ['invalid'] * 12):
            with self.subTest(projection=values):
                fields = fixture_fields()
                fields['P_rect_101'] = values
                self.assert_rejected(fields)

    def test_unsupported_projection_forms(self):
        # Skew, vertical translation, cross-axis term, and every noncanonical row term.
        for key in ('P_rect_101', 'P_rect_103'):
            for index, value in ((1, .5), (4, .5), (7, .5), (8, .5),
                                 (9, .5), (10, 2), (11, .5), (0, -1), (5, -1)):
                with self.subTest(key=key, index=index):
                    fields = fixture_fields()
                    fields[key][index] = value
                    self.assert_rejected(fields)
        for index, value in ((3, 1), (3, np.nan), (2, np.inf)):
            with self.subTest(left_index=index, value=value):
                fields = fixture_fields()
                fields['P_rect_101'][index] = value
                self.assert_rejected(fields)

    def test_pair_mismatch_and_baseline(self):
        for index, value in ((0, 2003), (5, 2003), (6, 389), (3, 0), (3, 1091.469)):
            with self.subTest(index=index, value=value):
                fields = fixture_fields()
                fields['P_rect_103'][index] = value
                self.assert_rejected(fields)

    def test_invalid_output_dimensions(self):
        self.path.write_text(calibration_text(fixture_fields()))
        for dimensions in ((0, 400), (-1, 400), (879, np.nan), (True, 400), (879.5, 400)):
            with self.subTest(dimensions=dimensions), self.assertRaises(ValueError):
                load_rectified_pair(self.path, *dimensions)

    def test_valid_projection_retains_exact_resize_parameters(self):
        fields = fixture_fields()
        self.path.write_text(calibration_text(fields))
        for width, height in ((879, 400), (1758, 800), (586, 200)):
            with self.subTest(dimensions=(width, height)):
                left, right, size, scale = load_rectified_pair(self.path, width, height)
                np.testing.assert_array_equal(size, [1758, 800])
                np.testing.assert_array_equal(scale, [width / 1758, height / 800])
                for key, actual in (('P_rect_101', left), ('P_rect_103', right)):
                    expected = np.array(fields[key]).reshape(3, 4)
                    expected[:2] *= scale[:, None]
                    np.testing.assert_array_equal(actual, expected)


class PreparationControlTests(unittest.TestCase):
    def test_nonfinite_controls_reject_before_source_read_or_geometry(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='controls-', dir=SCRATCH) as directory:
            directory = Path(directory)
            for option in ('--road-tolerance', '--max-depth', '--mesh-edge-limit'):
                for value in ('inf', 'nan', '0', '-1'):
                    with self.subTest(option=option, value=value):
                        output = directory / 'never-published'
                        result = run_preparation(directory / 'absent-source', output,
                                                 (option, value), True)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn('Processing tolerances must be finite and positive', result.stderr)
                        self.assertNotIn('REGRESSION_REACHED_GEOMETRY', result.stderr)
                        self.assertFalse(output.exists())


class RealPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = [str(DATA / p) for p in SOURCE_PATHS.values() if not (DATA / p).is_file()]
        missing += [str(REFERENCE / 'scene.json')] if not (REFERENCE / 'scene.json').is_file() else []
        missing += [str(REFERENCE / name) for name in
                    ('positions.bin', 'colors.bin', 'triangles.bin', 'depth.bin')
                    if not (REFERENCE / name).is_file()]
        if missing:
            raise unittest.SkipTest('Optional raw acquisition/reference unavailable: ' + ', '.join(missing))
        SCRATCH.mkdir(parents=True, exist_ok=True)
        cls.run_root = Path(tempfile.mkdtemp(prefix='real-', dir=SCRATCH))
        cls.original_hashes = {key: sha256(DATA / path) for key, path in SOURCE_PATHS.items()}
        cls.peer_reference = cls.run_root / 'prepared-peer' / PEER_FRAME
        peer_paths = [DATA / SOURCE_PATHS[key].with_name(PEER_FRAME + SOURCE_PATHS[key].suffix)
                      for key in ('left', 'right', 'depth')]
        cls.peer_available = all(path.is_file() for path in peer_paths)
        if cls.peer_available:
            command = [sys.executable, '-B', *prepare_command(DATA, cls.peer_reference.parent,
                                                           ('--frame', PEER_FRAME))]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
            (cls.run_root / 'peer-stdout.txt').write_text(result.stdout)
            (cls.run_root / 'peer-stderr.txt').write_text(result.stderr)
            if result.returncode:
                raise AssertionError('Genuine raw peer preparation failed: ' + result.stderr)

    @classmethod
    def tearDownClass(cls):
        for key, path in SOURCE_PATHS.items():
            if sha256(DATA / path) != cls.original_hashes[key]:
                raise AssertionError(f'Original source changed: {key}')

    def isolated_source(self, name, license=True):
        target = self.run_root / name / 'source'
        for key, relative in SOURCE_PATHS.items():
            if key == 'license' and not license:
                continue
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if key == 'calibration':
                path.write_bytes((DATA / relative).read_bytes())
            else:
                # These source links are read-only by use; tests never open them for writes.
                path.symlink_to(DATA / relative)
        return target

    def assert_early_rejection(self, source, output, message):
        result = run_preparation(source, output, reject_before_geometry=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(message, result.stderr)
        self.assertNotIn('REGRESSION_REACHED_GEOMETRY', result.stderr)
        self.assertFalse(output.exists())

    def test_unsupported_projection_rejects_before_geometry_and_publication(self):
        source = self.isolated_source('unsupported-projection')
        calibration = source / SOURCE_PATHS['calibration']
        lines = calibration.read_text().splitlines()
        for index, line in enumerate(lines):
            if line.startswith('P_rect_101:'):
                values = line.split(':', 1)[1].split()
                values[1] = '0.5'
                lines[index] = 'P_rect_101: ' + ' '.join(values)
        calibration.write_text('\n'.join(lines) + '\n')
        self.assert_early_rejection(source, source.parent / 'output', 'noncanonical')

    def test_missing_source_license_rejects_before_geometry_and_publication(self):
        source = self.isolated_source('missing-license', license=False)
        self.assert_early_rejection(source, source.parent / 'output', 'Required source artifact unavailable:')

    def test_existing_destination_is_retained(self):
        source = self.isolated_source('existing-destination')
        output = source.parent / 'output'
        scene = output / FRAME
        scene.mkdir(parents=True)
        (scene / 'scene.json').write_text('{"retained": true}\n')
        (scene / 'positions.bin').write_bytes(b'original destination bytes')
        (output / 'index.json').write_text('[{"retained": true}]\n')
        (output / 'LICENSE').write_text('existing collection license\n')
        before = {str(path.relative_to(output)): sha256(path) for path in output.rglob('*') if path.is_file()}
        result = run_preparation(source, output, reject_before_geometry=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Existing scene retained', result.stderr)
        self.assertNotIn('REGRESSION_REACHED_GEOMETRY', result.stderr)
        after = {str(path.relative_to(output)): sha256(path) for path in output.rglob('*') if path.is_file()}
        self.assertEqual(after, before)

    def test_corrupt_peer_rejects_before_geometry_and_publication(self):
        source = self.isolated_source('corrupt-peer')
        output = source.parent / 'output'
        peer = output / 'corrupt-peer'
        peer.mkdir(parents=True)
        (peer / 'scene.json').write_text('{')
        (output / 'index.json').write_text('[]\n')
        before = {str(p.relative_to(output)): sha256(p) for p in output.rglob('*') if p.is_file()}
        result = run_preparation(source, output, reject_before_geometry=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Invalid existing scene catalog metadata', result.stderr)
        self.assertNotIn('REGRESSION_REACHED_GEOMETRY', result.stderr)
        after = {str(p.relative_to(output)): sha256(p) for p in output.rglob('*') if p.is_file()}
        self.assertEqual(after, before)
        self.assertFalse((output / FRAME).exists())
        self.assertFalse((output / 'LICENSE').exists())

    def test_default_real_preparation_preserves_geometry_and_provenance(self):
        source = self.isolated_source('default-preparation')
        output = source.parent / 'output'
        result = run_preparation(source, output)
        self.assertEqual(result.returncode, 0, result.stderr)
        scene = output / FRAME
        metadata = strict_json(scene / 'scene.json')
        reference = strict_json(REFERENCE / 'scene.json')
        self.assertEqual((metadata['pointCount'], metadata['triangleCount']), (72707, 11990))
        for key, name in metadata['buffers'].items():
            with self.subTest(buffer=key):
                self.assertEqual((scene / name).read_bytes(), (REFERENCE / name).read_bytes())
                self.assertEqual(metadata['artifactHashes'][key], sha256(scene / name))
                self.assertEqual(metadata['artifactHashes'][key], reference['artifactHashes'][key])
        for key in ('camera', 'transform', 'ground', 'bounds', 'processing', 'coverage'):
            self.assertEqual(metadata[key], reference[key], key)
        self.assertEqual(metadata['sourceHashes'], self.original_hashes)
        self.assertEqual(metadata['processingCodeHashes'], {
            'preparation': sha256(ROOT / 'scripts/prepare-scene.py'),
            'calibration': sha256(ROOT / 'scripts/rectified_calibration.py'),
            'surface': sha256(ROOT / 'scripts/scene_surface.py'),
        })
        for license_path in (scene / 'LICENSE', output / 'LICENSE'):
            self.assertEqual(license_path.read_bytes(), (DATA / 'LICENSE').read_bytes())
        self.assertEqual(strict_json(output / 'index.json'), [{'id': FRAME, 'label': metadata['label']}])
        self.assertFalse(list(output.glob('.preparing-*')))

    def test_processing_uses_one_source_snapshot_despite_midrun_file_changes(self):
        source = self.run_root / 'source-snapshot' / 'source'
        original = {key: (DATA / relative).read_bytes() for key, relative in SOURCE_PATHS.items()}
        for key, relative in SOURCE_PATHS.items():
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(original[key])
        output = source.parent / 'output'
        paths = {key: str(source / relative) for key, relative in SOURCE_PATHS.items()}
        # Real decoding/calibration has finished at np.indices. Change only isolated copies.
        wrapper = f'''import numpy as np, runpy, sys
from pathlib import Path
from PIL import Image
paths = {paths!r}
original_indices = np.indices
def changed_indices(*args, **kwargs):
    for key, name in paths.items():
        path = Path(name)
        if key == 'calibration':
            lines = path.read_text().splitlines()
            for index, line in enumerate(lines):
                if line.startswith(('P_rect_101:', 'P_rect_103:')):
                    key_name, text = line.split(':', 1)
                    values = text.split()
                    values[0] = str(float(values[0]) * 1.1)
                    lines[index] = key_name + ': ' + ' '.join(values)
            path.write_text('\\n'.join(lines) + '\\n')
        elif key in ('left', 'right'):
            with Image.open(path) as image:
                pixels = np.array(image.convert('RGB'))
            pixels[:, :, 0] ^= 255
            Image.fromarray(pixels).save(path, format='JPEG', quality=90)
        elif key == 'depth':
            with Image.open(path) as image:
                pixels = np.array(image)
            pixels[pixels > 0] //= 2
            Image.fromarray(pixels).save(path, format='PNG')
        else:
            path.write_bytes(path.read_bytes() + b'\\nSOURCE_SNAPSHOT_TEST\\n')
    print('SOURCE_SNAPSHOT_MUTATION_APPLIED', flush=True)
    np.indices = original_indices
    return original_indices(*args, **kwargs)
np.indices = changed_indices
sys.argv = sys.argv[1:]
sys.path.insert(0, {str(ROOT / 'scripts')!r})
runpy.run_path(sys.argv[0], run_name='__main__')
'''
        result = subprocess.run([sys.executable, '-B', '-c', wrapper, *prepare_command(source, output)],
                                cwd=ROOT, capture_output=True, text=True, timeout=30)
        (source.parent / 'mutation-stdout.txt').write_text(result.stdout)
        (source.parent / 'mutation-stderr.txt').write_text(result.stderr)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('SOURCE_SNAPSHOT_MUTATION_APPLIED', result.stdout)
        for key, relative in SOURCE_PATHS.items():
            self.assertNotEqual((source / relative).read_bytes(), original[key], key)
            if key in ('left', 'right', 'depth'):
                with Image.open(source / relative) as image, Image.open(BytesIO(original[key])) as initial:
                    self.assertFalse(np.array_equal(np.asarray(image), np.asarray(initial)), key)
        scene = output / FRAME
        metadata = strict_json(scene / 'scene.json')
        self.assertEqual(metadata['sourceHashes'], self.original_hashes)
        self.assertEqual(metadata['camera'], strict_json(REFERENCE / 'scene.json')['camera'])
        for name, key in [('recorded.jpg', 'left'), ('right.jpg', 'right'), ('LICENSE', 'license')]:
            self.assertEqual((scene / name).read_bytes(), original[key], name)
        self.assertEqual((output / 'LICENSE').read_bytes(), original['license'])
        for name in metadata['buffers'].values():
            self.assertEqual((scene / name).read_bytes(), (REFERENCE / name).read_bytes(), name)

    def test_changed_processing_source_rejects_before_publication(self):
        for changed in ('prepare-scene.py', 'rectified_calibration.py', 'scene_surface.py'):
            with self.subTest(changed=changed):
                task = self.run_root / ('code-change-' + changed)
                scripts = task / 'scripts'
                scripts.mkdir(parents=True)
                original = {}
                for name in ('prepare-scene.py', 'rectified_calibration.py', 'scene_surface.py'):
                    original[name] = (ROOT / 'scripts' / name).read_bytes()
                    (scripts / name).write_bytes(original[name])
                code = scripts / changed
                if changed == 'prepare-scene.py':
                    modified = original[changed].replace(b'len(sample) < .7:', b'len(sample) < .8:')
                elif changed == 'rectified_calibration.py':
                    modified = original[changed].replace(b'scale = np.array', b'scale = 2 * np.array')
                else:
                    modified = original[changed].replace(b'<= mesh_edge_limit', b'<= mesh_edge_limit * 2')
                self.assertNotEqual(modified, original[changed])
                compile(modified, str(code), 'exec')
                output = task / 'output'
                output.mkdir()
                (output / 'LICENSE').write_bytes((DATA / 'LICENSE').read_bytes())
                (output / 'index.json').write_text('[]\n')
                before = directory_inventory(output)
                wrapper = f'''import numpy as np, runpy, sys
from pathlib import Path
original_indices = np.indices
def changed_indices(*args, **kwargs):
    Path({str(code)!r}).write_bytes({modified!r})
    print('PROCESSING_CODE_MUTATION_APPLIED', flush=True)
    np.indices = original_indices
    return original_indices(*args, **kwargs)
np.indices = changed_indices
sys.argv = sys.argv[1:]
sys.path.insert(0, {str(scripts)!r})
runpy.run_path(sys.argv[0], run_name='__main__')
'''
                command = prepare_command(DATA, output)
                command[0] = str(scripts / 'prepare-scene.py')
                result = subprocess.run([sys.executable, '-B', '-c', wrapper, *command],
                                        cwd=ROOT, capture_output=True, text=True, timeout=30)
                (task / 'stdout.txt').write_text(result.stdout)
                (task / 'stderr.txt').write_text(result.stderr)
                (task / 'inventory.json').write_text(json.dumps({
                    'exitCode': result.returncode, 'before': before, 'after': directory_inventory(output),
                    'originalCodeHashes': {name: hashlib.sha256(payload).hexdigest() for name, payload in original.items()},
                    'changedFile': changed, 'changedHash': sha256(code),
                }, indent=2) + '\n')
                self.assertIn('PROCESSING_CODE_MUTATION_APPLIED', result.stdout)
                self.assertEqual(code.read_bytes(), modified)
                self.assertNotEqual(result.returncode, 0, 'Changed code identity was published')
                self.assertIn('Processing code changed during preparation', result.stderr)
                self.assertEqual(directory_inventory(output), before)

    def test_publication_faults_preserve_collection_and_allow_retry(self):
        peer_id, peer_reference = PEER_FRAME, self.peer_reference
        if not self.peer_available:
            self.skipTest('Optional genuine raw peer acquisition unavailable')
        for existing_peer in (False, True):
            faults = ['catalog_write', 'catalog_replace', 'scene_rename']
            if not existing_peer:
                faults += ['license_write', 'license_replace']
            for fault in faults:
                with self.subTest(existing_peer=existing_peer, fault=fault):
                    source = self.isolated_source(f'publication-{existing_peer}-{fault}')
                    output = source.parent / 'output'
                    expected_catalog = []
                    if existing_peer:
                        shutil.copytree(peer_reference, output / peer_id)
                        (output / 'LICENSE').write_bytes((DATA / 'LICENSE').read_bytes())
                        peer = strict_json(peer_reference / 'scene.json')
                        expected_catalog.append({'id': peer_id, 'label': peer['label']})
                        (output / 'index.json').write_text(json.dumps(expected_catalog, indent=2) + '\n')
                    before = directory_inventory(output)
                    wrapper = f'''import runpy, sys, tempfile
from pathlib import Path
output = Path({str(output)!r})
fault = {fault!r}
applied = False
def fail():
    global applied
    if not applied:
        applied = True
        print('PUBLICATION_FAULT_' + fault, flush=True)
        raise OSError('Injected publication failure: ' + fault)
original_named = tempfile.NamedTemporaryFile
def named(*args, **kwargs):
    handle = original_named(*args, **kwargs)
    if (fault == 'catalog_write' and kwargs.get('prefix') == '.catalog-' or
            fault == 'license_write' and kwargs.get('prefix') == '.license-'):
        original_write = handle.write
        def write(*args, **kwargs):
            fail()
            return original_write(*args, **kwargs)
        handle.write = write
    return handle
tempfile.NamedTemporaryFile = named
original_replace = Path.replace
def replace(path, target):
    if (fault == 'catalog_replace' and Path(target) == output / 'index.json' or
            fault == 'license_replace' and Path(target) == output / 'LICENSE'):
        fail()
    return original_replace(path, target)
Path.replace = replace
original_rename = Path.rename
def rename(path, target):
    if fault == 'scene_rename' and Path(target) == output / {FRAME!r}:
        fail()
    return original_rename(path, target)
Path.rename = rename
sys.argv = sys.argv[1:]
sys.path.insert(0, {str(ROOT / 'scripts')!r})
runpy.run_path(sys.argv[0], run_name='__main__')
'''
                    result = subprocess.run([sys.executable, '-B', '-c', wrapper, *prepare_command(source, output)],
                                            cwd=ROOT, capture_output=True, text=True, timeout=30)
                    (source.parent / 'fault-stdout.txt').write_text(result.stdout)
                    (source.parent / 'fault-stderr.txt').write_text(result.stderr)
                    after = directory_inventory(output)
                    (source.parent / 'fault-inventory.json').write_text(json.dumps({'before': before, 'after': after}, indent=2))
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(result.stdout.count('PUBLICATION_FAULT_' + fault), 1)
                    self.assertIn('Injected publication failure: ' + fault, result.stderr)
                    self.assertEqual(after, before)
                    retry = subprocess.run([sys.executable, '-B', *prepare_command(source, output)],
                                           cwd=ROOT, capture_output=True, text=True, timeout=30)
                    (source.parent / 'retry-stdout.txt').write_text(retry.stdout)
                    (source.parent / 'retry-stderr.txt').write_text(retry.stderr)
                    self.assertEqual(retry.returncode, 0, retry.stderr)
                    scene = output / FRAME
                    metadata = strict_json(scene / 'scene.json')
                    expected_catalog.append({'id': FRAME, 'label': metadata['label']})
                    self.assertEqual(strict_json(output / 'index.json'), sorted(expected_catalog, key=lambda p: p['id']))
                    self.assertEqual(metadata['sourceHashes'], self.original_hashes)
                    for name in metadata['buffers'].values():
                        self.assertEqual((scene / name).read_bytes(), (REFERENCE / name).read_bytes(), name)
                    for name, digest in before['files'].items():
                        if name != 'index.json':
                            self.assertEqual(sha256(output / name), digest, name)
                    self.assertFalse(list(output.glob('.preparing-*')) + list(output.glob('.catalog-*')) + list(output.glob('.license-*')))

    def test_post_operation_interrupt_respects_catalog_commit(self):
        peer_id, peer_reference = PEER_FRAME, self.peer_reference
        if not self.peer_available:
            self.skipTest('Optional genuine raw peer acquisition unavailable')
        for existing_peer in (False, True):
            points = ['scene_rename', 'catalog_replace'] + ([] if existing_peer else ['license_replace'])
            for point in points:
                with self.subTest(existing_peer=existing_peer, point=point):
                    source = self.isolated_source(f'interrupt-{existing_peer}-{point}')
                    output = source.parent / 'output'
                    expected_catalog = []
                    if existing_peer:
                        shutil.copytree(peer_reference, output / peer_id)
                        (output / 'LICENSE').write_bytes((DATA / 'LICENSE').read_bytes())
                        peer = strict_json(peer_reference / 'scene.json')
                        expected_catalog.append({'id': peer_id, 'label': peer['label']})
                        (output / 'index.json').write_text(json.dumps(expected_catalog, indent=2) + '\n')
                    before = directory_inventory(output)
                    wrapper = f'''import runpy, sys
from pathlib import Path
output = Path({str(output)!r})
point = {point!r}
applied = False
def interrupt():
    global applied
    if not applied:
        applied = True
        print('POST_OPERATION_INTERRUPT_' + point, flush=True)
        raise KeyboardInterrupt('Injected after completed operation: ' + point)
original_replace = Path.replace
def replace(path, target):
    result = original_replace(path, target)
    if (point == 'catalog_replace' and Path(target) == output / 'index.json' or
            point == 'license_replace' and Path(target) == output / 'LICENSE'):
        interrupt()
    return result
Path.replace = replace
original_rename = Path.rename
def rename(path, target):
    result = original_rename(path, target)
    if point == 'scene_rename' and Path(target) == output / {FRAME!r}:
        interrupt()
    return result
Path.rename = rename
sys.argv = sys.argv[1:]
sys.path.insert(0, {str(ROOT / 'scripts')!r})
runpy.run_path(sys.argv[0], run_name='__main__')
'''
                    result = subprocess.run([sys.executable, '-B', '-c', wrapper, *prepare_command(source, output)],
                                            cwd=ROOT, capture_output=True, text=True, timeout=30)
                    (source.parent / 'interrupt-stdout.txt').write_text(result.stdout)
                    (source.parent / 'interrupt-stderr.txt').write_text(result.stderr)
                    after = directory_inventory(output)
                    (source.parent / 'interrupt-inventory.json').write_text(json.dumps({'point': point, 'before': before, 'after': after}, indent=2))
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(result.stdout.count('POST_OPERATION_INTERRUPT_' + point), 1)
                    self.assertIn('Injected after completed operation: ' + point, result.stderr)
                    if point != 'catalog_replace':
                        self.assertEqual(after, before)
                        retry = subprocess.run([sys.executable, '-B', *prepare_command(source, output)],
                                               cwd=ROOT, capture_output=True, text=True, timeout=30)
                        (source.parent / 'retry-stdout.txt').write_text(retry.stdout)
                        (source.parent / 'retry-stderr.txt').write_text(retry.stderr)
                        self.assertEqual(retry.returncode, 0, retry.stderr)
                    scene = output / FRAME
                    self.assertTrue(scene.is_dir())
                    metadata = strict_json(scene / 'scene.json')
                    self.assertEqual(metadata['sourceHashes'], self.original_hashes)
                    expected_catalog.append({'id': FRAME, 'label': metadata['label']})
                    self.assertEqual(strict_json(output / 'index.json'), sorted(expected_catalog, key=lambda p: p['id']))
                    for name in metadata['buffers'].values():
                        self.assertEqual((scene / name).read_bytes(), (REFERENCE / name).read_bytes(), name)
                    for name, digest in before['files'].items():
                        if name != 'index.json':
                            self.assertEqual(sha256(output / name), digest, name)
                    self.assertEqual((output / 'LICENSE').read_bytes(), (DATA / 'LICENSE').read_bytes())
                    self.assertFalse(list(output.glob('.preparing-*')) + list(output.glob('.catalog-*')) + list(output.glob('.license-*')))


if __name__ == '__main__':
    unittest.main()
