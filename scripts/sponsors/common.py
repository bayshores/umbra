"""Local sponsor artifact primitives. No transport, credentials or account access."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def parse(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def invalid(_):
        raise ValueError('Nonfinite JSON')
    value = json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)
    canonical(value)  # Also rejects numeric overflow such as 1e999.
    return value


def snapshot(path, expected=None, max_bytes=2_000_000):
    with Path(path).open('rb') as source:
        data = source.read(max_bytes + 1)
    require(len(data) <= max_bytes, 'Artifact exceeds byte ceiling')
    if expected is not None:
        require(isinstance(expected, str) and re.fullmatch('[a-f0-9]{64}', expected), 'Invalid SHA256')
        require(digest(data) == expected, 'Artifact SHA256 mismatch')
    return data


def record(path, data, parent, external=True):
    item = {'path': os.path.relpath(Path(path).resolve(), Path(parent).resolve()),
            'sha256': digest(data), 'bytes': len(data)}
    if external:
        item['readOnlyExternal'] = True
    return item


def load_script(name):
    path = Path(__file__).resolve().parents[1] / name
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def publish(output, files):
    """Reuse the preparation tool's exclusive directory publication, no overwrite."""
    output = Path(output).absolute()
    require(not os.path.lexists(output), 'Existing output retained')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.sponsor-', dir=output.parent) as temporary:
        staged = Path(temporary) / 'result'
        staged.mkdir()
        for name, data in files.items():
            require(Path(name).name == name, 'Output names must be flat')
            (staged / name).write_bytes(data)
        load_script('prepare-stereo-crop.py').publish_new_directory(staged, output)


def code_hashes():
    return {path.name: digest(path.read_bytes()) for path in sorted(Path(__file__).parent.glob('*.py'))}
