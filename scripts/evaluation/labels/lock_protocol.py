"""Freeze a labeling protocol: hash its parameters, its document, its labeler instructions and the kit code
before any reference or label is computed with it. Refuses to overwrite an existing lock (a change needs a new
protocol version), refuses while `pendingBeforeLock` lists open items, and copies the protocol's own
`lockStatement` (what was and was not computed before the lock) instead of a fixed claim.

  python -I lock_protocol.py .local/evaluation/pao-labels/v2/protocol-v2.json

(The v1 lock, written by the first version of this script, carries a fixed statement that overclaimed;
see protocol-v1.errata.md.)
"""
from __future__ import annotations

import datetime as dt
import importlib.metadata as metadata
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import ROOT, read_json, relative, require, sha256_file, write_json  # noqa: E402

PACKAGES = {'numpy': 'BSD-3-Clause', 'scipy': 'BSD-3-Clause', 'pyarrow': 'Apache-2.0', 'pillow': 'MIT-CMU (HPND)'}


def build_record(path):
    path = Path(path).resolve()
    params = read_json(path)
    require(not params.get('pendingBeforeLock'), f"{path.name} still lists pendingBeforeLock items: {params.get('pendingBeforeLock')}")
    require(params.get('lockStatement'), f'{path.name} needs a lockStatement saying what was computed before the lock')
    doc = path.with_name(params['document'])
    require(doc.exists(), f'missing protocol document {doc.name}')
    head = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    record = {'file': path.name, 'sha256': sha256_file(path), 'bytes': path.stat().st_size,
              'document': {'file': doc.name, 'sha256': sha256_file(doc)},
              'kitCodeSha256': {p.name: sha256_file(p) for p in sorted(HERE.glob('*.py'))},
              'environment': {'python': sys.version.split()[0], 'interpreter': relative(sys.executable),
                              'packages': {name: {'version': metadata.version(name), 'license': lic} for name, lic in PACKAGES.items()},
                              'notInstalled': 'No OpenCV (its wheels bundle LGPL FFmpeg); no segmentation model; no network service.'},
              'repositoryHead': head, 'frozenUtc': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
              'statement': params['lockStatement']}
    if params.get('labelerInstructions'):
        instructions = path.with_name(params['labelerInstructions'])
        require(instructions.exists(), f'missing labeler instructions {instructions.name}')
        record['labelerInstructions'] = {'file': instructions.name, 'sha256': sha256_file(instructions)}
    return record


def main(argv):
    path = Path(argv[0]).resolve()
    lock = path.with_name(path.stem + '.lock.json')
    require(not lock.exists(), f'{lock.name} exists; a changed protocol needs a new version')
    record = build_record(path)
    write_json(lock, record, overwrite=False)
    print(record['sha256'], relative(lock))


if __name__ == '__main__':
    main(sys.argv[1:])
