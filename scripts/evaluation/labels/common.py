"""Shared helpers for the evaluation-only reference-labeling kit (never shipped in the app).

Everything here is local development tooling for the Palo Alto (Argoverse 2) accuracy program.
AV2-derived data is CC BY-NC-SA 4.0, (c) 2021 Argo AI, LLC.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LABELS_DIR = ROOT / '.local/evaluation/pao-labels'
# Protocol v2 lives in its own folder. v1 (closed; gate FAIL) stays in LABELS_DIR with its byte-identical
# code snapshot in LABELS_DIR / 'v1-kit' (this kit no longer hashes to the v1 locks).
V2_DIR = LABELS_DIR / 'v2'
PUBLIC_TASKS_SCHEMA = 'pao-labels-tasks-v2-public'
INTERNAL_TASKS_SCHEMA = 'pao-labels-tasks-v2-internal'
INTERNAL_TASKS = Path('kit-internal') / 'tasks-internal.json'  # next to the public task list; labelers never open it

# Development logs only. Downloads and reads of any other log are refused by these tools.
DEV_LOGS = {
    '8c52d911': '8c52d911-fe34-3424-9864-d3fdfac38064',
    '453e5558': '453e5558-6363-38e3-bf9b-42b5ba0a6f1d',
    'c780d53a': 'c780d53a-2d37-3cd8-9e89-530966aef53e',
}
# Held-out (6) and reserve (3) logs from split.json. Never opened, listed in detail or downloaded here.
FORBIDDEN_PREFIXES = ('88f47a10', '3153b5b3', 'd33f667d', 'fd5c6932', '25e5c600', 'e1d68dde',
                      'a91d4c7b', '4a60c567', 'b6c4361a')


class ProtocolError(RuntimeError):
    """Raised when a tool would break the frozen protocol or the development-only rule."""


def require(condition, message):
    if not condition:
        raise ProtocolError(message)


def assert_development(text):
    """Refuse any path, key or id that names a held-out or reserve log."""
    lowered = str(text).lower()
    for prefix in FORBIDDEN_PREFIXES:
        require(prefix not in lowered, f'Refused: {prefix} is a held-out or reserve log')
    return text


def full_log_id(short_or_full):
    assert_development(short_or_full)
    key = str(short_or_full)[:8]
    require(key in DEV_LOGS, f'Not a development log: {short_or_full}')
    return DEV_LOGS[key]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value) -> bytes:
    """Deterministic JSON bytes (sorted keys, compact separators, trailing newline)."""
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n').encode()


def pretty_json(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False) + '\n').encode()


def write_bytes_atomic(path, data: bytes, overwrite=True):
    path = Path(path)
    require(overwrite or not path.exists(), f'Refusing to overwrite {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix='.tmp-')
    try:
        with os.fdopen(handle, 'wb') as out:
            out.write(data)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
    return sha256_bytes(data)


def write_json(path, value, overwrite=True, pretty=True):
    return write_bytes_atomic(path, pretty_json(value) if pretty else canonical_json(value), overwrite)


def read_json(path):
    return json.loads(Path(path).read_text())


def relative(path):
    """Repository-relative path string when possible (labels and receipts never store home paths)."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_protocol(path=None):
    """Load the frozen protocol parameters and check them against the lock file."""
    path = Path(path) if path else LABELS_DIR / 'protocol-v1.json'
    lock = path.with_name(path.stem + '.lock.json')
    data = path.read_bytes()
    require(lock.exists(), f'Protocol {path.name} is not locked (missing {lock.name})')
    locked = read_json(lock)
    require(locked.get('sha256') == sha256_bytes(data), f'Protocol {path.name} changed after it was locked')
    return json.loads(data), locked['sha256']


def stable_seed(*parts) -> int:
    """Process-independent seed (Python's hash() of a str is randomized per process, and -I ignores PYTHONHASHSEED)."""
    return int(sha256_bytes('|'.join(str(p) for p in parts).encode())[:8], 16)


def merge_task(public, internal):
    """Merge a labeler-visible task with its kit-only fields. Shared keys must agree (dicts merge key by key)."""
    merged = dict(public)
    for key, value in internal.items():
        if key not in merged:
            merged[key] = value
        elif isinstance(value, dict) and isinstance(merged[key], dict):
            for sub, item in value.items():
                require(sub not in merged[key] or merged[key][sub] == item, f'task {public["taskId"]}: {key}.{sub} differs between task files')
            merged[key] = {**merged[key], **value}
        else:
            require(merged[key] == value, f'task {public["taskId"]}: {key} differs between task files')
    return merged


def load_tasks(path):
    """Task list as the kit sees it. A v2 public list (what labelers read: no coordinates, columns, guides or hashes) is
    merged with kit-internal/tasks-internal.json next to it; any other task list is returned unchanged."""
    path = Path(path)
    doc = read_json(path)
    if doc.get('schema') != PUBLIC_TASKS_SCHEMA:
        return doc
    internal = read_json(path.parent / INTERNAL_TASKS)
    require(internal.get('schema') == INTERNAL_TASKS_SCHEMA, 'kit-internal task file has the wrong schema')
    require(internal.get('publicSha256') == sha256_file(path), 'public task list changed after the kit-internal file was written')
    by_id = internal['tasks']
    require(set(by_id) == {t['taskId'] for t in doc['tasks']}, 'public and kit-internal task lists name different tasks')
    merged = dict(doc, tasks=[merge_task(t, by_id[t['taskId']]) for t in doc['tasks']])
    merged['internal'] = {'path': relative(path.parent / INTERNAL_TASKS), 'sha256': sha256_file(path.parent / INTERNAL_TASKS)}
    return merged
