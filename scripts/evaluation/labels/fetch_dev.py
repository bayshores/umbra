"""Fetch public Argoverse 2 files for DEVELOPMENT logs only, with size/ETag/SHA-256 receipts.

Usage (evaluation-only, run with an isolated interpreter):
  python -I fetch_dev.py list <log> <subprefix> <out.json>
  python -I fetch_dev.py get <new-empty-directory> <log> <relative-key> [<relative-key> ...]

Every `get` writes into a directory that must not exist yet. Held-out and reserve logs are refused.
Downloaded files are data: they are never executed or imported.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import assert_development, full_log_id, require, write_json  # noqa: E402

BASE = 'https://argoverse.s3.amazonaws.com/'
SPLIT = 'train'  # all three development logs are in the public train split
MAX_FILE = 64 << 20


def log_prefix(log):
    return f'datasets/av2/sensor/{SPLIT}/{full_log_id(log)}/'


def list_keys(log, sub):
    """List object keys (with size and ETag) under a development log's sub-prefix."""
    assert_development(sub)
    prefix = log_prefix(log) + sub
    keys, token = [], None
    while True:
        url = f'{BASE}?list-type=2&prefix={prefix}'
        if token:
            url += '&continuation-token=' + urllib.request.quote(token, safe='')
        with urllib.request.urlopen(url, timeout=60) as response:
            text = response.read().decode()
        for block in re.findall(r'<Contents>(.*?)</Contents>', text, re.S):
            key = re.search(r'<Key>([^<]+)</Key>', block).group(1)
            size = int(re.search(r'<Size>(\d+)</Size>', block).group(1))
            etag = re.search(r'<ETag>([^<]+)</ETag>', block).group(1).replace('&quot;', '').strip('"')
            keys.append({'key': key, 'bytes': size, 'etag': etag})
        if '<IsTruncated>true</IsTruncated>' not in text:
            break
        token = re.search(r'<NextContinuationToken>([^<]+)</NextContinuationToken>', text).group(1)
    return keys


def fetch(directory, log, relatives):
    directory = Path(directory)
    assert_development(directory)
    require(not directory.exists(), f'Download directory must be new: {directory}')
    directory.mkdir(parents=True)
    records = []
    for rel in relatives:
        assert_development(rel)
        require('..' not in rel and not rel.startswith('/'), f'Bad relative key {rel}')
        key = log_prefix(log) + rel
        with urllib.request.urlopen(BASE + key, timeout=120) as response:
            etag = (response.headers.get('ETag') or '').strip('"')
            declared = int(response.headers.get('Content-Length') or -1)
            data = response.read(MAX_FILE + 1)
            status = response.status
        require(len(data) <= MAX_FILE, f'File too large: {key}')
        target = directory / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        md5 = hashlib.md5(data).hexdigest()
        records.append({'key': key, 'url': BASE + key, 'status': status, 'bytes': len(data),
                        'contentLength': declared, 'sizeMatches': declared == len(data), 'etag': etag,
                        'md5': md5, 'md5MatchesEtag': md5 == etag, 'sha256': hashlib.sha256(data).hexdigest()})
    receipt = {'log': full_log_id(log), 'split': SPLIT, 'fetchedUtc': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
               'files': len(records), 'totalBytes': sum(r['bytes'] for r in records),
               'allVerified': all(r['sizeMatches'] and r['md5MatchesEtag'] for r in records),
               'license': 'Argoverse 2 sensor data, CC BY-NC-SA 4.0, (c) 2021 Argo AI, LLC',
               'records': records}
    write_json(directory / 'receipt.json', receipt, overwrite=False)
    return receipt


def main(argv):
    if argv[:1] == ['list'] and len(argv) == 4:
        keys = list_keys(argv[1], argv[2])
        write_json(argv[3], {'log': full_log_id(argv[1]), 'prefix': log_prefix(argv[1]) + argv[2], 'count': len(keys), 'keys': keys})
        print(len(keys), 'keys')
    elif argv[:1] == ['get'] and len(argv) >= 4:
        receipt = fetch(argv[1], argv[2], argv[3:])
        print(receipt['files'], 'files', receipt['totalBytes'], 'bytes', 'verified' if receipt['allVerified'] else 'NOT VERIFIED')
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
