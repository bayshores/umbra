"""M7 (held-out run) shared helpers: the split roles, the lock and anchor checks, and receipted public AV2 downloads.

Nothing here opens a held-out file. preflight() is the only way into held-out work: protocol-lock.json at its fixed path
with the sha256 given, an approval block, the same sha256 in the tracked anchor on origin/main, an empty git status, a
HEAD whose tracked files equal pipeline.commit except the anchor files (ANCHOR_REL and ANCHOR_DIR/), every pinned
file unchanged, and the labels addendum equal to its committed anchor (null: no addendum file). The main checkout is used (ignored .local/ data and protocols live only there; a fresh worktree has none).
Each anchor file under ANCHOR_DIR is written once: exactly one commit on origin/main touches it, so a later change makes
every driver refuse. Held-out, each driver runs once per site: a run marker (RUN_MARKERS, O_EXCL) names its output folder
and a second run is refused. Rehearsal (development logs) skips the lock, anchor and marker checks and nothing else.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
SPLIT = ROOT / '.local/evaluation/pao-accuracy/split.json'
SPLIT_SHA = '47f64fe007b51d75f4bf1c9891d940fd6393c777e0232a0a4f9428bbbeaafab4'
LOCK_PATH = ROOT / '.local/evaluation/pao-accuracy/protocol-lock.json'
ANCHOR_REL = 'scripts/evaluation/pao-lock-anchor.json'
ANCHOR_DIR = 'scripts/evaluation/m7/anchors'  # tracked: lost-site list, labels addendum anchor and held-out label manifest
LOST_SITES_REL = f'{ANCHOR_DIR}/lost-sites.json'  # protocol-lock statistics.lostSiteRule: {"lostSites": [{logId, crossingId, evidence}]}
LABELS_ADDENDUM = ROOT / '.local/evaluation/pao-accuracy/lock/labels-addendum.json'  # protocol-lock labels.scoringCondition (4)
ADDENDUM_ANCHOR_REL = f'{ANCHOR_DIR}/labels-addendum-anchor.json'  # {"labelsAddendumSha256": <sha256> or null (D/E not scored)}
LABELS_MANIFEST_REL = f'{ANCHOR_DIR}/labels-manifest.json'  # {"labelsAddendumSha256", "sites": [{logId, crossingId, labelsSha256, reason}]}
HEX64 = re.compile('[0-9a-f]{64}')
RUN_MARKERS = ROOT / '.local/evaluation/pao-accuracy/m7/run-markers'  # held-out: <step>-<log8>-<crossing>.json, one per run
COORDINATOR_ENV = 'PAO_LABELS_COORDINATOR_DIR'  # the coordinator-only labels folder: never written into any file
DEVELOPMENT_ROLES = ('development', 'development-optional')
HELD_OUT_ROLES = ('held-out-blocked-test', 'held-out-left-side-case', 'held-out-control', 'held-out-control-declared-only')
BUCKET = 'https://argoverse.s3.amazonaws.com/'
LOG_PREFIX = 'datasets/av2/sensor/train/{log}/'
S3 = '{http://s3.amazonaws.com/doc/2006-03-01/}'
RETRIES = 3  # protocol-lock dataAccess: at most 3 retries, each into a new empty folder
CURL_SECONDS = 300


class SiteStop(Exception):
    """A step failed: the site stops with the step named (protocol-lock dataAccess); it is never rerun. The banned
    vehicle-status word is masked (p*rk) in the reason, which can quote a tool's error output."""

    def __init__(self, step, reason):
        reason = masked(str(reason))
        super().__init__(f'{step}: {reason}')
        self.step, self.reason = step, reason


def masked(text, folders=()):
    """text with the banned vehicle-status word masked (p*rk) and the coordinator-only folder (and any folder given)
    replaced by a placeholder: tool output can quote either."""
    for folder in [os.environ.get(COORDINATOR_ENV, ''), *folders]:
        folder = str(folder).rstrip('/')
        if len(folder) > 1:
            text = text.replace(folder, '<coordinator-only folder>')
    return re.sub('park', 'p*rk', text, flags=re.IGNORECASE)


def now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    return sha(Path(path).read_bytes())


def rel(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def dump(obj):
    return (json.dumps(obj, indent=1, ensure_ascii=False) + '\n').encode()


def write_new(path, data, mode=0o444):
    """New files only (O_EXCL); JSON objects are dumped; the banned vehicle-status word and the coordinator-only folder
    are refused."""
    data = dump(data) if isinstance(data, (dict, list)) else data
    if b'park' in data.lower():
        raise ValueError(f'{path}: banned vehicle-status word')
    folder = os.environ.get(COORDINATOR_ENV, '').rstrip('/')
    if len(folder) > 1 and folder.encode() in data:
        raise ValueError(f'{path}: names the coordinator-only folder')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(data)
    return sha(data)


def site(log, crossing):
    """The split.json entry of this site (split.json checked against its frozen sha256)."""
    hits = [s for s in split_sites() if s['logId'] == log and str(s['crossingId']) == str(crossing)]
    if len(hits) != 1:
        raise SystemExit(f'{log[:8]}/{crossing} is not one site of split.json')
    return hits[0]


def split_sites():
    """Every split.json site (split.json checked against its frozen sha256)."""
    raw = SPLIT.read_bytes()
    if sha(raw) != SPLIT_SHA:
        raise SystemExit('split.json differs from its frozen sha256')
    return json.loads(raw)['sites']


def committed(relative, what):
    """A tracked anchor file: committed, unchanged since its commit, pushed (equal to origin/main), and written once
    (exactly one commit on origin/main touches it: protocol-lock anchor.rule). Returns (doc, sha256)."""
    path = ROOT / relative
    if not path.is_file():
        raise SystemExit(f'{relative} does not exist: write {what} and commit it first')
    if subprocess.run(['git', 'ls-files', '--error-unmatch', relative], cwd=ROOT, capture_output=True).returncode != 0 \
            or git('status', '--porcelain', '--', relative).strip():
        raise SystemExit(f'{relative} is not committed unchanged')
    if subprocess.run(['git', 'diff', '--quiet', 'origin/main', '--', relative], cwd=ROOT, capture_output=True).returncode != 0:
        raise SystemExit(f'{relative} differs from origin/main: push it first')
    if git('rev-list', '--count', 'origin/main', '--', relative).strip() != '1':
        raise SystemExit(f'{relative} has more than one commit on origin/main: an anchor file is written once')
    return json.loads(path.read_bytes()), sha_file(path)


def lost_sites():
    """The committed lost-site list (statistics.lostSiteRule). Returns ({'<log8>/<crossing>', ...}, its sha256)."""
    doc, digest = committed(LOST_SITES_REL, 'the lost-site list (possibly empty)')
    return {f"{x['logId'][:8]}/{x['crossingId']}" for x in doc['lostSites']}, digest


def labels_addendum(held_out, path=LABELS_ADDENDUM):
    """The labels addendum sha256 this M7 run works under, or None (D and E not scored). Held-out: the value in the
    committed anchor, which the addendum file must equal (null: the file must not exist). Rehearsal: the file's sha256
    if it exists."""
    path = Path(path)
    current = sha_file(path) if path.exists() else None
    if not held_out:
        return current
    doc, _ = committed(ADDENDUM_ANCHOR_REL, 'the labels addendum anchor ({"labelsAddendumSha256": null} when D and E are not scored)')
    anchored = doc.get('labelsAddendumSha256', 'missing')
    if not (anchored is None or (isinstance(anchored, str) and HEX64.fullmatch(anchored))):
        raise SystemExit(f'{ADDENDUM_ANCHOR_REL}: labelsAddendumSha256 must be a sha256 or null')
    if anchored != current:
        raise SystemExit(f'the labels addendum ({current or "no file"}) differs from its committed anchor ({anchored or "null"})')
    return anchored


def labels_manifest(addendum_sha):
    """The committed held-out label manifest (labels.scoringCondition), for the anchored addendum: every held-out site of
    split.json once, with its label aggregate's sha256 or null and a reason. Returns ({'<log8>/<crossing>': sha256 or
    None}, its sha256)."""
    doc, digest = committed(LABELS_MANIFEST_REL, 'the held-out label manifest')
    if doc.get('labelsAddendumSha256') != addendum_sha:
        raise SystemExit(f'{LABELS_MANIFEST_REL} names another labels addendum')
    entries = {}
    for x in doc.get('sites', []):
        name, labels = f"{x['logId'][:8]}/{x['crossingId']}", x.get('labelsSha256')
        ok = HEX64.fullmatch(labels) if isinstance(labels, str) else labels is None and str(x.get('reason') or '').strip()
        if name in entries or not ok:
            raise SystemExit(f'{LABELS_MANIFEST_REL}: repeated entry, or neither a labels sha256 nor a reason, for {name}')
        entries[name] = labels
    held = {f"{x['logId'][:8]}/{x['crossingId']}" for x in split_sites() if x['role'] in HELD_OUT_ROLES}
    if set(entries) != held:
        raise SystemExit(f'{LABELS_MANIFEST_REL} must name every held-out site once; missing {sorted(held - set(entries))}, '
                         f'extra {sorted(set(entries) - held)}')
    return entries, digest


def marker_path(step, log, crossing):
    return RUN_MARKERS / f'{step}-{log[:8]}-{crossing}.json'


def run_marker(step, log, crossing, folder):
    """Held-out: the one run of this step (build, paint, evaluate) at this site, written before its output folder is
    made. A second run is refused (protocol-lock dataAccess: a site is never rerun)."""
    path = marker_path(step, log, crossing)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_new(path, {'kind': 'pao-m7-run-marker', 'step': step, 'site': {'logId': log, 'crossingId': str(crossing)},
                         'folder': rel(folder), 'utc': now()})
    except FileExistsError:
        raise SystemExit(f'{step} already ran for {log[:8]}/{crossing} ({rel(path)}): a held-out site is never rerun') from None


def marked_folder(step, log, crossing):
    """The output folder the run marker of this step names for this site, or None."""
    path = marker_path(step, log, crossing)
    return json.loads(path.read_bytes())['folder'] if path.is_file() else None


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def pinned_files(node, out=None):
    """Every {path, sha256} pair in the lock (the coordinator-only folder is skipped: M7 never opens it)."""
    out = [] if out is None else out
    if isinstance(node, dict):
        if isinstance(node.get('path'), str) and isinstance(node.get('sha256'), str) and re.fullmatch('[0-9a-f]{64}', node['sha256']):
            if not node['path'].startswith('$'):
                out.append((node['path'], node['sha256']))
        for value in node.values():
            pinned_files(value, out)
    elif isinstance(node, list):
        for value in node:
            pinned_files(value, out)
    return out


def preflight(lock_sha256):
    """Held-out only. Returns the lock. Raises SystemExit before any held-out file is touched."""
    if not (isinstance(lock_sha256, str) and re.fullmatch('[0-9a-f]{64}', lock_sha256)):
        raise SystemExit('Held-out runs need --protocol-lock-sha256')
    if not LOCK_PATH.is_file():
        raise SystemExit(f'{rel(LOCK_PATH)} does not exist (no approved lock)')
    raw = LOCK_PATH.read_bytes()
    if sha(raw) != lock_sha256:
        raise SystemExit('protocol-lock.json differs from the given sha256')
    lock = json.loads(raw)
    approval = lock.get('approval') or {}
    if not all(isinstance(approval.get(k), str) and approval[k].strip() for k in ('by', 'utc')):
        raise SystemExit('protocol-lock.json has no approval')
    commit = lock['pipeline']['commit']
    status = git('status', '--porcelain')
    if status.strip():
        raise SystemExit('The checkout has modified or untracked files:\n' + status)
    if subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=ROOT).returncode != 0:
        raise SystemExit(f'pipeline.commit {commit[:7]} is not an ancestor of HEAD')
    changed = [p for p in git('diff', '--name-only', commit, 'HEAD').split('\n') if p]
    extra = [p for p in changed if p != ANCHOR_REL and not p.startswith(ANCHOR_DIR + '/')]
    if extra:
        raise SystemExit('Tracked files differ from pipeline.commit: ' + ', '.join(extra))
    subprocess.run(['git', 'fetch', '--quiet', 'origin', 'main'], cwd=ROOT, check=True)
    try:
        anchor = json.loads(git('show', f'origin/main:{ANCHOR_REL}'))
    except subprocess.CalledProcessError:
        raise SystemExit(f'origin/main has no {ANCHOR_REL}: commit and push the anchor first') from None
    if anchor.get('protocolLockSha256') != lock_sha256:
        raise SystemExit('The anchor on origin/main names another lock')
    for path, value in pinned_files(lock):
        if sha_file(ROOT / path) != value:
            raise SystemExit(f'{path} differs from its sha256 in the lock')
    for group in (lock['pipeline']['scoringCode'], lock['pipeline']['m7Code'], lock['sceneBuilder']['code']):
        for path, value in group.items():
            if sha_file(ROOT / path) != value:
                raise SystemExit(f'{path} differs from its sha256 in the lock')
    labels_addendum(True)
    return lock


def guard(log, crossing, rehearse, lock_sha256):
    """The split role decides the mode: development logs only in rehearsal, held-out only after preflight, reserve never."""
    entry = site(log, crossing)
    role = entry['role']
    if rehearse:
        if role not in DEVELOPMENT_ROLES:
            raise SystemExit(f'Rehearsal runs on development sites only; {log[:8]} is {role}')
        return entry, None
    if role not in HELD_OUT_ROLES:
        raise SystemExit(f'{log[:8]} is {role}: M7 runs held-out sites only (reserve sites stay unopened)')
    return entry, preflight(lock_sha256)


# ------------------------------------------------------------------ public downloads
def curl(url, target, headers, max_bytes):
    subprocess.run(['curl', '--fail', '--silent', '--show-error', '--proto', '=https', '--max-time', str(CURL_SECONDS),
                    '--max-filesize', str(max_bytes), '--dump-header', str(headers), '--output', str(target), url], check=True)
    text = headers.read_text(errors='replace').lower().splitlines()
    head = {line.split(':', 1)[0].strip(): line.split(':', 1)[1].strip() for line in text if ':' in line}
    if 'content-length' not in head and head.get('transfer-encoding') == 'chunked':
        return None, head.get('etag', '').strip('"')  # S3 sends listings chunked, without a Content-Length
    return int(head['content-length']), head.get('etag', '').strip('"')


def fetch(url, folder, name, max_bytes, listing=None):
    """GET url into folder/attempt-<n>/name (a new empty folder per attempt; at most RETRIES retries). With a listing
    entry: bytes == HTTP Content-Length == listing Size, and MD5 == HTTP ETag == listing ETag for single-part objects
    (multi-part: the HTTP ETag equals the listing ETag and the size matches). Without one (a listing fetch) a chunked
    response without Content-Length is accepted; an object download always needs it. Returns (path, receipt)."""
    errors = []
    for attempt in range(RETRIES + 1):
        where = Path(folder) / f'attempt-{attempt}'
        where.mkdir(parents=True, exist_ok=False)
        target, headers = where / name, where / f'{name}.headers.txt'
        try:
            length, etag = curl(url, target, headers, max_bytes)
            data = target.read_bytes(); md5 = hashlib.md5(data).hexdigest()
            if length is None and listing is not None:
                raise ValueError('object download without Content-Length')
            if length is not None and len(data) != length:
                raise ValueError(f'{len(data)} bytes, Content-Length {length}')
            if listing is not None:
                if len(data) != listing['bytes'] or etag != listing['etag']:
                    raise ValueError(f'listing says {listing["bytes"]} bytes / ETag {listing["etag"]}, got {len(data)} / {etag}')
                if '-' not in etag and md5 != etag:
                    raise ValueError(f'MD5 {md5} != ETag {etag}')
            os.chmod(target, 0o444)
            return target, {'url': url, 'path': rel(target), 'bytes': len(data), 'httpContentLength': 'none (chunked)' if length is None else length,
                            'httpETag': etag, 'md5': md5, 'sha256': sha(data),
                            'md5Check': 'none (listing fetch)' if listing is None else
                            ('single-part: MD5 == ETag' if '-' not in etag else 'multi-part ETag: size and ETag only'),
                            'attempt': attempt, 'earlierErrors': errors, 'utc': now()}
        except (subprocess.CalledProcessError, ValueError, KeyError) as error:
            errors.append({'attempt': attempt, 'error': str(error)[:300]})
    raise SiteStop('download', f'{url}: no matching download after {RETRIES} retries ({errors[-1]["error"]})')


def listing_url(log, sub):
    prefix = (LOG_PREFIX.format(log=log) + sub).replace('/', '%2F')
    return f'{BUCKET}?list-type=2&delimiter=%2F&max-keys=1000&prefix={prefix}'


LISTINGS = {'root': '', 'calibration': 'calibration/', 'map': 'map/', 'lidar': 'sensors/lidar/',
            'stereo_front_left': 'sensors/cameras/stereo_front_left/', 'stereo_front_right': 'sensors/cameras/stereo_front_right/'}


def read_listing(path, log, sub):
    """Keys of one saved listing as {relative name: {key, bytes, etag}}; refuses a truncated or foreign listing."""
    root = ET.fromstring(Path(path).read_bytes())
    prefix = LOG_PREFIX.format(log=log) + sub
    if root.findtext(f'{S3}Prefix') != prefix or root.findtext(f'{S3}IsTruncated') != 'false':
        raise SiteStop('listings', f'{Path(path).name}: unexpected prefix or a truncated listing')
    out = {}
    for item in root.findall(f'{S3}Contents'):
        key, size, etag = (item.findtext(f'{S3}{tag}') or '' for tag in ('Key', 'Size', 'ETag'))
        if not (key.startswith(prefix) and size.isdigit() and etag):
            raise SiteStop('listings', f'{Path(path).name}: malformed entry')
        out[key[len(prefix):]] = {'key': key, 'bytes': int(size), 'etag': etag.strip('"')}
    return out
