"""shape-v2: the recorded stationary vehicle's LiDAR shape (one solid top per grid cell) for placement protocol v3.4.

The rules are the frozen shape-v2 protocol (.local/evaluation/pao-accuracy/shape-v2/protocol.json, sha256 70cbb24f...: the frozen
shape-v1 protocol with the sweep thinning step 0.25 m instead of 1.0 m and the shape-v2 names):
  6.1  sweep plan (metadata only): a listed sweep is a candidate when the vehicle's track has an annotation row at its
       timestamp, the timestamp lies inside the ego pose record and the ego origin is at most 30 m (horizontal) from the
       averaged box centre; candidates are thinned in time order (keep one when the ego moved at least 0.25 m since the
       last kept one), capped at 40 (evenly spaced positions) and split alternately into shape and scoring sweeps.
       The plan file is written and hashed before any sweep is downloaded.
  6.3  downloads: one file at a time, at most 4 attempts with waits of 5, 10 and 20 s; success = HTTP 200, the listing
       size and (single-part ETag) MD5 = ETag. A sweep that still fails is dropped and listed, never replaced.
  6.2  pose: the ego pose at the sweep timestamp, linearly interpolated (normalized linear quaternion interpolation).
  6.4  registration: ego frame -> city -> scene -> car-local frame of the upright averaged box.
  6.5  stationary: the placement protocol's stationary test (annotation span >= 3.0 s, largest horizontal move <= 1.0 m).
  5.3  returns kept: inside the box footprint, 0.25 m <= height <= box height, not inside another track's cuboid (grown
       0.10 m on every side) at that sweep. The vehicle's own per-sweep annotations never move or filter returns.
  5.4  solid top per cell: the second-highest of the per-sweep highest kept returns (two different sweeps needed),
       floored to the millimetre; otherwise null (not proven).
  10   the carShape block of ground.json (schemaVersion 3) and the two refusals (too-few-sweeps, not-stationary).
It also builds the evaluation-only shape of the approach-half split addendum (split-addendum.json, sha256 953dae5f...):
the same shape rules on the far half of the sweep list. That shape is a report file only, never written into ground.json.

Nothing here invents geometry: a cell without returns from two sweeps stays null (not proven), a refused shape has no
tops at all, and integrity failures (a changed plan, a changed or missing receipted sweep) stop with no output.
AV2 vehicles are called stationary. Held-out access loads M7's lock checks by path under a unique module name.

Commands (one at a time; the download is the only network step and reads only public, $0 files):
  plan            write .local/data/av2/shape-v2-<log>/sweep-plan.json from metadata and print its sha256
  download        download the planned sweeps with receipts (receipt.json)
  approach-split  write the evaluation-only approach-half shape for one scene (report only)
"""
import argparse
from datetime import datetime, timezone
import hashlib
import http.client
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
REPO = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS / 'sponsors'))
sys.path.insert(0, str(HERE))
from common import canonical, digest, parse, require, snapshot  # noqa: E402
import prepare_av2  # noqa: E402  (Feather reading, exact rigid poses)

# ------------------------------------------------------------------ frozen pins and numbers
SHAPE_PROTOCOL_PATH = '.local/evaluation/pao-accuracy/shape-v2/protocol.json'
SHAPE_PROTOCOL_SHA256 = '70cbb24fabd1509bcb797d651ba2ec6c094c28c869b4a6ecd061f57a5106a95a'
SPLIT_ADDENDUM_PATH = '.local/evaluation/pao-accuracy/shape-v2/split-addendum.json'
SPLIT_ADDENDUM_SHA256 = '953dae5f24b7ff9ac66fc53001d677bd60c159fbf647f39b9f7966d227f21838'
CODE_PATH = 'scripts/california/car_shape.py'
SCHEMA = 'shape-v2'
PLAN_SCHEMA = 'shape-v2-sweep-plan'
RECEIPT_SCHEMA = 'shape-v2-download-receipt'
APPROACH_SCHEMA = 'shape-v2-approach-split'
PLACEMENT_PROTOCOL_VERSION = '3.4'
# carShape.sweeps.rule (protocol dataSchema): the selection and download numbers, recorded with every shape.
RULE = {'maxEgoDistanceM': 30, 'minEgoStepM': 0.25, 'maxKept': 40, 'split': 'alternate', 'minShapeSweeps': 2,
        'downloadAttempts': 4, 'retryWaitsS': [5, 10, 20]}
MIN_HEIGHT_M = 0.25      # returns.minHeightM: a 0.15 m curb plus 0.10 m
OTHER_GROW_M = 0.10      # returns.otherObjectGrowM
MAX_CELL_M = 0.1         # representation.maxCellM
CEIL_TOLERANCE = 1e-9    # representation.ceilTolerance
BEYOND_ENDS_M = 0.5      # returns.beyondEnds: counted within 0.5 m of each end (diagnostics only)
PREFILTER_SLACK_M = 1e-6  # only widens a speed-up pre-filter; every kept/counted return passes the exact tests below
# Placement protocol stationary test (same numbers as prepare_av2_ground.STATIONARY_SPAN_S / _DISPLACEMENT_M).
STATIONARY_SPAN_S, STATIONARY_DISPLACEMENT_M = 3.0, 1.0
FRAME_TEXT = ('x forward along the upright box heading (front face at x = +L/2), y to the car\'s left, z above the box bottom '
              '(box centre Y - H/2). Same as metric F upright_frame/car_local.')
GRID_ORDER = 'index = i*lateral + j; i from the front face, j from the left side'
REFUSALS = {'too-few-sweeps': 'fewer than 2 LiDAR sweeps of it were available',
            'not-stationary': 'it did not pass the stationary test'}
# Held-out tests, controls and reserve of the frozen Palo Alto split (same tuple as prepare_av2_ground.py).
HELD_OUT_LOG_PREFIXES = ('88f47a10', '3153b5b3', 'd33f667d', 'fd5c6932', '25e5c600', 'e1d68dde', 'a91d4c7b', '4a60c567', 'b6c4361a')
DATA_ROOT = REPO / '.local/data/av2'
SPLIT_PATH = REPO / '.local/evaluation/pao-accuracy/split.json'
SPLIT_SHA256 = '47f64fe007b51d75f4bf1c9891d940fd6393c777e0232a0a4f9428bbbeaafab4'
RUNNABLE_ROLES = ('development', 'development-optional', 'held-out-blocked-test', 'held-out-left-side-case',
                  'held-out-control', 'held-out-control-declared-only')
LOCK_PATH = REPO / '.local/evaluation/pao-accuracy/lock-2/protocol-lock.json'
SUPERSEDED_LOCK_PATH = REPO / '.local/evaluation/pao-accuracy/protocol-lock.json'
SUPERSEDED_LOCK_SHA256 = '7aa2945e61e676ffc3a2a207895ca5639e4d9b4405893b648f4412891c00a475'
# The frozen ground builder cannot accept new flags. M7 passes its explicit lock hash to this child only;
# build_car_shape still runs the full preflight, rather than trusting the environment as approval.
GROUND_LOCK_ENV = 'PAO_M7_GROUND_PROTOCOL_LOCK_SHA256'
APPROACH_DIR = REPO / '.local/evaluation/pao-accuracy/metric-f-shape-v2/approach-split'
PLAN_NAME, RECEIPT_NAME, LISTING_NAME, SWEEP_DIR = 'sweep-plan.json', 'receipt.json', 'listing.xml', 'sweeps'
BUCKET_URL = 'https://argoverse.s3.amazonaws.com/'


class SensorTemplate(str):
    """Preserve .format(log=...) for existing callers; the frozen split supplies the publisher folder."""
    def format(self, *, log):
        return super().format(split=av2_split(log), log=log)


LIDAR_PREFIX = SensorTemplate('datasets/av2/sensor/{split}/{log}/sensors/lidar/')
LISTING_URL = SensorTemplate('https://argoverse.s3.amazonaws.com/?list-type=2&prefix=datasets%2Fav2%2Fsensor%2F{split}%2F{log}'
               '%2Fsensors%2Flidar%2F&delimiter=%2F&max-keys=1000')
S3_NS = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
MAX_JSON = 2_000_000
MAX_META = 16_000_000
MAX_SWEEP = 16_000_000
POSE_COLUMNS = ('timestamp_ns', 'qw', 'qx', 'qy', 'qz', 'tx_m', 'ty_m', 'tz_m')
ANN_COLUMNS = ('timestamp_ns', 'track_uuid', 'length_m', 'width_m', 'height_m', 'qw', 'qx', 'qy', 'qz', 'tx_m', 'ty_m', 'tz_m')
BANNED = b'park'


def utc_now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def no_banned_word(data, what):
    require(BANNED not in data.lower(), f'{what} must not use any form of the banned vehicle-status word')


def write_exclusive(path, data, mode=0o444):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(data)


def check_log(log):
    require(isinstance(log, str) and re.fullmatch('[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', log),
            'Canonical lower-case AV2 log UUID required')
    return log


def split_metadata():
    raw = SPLIT_PATH.read_bytes()
    if digest(raw) != SPLIT_SHA256:
        raise SystemExit('split.json differs from its frozen sha256')
    logs = set(re.findall(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', raw.decode('utf-8')))
    require(len({log[:8] for log in logs}) == len(logs), 'split.json logs share an 8-hex prefix')
    return parse(raw)['sites'], logs


def av2_split(log):
    """Same sha-checked split lookup as M7, without importing evaluation code into the scene builder."""
    check_log(log)
    sites, _logs = split_metadata()
    entries = [s for s in sites if s['logId'] == log]
    if not entries:
        raise SystemExit(f'{log[:8]} is not a log of split.json')
    for entry in entries:
        if entry['role'] not in RUNNABLE_ROLES:
            raise SystemExit(f"{log[:8]} is {entry['role']}: reserve and other non-runnable roles stay unopened")
        if entry.get('av2Split') not in ('train', 'val'):
            raise SystemExit(f"{log[:8]} has unsupported av2Split {entry.get('av2Split')!r}; expected train or val")
    splits = {s['av2Split'] for s in entries}
    if len(splits) != 1:
        raise SystemExit(f'{log[:8]} has conflicting av2Split entries in split.json')
    return splits.pop()


def lidar_prefix(log):
    return LIDAR_PREFIX.format(log=log)


def listing_url(log):
    return LISTING_URL.format(log=log)


def plan_dir_for(log, out_root=None):
    return Path(out_root if out_root is not None else DATA_ROOT) / f'shape-v2-{check_log(log)}'


def guard_input_paths(log, *paths):
    """After the log guard, reject another split log in either the supplied path or its resolved components."""
    _sites, logs = split_metadata()
    foreign = {token for other in logs if other != log for token in (other, other[:8])}
    for value in paths:
        if value is not None:
            path = Path(value)
            require(not any(token in part.lower() for part in (*path.parts, *path.resolve().parts) for token in foreign),
                    'Input path names another split.json log')


# ------------------------------------------------------------------ held-out guard (before any file of the log is read)
def load_m7_common():
    # Never bind M7 helpers to "common": the builder verifies that name belongs to scripts/sponsors/common.py.
    spec = importlib.util.spec_from_file_location('_pao_lock2_m7_common', REPO / 'scripts/evaluation/m7/common.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_held_out(log, held_out_run=False, lock_sha256=None, need_addendum=False, lock_path=None):
    """Held-out/control access requires the full lock-2 preflight and the shape pins; reserve stays unopened.
    No child fetches: origin/main must already hold the pushed anchors. Development logs need no lock."""
    check_log(log)
    av2_split(log)  # metadata only; refuse reserve/test before any log file is read
    if not log.startswith(HELD_OUT_LOG_PREFIXES):
        return None
    require(held_out_run, 'Held-out, control or reserve logs run only in M7 with --held-out-run and a valid protocol-lock.json')
    require(isinstance(lock_sha256, str) and re.fullmatch('[0-9a-f]{64}', lock_sha256),
            'Held-out runs need --protocol-lock-sha256 (the sha256 of protocol-lock.json recorded at the lock)')
    path = Path(lock_path) if lock_path is not None else LOCK_PATH
    if lock_sha256 == SUPERSEDED_LOCK_SHA256 or path.resolve() == SUPERSEDED_LOCK_PATH.resolve():
        raise SystemExit('Lock 1 is superseded; use lock-2/protocol-lock.json')
    guard_input_paths(log, path)
    require(path.is_file(), f'Held-out runs need {path.name} at {path} (written at the lock); it does not exist')
    lock = parse(snapshot(path, lock_sha256))
    pipeline = lock.get('pipeline') if isinstance(lock.get('pipeline'), dict) else {}
    pin = lambda name: (pipeline.get(name) if isinstance(pipeline.get(name), dict) else {}).get('sha256')  # noqa: E731
    require(pin('shapeProtocol') == SHAPE_PROTOCOL_SHA256, 'protocol-lock.json does not name the frozen shape-v2 protocol (pipeline.shapeProtocol.sha256)')
    if need_addendum:
        require(pin('shapeSplitAddendum') == SPLIT_ADDENDUM_SHA256,
                'protocol-lock.json does not name the approach-half split addendum (pipeline.shapeSplitAddendum.sha256)')
    approval = lock.get('approval') if isinstance(lock.get('approval'), dict) else {}
    require(all(isinstance(approval.get(k), str) and approval[k].strip() for k in ('by', 'utc')),
            'protocol-lock.json needs an approval block (approval.by and approval.utc)')
    m7 = load_m7_common()
    require(path.resolve() == m7.LOCK_PATH.resolve(), 'Held-out runs require the fixed lock-2/protocol-lock.json path')
    m7.preflight(lock_sha256, fetch=False)
    return {'path': str(path), 'sha256': lock_sha256}


def guard_plan_dir(plan_dir, log=None, *, held_out_run=False, lock_sha256=None, need_addendum=False):
    """Identify and guard a plan directory before opening its manifest or any payload. No ambient cached approval."""
    folder = Path(plan_dir)
    prefix = SCHEMA + '-'
    require(folder.name.startswith(prefix), 'A shape-v2-<log> plan directory is required before reading inputs')
    named_log = check_log(folder.name[len(prefix):])
    require(log is None or log == named_log, 'Plan directory is for another log')
    check_held_out(named_log, held_out_run, lock_sha256, need_addendum=need_addendum)
    require(folder.resolve().name == folder.name, 'Plan directory resolves to another log or directory')
    guard_input_paths(named_log, folder, *(folder / name for name in (PLAN_NAME, RECEIPT_NAME, LISTING_NAME)))
    return named_log


SWEEP_URL_RE = re.compile(re.escape(BUCKET_URL) + r'datasets/av2/sensor/(train|val)/([0-9a-f-]+)/sensors/lidar/[0-9]{1,32}\.feather')
LISTING_URL_RE = re.compile(re.escape(BUCKET_URL) + r'\?list-type=2&prefix=datasets%2Fav2%2Fsensor%2F(train|val)%2F([0-9a-f-]+)'
                            r'%2Fsensors%2Flidar%2F&delimiter=%2F&max-keys=1000')


def guard_url(url, *, held_out_run=False, lock_sha256=None):
    """Guard the log of the only two URL forms fetched, including direct network-helper calls: a lidar sweep object
    (BUCKET_URL + lidar_prefix(log) + <timestamp>.feather, no query) or the exact listing_url(log). Any other form (an
    object path with a query, a reordered or extra query field, a fragment, percent-encoding in an object path) is
    refused, so the log checked is the log requested."""
    require(isinstance(url, str) and url.startswith(BUCKET_URL), 'Only the public Argoverse bucket is fetched')
    match = SWEEP_URL_RE.fullmatch(url) or LISTING_URL_RE.fullmatch(url)
    require(match is not None, 'Only an AV2 lidar sweep URL or the exact lidar listing URL is fetched')
    log = check_log(match[2])
    check_held_out(log, held_out_run, lock_sha256)
    require(match[1] == av2_split(log), 'URL is for another publisher split')
    require(url == listing_url(log) if match.re is LISTING_URL_RE else url.startswith(BUCKET_URL + lidar_prefix(log)),
            'URL differs from the canonical URL for its log')


# ------------------------------------------------------------------ listing and downloads (protocol 6.3)
def _text(node, name):
    child = node.find(f's3:{name}', S3_NS)
    require(child is not None and child.text is not None, f'Listing field {name} missing')
    return child.text


def parse_listing(data, log):
    """Public S3 ListBucketResult of sensors/lidar/ for this log -> {timestamp: {key, bytes, etag}}. Complete listings
    only (not truncated, KeyCount equal to the entries); every key must be <prefix><timestamp>.feather."""
    prefix = lidar_prefix(log)
    doc = ET.fromstring(data)
    require(doc.tag == '{%s}ListBucketResult' % S3_NS['s3'], 'S3 ListBucketResult required')
    require(_text(doc, 'Prefix') == prefix, 'Listing is for another log or sensor')
    require(_text(doc, 'IsTruncated') == 'false', 'Truncated listing')
    contents = doc.findall('s3:Contents', S3_NS)
    require(re.fullmatch('[0-9]+', _text(doc, 'KeyCount')) and int(_text(doc, 'KeyCount')) == len(contents), 'Listing count mismatch')
    entries = {}
    for node in contents:
        key = _text(node, 'Key')
        match = re.fullmatch(re.escape(prefix) + r'([0-9]{1,32})\.feather', key)
        require(match is not None, 'Unexpected listing key')
        ts = int(match.group(1)); require(ts not in entries, 'Duplicate listing timestamp')
        etag = _text(node, 'ETag'); size = _text(node, 'Size')
        require(len(etag) >= 2 and etag[0] == etag[-1] == '"' and re.fullmatch('[0-9a-f]{32}(-[0-9]+)?', etag[1:-1]), 'Unexpected listing ETag')
        require(re.fullmatch('[0-9]+', size), 'Unexpected listing size')
        entries[ts] = {'key': key, 'bytes': int(size), 'etag': etag[1:-1]}
    return entries


def http_fetch(url, timeout=120, *, held_out_run=False, lock_sha256=None):
    """GET one public Argoverse URL -> (status, Content-Length or None, ETag header or None, body bytes).
    An HTTP error status is returned; a network error raises OSError."""
    guard_url(url, held_out_run=held_out_run, lock_sha256=lock_sha256)
    request = urllib.request.Request(url, headers={'User-Agent': 'umbra-car-shape/1'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(MAX_SWEEP + 1)
            length = response.headers.get('Content-Length')
            return response.status, int(length) if length and length.isdigit() else None, response.headers.get('ETag'), body
    except urllib.error.HTTPError as error:
        return error.code, None, None, b''
    except http.client.HTTPException as error:  # e.g. IncompleteRead: a network failure, not a status
        raise OSError(str(error)) from error


def fetch_with_rule(url, accept, fetch, sleep, *, held_out_run=False, lock_sha256=None):
    """The frozen attempt rule: at most RULE['downloadAttempts'] attempts, waiting retryWaitsS between them. An attempt
    succeeds only with status 200 and accept(body) true; a network error, any other status or a rejected body is a
    failed attempt (its bytes are discarded). Returns {ok, attempts, status, contentLength, data}."""
    guard_url(url, held_out_run=held_out_run, lock_sha256=lock_sha256)
    last = {'status': None, 'contentLength': None}
    for attempt in range(1, RULE['downloadAttempts'] + 1):
        try:
            if fetch is http_fetch:
                status, length, _etag, body = fetch(url, held_out_run=held_out_run, lock_sha256=lock_sha256)
            else:
                status, length, _etag, body = fetch(url)
        except OSError:
            status, length, body = None, None, None
        last = {'status': status, 'contentLength': length}
        if status == 200 and body is not None and accept(body):
            return {'ok': True, 'attempts': attempt, **last, 'data': body}
        if attempt < RULE['downloadAttempts']:
            sleep(RULE['retryWaitsS'][attempt - 1])
    return {'ok': False, 'attempts': RULE['downloadAttempts'], **last, 'data': None}


def byte_check(entry, data):
    """The download success test on bytes: size equals the listing size and, for a single-part ETag (32 hex, no '-'),
    MD5 equals the ETag; a multipart ETag checks the size only (md5MatchesEtag null)."""
    md5 = hashlib.md5(data).hexdigest()
    single = re.fullmatch('[0-9a-f]{32}', entry['etag']) is not None
    size_ok = len(data) == entry['bytes']
    md5_ok = (md5 == entry['etag']) if single else None
    return {'ok': size_ok and md5_ok is not False, 'sizeMatches': size_ok, 'md5': md5, 'md5MatchesEtag': md5_ok}


def fetch_listing(log, fetch, sleep, *, held_out_run=False, lock_sha256=None):
    """The lidar listing under the same attempt rule. Returns (data or None, {status 'ok' | 'failed', attempts})."""
    check_held_out(log, held_out_run, lock_sha256)
    def accept(body):
        try:
            parse_listing(body, log)
            return True
        except (ValueError, ET.ParseError):
            return False
    result = fetch_with_rule(listing_url(log), accept, fetch, sleep, held_out_run=held_out_run, lock_sha256=lock_sha256)
    return result['data'], {'status': 'ok' if result['ok'] else 'failed', 'attempts': result['attempts']}


# ------------------------------------------------------------------ metadata: poses, annotations, stationary test
def load_poses(data):
    """city_SE3_egovehicle rows: sorted arrays (timestamps strictly increasing, at least 2, finite) for the interpolated
    pose (6.2) and an exact rigid matrix per timestamp (prepare_av2.rigid, as prepare_av2_ground._poses)."""
    rows = prepare_av2.table(data, list(POSE_COLUMNS)).select(list(POSE_COLUMNS)).to_pylist()
    rigid = {}
    for row in rows:
        require(type(row['timestamp_ns']) is int and row['timestamp_ns'] not in rigid, 'Invalid/duplicate pose timestamp')
        rigid[row['timestamp_ns']] = prepare_av2.rigid(row)
    t = np.array([r['timestamp_ns'] for r in rows], np.int64); order = np.argsort(t, kind='stable'); t = t[order]
    q = np.array([[r[k] for k in ('qw', 'qx', 'qy', 'qz')] for r in rows], float).reshape(-1, 4)[order]
    p = np.array([[r[k] for k in ('tx_m', 'ty_m', 'tz_m')] for r in rows], float).reshape(-1, 3)[order]
    require(len(t) >= 2 and bool(np.all(np.diff(t) > 0)) and np.isfinite(q).all() and np.isfinite(p).all(),
            'Pose record needs at least 2 finite poses with strictly increasing timestamps')
    return {'t': t, 'q': q, 'p': p, 'rigid': rigid}


def ego_poses_at(times, poses):
    """Ego pose at each time, linearly interpolated between the two nearest recorded poses (normalized linear quaternion
    interpolation): metric F capture_poses applied to the sweep timestamp. Returns (quaternions w,x,y,z; translations)."""
    t, q, p = poses['t'], poses['q'], poses['p']
    times = np.asarray(times, np.int64).reshape(-1)
    require(bool((times >= t[0]).all() and (times <= t[-1]).all()), 'Sweep time outside the pose record')
    i = np.clip(np.searchsorted(t, times, side='right') - 1, 0, len(t) - 2)
    a = ((times - t[i]) / (t[i + 1] - t[i]))[:, None]
    q0, q1 = q[i], q[i + 1]; q1 = np.where((q0 * q1).sum(1, keepdims=True) < 0, -q1, q1)
    qi = (1 - a) * q0 + a * q1; qi /= np.linalg.norm(qi, axis=1, keepdims=True)
    return qi, (1 - a) * p[i] + a * p[i + 1]


def load_annotations(data):
    table = prepare_av2.table(data, list(ANN_COLUMNS))
    ann = {k: np.asarray(table.column(k).to_numpy(zero_copy_only=False)) for k in ANN_COLUMNS}
    ann['timestamp_ns'] = ann['timestamp_ns'].astype(np.int64)
    for k in ANN_COLUMNS[2:]:
        ann[k] = ann[k].astype(np.float64)
    ann['track_uuid'] = np.array([str(v) for v in ann['track_uuid']], dtype=object)
    return ann


def stationary(ann, rigid, track):
    """The placement protocol's stationary test, exactly as prepare_av2_ground.cuboids_from_annotations applies it: all
    annotations of the track in time order (stable), the exact ego pose at each; city XY = (R [tx, ty, tz] + t)[:2];
    span >= 3.0 s and the largest horizontal distance from the first position <= 1.0 m. A missing exact pose fails."""
    rows = np.flatnonzero(ann['track_uuid'] == track)
    rows = rows[np.argsort(ann['timestamp_ns'][rows], kind='stable')]
    stamps = [int(ann['timestamp_ns'][i]) for i in rows]
    if not stamps or not all(ts in rigid for ts in stamps):
        return {'passed': False, 'rows': len(stamps), 'spanSeconds': None, 'maxDisplacementFromFirstM': None}
    xy = np.array([(rigid[ts][:3, :3] @ np.array([ann['tx_m'][i], ann['ty_m'][i], ann['tz_m'][i]], float) + rigid[ts][:3, 3])[:2]
                   for ts, i in zip(stamps, rows)])
    span = (stamps[-1] - stamps[0]) / 1e9; moved = float(np.linalg.norm(xy - xy[0], axis=1).max())
    return {'passed': span >= STATIONARY_SPAN_S and moved <= STATIONARY_DISPLACEMENT_M, 'rows': len(stamps),
            'spanSeconds': span, 'maxDisplacementFromFirstM': moved}


def other_rows(ann, ts, track):
    """Every other track's annotation at this sweep (their cuboids remove returns; the vehicle's own rows never do)."""
    idx = np.flatnonzero((ann['timestamp_ns'] == ts) & (ann['track_uuid'] != track))
    return [{k: float(ann[k][i]) for k in ANN_COLUMNS[2:]} for i in idx]


# ------------------------------------------------------------------ frames (same arithmetic as metric F)
def quat_to_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rotate(q, v):
    """Rotate vectors v by unit quaternions q (w, x, y, z), row by row."""
    w, u = q[:, :1], q[:, 1:]
    v = np.broadcast_to(np.asarray(v, float), u.shape); uv = np.cross(u, v)
    return v + 2 * w * uv + 2 * np.cross(u, uv)


def world(city, origin):
    """City frame -> scene frame: X = cityX - originX, Y = cityZ - originZ (up), Z = -(cityY - originY)."""
    return np.column_stack([city[:, 0] - origin[0], city[:, 2] - origin[2], -(city[:, 1] - origin[1])])


def upright_frame(box):
    """The averaged box as the app tests it: cuboid centre, horizontal heading from worldFromBox column 0 (pitch and roll
    dropped), unrounded length, width and height."""
    m = np.asarray(box['worldFromBox'], float); n = math.hypot(m[0][0], m[2][0])
    heading = np.array([m[0][0] / n, 0., m[2][0] / n])
    return {'center': np.asarray(box['center'], float), 'heading': heading, 'left': np.array([heading[2], 0., -heading[0]]),
            'L': float(box['length']), 'W': float(box['width']), 'H': float(box['height'])}


def car_local(points, frame):
    """Scene points -> car-local [x forward, y left, z above the box bottom]."""
    d = np.asarray(points, float) - frame['center']
    return np.column_stack([d @ frame['heading'], d @ frame['left'], d[:, 1] + frame['H'] / 2])


def inside_cuboids(ego_points, rows, grow):
    """Returns (ego frame) inside any of these annotated cuboids grown by `grow` on every side."""
    inside = np.zeros(len(ego_points), bool)
    for r in rows:
        rot = quat_to_matrix([r['qw'], r['qx'], r['qy'], r['qz']]); p = (ego_points - np.array([r['tx_m'], r['ty_m'], r['tz_m']])) @ rot
        inside |= (np.abs(p) <= np.array([r['length_m'], r['width_m'], r['height_m']]) / 2 + grow).all(1)
    return inside


def grid_size(length, width):
    """Cells along (n) and across (m): ceil(size / 0.10 - 1e-9)."""
    return math.ceil(length / MAX_CELL_M - CEIL_TOLERANCE), math.ceil(width / MAX_CELL_M - CEIL_TOLERANCE)


def centre_city_xy(box, origin):
    """The averaged box centre in the city frame: cityX = X + originX, cityY = originY - Z."""
    return float(box['center'][0]) + float(origin[0]), float(origin[1]) - float(box['center'][2])


def box_record(box):
    """The averaged box values a sweep plan is made for (compared exactly at build time)."""
    return {'track': box['track'], 'center': [float(v) for v in box['center']], 'worldFromBox': [[float(v) for v in r] for r in box['worldFromBox']],
            'length': float(box['length']), 'width': float(box['width']), 'height': float(box['height'])}


# ------------------------------------------------------------------ sweep selection (protocol 6.1)
def select_sweeps(listing, ann, track, poses, centre_xy):
    """The frozen selection: candidates (track row at ts, ts inside the pose record, ego origin within 30 m of the box
    centre), thinning (0.25 m ego step), cap (40 evenly spaced positions, Python round) and the alternating split.
    Returns {'candidates', 'thinned', 'kept': [{timestampNs:int, set, egoDistanceM}]}."""
    own = set(int(v) for v in ann['timestamp_ns'][ann['track_uuid'] == track])
    t0, tn = int(poses['t'][0]), int(poses['t'][-1])
    stamps = [ts for ts in sorted(listing) if ts in own and t0 <= ts <= tn]
    candidates = []
    if stamps:
        _q, trans = ego_poses_at(stamps, poses)
        distance = np.hypot(trans[:, 0] - centre_xy[0], trans[:, 1] - centre_xy[1])
        candidates = [(ts, trans[k, :2], float(distance[k])) for k, ts in enumerate(stamps) if distance[k] <= RULE['maxEgoDistanceM']]
    thinned, last = [], None
    for ts, xy, dist in candidates:
        if last is None or float(np.hypot(xy[0] - last[0], xy[1] - last[1])) >= RULE['minEgoStepM']:
            thinned.append((ts, dist)); last = xy
    count, cap = len(thinned), RULE['maxKept']
    kept = thinned if count <= cap else [thinned[round(k * (count - 1) / (cap - 1))] for k in range(cap)]
    return {'candidates': len(candidates), 'thinned': count,
            'kept': [{'timestampNs': ts, 'set': 'shape' if k % 2 == 0 else 'scoring', 'egoDistanceM': dist} for k, (ts, dist) in enumerate(kept)]}


def plan_document(log, box, origin, listing, listing_status, listing_input, ann, poses, inputs):
    """The sweep plan (no times, no absolute paths: the same inputs always give the same bytes)."""
    track = box['track']
    chosen = select_sweeps(listing, ann, track, poses, centre_city_xy(box, origin))
    sweeps = [{'timestampNs': str(s['timestampNs']), 'set': s['set'], 'egoDistanceM': s['egoDistanceM'],
               **listing[s['timestampNs']]} for s in chosen['kept']]
    plan = {'schema': PLAN_SCHEMA, 'shapeProtocol': {'path': SHAPE_PROTOCOL_PATH, 'sha256': SHAPE_PROTOCOL_SHA256},
            'log': log, 'track': track, 'rule': dict(RULE), 'box': box_record(box), 'cityOrigin': [float(v) for v in origin],
            'boxCentreCityXY': list(centre_city_xy(box, origin)),
            'inputs': {**inputs, 'listing': listing_input}, 'listing': {**listing_status, 'objects': len(listing)},
            'candidates': chosen['candidates'], 'thinned': chosen['thinned'], 'kept': len(sweeps),
            'shape': [s['timestampNs'] for s in sweeps if s['set'] == 'shape'],
            'scoring': [s['timestampNs'] for s in sweeps if s['set'] == 'scoring'],
            'sweeps': sweeps,
            'note': ('Written and hashed before any sweep is downloaded (shape-v2 6.1). The split is never redone; a sweep '
                     'whose download fails is dropped and listed, not replaced. Choosing read only the public listing, '
                     'the annotations, the ego poses and the averaged box of ground.json.')}
    data = (json.dumps(plan, indent=1, sort_keys=True, allow_nan=False) + '\n').encode()
    no_banned_word(data, 'The sweep plan')
    return plan, data


def check_plan(plan, log=None):
    require(isinstance(plan, dict) and plan.get('schema') == PLAN_SCHEMA, 'Not a shape-v2 sweep plan')
    require(log is None or plan.get('log') == log, 'Sweep plan is for another log')
    require(plan.get('shapeProtocol', {}).get('sha256') == SHAPE_PROTOCOL_SHA256, 'Sweep plan made under another shape protocol')
    require(plan.get('rule') == RULE, 'Sweep plan rule numbers differ from shape-v2')
    prefix = lidar_prefix(plan['log'])
    require(plan['inputs']['listing']['url'] == listing_url(plan['log']), 'Sweep plan listing URL for another log or split')
    sweeps = plan['sweeps']
    require(isinstance(sweeps, list) and len(sweeps) == plan['kept'] <= RULE['maxKept'], 'Sweep plan kept count mismatch')
    stamps = [int(s['timestampNs']) for s in sweeps]
    require(stamps == sorted(set(stamps)) and all(s['timestampNs'] == str(int(s['timestampNs'])) for s in sweeps), 'Sweep plan timestamps must increase')
    require([s['set'] for s in sweeps] == ['shape' if k % 2 == 0 else 'scoring' for k in range(len(sweeps))], 'Sweep plan split is not alternating')
    require(all(s['key'] == prefix + f"{s['timestampNs']}.feather" for s in sweeps), 'Sweep plan key for another log or split')
    return plan


def make_plan(log, ground_path, ground_sha256, annotations_path, annotations_sha256, poses_path, poses_sha256,
              listing_path=None, listing_sha256=None, out_root=None, fetch=http_fetch, sleep=time.sleep, *,
              held_out_run=False, lock_sha256=None):
    """Write <out_root>/shape-v2-<log>/sweep-plan.json (exclusive) plus listing.xml; returns (plan_dir, plan sha256).
    Reads metadata only. The vehicle is the single averaged stationary vehicle (poseAverage cuboid) of ground.json."""
    plan_dir = plan_dir_for(log, out_root)
    guard_plan_dir(plan_dir, log, held_out_run=held_out_run, lock_sha256=lock_sha256)
    guard_input_paths(log, ground_path, annotations_path, poses_path, listing_path)
    url = listing_url(log)
    for path in (annotations_path, poses_path):
        require(Path(path).resolve().parent.name == log, 'Annotations and poses must come from the log directory of this log')
    ground_raw = snapshot(ground_path, ground_sha256, 1024 * 1024); ground = parse(ground_raw)
    ann_raw = snapshot(annotations_path, annotations_sha256, MAX_META); poses_raw = snapshot(poses_path, poses_sha256, MAX_META)
    sources = ground.get('sources') or {}
    require((sources.get('annotations') or {}).get('sha256') == annotations_sha256 and (sources.get('poses') or {}).get('sha256') == poses_sha256,
            'ground.json was built from other annotations or poses')
    averaged = [c for c in ground['cuboids']['items'] if 'poseAverage' in c]
    require(len(averaged) == 1, 'No single averaged stationary vehicle in ground.json: the car shape does not apply (no sweep plan, no download)')
    box = averaged[0]; origin = np.asarray(ground['frame']['cityOrigin'], float)
    require(origin.shape == (3,) and np.isfinite(origin).all(), 'ground.json city origin required')
    require(not os.path.lexists(plan_dir / PLAN_NAME), 'Existing sweep plan retained (a new plan needs a new directory)')
    if listing_path is not None:
        listing_raw = snapshot(listing_path, listing_sha256, MAX_JSON)
        listing_status = {'status': 'ok', 'attempts': 0}
        listing_input = {'source': 'saved file', 'file': LISTING_NAME, 'sha256': digest(listing_raw), 'url': url}
    else:
        listing_raw, listing_status = fetch_listing(log, fetch, sleep, held_out_run=held_out_run, lock_sha256=lock_sha256)
        listing_input = {'source': 'fetched', 'file': LISTING_NAME if listing_raw is not None else None,
                         'sha256': digest(listing_raw) if listing_raw is not None else None, 'url': url}
    listing = parse_listing(listing_raw, log) if listing_raw is not None else {}
    require(listing_raw is None or listing, 'The lidar listing of this log is empty (wrong log or split)')
    inputs = {'ground': {'sha256': ground_sha256}, 'annotations': {'sha256': annotations_sha256}, 'poses': {'sha256': poses_sha256}}
    _plan, data = plan_document(log, box, origin, listing, listing_status, listing_input, load_annotations(ann_raw), load_poses(poses_raw), inputs)
    plan_dir.mkdir(parents=True, exist_ok=True)
    if listing_raw is not None:
        write_exclusive(plan_dir / LISTING_NAME, listing_raw)
    write_exclusive(plan_dir / PLAN_NAME, data)
    return plan_dir, digest(data)


def download_sweeps(plan_dir, plan_sha256, fetch=http_fetch, sleep=time.sleep, *, log=None,
                    held_out_run=False, lock_sha256=None):
    """Download the planned sweeps one at a time under the frozen attempt rule and write receipt.json (exclusive).
    A sweep already on disk is reused only when it passes the same success test."""
    plan_dir = Path(plan_dir)
    log = guard_plan_dir(plan_dir, log, held_out_run=held_out_run, lock_sha256=lock_sha256)
    plan = check_plan(parse(snapshot(plan_dir / PLAN_NAME, plan_sha256, MAX_JSON)), log)
    require(not os.path.lexists(plan_dir / RECEIPT_NAME), 'Downloads already receipted for this plan')
    folder = plan_dir / SWEEP_DIR; folder.mkdir(exist_ok=True)
    started = utc_now(); records, dropped = [], []
    for entry in plan['sweeps']:
        ts = entry['timestampNs']; path = folder / f'{ts}.feather'; url = BUCKET_URL + entry['key']
        guard_input_paths(log, path)
        base = {'timestampNs': ts, 'set': entry['set'], 'file': f'{SWEEP_DIR}/{ts}.feather', 'key': entry['key'], 'url': url, 'etag': entry['etag']}
        if path.is_file():
            data = path.read_bytes(); check = byte_check(entry, data)
            if check['ok']:
                records.append({**base, 'source': 'disk', 'attempts': 0, 'status': None, 'contentLength': None, 'bytes': len(data),
                                'sha256': digest(data), **{k: check[k] for k in ('md5', 'md5MatchesEtag', 'sizeMatches')}})
                continue
        result = fetch_with_rule(url, lambda body, e=entry: byte_check(e, body)['ok'], fetch, sleep,
                                 held_out_run=held_out_run, lock_sha256=lock_sha256)
        if not result['ok']:
            dropped.append({'timestampNs': ts, 'set': entry['set'], 'reason': 'download-failed', 'attempts': RULE['downloadAttempts']})
            continue
        data = result['data']; check = byte_check(entry, data)
        with tempfile.NamedTemporaryFile(dir=folder, prefix='.partial-', delete=False) as handle:
            handle.write(data); temporary = handle.name
        os.replace(temporary, path)
        records.append({**base, 'source': 'download', 'attempts': result['attempts'], 'status': result['status'],
                        'contentLength': result['contentLength'], 'bytes': len(data), 'sha256': digest(data),
                        **{k: check[k] for k in ('md5', 'md5MatchesEtag', 'sizeMatches')}})
    receipt = {'schema': RECEIPT_SCHEMA, 'planSha256': plan_sha256, 'log': plan['log'], 'track': plan['track'], 'rule': dict(RULE),
               'startedUtc': started, 'finishedUtc': utc_now(), 'sweeps': records, 'downloadDropped': dropped,
               'bytesOnDisk': sum(r['bytes'] for r in records),
               'note': 'Public Argoverse 2 files ($0). Success = HTTP 200, listing size, and MD5 = ETag for a single-part ETag.'}
    data = (json.dumps(receipt, indent=1, sort_keys=True) + '\n').encode()
    no_banned_word(data, 'The download receipt')
    write_exclusive(plan_dir / RECEIPT_NAME, data)
    return receipt


# ------------------------------------------------------------------ the shape (protocol 5.3, 5.4, 6.2, 6.4)
def read_sweep(data):
    """x, y, z (ego frame, float16 in AV2) widened exactly to float64; every value must be finite."""
    table = prepare_av2.table(data, ['x', 'y', 'z'])
    points = np.column_stack([np.asarray(table.column(k).to_numpy(zero_copy_only=False), np.float64) for k in ('x', 'y', 'z')])
    require(np.isfinite(points).all(), 'Non-finite LiDAR return in a sweep')
    return points


def floor_mm(z):
    """floor(z * 1000) / 1000 (protocol representation.rounding), keeping its stated reason that rounding never enlarges
    the shape: for a z just below a whole millimetre the float product z * 1000 can round up to that millimetre, giving a
    value above z; then the millimetre below is used. Otherwise this is exactly the float formula (1.2 stays 1.2)."""
    k = math.floor(z * 1000)
    while k / 1000 > z:
        k -= 1
    return k / 1000


def shape_from_sweeps(sweeps, box, origin, ann, poses):
    """Solid top per cell from (timestamp, ego points) sweeps registered into the upright averaged box."""
    frame = upright_frame(box); L, W, H = frame['L'], frame['W'], frame['H']
    n, m = grid_size(L, W); a, b = L / n, W / m; cells = n * m; track = box['track']
    returns = np.zeros(cells, np.int64); maxima = np.full((len(sweeps), cells), -np.inf)
    removed = {'belowFloor': 0, 'otherObjects': 0, 'aboveBoxTop': 0}; beyond = {'front': 0, 'rear': 0}
    for k, (ts, ego) in enumerate(sweeps):
        q, t = ego_poses_at([ts], poses)
        city = rotate(np.repeat(q, len(ego), 0), ego) + t
        local = car_local(world(city, origin), frame)
        near = (np.abs(local[:, 1]) <= W / 2) & (np.abs(local[:, 0]) <= L / 2 + BEYOND_ENDS_M + PREFILTER_SLACK_M)
        x, y, z = local[near, 0], local[near, 1], local[near, 2]
        other = inside_cuboids(ego[near], other_rows(ann, ts, track), OTHER_GROW_M)
        lateral = np.abs(y) <= W / 2; high = z >= MIN_HEIGHT_M; capped = z <= H
        footprint = lateral & (x >= -L / 2) & (x <= L / 2)
        removed['belowFloor'] += int((footprint & ~high).sum())
        removed['otherObjects'] += int((footprint & high & other).sum())
        removed['aboveBoxTop'] += int((footprint & high & ~other & ~capped).sum())
        kept = footprint & high & capped & ~other
        usable = lateral & high & capped & ~other
        for end, dist in (('front', x - L / 2), ('rear', -L / 2 - x)):
            beyond[end] += int((usable & (dist > 0) & (dist <= BEYOND_ENDS_M)).sum())
        i = np.clip(np.floor((L / 2 - x[kept]) / a).astype(np.int64), 0, n - 1)
        j = np.clip(np.floor((W / 2 - y[kept]) / b).astype(np.int64), 0, m - 1)
        index = i * m + j
        returns += np.bincount(index, minlength=cells)
        np.maximum.at(maxima[k], index, z[kept])
    seen = (maxima > -np.inf).sum(0)
    second = np.sort(maxima, axis=0)[-2] if len(sweeps) >= 2 else np.full(cells, -np.inf)
    tops = [floor_mm(float(second[c])) if seen[c] >= 2 else None for c in range(cells)]
    first = next((i for i in range(n) if any(tops[i * m + j] is not None for j in range(m))), None)
    return {'solidTopM': tops, 'counts': {'returns': [int(v) for v in returns], 'returnSweeps': [int(v) for v in seen]},
            'removed': removed, 'diagnostics': {'frontFaceToFirstSolidCellM': None if first is None else first * a, 'returnsBeyondEnds': beyond}}


def car_shape_block(box, sweeps_record, placement_protocol_sha256, shape=None, refusal=None):
    """The carShape block (protocol dataSchema). A refused shape has every top null and every count zero."""
    L, W, H = float(box['length']), float(box['width']), float(box['height']); n, m = grid_size(L, W); cells = n * m
    require((shape is None) == (refusal is not None) and (refusal is None or refusal in REFUSALS), 'A shape is either measured or refused')
    if shape is None:
        shape = {'solidTopM': [None] * cells, 'counts': {'returns': [0] * cells, 'returnSweeps': [0] * cells},
                 'removed': {'belowFloor': 0, 'otherObjects': 0, 'aboveBoxTop': 0},
                 'diagnostics': {'frontFaceToFirstSolidCellM': None, 'returnsBeyondEnds': {'front': 0, 'rear': 0}}}
    block = {'schema': SCHEMA, 'status': 'measured' if refusal is None else 'refused',
             'refusal': None if refusal is None else {'code': refusal, 'reason': REFUSALS[refusal]},
             'track': box['track'], 'box': {'length': L, 'width': W, 'height': H}, 'frame': FRAME_TEXT,
             'grid': {'along': n, 'lateral': m, 'cellAlongM': L / n, 'cellLateralM': W / m, 'maxCellM': MAX_CELL_M, 'order': GRID_ORDER},
             'solidTopM': shape['solidTopM'], 'counts': shape['counts'], 'removed': shape['removed'], 'sweeps': sweeps_record,
             'diagnostics': shape['diagnostics'],
             'protocol': {'placementProtocolVersion': PLACEMENT_PROTOCOL_VERSION, 'sha256': placement_protocol_sha256,
                          'shapeProtocolSha256': SHAPE_PROTOCOL_SHA256},
             'code': {'path': CODE_PATH, 'sha256': digest(Path(__file__).read_bytes())}}
    validate_block(block, box)
    no_banned_word(canonical(block), 'The car shape')
    return block


def validate_block(s, box):
    """The loader's carShape checks (src/scene.mjs validateCarShape), applied before anything is written."""
    hex64 = lambda v: isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v) is not None  # noqa: E731
    count = lambda v: type(v) is int and 0 <= v <= 2 ** 53 - 1  # noqa: E731
    decimal = lambda v: isinstance(v, str) and re.fullmatch('[0-9]{1,32}', v) is not None  # noqa: E731
    keys = lambda d, names: isinstance(d, dict) and sorted(d) == sorted(names)  # noqa: E731
    require(keys(s, ['box', 'code', 'counts', 'diagnostics', 'frame', 'grid', 'protocol', 'refusal', 'removed', 'schema', 'solidTopM',
                     'status', 'sweeps', 'track']), 'car shape fields')
    L, W, H = box['length'], box['width'], box['height']; n, m = grid_size(L, W); cells = n * m
    require(s['schema'] == SCHEMA and s['status'] in ('measured', 'refused') and s['track'] == box['track'] and s['frame'] == FRAME_TEXT, 'car shape identity')
    require(s['box'] == {'length': L, 'width': W, 'height': H}, 'car shape box')
    require(s['grid'] == {'along': n, 'lateral': m, 'cellAlongM': L / n, 'cellLateralM': W / m, 'maxCellM': MAX_CELL_M, 'order': GRID_ORDER}, 'car shape grid')
    c = s['counts']
    require(keys(c, ['returnSweeps', 'returns']) and all(isinstance(c[k], list) and len(c[k]) == cells and all(count(v) for v in c[k]) for k in c)
            and all(k <= r for k, r in zip(c['returnSweeps'], c['returns'])), 'car shape counts')
    require(isinstance(s['solidTopM'], list) and len(s['solidTopM']) == cells and
            all(v is None or (type(v) is float and math.isfinite(v) and MIN_HEIGHT_M <= v <= H and c['returnSweeps'][i] >= 2)
                for i, v in enumerate(s['solidTopM'])), 'car shape solid tops')
    require(keys(s['removed'], ['aboveBoxTop', 'belowFloor', 'otherObjects']) and all(count(v) for v in s['removed'].values()), 'car shape removed counts')
    w = s['sweeps']
    require(keys(w, ['candidates', 'downloadDropped', 'kept', 'listing', 'planSha256', 'rule', 'scoring', 'shape', 'thinned']) and w['rule'] == RULE, 'car shape sweep rule')
    require(hex64(w['planSha256']) and keys(w['listing'], ['attempts', 'status']) and w['listing']['status'] in ('ok', 'failed') and
            count(w['listing']['attempts']) and all(count(w[k]) for k in ('candidates', 'thinned', 'kept')), 'car shape sweep plan record')
    used = lambda lst: (isinstance(lst, list) and all(keys(x, ['sha256', 'timestampNs']) and decimal(x['timestampNs']) and hex64(x['sha256']) for x in lst)  # noqa: E731
                        and all(int(lst[i]['timestampNs']) > int(lst[i - 1]['timestampNs']) for i in range(1, len(lst))))
    require(used(w['shape']) and used(w['scoring']), 'car shape sweep lists')
    require(isinstance(w['downloadDropped'], list) and all(keys(x, ['attempts', 'reason', 'set', 'timestampNs']) and decimal(x['timestampNs']) and
                                                           x['set'] in ('shape', 'scoring') and x['reason'] == 'download-failed' and x['attempts'] == 4
                                                           for x in w['downloadDropped']), 'car shape dropped downloads')
    stamps = [int(x['timestampNs']) for x in w['shape'] + w['scoring'] + w['downloadDropped']]
    require(len(set(stamps)) == len(stamps) and len(stamps) == w['kept'], 'car shape sweep lists must be disjoint and add up to kept')
    if s['status'] == 'measured':
        require(s['refusal'] is None and len(w['shape']) >= RULE['minShapeSweeps'], 'measured needs no refusal and at least 2 shape sweeps')
    else:
        require(keys(s['refusal'], ['code', 'reason']) and REFUSALS.get(s['refusal']['code']) == s['refusal']['reason'] and
                all(v is None for v in s['solidTopM']) and not any(c['returns']) and not any(c['returnSweeps']), 'refused shape content')
    d = s['diagnostics']
    require(keys(d, ['frontFaceToFirstSolidCellM', 'returnsBeyondEnds']) and
            (d['frontFaceToFirstSolidCellM'] is None or (type(d['frontFaceToFirstSolidCellM']) is float and math.isfinite(d['frontFaceToFirstSolidCellM']))) and
            keys(d['returnsBeyondEnds'], ['front', 'rear']) and all(count(v) for v in d['returnsBeyondEnds'].values()), 'car shape diagnostics')
    require(keys(s['protocol'], ['placementProtocolVersion', 'sha256', 'shapeProtocolSha256']) and s['protocol']['placementProtocolVersion'] == PLACEMENT_PROTOCOL_VERSION
            and hex64(s['protocol']['sha256']) and s['protocol']['shapeProtocolSha256'] == SHAPE_PROTOCOL_SHA256, 'car shape protocol record')
    require(keys(s['code'], ['path', 'sha256']) and s['code']['path'] == CODE_PATH and hex64(s['code']['sha256']), 'car shape code record')


def load_receipted(plan_dir, plan_sha256, plan, box, origin, annotations_data, poses_data, *, log=None,
                  held_out_run=False, lock_sha256=None):
    """Integrity checks shared by the app shape and the approach split: the plan (by its sha256) was made for this box,
    these annotations and poses, and re-applying the frozen selection to its saved listing gives the same lists; every
    receipted sweep is on disk with its receipted sha256. Returns (ann, poses, receipted {ts: (entry, data)}, dropped)."""
    plan_dir = Path(plan_dir)
    log = guard_plan_dir(plan_dir, log, held_out_run=held_out_run, lock_sha256=lock_sha256)
    check_plan(plan, log)
    require((plan_dir / RECEIPT_NAME).is_file(), 'The planned sweeps have no download receipt (run the download first)')
    receipt = parse(snapshot(plan_dir / RECEIPT_NAME, max_bytes=MAX_JSON))
    require(receipt.get('log') == log, 'The download receipt is for another log')
    require(receipt.get('schema') == RECEIPT_SCHEMA and receipt.get('planSha256') == plan_sha256, 'The download receipt is for another sweep plan')
    require(receipt.get('track') == plan['track'] and receipt.get('rule') == RULE, 'The download receipt names another track or rule')
    require(plan['track'] == box['track'] and plan['box'] == box_record(box) and plan['cityOrigin'] == [float(v) for v in origin],
            'The sweep plan was made for another averaged box')
    require(plan['inputs']['annotations']['sha256'] == digest(annotations_data) and plan['inputs']['poses']['sha256'] == digest(poses_data),
            'The sweep plan was made from other annotations or poses')
    ann, poses = load_annotations(annotations_data), load_poses(poses_data)
    listing = {}
    if plan['inputs']['listing']['file'] is not None:
        listing = parse_listing(snapshot(plan_dir / LISTING_NAME, plan['inputs']['listing']['sha256'], MAX_JSON), plan['log'])
    require(plan['listing']['objects'] == len(listing) and (plan['listing']['status'] == 'ok') == bool(listing), 'Sweep plan listing record mismatch')
    again = select_sweeps(listing, ann, box['track'], poses, centre_city_xy(box, origin))
    require(again['candidates'] == plan['candidates'] and again['thinned'] == plan['thinned'] and
            [(str(s['timestampNs']), s['set'], s['egoDistanceM']) for s in again['kept']] ==
            [(s['timestampNs'], s['set'], s['egoDistanceM']) for s in plan['sweeps']] and
            all(s['key'] == listing[int(s['timestampNs'])]['key'] and s['bytes'] == listing[int(s['timestampNs'])]['bytes'] and
                s['etag'] == listing[int(s['timestampNs'])]['etag'] for s in plan['sweeps']),
            'The sweep plan differs from the frozen selection rule applied to its own inputs')
    planned = {s['timestampNs']: s for s in plan['sweeps']}
    dropped = receipt['downloadDropped']; records = receipt['sweeps']
    names = [r['timestampNs'] for r in records] + [d['timestampNs'] for d in dropped]
    require(sorted(names) == sorted(planned) and len(set(names)) == len(names), 'The receipt does not cover exactly the planned sweeps')
    for d in dropped:
        require(d == {'timestampNs': d['timestampNs'], 'set': planned[d['timestampNs']]['set'], 'reason': 'download-failed',
                      'attempts': RULE['downloadAttempts']}, 'Malformed dropped-download record')
    receipted = {}
    for r in records:
        entry = planned[r['timestampNs']]
        require(r['set'] == entry['set'] and r['key'] == entry['key'] and r['etag'] == entry['etag'] and r['bytes'] == entry['bytes'] and
                r['sizeMatches'] is True and r['md5MatchesEtag'] is (True if re.fullmatch('[0-9a-f]{32}', entry['etag']) else None),
                'A receipt record does not pass the download success test')
        path = plan_dir / SWEEP_DIR / f"{r['timestampNs']}.feather"
        guard_input_paths(log, path)
        require(path.is_file(), f"A planned and receipted sweep is missing: {r['timestampNs']}")
        data = snapshot(path, max_bytes=MAX_SWEEP)
        require(digest(data) == r['sha256'], f"A receipted sweep changed on disk: {r['timestampNs']}")
        receipted[int(r['timestampNs'])] = (r, data)
    return ann, poses, receipted, dropped


def build_car_shape(plan_dir, plan_sha256, box, origin, annotations_data, poses_data, placement_protocol_sha256, *, log=None,
                    held_out_run=False, lock_sha256=None):
    """The carShape block for ground.json (placement protocol v3.4) from a hashed plan and its receipted sweeps.
    Integrity failures stop with ValueError; not-stationary and too-few-sweeps are refusals (a block is returned)."""
    if not held_out_run and lock_sha256 is None:
        lock_sha256 = os.environ.get(GROUND_LOCK_ENV)
        held_out_run = lock_sha256 is not None
    log = guard_plan_dir(plan_dir, log, held_out_run=held_out_run, lock_sha256=lock_sha256)
    origin = np.asarray(origin, float)
    plan = check_plan(parse(snapshot(Path(plan_dir) / PLAN_NAME, plan_sha256, MAX_JSON)), log)
    ann, poses, receipted, dropped = load_receipted(plan_dir, plan_sha256, plan, box, origin, annotations_data, poses_data,
                                                 log=log, held_out_run=held_out_run, lock_sha256=lock_sha256)
    listed = lambda name: [{'timestampNs': str(ts), 'sha256': r['sha256']} for ts, (r, _d) in sorted(receipted.items()) if r['set'] == name]  # noqa: E731
    record = {'rule': dict(RULE), 'planSha256': plan_sha256, 'listing': {'status': plan['listing']['status'], 'attempts': plan['listing']['attempts']},
              'candidates': plan['candidates'], 'thinned': plan['thinned'], 'kept': plan['kept'],
              'shape': listed('shape'), 'scoring': listed('scoring'),
              'downloadDropped': sorted(dropped, key=lambda d: int(d['timestampNs']))}
    if not stationary(ann, poses['rigid'], box['track'])['passed']:
        return car_shape_block(box, record, placement_protocol_sha256, refusal='not-stationary')
    if len(record['shape']) < RULE['minShapeSweeps']:
        return car_shape_block(box, record, placement_protocol_sha256, refusal='too-few-sweeps')
    sweeps = [(ts, read_sweep(data)) for ts, (r, data) in sorted(receipted.items()) if r['set'] == 'shape']
    return car_shape_block(box, record, placement_protocol_sha256, shape=shape_from_sweeps(sweeps, box, origin, ann, poses))


# ------------------------------------------------------------------ approach-half split addendum (report only)
def approach_halves(entries):
    """entries: [{timestampNs:int, distanceM, ...}] (the shape and scoring sweeps, dropped ones left out). Sorted by
    distanceM from largest to smallest, ties by timestampNs ascending; h = K // 2; far = the h farthest, near = the h
    nearest; with an odd K the middle sweep is unused."""
    ordered = sorted(entries, key=lambda e: (-e['distanceM'], int(e['timestampNs'])))
    count = len(ordered); half = count // 2
    return {'K': count, 'h': half, 'far': ordered[:half], 'near': ordered[count - half:], 'unused': ordered[half:count - half]}


def approach_split_document(ground_raw, plan_dir, annotations_data, poses_data, ground_name=None, *, log=None,
                            held_out_run=False, lock_sha256=None):
    """The evaluation-only approach-half shape of one scene (split addendum). Returns (document, bytes). Never part of
    ground.json; the app never loads it."""
    log = guard_plan_dir(plan_dir, log, held_out_run=held_out_run, lock_sha256=lock_sha256, need_addendum=True)
    ground = parse(ground_raw); shape = ground.get('carShape')
    require(ground.get('schemaVersion') == 3 and 'carShape' in ground, 'A placement protocol v3.4 ground layer (schemaVersion 3) is required')
    require(shape is not None, 'This scene has no annotated stationary vehicle: the approach split does not apply')
    validate_block(shape, next(c for c in ground['cuboids']['items'] if 'poseAverage' in c and c['track'] == shape['track']))
    box = next(c for c in ground['cuboids']['items'] if 'poseAverage' in c and c['track'] == shape['track'])
    origin = np.asarray(ground['frame']['cityOrigin'], float)
    plan_sha = shape['sweeps']['planSha256']
    plan = check_plan(parse(snapshot(Path(plan_dir) / PLAN_NAME, plan_sha, MAX_JSON)), log)
    ann, poses = load_annotations(annotations_data), load_poses(poses_data)
    require(plan['inputs']['annotations']['sha256'] == digest(annotations_data) and plan['inputs']['poses']['sha256'] == digest(poses_data),
            'The sweep plan was made from other annotations or poses')
    require(plan['box'] == box_record(box) and plan['cityOrigin'] == [float(v) for v in origin], 'The sweep plan was made for another averaged box')
    listed = {int(x['timestampNs']): (x['sha256'], name) for name in ('shape', 'scoring') for x in shape['sweeps'][name]}
    stamps = sorted(listed)
    distance = {}
    if stamps:
        _q, trans = ego_poses_at(stamps, poses)
        cx, cy = centre_city_xy(box, origin)
        distance = {ts: float(d) for ts, d in zip(stamps, np.hypot(trans[:, 0] - cx, trans[:, 1] - cy))}
    by_plan = {int(s['timestampNs']): s['egoDistanceM'] for s in plan['sweeps']}
    require(all(by_plan.get(ts) == distance[ts] for ts in stamps), 'Approach distances differ from the sweep plan')
    halves = approach_halves([{'timestampNs': ts, 'distanceM': distance[ts], 'sha256': listed[ts][0], 'appSet': listed[ts][1]} for ts in stamps])
    refusal = None
    if shape['status'] == 'refused' and shape['refusal']['code'] == 'not-stationary':
        refusal = 'not-stationary'
    elif halves['h'] < RULE['minShapeSweeps']:
        refusal = 'too-few-sweeps'
    used = halves['far'] + halves['near'] if refusal is None else []
    data_by_ts = {}
    for e in used:  # every sweep used must be present with its listed sha256, else no output at all
        path = Path(plan_dir) / SWEEP_DIR / f"{e['timestampNs']}.feather"
        guard_input_paths(log, path)
        require(path.is_file(), f"Approach split sweep missing: {e['timestampNs']}")
        data = snapshot(path, max_bytes=MAX_SWEEP)
        require(digest(data) == e['sha256'], f"Approach split sweep changed on disk: {e['timestampNs']}")
        data_by_ts[e['timestampNs']] = data
    by_time = lambda lst: [{'timestampNs': str(e['timestampNs']), 'sha256': e['sha256']} for e in sorted(lst, key=lambda e: e['timestampNs'])]  # noqa: E731
    record = {**shape['sweeps'], 'shape': by_time(halves['far']), 'scoring': by_time(halves['near']),
              'kept': len(halves['far']) + len(halves['near']) + len(shape['sweeps']['downloadDropped'])}
    if refusal is None:
        far = [(e['timestampNs'], read_sweep(data_by_ts[e['timestampNs']])) for e in sorted(halves['far'], key=lambda e: e['timestampNs'])]
        block = car_shape_block(box, record, shape['protocol']['sha256'], shape=shape_from_sweeps(far, box, origin, ann, poses))
    else:
        block = car_shape_block(box, record, shape['protocol']['sha256'], refusal=refusal)
    span = lambda lst: None if not lst else [min(e['distanceM'] for e in lst), max(e['distanceM'] for e in lst)]  # noqa: E731
    listing = lambda lst: [{'timestampNs': str(e['timestampNs']), 'distanceM': e['distanceM'], 'sha256': e['sha256'], 'appSet': e['appSet']} for e in lst]  # noqa: E731
    doc = {'schema': APPROACH_SCHEMA, 'kind': 'evaluation-only shape (report only; never written into ground.json, never loaded by the app)',
           'addendum': {'path': SPLIT_ADDENDUM_PATH, 'sha256': SPLIT_ADDENDUM_SHA256},
           'shapeProtocol': {'path': SHAPE_PROTOCOL_PATH, 'sha256': SHAPE_PROTOCOL_SHA256},
           'scene': {'sceneId': ground.get('sceneId'), 'ground': {'file': ground_name, 'sha256': digest(ground_raw)},
                     'appCarShapeSha256': digest(canonical(shape)), 'appStatus': shape['status'],
                     'appRefusal': None if shape['refusal'] is None else shape['refusal']['code']},
           'approachSplit': {'K': halves['K'], 'h': halves['h'], 'far': listing(halves['far']), 'near': listing(halves['near']),
                             'unused': listing(halves['unused']), 'farRangeM': span(halves['far']), 'nearRangeM': span(halves['near']),
                             'order': 'distanceM (ego origin at ts to the averaged box centre, city frame) from largest to smallest; '
                                      'ties by timestampNs ascending',
                             'carShapeFields': ('carShape.sweeps.shape is the far half (the shape sweeps of this check) and '
                                                'carShape.sweeps.scoring the near half (its scoring sweeps); kept counts these two '
                                                'halves and the dropped downloads, not the unused middle sweep.')},
           'carShape': block}
    data = (json.dumps(doc, indent=1, sort_keys=True, allow_nan=False) + '\n').encode()
    no_banned_word(data, 'The approach split')
    return doc, data


# ------------------------------------------------------------------ command line
def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    plan = sub.add_parser('plan', help='write the sweep plan (metadata only) and print its sha256')
    for name in ('log', 'ground', 'ground-sha256', 'annotations', 'annotations-sha256', 'poses', 'poses-sha256'):
        plan.add_argument('--' + name, required=True)
    plan.add_argument('--listing-file', help='a saved public S3 listing of sensors/lidar/ (else it is fetched)')
    plan.add_argument('--listing-sha256')
    down = sub.add_parser('download', help='download the planned sweeps with receipts')
    down.add_argument('--log', required=True); down.add_argument('--plan-sha256', required=True)
    split = sub.add_parser('approach-split', help='write the evaluation-only approach-half shape (report only)')
    for name in ('ground', 'ground-sha256', 'annotations', 'annotations-sha256', 'poses', 'poses-sha256', 'name'):
        split.add_argument('--' + name, required=True)
    split.add_argument('--output-dir', help=f'default {APPROACH_DIR.relative_to(REPO)}')
    for command in (plan, down, split):
        command.add_argument('--out-root', help=f'default {DATA_ROOT.relative_to(REPO)}')
        command.add_argument('--held-out-run', action='store_true', help='M7 only: needs protocol-lock.json')
        command.add_argument('--protocol-lock-sha256')
    args = parser.parse_args()
    try:
        if args.command == 'plan':
            require((args.listing_file is None) == (args.listing_sha256 is None), '--listing-file and --listing-sha256 go together')
            check_held_out(args.log, args.held_out_run, args.protocol_lock_sha256)
            plan_dir, sha = make_plan(args.log, args.ground, args.ground_sha256, args.annotations, args.annotations_sha256, args.poses,
                                      args.poses_sha256, args.listing_file, args.listing_sha256, args.out_root,
                                      held_out_run=args.held_out_run, lock_sha256=args.protocol_lock_sha256)
            summary = json.loads(snapshot(plan_dir / PLAN_NAME, sha, MAX_JSON))
            print(canonical({'planSha256': sha, 'plan': str(plan_dir / PLAN_NAME), 'candidates': summary['candidates'],
                             'thinned': summary['thinned'], 'kept': summary['kept'], 'shape': len(summary['shape']),
                             'scoring': len(summary['scoring']), 'listing': summary['listing']}).decode())
        elif args.command == 'download':
            check_held_out(args.log, args.held_out_run, args.protocol_lock_sha256)
            receipt = download_sweeps(plan_dir_for(args.log, args.out_root), args.plan_sha256, log=args.log,
                                       held_out_run=args.held_out_run, lock_sha256=args.protocol_lock_sha256)
            print(canonical({'receipted': len(receipt['sweeps']), 'dropped': len(receipt['downloadDropped']),
                             'bytesOnDisk': receipt['bytesOnDisk']}).decode())
        else:
            require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]{0,80}', args.name) is not None, 'Plain file name required for --name')
            log = Path(args.annotations).resolve().parent.name
            require(Path(args.poses).resolve().parent.name == log, 'Annotations and poses must come from the same log directory')
            check_held_out(log, args.held_out_run, args.protocol_lock_sha256, need_addendum=True)
            guard_input_paths(log, args.ground, args.annotations, args.poses, plan_dir_for(log, args.out_root))
            ground_raw = snapshot(args.ground, args.ground_sha256, 1024 * 1024); ground = parse(ground_raw)
            shape = ground.get('carShape')
            require(isinstance(shape, dict), 'This scene has no annotated stationary vehicle: the approach split does not apply')
            plan_dir = plan_dir_for(log, args.out_root)
            guard_plan_dir(plan_dir, log, held_out_run=args.held_out_run, lock_sha256=args.protocol_lock_sha256, need_addendum=True)
            check_plan(parse(snapshot(plan_dir / PLAN_NAME, shape['sweeps']['planSha256'], MAX_JSON)), log)
            ann_raw = snapshot(args.annotations, args.annotations_sha256, MAX_META); poses_raw = snapshot(args.poses, args.poses_sha256, MAX_META)
            doc, data = approach_split_document(ground_raw, plan_dir, ann_raw, poses_raw, Path(args.ground).name,
                                                log=log, held_out_run=args.held_out_run, lock_sha256=args.protocol_lock_sha256)
            folder = Path(args.output_dir) if args.output_dir else APPROACH_DIR
            folder.mkdir(parents=True, exist_ok=True)
            write_exclusive(folder / f'{args.name}.json', data)
            print(canonical({'file': str(folder / f'{args.name}.json'), 'sha256': digest(data), 'K': doc['approachSplit']['K'],
                             'h': doc['approachSplit']['h'], 'status': doc['carShape']['status']}).decode())
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f'Car shape rejected: {error}\n')


if __name__ == '__main__':
    main()
