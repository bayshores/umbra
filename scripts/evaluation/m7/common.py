"""M7 (held-out run) shared helpers: the split roles, the lock and anchor checks, and receipted public AV2 downloads.

Nothing here opens a held-out file. preflight() is the only way into held-out work: protocol-lock.json at its fixed path
with the sha256 given, an approval block, the same bytes as the candidate its approval names once status and approval are
put back as shown, the lock record naming both sha256 values, the same sha256 in the tracked anchor on origin/main
(written once, like the anchors under ANCHOR_DIR), an empty git status, a
HEAD whose tracked files equal pipeline.commit except the anchor files (ANCHOR_REL and ANCHOR_DIR/), every pinned
file unchanged, and the labels addendum equal to its committed anchor and to the one the lock pins (labels.addendum;
null only with labels.noAddendum, Sean's recorded answer: no addendum file). The main checkout is used (ignored .local/ data and protocols live only there; a fresh worktree has none).
Each anchor file under ANCHOR_DIR is written once: exactly one commit on origin/main touches it, so a later change makes
every driver refuse. Held-out, each driver runs once per site: a run marker (RUN_MARKERS, O_EXCL) names its output folder
and the driver's process id, and a second run is refused. A marked run that ended (its process is gone) without its own
record is recorded once as stopped by its driver (ended_run), so every site still reaches the summary. Rehearsal
(development logs) skips the lock, anchor and marker checks and nothing else.
"""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time
import unicodedata
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
SPLIT = ROOT / '.local/evaluation/pao-accuracy/split.json'
SPLIT_SHA = '47f64fe007b51d75f4bf1c9891d940fd6393c777e0232a0a4f9428bbbeaafab4'
LOCK_PATH = ROOT / '.local/evaluation/pao-accuracy/lock-2/protocol-lock.json'
LOCK_RECORD = ROOT / '.local/evaluation/pao-accuracy/lock-2/protocol-lock.record.json'  # written by write_protocol_lock.py approve
SUPERSEDED_LOCK_REL = '.local/evaluation/pao-accuracy/protocol-lock.json'
SUPERSEDED_LOCK_SHA = '7aa2945e61e676ffc3a2a207895ca5639e4d9b4405893b648f4412891c00a475'
CANDIDATE_STATUS = 'CANDIDATE (not approved; no held-out run may use it)'  # the candidate's status, as Sean was shown it
ANCHOR_REL = 'scripts/evaluation/m7/anchors-2/pao-lock-anchor.json'
ANCHOR_DIR = 'scripts/evaluation/m7/anchors-2'  # tracked: lock, lost-site list, labels addendum, held-out label manifest, runs list
LOST_SITES_REL = f'{ANCHOR_DIR}/lost-sites.json'  # protocol-lock statistics.lostSiteRule: {"lostSites": [{logId, crossingId, evidence}]}
LABELS_ADDENDUM = ROOT / '.local/evaluation/pao-accuracy/lock/labels-addendum.json'  # protocol-lock labels.scoringCondition (4)
ADDENDUM_ANCHOR_REL = f'{ANCHOR_DIR}/labels-addendum-anchor.json'  # {"labelsAddendumSha256": <sha256> or null (D/E not scored)}
RUNS_REL = f'{ANCHOR_DIR}/runs.json'  # {"runs": [{logId, crossingId, siteRecordSha256, paintRecordSha256}]}: score.py runs-list
LABELS_MANIFEST_REL = f'{ANCHOR_DIR}/labels-manifest.json'  # {"labelsAddendumSha256", "sites": [{logId, crossingId, labelsSha256, reason}]}
HEX64 = re.compile('[0-9a-f]{64}')
RUN_MARKERS = ROOT / '.local/evaluation/pao-accuracy/m7/lock-2/run-markers'  # held-out: <step>-<log8>-<crossing>.json, one per run
COORDINATOR_ENV = 'PAO_LABELS_COORDINATOR_DIR'  # the coordinator-only labels folder: never written into any file
DE_CODE = ROOT / '.local/evaluation/pao-accuracy/lock/de-code'  # ignored: the D/E code, every file pinned by the labels addendum
FETCH_TRIES = 3  # preflight's git fetch of origin/main: a dropped connection is tried again, at most 3 times in all
FETCH_SECONDS = 120  # each git fetch try is stopped after this long (git has no network time limit of its own)
DEVELOPMENT_ROLES = ('development', 'development-optional')
HELD_OUT_ROLES = ('held-out-blocked-test', 'held-out-left-side-case', 'held-out-control', 'held-out-control-declared-only')
BUCKET = 'https://argoverse.s3.amazonaws.com/'
LOG_PREFIX = 'datasets/av2/sensor/{split}/{log}/'
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


def coordinator_folders():
    """The coordinator-only folder as PAO_LABELS_COORDINATOR_DIR gives it and resolved (empty when it is not set)."""
    value = os.environ.get(COORDINATOR_ENV, '').rstrip('/')
    return list(dict.fromkeys([value, str(Path(value).resolve())])) if len(value) > 1 else []


def folder_forms(folders=()):
    """The coordinator-only folder and any folder given, each as given, resolved and (when inside the repository)
    relative to the repository, longest first."""
    forms = set()
    for folder in [*coordinator_folders(), *folders]:
        for form in (str(folder).rstrip('/'), str(Path(folder).resolve())):
            if len(form) > 1:
                forms.add(form)
                inside = Path(form).relative_to(ROOT) if Path(form).is_relative_to(ROOT) else None
                if inside is not None and len(str(inside)) > 1:
                    forms.add(str(inside))
    return sorted(forms, key=len, reverse=True)


def fold(text):
    """text for a folder comparison: Unicode NFC and case-folded (macOS folder names ignore case)."""
    return unicodedata.normalize('NFC', text).casefold()


def json_strings(node):
    """Every string (keys included) inside a decoded JSON value."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from json_strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from json_strings(value)


JSON_STRING = re.compile(r'"(?:[^"\\\n]|\\.)*"')  # one JSON string literal inside other text
WITHHELD = '<text withheld: it names the coordinator-only folder>'


def unique_keys(pairs):
    """json object_pairs_hook: a repeated key is refused (json.loads would keep only its last value)."""
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise SystemExit('a JSON object repeats a key')
    return dict(pairs)


def spellings(form):
    """form as plain text and as JSON writes it, escaped once or twice (\\u00e9 for non-ASCII, \\/ for /), NFC and NFD."""
    out = set()
    for text in {unicodedata.normalize('NFC', form), unicodedata.normalize('NFD', form)}:
        level = {text}
        for _ in range(3):
            out |= level
            level = {e for t in level for d in (json.dumps(t)[1:-1], json.dumps(t, ensure_ascii=False)[1:-1])
                     for e in (d, d.replace('/', '\\/'))}
    return sorted(out, key=len, reverse=True)


def decoded_texts(text, depth=3):
    """text, the strings of text read as one JSON value, and every JSON string literal inside it, decoded to depth levels
    (a JSON document quoted inside a log line, or inside another JSON string)."""
    texts, todo = [text], [text]
    for _ in range(depth):
        found = []
        for item in todo:
            try:
                found += list(json_strings(json.loads(item, strict=False)))
            except (ValueError, RecursionError):
                pass
            for literal in JSON_STRING.findall(item):
                try:
                    found.append(json.loads(literal, strict=False))
                except ValueError:
                    pass
        texts += found
        todo = [t for t in found if '"' in t or '\\' in t]
    return texts


def names_folder(data, folders=()):
    """True when data (bytes or text) names the coordinator-only folder or a folder given, in any form folder_forms
    lists, as plain text, JSON-escaped (spellings) or inside JSON strings (decoded_texts), ignoring case and Unicode
    normalization."""
    text = data.decode('utf-8', 'replace') if isinstance(data, bytes) else data
    forms = {fold(s) for f in folder_forms(folders) for s in spellings(f)}
    if not forms:
        return False
    texts = fold('\0'.join(decoded_texts(text)))  # no form holds a NUL, so no match spans two texts
    return any(form in texts for form in forms)


def masked(text, folders=()):
    """text with the banned vehicle-status word masked (p*rk) and the coordinator-only folder (and any folder given, in
    every form folder_forms lists and every spelling, any case) replaced by a placeholder: tool output can quote
    either. Text that still names a folder (names_folder: a case match only casefold finds, or a quoted JSON document)
    is withheld whole, so a masked text can always be written."""
    text = unicodedata.normalize('NFC', text)
    for form in sorted({s for f in folder_forms(folders) for s in spellings(f)}, key=len, reverse=True):
        text = re.sub(re.escape(form), '<coordinator-only folder>', text, flags=re.IGNORECASE)
    if names_folder(text, folders):
        text = WITHHELD
    return re.sub('park', 'p*rk', text, flags=re.IGNORECASE)


def child_env(**extra):
    """The environment of a tool step (a child process): this one without the coordinator-only folder's variable, plus
    extra. No tool is told where that folder is; the D/E scorer gets only its --labels file."""
    return {**{k: v for k, v in os.environ.items() if k != COORDINATOR_ENV}, **extra}


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
    """New files only; JSON objects are dumped; the banned vehicle-status word and the coordinator-only folder (any form,
    names_folder) are refused. The bytes go to a temporary file in the same folder, flushed to the disk, which is then
    linked to the new name (a link fails if the name exists): after a crash or a power loss the file is whole or absent,
    never empty or cut short."""
    data = dump(data) if isinstance(data, (dict, list)) else data
    if b'park' in data.lower():
        raise ValueError(f'{path}: banned vehicle-status word')
    if names_folder(data):
        raise ValueError(f'{path}: names the coordinator-only folder')
    path = Path(path)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.', suffix='.part')
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            handle.flush()
            durable(handle.fileno())
        os.chmod(temp, mode)
        os.link(temp, path)
    finally:
        os.unlink(temp)
    folder = os.open(path.parent, os.O_RDONLY)
    try:
        durable(folder)  # the new name itself
    finally:
        os.close(folder)
    return sha(data)


def durable(fd):
    """Flush fd to the disk itself (macOS F_FULLFSYNC, which a power loss survives; fsync where that is missing)."""
    try:
        fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
    except (AttributeError, OSError):
        os.fsync(fd)


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


def av2_split(log):
    """The publisher folder of a runnable log, from sha-checked split.json; reserve and test stay unopened."""
    entries = [s for s in split_sites() if s['logId'] == log]
    if not entries:
        raise SystemExit(f'{log[:8]} is not a log of split.json')
    for entry in entries:
        if entry['role'] not in DEVELOPMENT_ROLES + HELD_OUT_ROLES:
            raise SystemExit(f"{log[:8]} is {entry['role']}: reserve and other non-runnable roles stay unopened")
        if entry.get('av2Split') not in ('train', 'val'):
            raise SystemExit(f"{log[:8]} has unsupported av2Split {entry.get('av2Split')!r}; expected train or val")
    splits = {s['av2Split'] for s in entries}
    if len(splits) != 1:
        raise SystemExit(f'{log[:8]} has conflicting av2Split entries in split.json')
    return splits.pop()


def log_prefix(log):
    return LOG_PREFIX.format(split=av2_split(log), log=log)


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
    """The committed lost-site list (statistics.lostSiteRule), needed before any held-out build. Each entry names a site
    once, with non-empty evidence text and a date (YYYY-MM-DD) no later than the UTC day of the list's one commit.
    Returns ({'<log8>/<crossing>', ...}, its sha256)."""
    doc, digest = committed(LOST_SITES_REL, 'the lost-site list (possibly empty)')
    stamp = git('log', '-1', '--format=%cI', 'origin/main', '--', LOST_SITES_REL).strip()
    day = datetime.fromisoformat(stamp).astimezone(timezone.utc).date().isoformat()
    names = set()
    for x in doc['lostSites']:
        name, date = f"{x['logId'][:8]}/{x['crossingId']}", x.get('date')
        if name in names or not str(x.get('evidence') or '').strip() or not iso_day(date) or date > day:
            raise SystemExit(f'{LOST_SITES_REL}: {name} must appear once, with evidence text and a date (YYYY-MM-DD) no later '
                             f'than the list\'s commit ({day})')
        names.add(name)
    return names, digest


def iso_day(value):
    """True for a real calendar day written YYYY-MM-DD."""
    try:
        return isinstance(value, str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}', value) and bool(datetime.strptime(value, '%Y-%m-%d'))
    except ValueError:
        return False


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


def held_out_names():
    """{'<log8>/<crossing>': split.json entry} of every held-out site minus the committed lost-site list."""
    lost, _ = lost_sites()
    return {n: x for x in split_sites() if x['role'] in HELD_OUT_ROLES and (n := f"{x['logId'][:8]}/{x['crossingId']}") not in lost}


def runs_list():
    """The committed held-out runs list (statistics.oneRunRule): written once by score.py runs-list after every held-out
    build and paint run has its record and before any evaluate_site.py run, it names each held-out site minus the
    lost-site list once, with the sha256 of its one site record and one paint record. Returns ({'<log8>/<crossing>':
    (site record sha256, paint record sha256)}, its sha256)."""
    doc, digest = committed(RUNS_REL, 'the held-out runs list (score.py runs-list)')
    keys = {'logId', 'crossingId', 'siteRecordSha256', 'paintRecordSha256'}
    runs = {}
    for x in doc.get('runs') if isinstance(doc.get('runs'), list) else [None]:
        ok = isinstance(x, dict) and set(x) == keys and all(isinstance(x[k], str) for k in keys) \
            and all(HEX64.fullmatch(x[k]) for k in ('siteRecordSha256', 'paintRecordSha256'))
        name = f"{x['logId'][:8]}/{x['crossingId']}" if ok else None
        if not ok or name in runs:
            raise SystemExit(f'{RUNS_REL}: each entry is {{logId, crossingId, siteRecordSha256, paintRecordSha256}} and names a site once')
        runs[name] = (x['siteRecordSha256'], x['paintRecordSha256'])
    held = held_out_names()
    if set(runs) != set(held) or any(held[n]['logId'] != x['logId'] for x in doc['runs'] for n in [f"{x['logId'][:8]}/{x['crossingId']}"]):
        raise SystemExit(f'{RUNS_REL} must name every held-out site minus the lost-site list once; missing '
                         f'{sorted(set(held) - set(runs))}, extra {sorted(set(runs) - set(held))}')
    return runs, digest


def marker_path(step, log, crossing):
    return RUN_MARKERS / f'{step}-{log[:8]}-{crossing}.json'


def run_marker(step, log, crossing, folder):
    """Held-out: the one run of this step (build, paint, evaluate) at this site, written before its output folder is
    made. A second run is refused (protocol-lock dataAccess: a site is never rerun)."""
    path = marker_path(step, log, crossing)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_new(path, {'kind': 'pao-m7-run-marker', 'step': step, 'site': {'logId': log, 'crossingId': str(crossing)},
                         'folder': rel(folder), 'pid': os.getpid(), 'utc': now()})
    except FileExistsError:
        raise SystemExit(f'{step} already ran for {log[:8]}/{crossing} ({rel(path)}): a held-out site is never rerun') from None


def marked_folder(step, log, crossing):
    """The output folder the run marker of this step names for this site, or None."""
    path = marker_path(step, log, crossing)
    return json.loads(path.read_bytes())['folder'] if path.is_file() else None


def ended_run(step, log, crossing):
    """Held-out: the output folder (a Path) the run marker of this step names for this site, once that run has ended (the
    process the marker names is gone, or its id now belongs to a process that started after the marker was written). Used
    only to record a stop for a marked run that left no record of its own."""
    path = marker_path(step, log, crossing)
    if not path.is_file():
        raise SystemExit(f'{step} never ran for {log[:8]}/{crossing}: there is nothing to record')
    marker = json.loads(path.read_bytes())
    pid = marker.get('pid')
    if not isinstance(pid, int) or pid <= 0:
        raise SystemExit(f'{rel(path)} names no process')
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return ROOT / marker['folder']
    except PermissionError:
        pass
    # The marked run wrote its marker after it started; a process with that id that started later reuses the id.
    started, marked = started_utc(pid), datetime.strptime(marker['utc'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
    if started is not None and started > marked:
        return ROOT / marker['folder']
    raise SystemExit(f'the {step} run for {log[:8]}/{crossing} (process {pid}) may still be running: wait until it ends')


def started_utc(pid):
    """When process pid started (UTC, whole seconds), or None when ps cannot tell."""
    result = subprocess.run(['ps', '-o', 'lstart=', '-p', str(pid)], capture_output=True, text=True, env={**os.environ, 'LC_ALL': 'C'})
    try:
        return datetime.strptime(' '.join(result.stdout.split()), '%a %b %d %H:%M:%S %Y').astimezone(timezone.utc)
    except ValueError:
        return None


def de_code(addendum):
    """The D/E scorer ({file, sha256}) the labels addendum pins, after checking the D/E code (labels.scoringCondition):
    every file in DE_CODE is pinned in the addendum's deCode.files ({repository path: sha256}) with that sha256, nothing
    else is in that folder (no link), and the scorer is one of those files. The scorer runs with -I -B: it can import only
    the standard library, this folder (by adding it to its own path) and files the protocol lock pins."""
    code = addendum.get('deCode') or {}
    files, scorer = code.get('files') or {}, code.get('scorer') or {}
    found = sorted(DE_CODE.rglob('*')) if DE_CODE.is_dir() else []
    present = sorted(rel(p) for p in found if p.is_file() or p.is_symlink())
    problems = [what for what, ok in (
        ('a link in the D/E code folder', not any(p.is_symlink() for p in found)),
        (f'the files in {rel(DE_CODE)} differ from the pinned ones', present == sorted(files)),
        ('a pinned D/E file differs from its sha256', all((ROOT / f).is_file() and sha_file(ROOT / f) == v for f, v in files.items())),
        ('the scorer is not one of the pinned D/E files', bool(scorer.get('file')) and files.get(scorer['file']) == scorer.get('sha256')))
        if not ok]
    if problems:
        raise SystemExit('the D/E code differs from the labels addendum: ' + '; '.join(problems))
    return {'file': scorer['file'], 'sha256': scorer['sha256']}


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


def fetch_once():
    """One git fetch origin main, stopped after FETCH_SECONDS. It runs in its own process group with no shared output
    and no prompt, and on the time limit the whole group (git and its https helper) is killed, so nothing it started
    keeps a caller's captured output open past the limit."""
    proc = subprocess.Popen(['git', 'fetch', '--quiet', 'origin', 'main'], cwd=ROOT, stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
                            env={**child_env(), 'GIT_TERMINAL_PROMPT': '0'})
    try:
        return proc.wait(timeout=FETCH_SECONDS) == 0
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        return False


def fetch_origin():
    """git fetch origin main (fetch_once), tried again after a failure (FETCH_TRIES in all, 10 s apart): a dropped
    connection should not stop a site."""
    for attempt in range(FETCH_TRIES):
        if fetch_once():
            return
        if attempt + 1 < FETCH_TRIES:
            time.sleep(10)
    raise SystemExit(f'git fetch origin main failed {FETCH_TRIES} times')


def wait_for_origin(limit=3600, pause=60):
    """Held-out evaluate_site.py, just before its score step (nothing scored yet, so this is no rerun): wait up to limit
    seconds, trying every pause seconds, until git fetch origin main works, so a network outage does not fail the score
    step. Returns whether it works."""
    end = time.time() + limit
    while True:
        try:
            fetch_origin()
            return True
        except SystemExit:
            if time.time() + pause > end:
                return False
            time.sleep(pause)


def preflight(lock_sha256):
    """Held-out only. Returns the lock. Raises SystemExit before any held-out file is touched."""
    if lock_sha256 == SUPERSEDED_LOCK_SHA or LOCK_PATH.resolve() == (ROOT / SUPERSEDED_LOCK_REL).resolve():
        raise SystemExit(f'Lock 1 ({SUPERSEDED_LOCK_REL}, sha256 {SUPERSEDED_LOCK_SHA}) is superseded; use lock 2')
    if not (isinstance(lock_sha256, str) and re.fullmatch('[0-9a-f]{64}', lock_sha256)):
        raise SystemExit('Held-out runs need --protocol-lock-sha256')
    if not LOCK_PATH.is_file():
        raise SystemExit(f'{rel(LOCK_PATH)} does not exist (no approved lock)')
    raw = LOCK_PATH.read_bytes()
    if sha(raw) == SUPERSEDED_LOCK_SHA:
        raise SystemExit(f'Lock 1 ({SUPERSEDED_LOCK_REL}) is superseded; use lock 2')
    if sha(raw) != lock_sha256:
        raise SystemExit('protocol-lock.json differs from the given sha256')
    lock = json.loads(raw)
    approval = lock.get('approval') or {}
    if not all(isinstance(approval.get(k), str) and approval[k].strip() for k in ('by', 'utc', 'how', 'candidateSha256', 'recordedUtc')):
        raise SystemExit('protocol-lock.json has no approval')
    # The lock is the candidate Sean approved: the same bytes once its status and approval are put back as shown.
    shown = (json.dumps({**lock, 'status': CANDIDATE_STATUS, 'approval': None}, indent=1, ensure_ascii=False) + '\n').encode()
    if sha(shown) != approval['candidateSha256']:
        raise SystemExit('protocol-lock.json differs from the candidate its approval names (approval.candidateSha256)')
    record = json.loads(LOCK_RECORD.read_bytes()) if LOCK_RECORD.is_file() else {}
    if (record.get('file'), record.get('sha256'), record.get('candidateSha256')) != (rel(LOCK_PATH), lock_sha256, approval['candidateSha256']):
        raise SystemExit(f'{rel(LOCK_RECORD)} is missing or names another lock')
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
    fetch_origin()
    anchor, _ = committed(ANCHOR_REL, 'the lock anchor (write_protocol_lock.py approve)')  # pushed and written once
    if anchor.get('protocolLockSha256') != lock_sha256:
        raise SystemExit('The anchor on origin/main names another lock')
    for path, value in pinned_files(lock):
        if sha_file(ROOT / path) != value:
            raise SystemExit(f'{path} differs from its sha256 in the lock')
    for group in (lock['pipeline']['scoringCode'], lock['pipeline']['m7Code'], lock['sceneBuilder']['code']):
        for path, value in group.items():
            if sha_file(ROOT / path) != value:
                raise SystemExit(f'{path} differs from its sha256 in the lock')
    # The labels addendum Sean approved inside the lock (labels.addendum), or none only with his recorded answer.
    labels = lock.get('labels') or {}
    pinned, answer, addendum = labels.get('addendum'), labels.get('noAddendum'), labels_addendum(True)
    if not (addendum is None and pinned is None and isinstance(answer, str) and answer.strip()
            or addendum is not None and pinned == {'path': rel(LABELS_ADDENDUM), 'sha256': addendum} and answer is None):
        raise SystemExit('the labels addendum differs from the one the lock pins (labels.addendum {path, sha256}, or null '
                         'with the answer in labels.noAddendum)')
    return lock


def guard(log, crossing, rehearse, lock_sha256):
    """The split role decides the mode: development logs only in rehearsal, held-out only after preflight, reserve never."""
    entry = site(log, crossing)
    av2_split(log)  # refuse unsupported publisher folders before a marker or any data access
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
                    '--max-filesize', str(max_bytes), '--dump-header', str(headers), '--output', str(target), url], check=True, env=child_env())
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
    prefix = (log_prefix(log) + sub).replace('/', '%2F')
    return f'{BUCKET}?list-type=2&delimiter=%2F&max-keys=1000&prefix={prefix}'


LISTINGS = {'root': '', 'calibration': 'calibration/', 'map': 'map/', 'lidar': 'sensors/lidar/',
            'stereo_front_left': 'sensors/cameras/stereo_front_left/', 'stereo_front_right': 'sensors/cameras/stereo_front_right/'}


def read_listing(path, log, sub):
    """Keys of one saved listing as {relative name: {key, bytes, etag}}; refuses a truncated or foreign listing."""
    prefix = log_prefix(log) + sub
    root = ET.fromstring(Path(path).read_bytes())
    if root.findtext(f'{S3}Prefix') != prefix or root.findtext(f'{S3}IsTruncated') != 'false':
        raise SiteStop('listings', f'{Path(path).name}: unexpected prefix or a truncated listing')
    out = {}
    for item in root.findall(f'{S3}Contents'):
        key, size, etag = (item.findtext(f'{S3}{tag}') or '' for tag in ('Key', 'Size', 'ETag'))
        if not (key.startswith(prefix) and size.isdigit() and etag):
            raise SiteStop('listings', f'{Path(path).name}: malformed entry')
        out[key[len(prefix):]] = {'key': key, 'bytes': int(size), 'etag': etag.strip('"')}
    return out
