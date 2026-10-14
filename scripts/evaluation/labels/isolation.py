"""Keep answers away from labelers (protocol `isolation`, v2 draft onward). Evaluation-only; never shipped in the app.

Every agent runs as the same user, so file permissions cannot stop a labeler from opening an answer. A protocol with
an `isolation` block therefore adds three recorded controls to the labeler instructions:

1. Answer-bearing outputs (sealed answer key and coverage, aggregates, scores, oracle self-test outputs, report rows)
   are written only to a coordinator-only folder OUTSIDE the repository. Its path comes from the environment variable
   named in `isolation.coordinatorRootEnv`, so no file inside the repository records where it is. Tools refuse to
   write these outputs inside the repository.
2. `labelkit.py` refuses every command while a file matching `isolation.answerBearingGlobs` is still inside the
   repository (`check`). `move` relocates such files into the coordinator folder with a hash manifest.
3. Before aggregation the coordinator scans every labeler transcript (`scan`). Each tool-call input is matched against
   `isolation.forbiddenToolInputPatterns`, the own-session rule (only `v2/sessions/<own labeler>/<own task>` and the
   matching scratch folder) and the one-agent-per-task rule (no `--labeler` or `--task` other than its own).
   aggregate.py refuses unless the scan covers every labeled session with zero hits, and hashes the scan into the
   labels manifest.

  python -I isolation.py check --protocol P
  python -I isolation.py move --protocol P [--apply]          # dry run unless --apply
  python -I isolation.py scan --protocol P --transcripts transcripts.json --out scan.json

transcripts.json (coordinator-owned): {"sessions": {"L1/V01": ["<agent transcript .jsonl>", ...], ...}}

Limits: the scan reads tool-call inputs only (Claude Code `tool_use`, Codex `function_call`, `custom_tool_call`,
`local_shell_call`). The agent's final report (Claude Code `SubagentHandback`, REPORT_TOOLS) is the reply the
labeler instructions ask for, opens no file and is not scanned; reports are counted (`reportCalls`). A path assembled at run time, or a relative path after `cd`, can evade the path rules; the name
rules (sealed, answer-key, selftest, ...), the parent-folder (`..`) rule and the directory-walk rule narrow this but
do not close it. The scan is recorded evidence, not proof.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (LABELS_DIR, ROOT, canonical_json, load_protocol, read_json, relative, require,  # noqa: E402
                    sha256_bytes, sha256_file, write_json)

SCAN_SCHEMA = 'pao-labels-transcript-scan-v1'
TOOL_CALL_FIELDS = {'tool_use': ('input',), 'server_tool_use': ('input',), 'function_call': ('arguments',),
                    'custom_tool_call': ('input',), 'local_shell_call': ('action',)}
REPORT_TOOLS = ('SubagentHandback',)  # the final reply to the coordinator (section 7), not a file access
ID = r'[A-Za-z0-9_-]+'
SESSION_WORD = re.compile(r'(?<![\w-])(?:sessions|scratch)(?![\w-])', re.IGNORECASE)
FLAG = re.compile(r'--(labeler|task)(?:=|\s+)[\'"]?(' + ID + ')')


def settings(protocol):
    return protocol.get('isolation') or {}


def inside_repo(path):
    path = Path(path).expanduser().resolve()
    return path == ROOT or ROOT in path.parents


def coordinator_root(env_name, must_exist=True):
    """The coordinator-only folder named by the environment variable `env_name` (never inside the repository)."""
    value = os.environ.get(env_name)
    require(bool(value), f'Set {env_name} to the coordinator-only folder outside the repository')
    path = Path(value).expanduser().resolve()
    require(not inside_repo(path), f'{env_name} must point outside the repository')
    require(path.is_dir() or not must_exist, f'{env_name} does not name an existing folder')
    return path


def protocol_store(protocol, must_exist=True):
    """Coordinator-only folder for this protocol version, or None when the protocol has no isolation block."""
    env = settings(protocol).get('coordinatorRootEnv')
    if not env:
        return None
    return coordinator_root(env, must_exist) / 'pao-labels' / protocol['version']


def sealed_spec(protocol):
    """Where a set built under `protocol` keeps its sealed files (recorded in validation-lock.json, no path)."""
    env = settings(protocol).get('coordinatorRootEnv')
    if not env:
        return None
    return {'location': 'coordinator', 'env': env, 'subdir': f"pao-labels/{protocol['version']}/sealed"}


def sealed_dir(base, lock):
    """Sealed folder of a validation set: base/sealed (v1) or the coordinator folder named in its lock."""
    spec = lock.get('sealed')
    if not spec:
        return Path(base) / 'sealed'
    require(spec.get('location') == 'coordinator', 'unknown sealed location in validation-lock.json')
    return coordinator_root(spec['env']) / spec['subdir']


def output_dir(protocol, path, what):
    """Refuse an answer-bearing output folder inside the repository when the protocol isolates answers."""
    if settings(protocol).get('coordinatorRootEnv'):
        require(not inside_repo(path), f'{what} holds per-feature values; protocol {protocol["version"]} keeps it outside '
                                       f'the repository (under ${settings(protocol)["coordinatorRootEnv"]})')
    return Path(path)


def default_output(protocol, name, fallback):
    store = protocol_store(protocol, must_exist=True)
    return (store / name) if store else Path(fallback)


# ----- clean tree ---------------------------------------------------------------------------------------

def answer_bearing(protocol):
    """Repository-relative files that match `isolation.answerBearingGlobs` (hidden files included)."""
    found = set()
    for pattern in settings(protocol).get('answerBearingGlobs', []):
        require(not pattern.startswith('/') and '..' not in Path(pattern).parts, f'glob must be repository-relative: {pattern}')
        for name in glob.glob(str(ROOT / pattern), recursive=True, include_hidden=True):
            path = Path(name)
            if path.is_file() and not path.is_symlink():
                found.add(relative(path))
    return sorted(found)


def check_clean_tree(protocol):
    """Refuse while answer-bearing files sit inside the repository (protocol isolation.requireCleanTree)."""
    if not settings(protocol).get('requireCleanTree'):
        return []
    found = answer_bearing(protocol)
    require(not found, f'{len(found)} answer-bearing file(s) are still inside the repository, e.g. {found[:3]}; '
                       'the coordinator must move them out first (isolation.py move)')
    return found


def move(protocol, apply=False):
    """Move answer-bearing files into <coordinator store>/moved-from-repo/<same relative path>; hash-checked."""
    files = answer_bearing(protocol)
    plan = [{'path': f, 'sha256': sha256_file(ROOT / f), 'bytes': (ROOT / f).stat().st_size} for f in files]
    if not apply or not plan:
        return {'apply': False, 'files': plan}
    dest = protocol_store(protocol, must_exist=False) / 'moved-from-repo'
    for item in plan:
        require(not (dest / item['path']).exists(), f"{item['path']} already exists in the coordinator folder")
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    for item in plan:
        target = dest / item['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(ROOT / item['path']), str(target))
        require(sha256_file(target) == item['sha256'], f"{item['path']} changed while moving")
        folder = (ROOT / item['path']).parent
        while folder != ROOT and folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()  # drop folders the move left empty, so later lookups fall through to the moved copy
            folder = folder.parent
    record = {'schema': 'pao-labels-isolation-moves-v1', 'movedUtc': stamp, 'protocolVersion': protocol['version'],
              'destination': f"${settings(protocol)['coordinatorRootEnv']}/pao-labels/{protocol['version']}/moved-from-repo",
              'files': plan}
    write_json(dest / f'moves-{stamp}.json', record, overwrite=False)
    write_json(LABELS_DIR / f'isolation-moves-{stamp}.json', record, overwrite=False)  # paths and hashes only, no values
    return {'apply': True, **record}


# ----- transcript scan ----------------------------------------------------------------------------------

def scan_rules(protocol):
    iso = settings(protocol)
    patterns = iso.get('forbiddenToolInputPatterns')
    require(bool(patterns), 'protocol isolation.forbiddenToolInputPatterns is missing')
    own = iso.get('ownSessionPrefix')
    require(bool(own), 'protocol isolation.ownSessionPrefix is missing')
    return {'forbidden': list(patterns), 'ownSessionPrefix': own, 'oneTaskPerAgent': True}


def tool_inputs(node):
    """(tool name, input text) for every tool call anywhere in one transcript record."""
    stack = [node]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            kind = item.get('type')
            if isinstance(kind, str) and kind in TOOL_CALL_FIELDS:  # some records carry a non-text 'type'
                parts = [item[f] for f in TOOL_CALL_FIELDS[kind] if item.get(f) is not None]
                yield str(item.get('name', kind)), ' '.join(p if isinstance(p, str) else json.dumps(p, sort_keys=True) for p in parts)
                continue
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)


def _excerpt(text, start, end):
    snippet = text[max(0, start - 60):end + 60].replace(str(Path.home()), '~')
    return snippet if len(snippet) <= 200 else snippet[:200]


def scan_text(text, labeler, task, compiled, own):
    """Rule hits in one tool-call input for the session (labeler, task)."""
    hits = []
    for rule, regex in compiled:
        for m in regex.finditer(text):
            hits.append({'rule': rule, 'excerpt': _excerpt(text, m.start(), m.end())})
    own_spans = [m.span() for m in own.finditer(text)]
    for m in SESSION_WORD.finditer(text):
        if not any(a <= m.start() < b for a, b in own_spans):
            hits.append({'rule': 'session or scratch path other than its own', 'excerpt': _excerpt(text, m.start(), m.end())})
    for m in FLAG.finditer(text):
        expected = labeler if m.group(1) == 'labeler' else task
        if m.group(2) != expected:
            hits.append({'rule': f'--{m.group(1)} other than its own', 'excerpt': _excerpt(text, m.start(), m.end())})
    return hits


def own_regex(rules, labeler, task):
    prefix = re.escape(rules['ownSessionPrefix'])
    return re.compile(rf'(?<![\w-])(?:{prefix})?(?:sessions|scratch)/{re.escape(labeler)}/{re.escape(task)}(?=/|$|[\s\'"\\,;)])')


def scan_transcript(path, labeler, task, rules):
    compiled = [(p, re.compile(p, re.IGNORECASE)) for p in rules['forbidden']]
    own = own_regex(rules, labeler, task)
    binding = re.compile(rf'--labeler(?:=|\s+)[\'"]?{re.escape(labeler)}\b.*--task(?:=|\s+)[\'"]?{re.escape(task)}\b|'
                         rf'--task(?:=|\s+)[\'"]?{re.escape(task)}\b.*--labeler(?:=|\s+)[\'"]?{re.escape(labeler)}\b', re.DOTALL)
    out = {'name': Path(path).name, 'sha256': sha256_file(path), 'lines': 0, 'toolCalls': 0, 'reportCalls': 0, 'bindingCalls': 0,
           'hits': []}
    with open(path, encoding='utf-8') as handle:
        for number, line in enumerate(handle, 1):
            out['lines'] += 1
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                out['hits'].append({'line': number, 'tool': None, 'rule': 'unreadable transcript line', 'excerpt': line[:120]})
                continue
            for tool, text in tool_inputs(record):
                if tool in REPORT_TOOLS:
                    out['reportCalls'] += 1
                    continue
                out['toolCalls'] += 1
                if binding.search(text):
                    out['bindingCalls'] += 1
                for hit in scan_text(text, labeler, task, compiled, own):
                    out['hits'].append({'line': number, 'tool': tool, **hit})
    return out


def scan(protocol, phash, transcripts_doc):
    rules = scan_rules(protocol)
    sessions = {}
    for name, paths in sorted(transcripts_doc['sessions'].items()):
        labeler, task = name.split('/')
        require(bool(paths), f'{name}: no transcript listed')
        items = [scan_transcript(p, labeler, task, rules) for p in paths]
        sessions[name] = {'transcripts': items, 'hits': sum(len(i['hits']) for i in items)}
    return {'schema': SCAN_SCHEMA, 'protocolSha256': phash, 'rules': rules, 'rulesSha256': sha256_bytes(canonical_json(rules)),
            'scannedUtc': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), 'sessions': sessions,
            'totalHits': sum(s['hits'] for s in sessions.values()),
            'statement': 'Tool-call inputs of every listed labeler transcript, scanned before aggregation and scoring.'}


def verify_scan(path, protocol, phash, pairs):
    """Used by aggregate.py: the scan must use this protocol's rules, cover every labeled session and have no hit."""
    doc = read_json(path)
    require(doc.get('schema') == SCAN_SCHEMA, 'not a transcript scan file')
    require(doc.get('protocolSha256') == phash, 'transcript scan was made under another protocol')
    rules = scan_rules(protocol)
    require(doc.get('rules') == rules and doc.get('rulesSha256') == sha256_bytes(canonical_json(rules)),
            'transcript scan used other rules than this protocol')
    for name in pairs:
        entry = doc['sessions'].get(name)
        require(entry is not None and entry['transcripts'], f'{name}: no scanned transcript')
        require(all(t['toolCalls'] > 0 for t in entry['transcripts']), f'{name}: a transcript has no tool call (wrong format?)')
        require(sum(t['bindingCalls'] for t in entry['transcripts']) > 0, f'{name}: no transcript runs this labeler and task')
    hits = [(name, h) for name, s in doc['sessions'].items() for t in s['transcripts'] for h in t['hits']]
    require(doc.get('totalHits') == len(hits), 'transcript scan hit count is inconsistent')
    require(not hits, f'transcript scan found {len(hits)} forbidden tool call(s), e.g. {hits[:2]}')
    return sha256_file(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('check', 'move', 'scan'))
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--draft', action='store_true', help='read an unlocked protocol (check/move only)')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--transcripts')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    if args.draft:
        require(args.command != 'scan', 'a transcript scan needs a locked protocol')
        protocol, phash = read_json(args.protocol), 'DRAFT-' + sha256_file(args.protocol)
    else:
        protocol, phash = load_protocol(args.protocol)
    if args.command == 'check':
        found = answer_bearing(protocol)
        print(json.dumps({'answerBearingInRepository': len(found), 'files': found}, indent=1))
        return 1 if found else 0
    if args.command == 'move':
        result = move(protocol, args.apply)
        print(json.dumps({'apply': result['apply'], 'files': len(result['files']),
                          'bytes': sum(f['bytes'] for f in result['files'])}, indent=1))
        return 0
    require(args.transcripts and args.out, 'scan needs --transcripts and --out')
    output_dir(protocol, Path(args.out).parent, 'the transcript scan')
    result = scan(protocol, phash, read_json(args.transcripts))
    write_json(Path(args.out), result, overwrite=False)
    print(json.dumps({'sessions': len(result['sessions']), 'totalHits': result['totalHits']}, indent=1))
    return 0 if result['totalHits'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
