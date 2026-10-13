"""Aggregate independent labelers: per-feature median, agreement threshold, disagreement = unresolved.

  python -I aggregate.py --labelers L1,L2,L3 [--heads reported-heads.json] [--transcript-scan scan.json]
                         [--tasks validation-tasks.json] [--sessions DIR] [--out DIR]

Every label is checked before it is used: label.json against its own label.sha256, the task hash and the
protocol hash; the answer log (answers.jsonl) is re-read and its hash chain rebuilt; the label's answer count
and head must equal the log's; and the session is replayed from the logged answers, which must reproduce
label.json's result exactly. So an edited value in label.json is refused even when its sha256 file and head
were rewritten to match. The output folder gets labels-aggregate.json and labels-manifest.json (hashes of
every label, every answer log and the aggregate). The manifest is written once: labels are hashed before scoring.

A whole session rewritten consistently (log, label and sha256 together) still replays cleanly, so protocol v2
also requires --heads: a coordinator-owned file with the session head each labeler reported when its session
closed (labelkit prints it). Any missing or different head is refused. With `isolation.requireTranscriptScan`
the protocol also requires --transcript-scan (isolation.py scan): it must cover every labeled session with no
forbidden tool call, and its sha256 goes into the manifest.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import imaging  # noqa: E402
import isolation  # noqa: E402
import labelkit  # noqa: E402
import session  # noqa: E402
from common import (ROOT, V2_DIR, canonical_json, load_protocol, load_tasks, read_json, relative, require,  # noqa: E402
                    sha256_bytes, sha256_file, write_json)

LINE = ('L', 'R', 'T', 'B')


def family(name):
    if name in LINE:
        return 'line'
    return {'S': 'profile', 'W': 'contact', 'P': 'paint'}[name[0]]


def task_image(task, cache):
    """Decoded task image (checked against the task's sha256), cached per task."""
    if task['taskId'] not in cache:
        path = ROOT / task['image']['path']
        require(sha256_file(path) == task['image']['sha256'], f"{task['taskId']}: task image changed")
        cache[task['taskId']] = imaging.load_rgb(path)
    return cache[task['taskId']]


def load_label(sessions, labeler, task, protocol, protocol_hash, images=None):
    """(label, label sha256, answer-log sha256), or (None, None, None) when the session has no label yet.

    Refuses unless label.json matches label.sha256, the task and protocol hashes, the rebuilt answer-log chain
    (count, head, per-record task and protocol hashes) and a deterministic replay of the logged answers."""
    name = f'{labeler}/{task["taskId"]}'
    folder = Path(sessions) / labeler / task['taskId']
    path = folder / 'label.json'
    if not path.exists():
        return None, None, None
    digest = sha256_file(path)
    recorded = (folder / 'label.sha256').read_text().split()[0]
    require(digest == recorded, f'{name}: label.json does not match label.sha256')
    label = read_json(path)
    task_hash = sha256_bytes(canonical_json(task))
    require(label['taskSha256'] == task_hash, f'{name}: task changed after labeling')
    require(label['protocolSha256'] == protocol_hash, f'{name}: different protocol')
    store = labelkit.Store(sessions, labeler, task['taskId'])
    require(store.log.exists(), f'{name}: no answer log (answers.jsonl)')
    records = store.records()  # refuses a broken chain
    require(label.get('answers') == len(records), f'{name}: label answer count differs from the answer log')
    require(label.get('answerLogHead') == (records[-1]['hash'] if records else None), f'{name}: label head differs from the answer log')
    for record in records:
        require(record.get('taskSha256') == task_hash and record.get('protocolSha256') == protocol_hash,
                f'{name}: an answer record names another task or protocol')
    rgb = task_image(task, {} if images is None else images)
    pending, result, _ = session.run(task, protocol, labeler, rgb, [r['answer'] for r in records])
    require(pending is None, f'{name}: replaying the answer log leaves a question pending')
    require(json.loads(json.dumps(result, default=float)) == label['result'], f'{name}: replaying the answer log does not give label.json')
    return label, digest, sha256_file(store.log)


def combine(feature, entries, spread_max, column_spread_max=None):
    """entries: {labeler: feature result or None}. Returns the aggregated record.

    column_spread_max (protocol v2): roofline stations sit at each labeler's own columns; they are combined only
    when those columns differ by at most this many pixels.
    """
    values, columns = {}, {}
    for labeler, result in entries.items():
        if result and result.get('status') == 'resolved' and result.get('value') is not None:
            values[labeler] = float(result['value'])
            if result.get('fixed') is not None:
                columns[labeler] = int(result['fixed'])
    record = {'family': family(feature), 'values': values, 'columns': columns}
    if len(values) < len(entries):
        missing = sorted(set(entries) - set(values))
        record.update(status='unresolved', reason=f'not resolved by {",".join(missing)}')
        return record
    if column_spread_max is not None and feature.startswith('S') and len(columns) == len(values):
        column_spread = max(columns.values()) - min(columns.values())
        record['columnSpread'] = column_spread
        if column_spread > column_spread_max:
            record.update(status='unresolved', reason=f'station columns differ by {column_spread} px > {column_spread_max} px')
            return record
    spread = max(values.values()) - min(values.values())
    record['spread'] = round(spread, 4)
    limit = spread_max['line' if feature in LINE else 'pixel']
    if spread > limit:
        record.update(status='unresolved', reason=f'spread {spread:.2f} px > {limit} px')
        return record
    record.update(status='resolved', value=float(np.median(list(values.values()))))
    if columns:
        record['column'] = int(np.median(list(columns.values())))
    return record


def aggregate(tasks_path, sessions, labelers, protocol, protocol_hash, heads=None, transcript_scan=None):
    """Returns (per-task result, label sha256 by session, evidence). evidence holds the answer-log sha256 of every
    session and the transcript-scan sha256 (or None)."""
    tasks = load_tasks(tasks_path)
    require(tasks['protocol']['sha256'] == protocol_hash, 'Task list was built for a different protocol')
    spread_max = protocol['aggregation']['spreadMaxPx']
    column_spread_max = protocol['aggregation'].get('columnSpreadMaxPx')
    if protocol['aggregation'].get('requireReportedHeads'):
        require(heads is not None, 'This protocol requires --heads (session heads recorded by the coordinator)')
    if isolation.settings(protocol).get('requireTranscriptScan'):
        require(transcript_scan is not None, 'This protocol requires --transcript-scan (isolation.py scan of every labeler transcript)')
    out, hashes, logs, images = {}, {}, {}, {}
    for task in tasks['tasks']:
        labels = {}
        for labeler in labelers:
            name = f'{labeler}/{task["taskId"]}'
            label, digest, log_digest = load_label(sessions, labeler, task, protocol, protocol_hash, images)
            labels[labeler] = label
            if digest:
                hashes[name] = digest
                logs[name] = log_digest
            if label is not None and heads is not None:
                require(name in heads, f'{name}: no session head recorded by the coordinator')
                require(label.get('answerLogHead') == heads[name], f'{name}: answer log head differs from the reported head')
        images.pop(task['taskId'], None)
        if not all(labels.values()):
            out[task['taskId']] = {'status': 'incomplete', 'missing': [k for k, v in labels.items() if not v]}
            continue
        names = sorted({name for label in labels.values() for name in label['result'].get('features', {})})
        features = {}
        for name in names:
            features[name] = combine(name, {k: v['result'].get('features', {}).get(name) for k, v in labels.items()}, spread_max,
                                     column_spread_max)
        counts = {k: v['result'].get('wheelCount') for k, v in labels.items()}
        if task['kind'] == 'car' and len(set(counts.values())) > 1:
            for name in [n for n in features if n.startswith('W')]:
                features[name].update(status='unresolved', reason=f'wheel counts differ {counts}')
        out[task['taskId']] = {'status': 'complete', 'kind': task['kind'], 'features': features, 'wheelCounts': counts}
    scan_hash = None
    if transcript_scan is not None:
        scan_hash = isolation.verify_scan(transcript_scan, protocol, protocol_hash, sorted(hashes))
    return out, hashes, {'answerLogSha256': logs, 'transcriptScanSha256': scan_hash}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--labelers', required=True)
    parser.add_argument('--heads', help='coordinator-owned JSON {"heads": {"<labeler>/<task>": "<session head sha256>"}}')
    parser.add_argument('--transcript-scan', help='isolation.py scan output covering every labeler session')
    parser.add_argument('--tasks', default=str(V2_DIR / 'validation-tasks.json'))
    parser.add_argument('--protocol', default=str(V2_DIR / 'protocol-v2.json'))
    parser.add_argument('--sessions', default=str(V2_DIR / 'sessions'))
    parser.add_argument('--out', help='default: <coordinator folder>/aggregate under isolation, else v2/aggregate')
    args = parser.parse_args(argv)
    protocol, protocol_hash = load_protocol(args.protocol)
    labelers = args.labelers.split(',')
    require(len(labelers) == protocol['aggregation']['labelers'], f"Protocol requires {protocol['aggregation']['labelers']} labelers")
    out_dir = isolation.output_dir(protocol, args.out or isolation.default_output(protocol, 'aggregate', V2_DIR / 'aggregate'),
                                   'the label aggregate')
    require(not (out_dir / 'labels-manifest.json').exists(), 'labels-manifest.json exists; aggregates are written once')
    heads = read_json(args.heads)['heads'] if args.heads else None
    result, hashes, evidence = aggregate(args.tasks, args.sessions, labelers, protocol, protocol_hash, heads, args.transcript_scan)
    agg_hash = write_json(out_dir / 'labels-aggregate.json', {'schema': 'pao-labels-aggregate-v1', 'protocolSha256': protocol_hash,
                                                             'labelers': labelers, 'tasks': result}, overwrite=False)
    manifest = {'schema': 'pao-labels-manifest-v1', 'protocolSha256': protocol_hash, 'tasksSha256': sha256_file(args.tasks),
                'reportedHeadsSha256': sha256_file(args.heads) if args.heads else None,
                'transcriptScanSha256': evidence['transcriptScanSha256'],
                'labelSha256': hashes, 'answerLogSha256': evidence['answerLogSha256'], 'aggregateSha256': agg_hash,
                'sessions': relative(args.sessions), 'writtenUtc': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                'statement': 'Hashes recorded before any scoring against the sealed answer key. Every label was checked against '
                             'its rebuilt answer-log chain and a deterministic replay before aggregation.'}
    write_json(out_dir / 'labels-manifest.json', manifest, overwrite=False)
    done = sum(1 for v in result.values() if v['status'] == 'complete')
    print(f'{done}/{len(result)} tasks complete; manifest {relative(out_dir / "labels-manifest.json")}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
