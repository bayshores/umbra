"""Score labels against the sealed known answers (DEVELOPMENT validation only).

  python -I score.py [--aggregate DIR] [--sessions DIR] [--out DIR]

Refuses unless: the protocol is locked, validation-lock.json matches the task list, answer key and
coverage files, and labels-manifest.json matches every label file, every answer log (when recorded) and the
aggregate (labels were hashed before scoring). Errors are in full-resolution pixels (1024-width = /2).
Under a protocol with `isolation` the sealed files are read from, and the score is written to, the
coordinator-only folder outside the repository (isolation.py).
Synthetic: exact signed error. LiDAR: coarse consistency with an inner-bound reference plus a
registration allowance (protocol scoring.lidarCoarse); B and wheel contacts are one-sided.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import isolation  # noqa: E402
from common import ROOT, V2_DIR, load_protocol, load_tasks, read_json, relative, require, sha256_file, write_json  # noqa: E402
from truth import Truth  # noqa: E402

OUTWARD = {'L': -1, 'R': 1, 'T': -1, 'B': 1}


def reference(truth, feature, column):
    """(reference value, kind) where kind is exact | inner | onesided; None when undefined."""
    if feature in OUTWARD:
        value = truth.extreme(feature)
        if truth.source == 'synthetic':
            return value, 'exact'
        return value, 'onesided' if feature == 'B' else 'inner'
    if feature[0] in 'SW':
        if column is None:
            return None, None
        span = truth.column_span(column)
        if span is None:
            return None, None
        value = span[0] if feature[0] == 'S' else span[1]
        if truth.source == 'synthetic':
            return value, 'exact'
        return value, 'inner' if feature[0] == 'S' else 'onesided'
    row = truth.paint_near(column)
    if row is None:
        return None, None
    return row, 'exact' if truth.source == 'synthetic' else 'inner'


def outward_sign(feature):
    if feature in OUTWARD:
        return OUTWARD[feature]
    return {'S': -1, 'W': 1, 'P': 1}[feature[0]]


def evaluate(value, ref, kind, feature, entry, coarse):
    e = value - ref
    row = {'value': round(value, 4), 'reference': round(ref, 4), 'error': round(e, 4), 'referenceKind': kind}
    if kind != 'exact':
        sign = outward_sign(feature)
        outward = sign * e
        allow = coarse['registrationAllowancePx']
        spacing = (entry.get('spacingPx') or {}).get(feature if feature in OUTWARD else ('T' if feature[0] == 'S' else 'B')) or 0.0
        if feature[0] == 'P':
            spacing = 0.0
        if kind == 'onesided':
            row['consistent'] = bool(outward >= -allow)
        elif feature[0] == 'P':
            row['consistent'] = bool(abs(e) <= allow)
        else:
            row['consistent'] = bool(-allow <= outward <= spacing + allow)
        row['outwardError'] = round(outward, 4)
    return row


def stats(rows):
    errs = np.array([abs(r['error']) for r in rows if r['referenceKind'] == 'exact'])
    out = {'n': len(rows)}
    if len(errs):
        out.update(exactN=int(len(errs)), medianAbs=round(float(np.median(errs)), 3), p90Abs=round(float(np.percentile(errs, 90)), 3),
                   maxAbs=round(float(errs.max()), 3), meanSigned=round(float(np.mean([r['error'] for r in rows if r['referenceKind'] == 'exact'])), 3))
    ranged = [r for r in rows if 'rangeHalfWidthPx' in r]
    if ranged:  # protocol v2 (search.ambiguity): how many of these resolved features are range results (+/- > 0)
        out['rangeN'] = sum(1 for r in ranged if r['rangeHalfWidthPx'] > 0)
    coarse = [r for r in rows if 'consistent' in r]
    if coarse:
        out.update(lidarN=len(coarse), lidarConsistentShare=round(sum(r['consistent'] for r in coarse) / len(coarse), 3),
                   lidarMedianOutward=round(float(np.median([r['outwardError'] for r in coarse])), 3))
    return out


def family(feature):
    return {'L': 'extreme', 'R': 'extreme', 'T': 'extreme', 'B': 'extreme'}.get(feature) or {'S': 'profile', 'W': 'contact', 'P': 'paint'}[feature[0]]


def score(protocol, phash, base, aggregate_dir, sessions, set_protocol_sha=None):
    """set_protocol_sha: only for tool self-tests of a DRAFT protocol on an existing set (the set's own protocol hash)."""
    lock = read_json(base / 'validation-lock.json')
    require(lock['protocolSha256'] == (set_protocol_sha or phash), 'Validation set was built with another protocol')
    require(sha256_file(base / 'validation-tasks.json') == lock['validationTasksSha256'], 'validation-tasks.json changed after the lock')
    if 'tasksInternalSha256' in lock:
        require(sha256_file(base / 'kit-internal/tasks-internal.json') == lock['tasksInternalSha256'], 'kit-internal task file changed after the lock')
    if 'labelerInstructions' in lock:
        require(sha256_file(ROOT / lock['labelerInstructions']['path']) == lock['labelerInstructions']['sha256'],
                'labeler instructions changed after the lock')
    sealed = isolation.sealed_dir(base, lock)
    require(sha256_file(sealed / 'answer-key.json') == lock['answerKeySha256'], 'answer key changed after the lock')
    for task_id, digest in lock['coverageSha256'].items():
        require(sha256_file(sealed / f'coverage/{task_id}.npz') == digest, f'coverage {task_id} changed after the lock')
    manifest = read_json(aggregate_dir / 'labels-manifest.json')
    require(manifest['protocolSha256'] == phash, 'labels were made under another protocol')
    if protocol.get('aggregation', {}).get('requireReportedHeads'):
        require(manifest.get('reportedHeadsSha256'), 'this protocol requires labels aggregated with coordinator-recorded heads')
    if isolation.settings(protocol).get('requireTranscriptScan'):
        require(manifest.get('transcriptScanSha256'), 'this protocol requires a transcript scan recorded before scoring')
        require(manifest.get('answerLogSha256'), 'this protocol requires answer-log hashes in the labels manifest')
    require(sha256_file(aggregate_dir / 'labels-aggregate.json') == manifest['aggregateSha256'], 'aggregate changed after hashing')
    for name, digest in manifest['labelSha256'].items():
        labeler, task_id = name.split('/')
        require(sha256_file(Path(sessions) / labeler / task_id / 'label.json') == digest, f'label {name} changed after hashing')
    for name, digest in (manifest.get('answerLogSha256') or {}).items():
        labeler, task_id = name.split('/')
        require(sha256_file(Path(sessions) / labeler / task_id / 'answers.jsonl') == digest, f'answer log {name} changed after hashing')
    key = read_json(sealed / 'answer-key.json')['tasks']
    aggregate = read_json(aggregate_dir / 'labels-aggregate.json')
    coarse = protocol['scoring']['lidarCoarse']
    tasks = {t['taskId']: t for t in load_tasks(base / 'validation-tasks.json')['tasks']}
    rows = {'aggregate': []}
    resolution = {'aggregate': [0, 0]}
    snapping = []
    for task_id, entry in key.items():
        if entry['source'] == 'synthetic':
            cov = np.load(sealed / f'coverage/{task_id}.npz')
            for name in ('coverage', 'paintCoverage'):
                if name in cov:
                    entry['_' + name] = cov[name]
        truth = Truth(entry)
        agg = aggregate['tasks'].get(task_id, {})
        if agg.get('status') != 'complete':
            continue
        for labeler in aggregate['labelers']:
            rows.setdefault(labeler, [])
            resolution.setdefault(labeler, [0, 0])
            label = read_json(Path(sessions) / labeler / task_id / 'label.json')
            for feature, result in label['result'].get('features', {}).items():
                resolution[labeler][1] += 1
                if result.get('status') != 'resolved':
                    continue
                resolution[labeler][0] += 1
                column = result.get('fixed') if feature not in OUTWARD else None
                ref, kind = reference(truth, feature, column)
                if ref is None:
                    continue
                row = evaluate(float(result['value']), ref, kind, feature, entry, coarse)
                row.update(task=task_id, feature=feature, family=family(feature), source=entry['source'], labeler=labeler)
                if 'rangeHalfWidthPx' in result:
                    row['rangeHalfWidthPx'] = result['rangeHalfWidthPx']
                rows[labeler].append(row)
                if kind == 'exact' and feature not in OUTWARD:
                    raw = evaluate(float(result['estimate']), ref, kind, feature, entry, coarse)
                    snapping.append({'task': task_id, 'feature': feature, 'labeler': labeler, 'rawError': raw['error'],
                                     'snappedError': row['error'], 'snapped': result.get('snapped') is not None})
        for feature, record in agg['features'].items():
            resolution['aggregate'][1] += 1
            if record['status'] != 'resolved':
                continue
            resolution['aggregate'][0] += 1
            column = record.get('column')
            if feature[0] in 'SP' and column is None:
                column = tasks[task_id]['stationColumns'][int(feature[1:]) - 1]
            ref, kind = reference(truth, feature, column)
            if ref is None:
                continue
            row = evaluate(record['value'], ref, kind, feature, entry, coarse)
            row.update(task=task_id, feature=feature, family=family(feature), source=entry['source'], labeler='aggregate',
                       spread=record.get('spread'))
            if 'rangeHalfWidthPx' in record:
                row['rangeHalfWidthPx'] = record['rangeHalfWidthPx']
            rows['aggregate'].append(row)
    summary = {}
    for who, items in rows.items():
        summary[who] = {'resolvedShare': round(resolution[who][0] / resolution[who][1], 3) if resolution[who][1] else None,
                        'resolved': resolution[who][0], 'features': resolution[who][1]}
        for fam in ('extreme', 'profile', 'contact', 'paint'):
            for source in ('synthetic', 'lidar'):
                subset = [r for r in items if r['family'] == fam and r['source'] == source]
                if subset:
                    summary[who][f'{fam}/{source}'] = stats(subset)
    gate = gate_check(protocol, summary.get('aggregate', {}), resolution['aggregate'], rows['aggregate'])
    repeats = repeat_precision(key, aggregate, sessions, protocol['aggregation'].get('columnSpreadMaxPx'))
    snap_summary = {}
    if snapping:
        snap_summary = {'n': len(snapping), 'snappedShare': round(np.mean([s['snapped'] for s in snapping]), 3),
                        'medianAbsRaw': round(float(np.median([abs(s['rawError']) for s in snapping])), 3),
                        'medianAbsSnapped': round(float(np.median([abs(s['snappedError']) for s in snapping])), 3)}
    return {'summary': summary, 'gate': gate, 'snapping': snap_summary, 'repeatPrecision': repeats, 'rows': rows,
            'hashes': {'manifest': sha256_file(aggregate_dir / 'labels-manifest.json'), 'validationLock': sha256_file(base / 'validation-lock.json'),
                       'protocol': phash}}


def repeat_precision(key, aggregate, sessions, column_spread_max=None):
    """Within-labeler differences between a task and its blind repeat.

    A station feature (S, P) is compared only when the two sessions measured it at columns no more than
    `column_spread_max` px apart (protocol v2 aggregation.columnSpreadMaxPx; v1 stations shared their columns, so
    there the columns must be equal). Pairs at more distant columns measure different points of the object: they are
    listed in `columnMismatched` and kept out of n, median and max."""
    tolerance = 0 if column_spread_max is None else column_spread_max
    diffs, mismatched = {}, {}
    for task_id, entry in key.items():
        original = entry.get('repeatOf')
        if not original:
            continue
        if aggregate['tasks'].get(task_id, {}).get('status') != 'complete' or aggregate['tasks'].get(original, {}).get('status') != 'complete':
            continue
        for labeler in aggregate['labelers']:
            a = read_json(Path(sessions) / labeler / original / 'label.json')['result'].get('features', {})
            b = read_json(Path(sessions) / labeler / task_id / 'label.json')['result'].get('features', {})
            for feature in sorted(set(a) & set(b)):
                if feature.startswith('W'):
                    continue  # wheel columns are labeler-chosen and differ between sessions
                if a[feature].get('status') == 'resolved' and b[feature].get('status') == 'resolved':
                    item = {'task': task_id, 'feature': feature, 'difference': round(float(b[feature]['value']) - float(a[feature]['value']), 4)}
                    ca, cb = a[feature].get('fixed'), b[feature].get('fixed')
                    if ca != cb:
                        # v2 roofline stations follow each session's own left/right answers, so columns can differ
                        item['columns'] = [ca, cb]
                        if ca is None or cb is None or abs(int(ca) - int(cb)) > tolerance:
                            mismatched.setdefault(labeler, []).append(item)
                            continue
                    diffs.setdefault(labeler, []).append(item)
    out = {}
    for labeler in sorted(set(diffs) | set(mismatched)):
        items = diffs.get(labeler, [])
        values = np.abs([d['difference'] for d in items])
        out[labeler] = {'n': len(items), 'medianAbs': round(float(np.median(values)), 3) if len(items) else None,
                        'maxAbs': round(float(values.max()), 3) if len(items) else None, 'items': items}
        if labeler in mismatched:
            out[labeler]['columnMismatched'] = mismatched[labeler]
    return out


def gate_check(protocol, agg_summary, resolution, rows):
    g = protocol['scoring']['syntheticGate']
    min_n = g.get('minExactPerFamily')  # protocol v2 draft onward; v1 had no minimum (it "passed" families on n = 2 and 4)
    checks = {}
    def check(name, family_key, med, mx):
        s = agg_summary.get(family_key)
        if not s or 'medianAbs' not in s:
            checks[name] = {'pass': False, 'reason': 'no resolved synthetic features'}
            return
        checks[name] = {'pass': s['medianAbs'] <= med and s['maxAbs'] <= mx, 'medianAbs': s['medianAbs'], 'maxAbs': s['maxAbs'],
                        'limits': [med, mx], 'n': s['exactN']}
        if min_n is not None:
            checks[name]['minN'] = min_n
            if s['exactN'] < min_n:
                checks[name].update({'pass': False, 'reason': f"only {s['exactN']} exact features (< {min_n})"})
    check('carExtremes', 'extreme/synthetic', g['carExtremeMedianAbsPx'], g['carExtremeMaxAbsPx'])
    check('carProfile', 'profile/synthetic', g['carProfileMedianAbsPx'], g['carProfileMaxAbsPx'])
    check('wheelContacts', 'contact/synthetic', g['wheelContactMedianAbsPx'], g['wheelContactMaxAbsPx'])
    check('paint', 'paint/synthetic', g['paintMedianAbsPx'], g['paintMaxAbsPx'])
    share = resolution[0] / resolution[1] if resolution[1] else 0.0
    checks['resolvedShare'] = {'pass': share >= g['minResolvedShare'], 'value': round(share, 3), 'limit': g['minResolvedShare']}
    coarse = [r for r in rows if 'consistent' in r]
    lim = protocol['scoring']['lidarCoarse']['minConsistentShare']
    cshare = sum(r['consistent'] for r in coarse) / len(coarse) if coarse else 0.0
    checks['lidarCoarse'] = {'pass': bool(coarse) and cshare >= lim, 'value': round(cshare, 3), 'limit': lim, 'n': len(coarse)}
    return {'pass': all(c['pass'] for c in checks.values()), 'checks': checks}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default=str(V2_DIR))
    parser.add_argument('--protocol', default=str(V2_DIR / 'protocol-v2.json'))
    parser.add_argument('--aggregate', help='default: <coordinator folder>/aggregate under isolation, else v2/aggregate')
    parser.add_argument('--sessions', default=str(V2_DIR / 'sessions'))
    parser.add_argument('--out', help='default: <coordinator folder>/scores under isolation, else v2/scores')
    args = parser.parse_args(argv)
    protocol, phash = load_protocol(args.protocol)
    aggregate_dir = Path(args.aggregate) if args.aggregate else isolation.default_output(protocol, 'aggregate', V2_DIR / 'aggregate')
    out_dir = isolation.output_dir(protocol, args.out or isolation.default_output(protocol, 'scores', V2_DIR / 'scores'), 'the score')
    result = score(protocol, phash, Path(args.base), aggregate_dir, args.sessions)
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    result['scoredUtc'] = stamp
    path = Path(out_dir) / f'score-{stamp}.json'
    write_json(path, json.loads(json.dumps(result, default=float)), overwrite=False)
    print(json.dumps({'gate': result['gate'], 'aggregate': result['summary'].get('aggregate')}, indent=1))
    print('written', relative(path))
    return 0


if __name__ == '__main__':
    sys.exit(main())
