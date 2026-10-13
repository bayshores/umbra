"""Tool self-test: run oracle 'labelers' (known answers, optional answer flips) through the real session
logic, then aggregate and score. Measures the method floor (quantisation + snapping with a perfect
labeler) and robustness to random answer errors. Writes ONLY into the given output folder; never into
the labels area. Not a labeling run.

  python -I selftest.py BASE OUTDIR [--protocol PATH [--draft]] [--flips 0,0,0] [--seeds 0-9 | --seed N]
                                    [--tasks V01,V02] [--keep-sessions]

Each oracle's answer flips come from random.Random(stable_seed(taskId, labeler, seed)): the same seed gives
the same flips in every process. (Until 2026-10-12 the seed was Python's hash() of a tuple of strings, which
changes per process, so the stored v1 flip-0.05 / flip-0.10 results were single unrepeatable draws.)
With several seeds the summary reports the gate pass count and the range of every number.
--draft runs an unlocked protocol on an existing (locked) set: the result is a tool check, never a reference.
Each oracle session is written like a labelkit session (hash-chained answers.jsonl, label.json, label.sha256), so
aggregation checks it the same way (chain, head, deterministic replay). Under a protocol with `isolation` the
output folder must be outside the repository: rows and aggregates hold exact reference values.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import aggregate as agg  # noqa: E402
import imaging  # noqa: E402
import isolation  # noqa: E402
import labelkit  # noqa: E402
import score as scoring  # noqa: E402
import session  # noqa: E402
from common import (ROOT, canonical_json, load_protocol, load_tasks, read_json, require, sha256_bytes, sha256_file,  # noqa: E402
                    stable_seed, write_bytes_atomic, write_json)
from oracle import Oracle  # noqa: E402
from truth import Truth  # noqa: E402

FAMILIES = ('extreme/synthetic', 'profile/synthetic', 'contact/synthetic', 'paint/synthetic')


def oracle_seed(task_id, labeler, seed):
    return stable_seed(task_id, labeler, seed)


def drive(task, protocol, labeler, rgb, oracle):
    ctx = session.Context(task, protocol, labeler, rgb)
    gen = session.GENERATORS[task['kind']](ctx)
    answers = []
    try:
        question = next(gen)
        while True:
            answer = session.normalize_answer(question, oracle.answer(question))
            answers.append(answer)
            question = gen.send(answer)
    except StopIteration as stop:
        return stop.value, answers


def write_answer_log(store, task_hash, protocol_hash, asked, answers):
    """Write a labelkit-format hash-chained answers.jsonl in one go (no clock fields, so it is deterministic)."""
    lines, prev = [], None
    for seq, (question, answer) in enumerate(zip(asked, answers), 1):
        record = {'seq': seq, 'questionId': question['id'], 'feature': question['feature'], 'kind': question['kind'],
                  'answer': answer, 'taskSha256': task_hash, 'protocolSha256': protocol_hash, 'selftest': True, 'prev': prev}
        record['hash'] = prev = sha256_bytes(canonical_json(record))
        lines.append(json.dumps(record, sort_keys=True))
    store.dir.mkdir(parents=True, exist_ok=True)
    write_bytes_atomic(store.log, ('\n'.join(lines) + '\n').encode() if lines else b'', overwrite=False)
    records = store.records()
    require(len(records) == len(answers), 'self-test answer log did not round-trip')
    return records[-1]['hash'] if records else None


def write_label(sessions, labeler, task, protocol_hash, result, answers, asked, seed):
    folder = Path(sessions) / labeler / task['taskId']
    task_hash = sha256_bytes(canonical_json(task))
    head = write_answer_log(labelkit.Store(sessions, labeler, task['taskId']), task_hash, protocol_hash, asked, answers)
    label = {'schema': 'pao-labels-label-v1', 'taskId': task['taskId'], 'labeler': labeler, 'protocolSha256': protocol_hash,
             'taskSha256': task_hash, 'imageSha256': task['image']['sha256'], 'answers': len(answers),
             'answerLogHead': head, 'selftest': True, 'oracleSeed': seed,
             'finalizedUtc': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
             'result': json.loads(json.dumps(result, default=float))}
    digest = write_json(folder / 'label.json', label)
    write_bytes_atomic(folder / 'label.sha256', (digest + '  label.json\n').encode())
    return label['answerLogHead']


def load_set(base, wanted):
    tasks_doc = load_tasks(base / 'validation-tasks.json')
    sealed = isolation.sealed_dir(base, read_json(base / 'validation-lock.json'))
    key = read_json(sealed / 'answer-key.json')['tasks']
    prepared = []
    for task in tasks_doc['tasks']:
        if wanted and task['taskId'] not in wanted:
            continue
        entry = dict(key[task['taskId']])
        if entry['source'] == 'synthetic':
            cov = np.load(sealed / f"coverage/{task['taskId']}.npz")
            for name in ('coverage', 'paintCoverage'):
                if name in cov:
                    entry['_' + name] = cov[name]
        prepared.append((task, entry, Truth(entry), imaging.load_rgb(ROOT / task['image']['path'])))
    return tasks_doc, prepared


def run_once(base, out, protocol, phash, flips, seed, tasks_doc, prepared, sessions, set_protocol_sha=None, quiet=False):
    labelers = [f'O{k + 1}' for k in range(len(flips))]
    per_task, heads = {}, {}
    for task, entry, truth, rgb in prepared:
        counts = []
        for labeler, flip in zip(labelers, flips):
            oracle = Oracle(truth, task, flip=flip, seed=oracle_seed(task['taskId'], labeler, seed))
            result, answers = drive(task, protocol, labeler, rgb, oracle)
            replay_pending, replay_result, asked = session.run(task, protocol, labeler, rgb, answers)
            assert replay_pending is None and json.dumps(replay_result, default=float) == json.dumps(result, default=float), 'replay mismatch'
            heads[f"{labeler}/{task['taskId']}"] = write_label(sessions, labeler, task, phash, result, answers, asked, seed)
            counts.append(len(answers))
        per_task[task['taskId']] = {'questions': counts, 'kind': task['kind'], 'source': entry['source']}
        if not quiet:
            print(task['taskId'], task['kind'], entry['source'], 'questions', counts, flush=True)
    copy = dict(tasks_doc, schema='pao-labels-tasks-selftest-copy', tasks=[t for t, *_ in prepared])
    copy.pop('internal', None)
    if set_protocol_sha:
        copy['protocol'] = {'sha256': phash, 'note': f'self-test copy; set built with {set_protocol_sha}'}
    sub_tasks = out / 'validation-tasks.subset.json'
    write_json(sub_tasks, copy)
    result, label_hashes, evidence = agg.aggregate(sub_tasks, sessions, labelers, protocol, phash, heads)
    agg_dir = out / 'aggregate'
    agg_hash = write_json(agg_dir / 'labels-aggregate.json', {'schema': 'pao-labels-aggregate-v1', 'protocolSha256': phash,
                                                             'labelers': labelers, 'tasks': result})
    write_json(agg_dir / 'labels-manifest.json', {'protocolSha256': phash, 'labelSha256': label_hashes,
                                                  'answerLogSha256': evidence['answerLogSha256'], 'aggregateSha256': agg_hash})
    scored = scoring.score(protocol, phash, base, agg_dir, sessions, set_protocol_sha=set_protocol_sha)
    report = {'flips': dict(zip(labelers, flips)), 'seed': seed, 'seedRule': 'random.Random(int(sha256(f"{taskId}|{labeler}|{seed}")[:8], 16))',
              'protocolSha256': phash, 'aggregateSha256': agg_hash, 'questions': per_task, 'summary': scored['summary'],
              'gate': scored['gate'], 'snapping': scored['snapping']}
    write_json(out / 'selftest-report.json', report)
    write_json(out / 'selftest-rows.json', scored['rows'])
    return report


def parse_seeds(args):
    if args.seeds:
        if '-' in args.seeds:
            lo, hi = (int(x) for x in args.seeds.split('-'))
            return list(range(lo, hi + 1))
        return [int(x) for x in args.seeds.split(',')]
    return [args.seed]


def summarize(reports):
    def rng(values):
        values = [v for v in values if v is not None]
        return None if not values else {'min': round(min(values), 3), 'median': round(float(np.median(values)), 3), 'max': round(max(values), 3)}
    agg_summaries = [r['summary']['aggregate'] for r in reports]
    out = {'runs': len(reports), 'gatePass': sum(r['gate']['pass'] for r in reports),
           'resolvedShare': rng([s['resolvedShare'] for s in agg_summaries]),
           'checkPass': {name: sum(r['gate']['checks'][name]['pass'] for r in reports) for name in reports[0]['gate']['checks']}}
    for fam in FAMILIES:
        out[fam] = {'medianAbs': rng([s.get(fam, {}).get('medianAbs') for s in agg_summaries]),
                    'maxAbs': rng([s.get(fam, {}).get('maxAbs') for s in agg_summaries]),
                    'n': rng([s.get(fam, {}).get('exactN') for s in agg_summaries])}
    out['perSeed'] = [{'seed': r['seed'], 'gatePass': r['gate']['pass'], 'resolvedShare': r['summary']['aggregate']['resolvedShare'],
                       'aggregateSha256': r['aggregateSha256']} for r in reports]
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('base')
    parser.add_argument('out')
    parser.add_argument('--protocol')
    parser.add_argument('--draft', action='store_true')
    parser.add_argument('--flips', default='0,0,0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seeds', default='', help='e.g. 0-19 or 3,5,8; one output folder per seed plus summary.json')
    parser.add_argument('--tasks', default='')
    parser.add_argument('--keep-sessions', action='store_true')
    args = parser.parse_args(argv)
    base, out = Path(args.base), Path(args.out)
    set_protocol_sha = None
    if args.draft:
        protocol = read_json(args.protocol)
        phash = 'DRAFT-' + sha256_file(args.protocol)
        set_protocol_sha = read_json(base / 'validation-lock.json')['protocolSha256']
    else:
        protocol, phash = load_protocol(args.protocol)
    isolation.output_dir(protocol, out, 'self-test output (rows and aggregates hold exact reference values)')
    flips = [float(x) for x in args.flips.split(',')]
    wanted = set(filter(None, args.tasks.split(',')))
    tasks_doc, prepared = load_set(base, wanted)
    seeds = parse_seeds(args)
    multi = len(seeds) > 1 or bool(args.seeds)
    reports = []
    for seed in seeds:
        run_dir = out / f'seed-{seed}' if multi else out
        run_dir.mkdir(parents=True, exist_ok=True)
        if args.keep_sessions:
            report = run_once(base, run_dir, protocol, phash, flips, seed, tasks_doc, prepared, run_dir / 'sessions', set_protocol_sha, multi)
        else:
            with tempfile.TemporaryDirectory(dir=run_dir, prefix='.sessions-') as tmp:
                report = run_once(base, run_dir, protocol, phash, flips, seed, tasks_doc, prepared, Path(tmp), set_protocol_sha, multi)
        reports.append(report)
        agg_s = report['summary']['aggregate']
        print(json.dumps({'seed': seed, 'gate': report['gate']['pass'], 'resolvedShare': agg_s['resolvedShare'],
                          'aggregateSha256': report['aggregateSha256'][:12]}), flush=True)
    if multi:
        summary = {'flips': flips, 'seeds': seeds, 'protocolSha256': phash, 'draft': bool(args.draft), **summarize(reports)}
        write_json(out / 'summary.json', summary)
        print(json.dumps({k: v for k, v in summary.items() if k != 'perSeed'}, indent=1))
    else:
        print(json.dumps({'gate': reports[0]['gate'], 'aggregate': reports[0]['summary']['aggregate'], 'snapping': reports[0]['snapping']},
                         indent=1))


if __name__ == '__main__':
    main()
