"""Labeler CLI for the Palo Alto reference-labeling protocol (evaluation-only; never part of the app).

  python -I scripts/evaluation/labels/labelkit.py --labeler L1 --task V01 next
  python -I scripts/evaluation/labels/labelkit.py --labeler L1 --task V01 answer yes
  python -I scripts/evaluation/labels/labelkit.py --labeler L1 --task V01 status

Every answer is appended to a hash-chained log. Answers cannot be edited or undone. When the last
question is answered the label is written and hashed automatically and the session head hash is printed.
The chain is tamper-evident for partial edits only (a whole log could be rewritten consistently), so in
protocol v2 the labeler reports the printed head and aggregation refuses any session whose head differs.
The labeler only ever answers grid cells, yes / no / cannot_tell, counts and candidate numbers: never
pixel coordinates. Under a protocol with `isolation.requireCleanTree`, every command is refused while an
answer-bearing file (isolation.answerBearingGlobs) is still inside the repository.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import imaging  # noqa: E402
import isolation  # noqa: E402
import session  # noqa: E402
from common import (ROOT, V2_DIR, ProtocolError, canonical_json, load_protocol, load_tasks, read_json,  # noqa: E402
                    relative, require, sha256_bytes, sha256_file, write_bytes_atomic, write_json)

CODE_FILES = ('labelkit.py', 'session.py', 'imaging.py', 'snap.py', 'common.py', 'isolation.py')


def utc():
    return dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def code_hashes():
    return {name: sha256_file(HERE / name) for name in CODE_FILES}


def load_task(tasks_path, task_id):
    tasks = load_tasks(tasks_path)
    matches = [t for t in tasks['tasks'] if t['taskId'] == task_id]
    require(len(matches) == 1, f'Unknown task {task_id}')
    return tasks, matches[0]


class Store:
    """Session folder: answers.jsonl (hash chain), questions/NNNN.png, label.json, label.sha256."""

    def __init__(self, sessions_dir, labeler, task_id):
        require(labeler.replace('-', '').replace('_', '').isalnum() and len(labeler) <= 32, 'Labeler id must be short and alphanumeric')
        self.dir = Path(sessions_dir) / labeler / task_id
        self.log = self.dir / 'answers.jsonl'

    def records(self):
        if not self.log.exists():
            return []
        records, prev = [], None
        for line in self.log.read_text().splitlines():
            record = json.loads(line)
            body = {k: v for k, v in record.items() if k != 'hash'}
            require(record['prev'] == prev, 'Answer log chain broken (prev mismatch)')
            require(record['hash'] == sha256_bytes(canonical_json(body)), 'Answer log chain broken (hash mismatch)')
            prev = record['hash']
            records.append(record)
        return records

    def append(self, record):
        records = self.records()
        record['prev'] = records[-1]['hash'] if records else None
        record['hash'] = sha256_bytes(canonical_json(record))
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.log, 'a') as out:
            out.write(json.dumps(record, sort_keys=True) + '\n')
        return record


def prepare(args):
    protocol, protocol_hash = load_protocol(args.protocol)
    isolation.check_clean_tree(protocol)
    tasks, task = load_task(args.tasks, args.task)
    require(tasks['protocol']['sha256'] == protocol_hash, 'Task list was built for a different protocol version')
    image_path = ROOT / task['image']['path']
    require(sha256_file(image_path) == task['image']['sha256'], 'Task image changed since the task list was built')
    if task.get('instructions'):
        require(sha256_file(ROOT / task['instructions']['path']) == task['instructions']['sha256'],
                'Labeler instructions changed since the task list was locked')
    rgb = imaging.load_rgb(image_path)
    store = Store(args.sessions, args.labeler, args.task)
    records = store.records()
    task_hash = sha256_bytes(canonical_json(task))
    if records:
        require(records[0]['taskSha256'] == task_hash and records[0]['protocolSha256'] == protocol_hash,
                'Task or protocol changed since this session started')
    return protocol, protocol_hash, task, task_hash, rgb, store, records


def question_image(args, task, rgb, store, question, seq):
    render = question['render']
    if render['type'] == 'overview':
        path = ROOT / task['overview']['path']
        require(sha256_file(path) == task['overview']['sha256'], 'Overview image changed since the task list was built')
        return path, task['overview']['sha256']
    data = imaging.render_question(rgb, render['context'], render['window'], render['zoom'], render['marker'],
                                   candidates=render.get('candidates', ()), guides=render.get('guides', ()))
    path = store.dir / 'questions' / f'{seq:04d}.png'
    write_bytes_atomic(path, data)
    return path, sha256_bytes(data)


def scratch_dir(args, protocol):
    """Protocol v2: a private scratch folder per (labeler, task); helpers outside it must not be used."""
    if not protocol.get('labelers', {}).get('privateScratch'):
        return None
    folder = Path(args.sessions).parent / 'scratch' / args.labeler / args.task
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def describe(args, task, question, image_path, seq, total_hint, scratch=None):
    kind = question['kind']
    how = {'yesno': 'yes | no | cannot_tell', 'box': 'TOPLEFT:BOTTOMRIGHT cells (e.g. C4:H9) | none',
           'cell': 'one cell (e.g. C4) | none', 'count': f"a whole number 0..{question['options']}",
           'pick': f"a candidate number 1..{question['options']} | 0"}[kind]
    cmd = (f"python -I scripts/evaluation/labels/labelkit.py --labeler {args.labeler} --task {args.task} answer <ANSWER>")
    lines = [
        f"TASK {task['taskId']}  question {seq} (about {total_hint} in total)  feature {question['feature']}",
        f"IMAGE: {image_path}",
        f"QUESTION: {question['text']}",
        f"ANSWER FORMAT: {how}",
        f"ANSWER WITH: {cmd}",
    ]
    if scratch is not None:
        lines.append(f"SCRATCH (yours only, for this task): {relative(scratch)}")
    return '\n'.join(lines)


def finalize(args, protocol_hash, task, task_hash, store, records, result):
    label_path = store.dir / 'label.json'
    if label_path.exists():
        return label_path
    label = {'schema': 'pao-labels-label-v1', 'taskId': task['taskId'], 'labeler': args.labeler, 'protocolSha256': protocol_hash,
             'taskSha256': task_hash, 'imageSha256': task['image']['sha256'], 'answers': len(records),
             'answerLogHead': records[-1]['hash'] if records else None, 'codeSha256': code_hashes(), 'finalizedUtc': utc(),
             'result': json.loads(json.dumps(result, default=float))}
    digest = write_json(label_path, label, overwrite=False)
    write_bytes_atomic(store.dir / 'label.sha256', (digest + '  label.json\n').encode(), overwrite=False)
    return label_path


def command_next(args, echo=True):
    protocol, protocol_hash, task, task_hash, rgb, store, records = prepare(args)
    answers = [r['answer'] for r in records]
    pending, result, asked = session.run(task, protocol, args.labeler, rgb, answers)
    if pending is None:
        path = finalize(args, protocol_hash, task, task_hash, store, records, result)
        if echo:
            print(f"TASK {task['taskId']} is complete for labeler {args.labeler}. Label written: {relative(path)}")
            print(f"SESSION HEAD {args.labeler}/{task['taskId']}: {records[-1]['hash'] if records else None} "
                  "(copy this line into your final report)")
        return None
    seq = len(asked)
    image_path, image_hash = question_image(args, task, rgb, store, pending, seq)
    state = {'seq': seq, 'questionId': pending['id'], 'imageSha256': image_hash, 'textSha256': sha256_bytes(pending['text'].encode())}
    write_json(store.dir / 'pending.json', state)
    if echo:
        print(describe(args, task, pending, image_path, seq, task.get('estimatedQuestions', '?'), scratch_dir(args, protocol)))
    return pending, image_path


def command_answer(args):
    protocol, protocol_hash, task, task_hash, rgb, store, records = prepare(args)
    require(not (store.dir / 'label.json').exists(), 'This task is already complete; answers are closed')
    answers = [r['answer'] for r in records]
    pending, result, asked = session.run(task, protocol, args.labeler, rgb, answers)
    require(pending is not None, 'No question is pending')
    pending_file = store.dir / 'pending.json'
    require(pending_file.exists(), 'Run `next` first so the question image is shown before answering')
    shown = read_json(pending_file)
    require(shown['seq'] == len(asked) and shown['textSha256'] == sha256_bytes(pending['text'].encode()),
            'The pending question changed; run `next` again')
    try:
        value = session.normalize_answer(pending, args.value)
    except ValueError as error:
        print(f'INVALID ANSWER (not recorded): {error}')
        return 1
    record = {'seq': len(asked), 'utc': utc(), 'questionId': pending['id'], 'feature': pending['feature'], 'kind': pending['kind'],
              'purpose': pending['render'].get('purpose', pending['render'].get('type')),
              'position': pending['render'].get('position'), 'imageSha256': shown['imageSha256'],
              'textSha256': shown['textSha256'], 'answer': value, 'taskSha256': task_hash, 'protocolSha256': protocol_hash}
    store.append(record)
    pending_file.unlink()
    print(f'Recorded answer {value!r} for question {len(asked)}.')
    command_next(args)
    return 0


def command_status(args):
    protocol, protocol_hash, task, task_hash, rgb, store, records = prepare(args)
    done = (store.dir / 'label.json').exists()
    print(json.dumps({'task': task['taskId'], 'labeler': args.labeler, 'answers': len(records), 'complete': done,
                      'estimatedQuestions': task.get('estimatedQuestions')}, indent=1))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--labeler', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--tasks', default=str(V2_DIR / 'validation-tasks.json'))
    parser.add_argument('--protocol', default=str(V2_DIR / 'protocol-v2.json'))
    parser.add_argument('--sessions', default=str(V2_DIR / 'sessions'))
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('next')
    answer = sub.add_parser('answer')
    answer.add_argument('value')
    sub.add_parser('status')
    args = parser.parse_args(argv)
    try:
        if args.command == 'next':
            command_next(args)
            return 0
        if args.command == 'answer':
            return command_answer(args)
        return command_status(args)
    except ProtocolError as error:
        print(f'REFUSED: {error}')
        return 2


if __name__ == '__main__':
    sys.exit(main())
