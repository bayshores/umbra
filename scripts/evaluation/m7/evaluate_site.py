"""M7 per-site evaluation: the pinned scoring code on one site that run_heldout.py built, then score.py site.

  nice -n 10 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -I -B scripts/evaluation/m7/evaluate_site.py \
      --site .local/evaluation/pao-accuracy/m7/<log8>-<crossing> --out .local/evaluation/pao-accuracy/m7/<log8>-<crossing>-eval \
      [--paint <paint_heldout.py folder>] [--labels <held-out label aggregate>] (--protocol-lock-sha256 <sha256> | --rehearse)

Steps, each with nice -n 10 from the repository root, stdout and stderr kept in the output folder:
  1. pao-baseline.mjs --scene (metrics B, C, I; J's Node values)
  2. a site with the recorded vehicle's LiDAR shape (built-shape): metric-f.mjs rows, metric_f_car_top.py profile (the
     scoring sweeps that ground.json carShape.sweeps.scoring lists, nothing else) and score, under metric F protocol v4
     (pinned in metricProtocols.F). No --box-rows-from: a held-out scene has no metric F v2 rows to reproduce.
  3. metric_k.py (the scene sweep; on a built-shape site also the scoring sweeps)
  4. only when the labels addendum exists (protocol-lock labels.scoringCondition): the D/E scorer it pins, after its sha256
     is checked, with --site, --baseline, --metric-f-results (those that exist), --labels and --out labels-de.json.
     Held-out: --labels is required exactly for the sites the committed label manifest lists with labels, and must
     have that sha256 (checked before the output folder is made); the receipt records its file name and sha256.
  5. score.py site with the inputs that exist and the failed steps named (--failed-step)
A failed step's output is left out of score.py, which handles it like a site with no feasible frame for its targets
(B cannot pass, H is a miss at a Blocked-test site, the rest not scored); it is listed in the receipt and the other
steps still run. A site without a built scene goes straight to score.py. Nothing is downloaded here. Step logs and
error tails have the banned vehicle-status word masked (p*rk) before they are written.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import score  # noqa: E402
from common import ROOT, now, rel, sha_file, write_new  # noqa: E402

NICE = ['nice', '-n', '10']
VARIANT = 'v4-18.5'  # run_heldout.py VARIANT
F_DIR = '.local/evaluation/pao-accuracy/metric-f-shape-v2'
F_PROTOCOL = ['--protocol', f'{F_DIR}/protocol-v4.json', '--protocol-lock', f'{F_DIR}/protocol-v4.lock.json']
BUILT = ('built-shape', 'built-box')
BANNED = re.compile('park', re.IGNORECASE)  # common.write_new refuses it; node traces can quote source lines


def masked(text):
    return BANNED.sub('p*rk', text)


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or '').split('\n')[0])
    parser.add_argument('--site', required=True, help='the run_heldout.py output folder of this site')
    parser.add_argument('--out', required=True)
    parser.add_argument('--paint', help='the paint_heldout.py output folder of this crossing')
    parser.add_argument('--labels', help='the held-out label aggregate for this site (only with the labels addendum)')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--protocol-lock-sha256')
    mode.add_argument('--rehearse', action='store_true')
    args = parser.parse_args()
    site, out = Path(args.site).resolve(), Path(args.out).resolve()
    if not out.is_relative_to(ROOT / '.local') or os.path.lexists(out):
        parser.error('--out must be a new folder under .local/')
    found = [p for p in (site / 'build-receipt.json', site / 'site-stopped.json') if p.is_file()]
    if len(found) != 1:
        parser.error(f'{site} has no single site record (build-receipt.json or site-stopped.json)')
    record = json.loads(found[0].read_bytes())
    log, crossing = record['site']['logId'], record['site']['crossingId']
    if record['mode'].startswith('held-out') == bool(args.rehearse):
        parser.error('the site folder was built in the other mode')
    if not args.rehearse and record['protocolLockSha256'] != args.protocol_lock_sha256:
        parser.error('the site folder was built under another lock')
    common.guard(log, crossing, args.rehearse, args.protocol_lock_sha256)
    # Labels: everything is checked before the output folder is made.
    addendum_sha = common.labels_addendum(not args.rehearse, score.LABELS_ADDENDUM)
    if not args.rehearse and record.get('labelsAddendumSha256', 'missing') != addendum_sha:
        parser.error('the site was built under another labels addendum than the committed anchor')
    if args.labels and addendum_sha is None:
        parser.error('--labels needs the labels addendum (protocol-lock labels.scoringCondition)')
    labels = {'name': Path(args.labels).name, 'sha256': sha_file(args.labels)} if args.labels else None  # never the folder
    manifest = None
    if not args.rehearse and addendum_sha:
        entries, manifest_sha = common.labels_manifest(addendum_sha)
        wanted = entries[f'{log[:8]}/{crossing}']
        if (labels or {}).get('sha256') != wanted:
            parser.error('--labels must be given exactly when the committed label manifest lists labels for this site, with that sha256')
        manifest = {'file': common.LABELS_MANIFEST_REL, 'sha256': manifest_sha}
    pinned = None
    if labels and record['outcome'] in BUILT:
        pinned = json.loads(score.LABELS_ADDENDUM.read_bytes())['deCode']['scorer']
        if sha_file(ROOT / pinned['file']) != pinned['sha256']:
            parser.error(f'{pinned["file"]} differs from the sha256 the labels addendum pins')
    out.mkdir(parents=True)
    steps = []

    def run(step, argv):
        start = time.time()
        result = subprocess.run(NICE + argv, cwd=ROOT, capture_output=True, text=True)
        for kind, text in (('stdout', result.stdout), ('stderr', result.stderr)):
            fd = os.open(out / f'{step}-{kind}.txt', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
            with os.fdopen(fd, 'wb') as handle:
                handle.write(masked(text).encode())
        steps.append({'step': step, 'exit': result.returncode, 'seconds': round(time.time() - start, 2),
                      **({'error': masked((result.stderr.strip() or result.stdout.strip())[-400:])} if result.returncode else {})})
        return result.returncode == 0

    py = [sys.executable, '-I', '-B']
    inputs = {}
    if record['outcome'] in BUILT:
        grounded, assembled, prepared = site / 'grounded', site / 'assembled', site / 'prepared'
        shape = record['outcome'] == 'built-shape'
        if run('baseline', ['node', 'scripts/evaluation/pao-baseline.mjs', '--scene', str(site), '--out', str(out / 'baseline')]):
            files = sorted((out / 'baseline').glob('baseline-*.json'))
            if len(files) == 1:
                inputs['baseline'] = files[0]
        data = site / 'data' / log
        sweeps = site / 'shape' / f'shape-v2-{log}'
        name = f'{log[:8]}-{crossing}-{VARIANT}'
        if shape and run('metric-f-rows', ['node', 'scripts/evaluation/metric-f.mjs', '--assembled', f'{assembled}/', '--grounded', f'{grounded}/',
                                           *F_PROTOCOL, '--scene', name, '--out', str(out / 'rows.json')]) \
                and run('metric-f-profile', py + ['scripts/evaluation/metric_f_car_top.py', 'profile', *F_PROTOCOL,
                                                  '--ground', str(grounded / 'ground.json'), '--suggestions', str(grounded / 'suggestions.json'),
                                                  '--scene', name, '--annotations', str(data / 'annotations.feather'),
                                                  '--poses', str(data / 'city_SE3_egovehicle.feather'),
                                                  '--calibration', str(data / 'calibration/egovehicle_SE3_sensor.feather'),
                                                  '--log-id', log, '--sweep-dir', str(sweeps / 'sweeps'), '--receipt', str(sweeps / 'receipt.json'),
                                                  '--scoring-sweeps-from', str(grounded / 'ground.json'), '--out', str(out / 'profile.json')]) \
                and run('metric-f-score', py + ['scripts/evaluation/metric_f_car_top.py', 'score', *F_PROTOCOL, '--profile', str(out / 'profile.json'),
                                                '--rows', str(out / 'rows.json'), '--scene', name, '--out', str(out / 'results.json')]):
            inputs['metric-f-results'] = out / 'results.json'
        k_sweeps = ['--sweep-dir', str(sweeps / 'sweeps'), '--receipt', str(sweeps / 'receipt.json')] if shape else []
        if run('metric-k', py + ['scripts/evaluation/metric_k.py', '--grounded', str(grounded), '--assembled', str(assembled),
                                 '--prepared', str(prepared), '--out', str(out / 'metric-k.json'), *k_sweeps]):
            inputs['metric-k'] = out / 'metric-k.json'
        if pinned:
            given = [x for k in ('baseline', 'metric-f-results') if k in inputs for x in (f'--{k}', str(inputs[k]))]
            if run('labels-de', py + [pinned['file'], '--site', str(site), *given, '--labels', str(Path(args.labels).resolve()),
                                      '--out', str(out / 'labels-de.json')]):
                inputs['labels-de'] = out / 'labels-de.json'
    if args.paint:
        inputs['paint'] = Path(args.paint).resolve()
    failed = [x for s in steps if s['exit'] for x in ('--failed-step', s['step'])]
    scored = run('score', py + ['scripts/evaluation/m7/score.py', 'site', '--site', str(site), '--out', str(out / 'score.json'),
                                *[x for k, v in inputs.items() for x in (f'--{k}', str(v))], *failed])
    doc = {'kind': 'pao-m7-site-evaluation', 'mode': record['mode'], 'site': record['site'],
           'siteRecord': {'file': rel(found[0]), 'sha256': sha_file(found[0])},
           'protocolLockSha256': None if args.rehearse else args.protocol_lock_sha256, 'buildOutcome': record['outcome'],
           'labelsAddendumSha256': addendum_sha, 'labels': labels, 'labelsManifest': manifest,
           'paint': rel(inputs['paint']) if 'paint' in inputs else None,
           'steps': steps, 'notScoredBecauseAStepFailed': [s['step'] for s in steps if s['exit']],
           'outputs': {rel(p): sha_file(p) for p in sorted(out.rglob('*.json'))},
           'driver': {'file': 'scripts/evaluation/m7/evaluate_site.py', 'sha256': sha_file(__file__)},
           'python': {'executable': sys.executable, 'version': sys.version.split()[0]}, 'finishedUtc': now()}
    digest = write_new(out / 'evaluate-receipt.json', doc)
    print(json.dumps({'site': f'{log[:8]}/{crossing}', 'score': rel(out / 'score.json') if scored else None,
                      'failedSteps': doc['notScoredBecauseAStepFailed'], 'receipt': rel(out / 'evaluate-receipt.json'), 'sha256': digest}))
    sys.exit(0 if scored else 3)


if __name__ == '__main__':
    main()
