"""M7 per-site evaluation: the pinned scoring code on one site that run_heldout.py built, then score.py site.

  nice -n 10 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -I -B scripts/evaluation/m7/evaluate_site.py \
      --site .local/evaluation/pao-accuracy/m7/<log8>-<crossing> --out .local/evaluation/pao-accuracy/m7/<log8>-<crossing>-eval \
      [--paint <paint_heldout.py folder>] [--labels <held-out label aggregate>] (--protocol-lock-sha256 <sha256> | --rehearse)
  ... evaluate_site.py --site <the marked build folder> --record-stopped --protocol-lock-sha256 <sha256>

Steps, each with nice -n 10 from the repository root, stdout and stderr kept in the output folder:
  1. pao-baseline.mjs --scene (metrics B, C, I; J's Node values)
  2. a site with the recorded vehicle's LiDAR shape (built-shape): metric-f.mjs rows, metric_f_car_top.py profile (the
     scoring sweeps that ground.json carShape.sweeps.scoring lists, nothing else) and score, under metric F protocol v4
     (pinned in metricProtocols.F). No --box-rows-from: a held-out scene has no metric F v2 rows to reproduce.
  3. metric_k.py (the scene sweep; on a built-shape site also the scoring sweeps)
  4. only when the labels addendum exists (protocol-lock labels.scoringCondition): the D/E scorer it pins, after the D/E
     code folder is checked against the addendum (common.de_code), with --site, --baseline, --metric-f-results (those
     that exist), --labels and --out <a new file in a temporary folder outside the repository>; that file is copied to
     labels-de.json only if it names neither the coordinator-only folder nor the labels folder (and holds no banned
     word), otherwise the step is recorded as refused (a failed step). Held-out: --labels is required exactly for the
     sites the committed label manifest lists with labels, must have that sha256 and must lie inside the coordinator-only
     folder named by PAO_LABELS_COORDINATOR_DIR (checked before the output folder is made); the receipt records its file
     name and sha256. Step logs and error tails replace both folders with a placeholder.
  5. score.py site with the inputs that exist and the failed steps named (--failed-step)
A failed step's output is left out of score.py, which handles it like a site with no feasible frame for its targets
(B cannot pass, H is a miss at a Blocked-test site, the rest not scored); it is listed in the receipt and the other
steps still run. A site without a built scene goes straight to score.py. Nothing is downloaded here. Step logs and
error tails have the banned vehicle-status word masked (p*rk) before they are written. Held-out: the committed lost-site
list must exist before this first scoring run (statistics.lostSiteRule; the receipt records its sha256 and the label
manifest's), and a run marker allows one evaluation per site. Before that marker, --site must be the folder the build
marker names and --paint the folder the paint marker names, holding one paint record made from this site record (a paint
run that ended without one is first recorded with paint_heldout.py --record-interrupted).

--record-stopped (held-out): a marked evaluation that has ended (its process is gone) without a score, because its score
step failed or it left no receipt, is recorded once in its folder (evaluate-stopped.json: the reason and the receipt's
sha256, or null) and scored by score.py site --evaluation-stopped into stopped-score.json: handled like a site with no
feasible frame, with A from the marked paint record. It is never rerun.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
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

def site_record(parser, args, site):
    """The site's single record, built in this mode under this lock; then the guard (held-out: preflight)."""
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
    if not args.rehearse and common.marked_folder('build', log, crossing) != rel(site):
        parser.error('--site must be the folder the build marker names')
    return record, found[0], log, crossing


def copy_de(made, target, hidden):
    """The D/E scorer's output, copied to target unless it names the coordinator-only folder or a hidden folder (or holds
    the banned word). Returns None, or why it was refused."""
    if not made.is_file():
        return 'the D/E scorer wrote no output'
    data = made.read_bytes()
    if any(str(f).rstrip('/').encode() in data for f in [*common.coordinator_folders(), *hidden] if len(str(f).rstrip('/')) > 1):
        return 'the D/E output names the coordinator-only folder'
    try:
        write_new(target, data)
    except ValueError as error:
        return common.masked(str(error), hidden)
    return None


def record_stopped(parser, args):
    """--record-stopped: see the module docstring."""
    site = Path(args.site).resolve()
    record, record_path, log, crossing = site_record(parser, args, site)
    folder = common.ended_run('evaluate', log, crossing)
    paint = common.marked_folder('paint', log, crossing)
    if paint is None:
        parser.error('no paint run is marked for this site')
    receipt_path = folder / 'evaluate-receipt.json'
    receipt, reason = None, 'the marked evaluation ended without a receipt'
    if receipt_path.is_file():
        doc = json.loads(receipt_path.read_bytes())
        steps = [s for s in doc.get('steps', []) if s.get('step') == 'score']
        if len(steps) == 1 and not score.failed_step(steps[0]):
            parser.error('the marked evaluation was scored: there is nothing to record')
        receipt = {'file': rel(receipt_path), 'sha256': sha_file(receipt_path)}
        reason = 'the score step failed' + (f": {steps[0].get('error')}" if steps and steps[0].get('error') else '')
    stop = {'kind': score.STOPPED_KIND, 'mode': record['mode'], 'site': record['site'],
            'siteRecord': {'file': rel(record_path), 'sha256': sha_file(record_path)}, 'protocolLockSha256': args.protocol_lock_sha256,
            'evaluateReceipt': receipt, 'reason': common.masked(reason),
            'rule': 'handled like a site with no feasible frame (B cannot pass, H a miss at a Blocked-test site, D/E and the '
                    'rest not scored); A from the marked paint record; never rerun',
            'driver': {'file': 'scripts/evaluation/m7/evaluate_site.py', 'sha256': sha_file(__file__)}}
    stop_path, score_path = folder / 'evaluate-stopped.json', folder / 'stopped-score.json'
    if os.path.lexists(score_path):
        parser.error(f'{rel(score_path)} exists: the stopped evaluation is already scored')
    if stop_path.is_file():  # written by an earlier --record-stopped whose score step did not finish
        if {k: v for k, v in json.loads(stop_path.read_bytes()).items() if k != 'finishedUtc'} != stop:
            parser.error(f'{rel(stop_path)} differs from this record')
    else:
        folder.mkdir(parents=True, exist_ok=True)
        write_new(stop_path, {**stop, 'finishedUtc': now()})
    result = subprocess.run(NICE + [sys.executable, '-I', '-B', 'scripts/evaluation/m7/score.py', 'site', '--site', str(site),
                                    '--paint', str(ROOT / paint), '--evaluation-stopped', str(stop_path), '--out', str(score_path)],
                            cwd=ROOT, capture_output=True, text=True)
    print(common.masked(result.stdout.strip() or result.stderr.strip()[-400:]))
    sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or '').split('\n')[0])
    parser.add_argument('--site', required=True, help='the run_heldout.py output folder of this site')
    parser.add_argument('--out', help='a new folder under .local/ (not with --record-stopped)')
    parser.add_argument('--paint', help='the paint_heldout.py output folder of this crossing')
    parser.add_argument('--labels', help='the held-out label aggregate for this site (only with the labels addendum)')
    parser.add_argument('--record-stopped', action='store_true', help='held-out: record a marked evaluation that ended without a score')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--protocol-lock-sha256')
    mode.add_argument('--rehearse', action='store_true')
    args = parser.parse_args()
    if args.record_stopped:
        if args.rehearse or args.out or args.paint or args.labels:
            parser.error('--record-stopped takes only --site and --protocol-lock-sha256')
        return record_stopped(parser, args)
    if not args.out:
        parser.error('--out is required')
    site, out = Path(args.site).resolve(), Path(args.out).resolve()
    if not out.is_relative_to(ROOT / '.local') or os.path.lexists(out):
        parser.error('--out must be a new folder under .local/')
    record, record_path, log, crossing = site_record(parser, args, site)
    found = [record_path]
    if not args.rehearse:  # the marked paint run, with its record made from this site record
        paint = Path(args.paint).resolve() if args.paint else None
        if paint is None or common.marked_folder('paint', log, crossing) != rel(paint):
            parser.error('--paint must be the folder the paint marker names (run paint_heldout.py first)')
        try:
            made_from = json.loads(score.paint_record(paint).read_bytes())['siteRecord']['sha256']
        except SystemExit:
            parser.error('the marked paint folder has no single paint record: wait for paint_heldout.py to end, or record '
                         'a paint run that ended without one with paint_heldout.py --record-interrupted')
        if made_from != sha_file(record_path):
            parser.error('the paint record was made from another site record')
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
    lost = None
    if not args.rehearse:  # statistics.lostSiteRule: the lost-site list is committed before any held-out build
        names, lost_sha = common.lost_sites()
        if f'{log[:8]}/{crossing}' in names:
            parser.error('this site is on the committed lost-site list: it is not scored')
        lost = {'file': common.LOST_SITES_REL, 'sha256': lost_sha}
    labels_path = Path(args.labels).resolve() if args.labels else None
    if labels_path and not args.rehearse and not any(labels_path.is_relative_to(Path(f).resolve()) for f in common.coordinator_folders()):
        parser.error(f'held-out --labels must lie inside the coordinator-only folder named by {common.COORDINATOR_ENV}')
    hidden = [labels_path.parent] if labels_path else []  # never written into a log, an output or the receipt
    pinned = None
    if labels and record['outcome'] in BUILT:
        try:
            pinned = common.de_code(json.loads(score.LABELS_ADDENDUM.read_bytes()))
        except SystemExit as error:
            parser.error(str(error))
    if not args.rehearse:
        common.run_marker('evaluate', log, crossing, out)  # one evaluation per held-out site
    out.mkdir(parents=True)
    steps = []

    def run(step, argv):
        start = time.time()
        result = subprocess.run(NICE + argv, cwd=ROOT, capture_output=True, text=True)
        for kind, text in (('stdout', result.stdout), ('stderr', result.stderr)):
            fd = os.open(out / f'{step}-{kind}.txt', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
            with os.fdopen(fd, 'wb') as handle:
                handle.write(common.masked(text, hidden).encode())
        steps.append({'step': step, 'exit': result.returncode, 'seconds': round(time.time() - start, 2),
                      **({'error': common.masked((result.stderr.strip() or result.stdout.strip())[-400:], hidden)} if result.returncode else {})})
        return result.returncode == 0

    def refuse(why):
        steps[-1]['refused'] = why  # the step ran but its output was not accepted: a failed step
        return False

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
            temp = Path(tempfile.mkdtemp(prefix='pao-m7-de-'))  # outside the repository: checked before anything is kept
            try:
                made = temp / 'labels-de.json'
                if run('labels-de', py + [pinned['file'], '--site', str(site), *given, '--labels', str(labels_path), '--out', str(made)]):
                    why = copy_de(made, out / 'labels-de.json', hidden)
                    if why is None:
                        inputs['labels-de'] = out / 'labels-de.json'
                    else:
                        refuse(why)
            finally:
                shutil.rmtree(temp, ignore_errors=True)
    if args.paint:
        inputs['paint'] = Path(args.paint).resolve()
    failed = [x for s in steps if score.failed_step(s) for x in ('--failed-step', s['step'])]
    scored = run('score', py + ['scripts/evaluation/m7/score.py', 'site', '--site', str(site), '--out', str(out / 'score.json'),
                                *[x for k, v in inputs.items() for x in (f'--{k}', str(v))], *failed])
    doc = {'kind': 'pao-m7-site-evaluation', 'mode': record['mode'], 'site': record['site'],
           'siteRecord': {'file': rel(found[0]), 'sha256': sha_file(found[0])},
           'protocolLockSha256': None if args.rehearse else args.protocol_lock_sha256, 'buildOutcome': record['outcome'],
           'labelsAddendumSha256': addendum_sha, 'labels': labels, 'labelsManifest': manifest, 'lostSiteList': lost,
           'paint': rel(inputs['paint']) if 'paint' in inputs else None,
           'steps': steps, 'notScoredBecauseAStepFailed': [s['step'] for s in steps if score.failed_step(s)],
           'outputs': {rel(p): sha_file(p) for p in sorted(out.rglob('*.json'))},
           'driver': {'file': 'scripts/evaluation/m7/evaluate_site.py', 'sha256': sha_file(__file__)},
           'python': {'executable': sys.executable, 'version': sys.version.split()[0]}, 'finishedUtc': now()}
    digest = write_new(out / 'evaluate-receipt.json', doc)
    print(json.dumps({'site': f'{log[:8]}/{crossing}', 'score': rel(out / 'score.json') if scored else None,
                      'failedSteps': doc['notScoredBecauseAStepFailed'], 'receipt': rel(out / 'evaluate-receipt.json'), 'sha256': digest}))
    sys.exit(0 if scored else 3)


if __name__ == '__main__':
    main()
