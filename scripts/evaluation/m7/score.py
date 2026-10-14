"""M7 scoring: one site's metric outcomes from its run files, then the summary over the scored sites.

  nice -n 10 python3 -I -B scripts/evaluation/m7/score.py site --site <run_heldout.py folder> --out <new file> \
      [--baseline <pao-baseline.mjs --scene output>] [--metric-f-results <metric_f_car_top.py score output>] \
      [--metric-k <metric_k.py output>] [--paint <paint_heldout.py folder>]
  nice -n 10 python3 -I -B scripts/evaluation/m7/score.py summary --scores <site score> ... --out <new file>

Every target is written as pass, fail or not scored (with its reason), using the locked rules (protocol-lock.json
metrics, referenceBlocked and statistics). Each input is checked against the site: its ground.json sha256 (baseline,
metric F rows, metric K) or its site record sha256 (paint). A missing input leaves its metrics not scored with that
reason; it never counts as a pass. Nothing here changes a result the app shows.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEVELOPMENT_ROLES, HELD_OUT_ROLES, ROOT, now, rel, sha_file, write_new  # noqa: E402

LABEL = 'AV2-referenced, not surveyed'
B_LIMIT_M, B_COVERAGE = .76, .95
A_OFFSET_M, A_AGREE_M = .30, .15
ALPHA = .05
BLOCKED_TEST = 'held-out-blocked-test'
BUILT = ('built-shape', 'built-box')
LABELS_ADDENDUM = ROOT / '.local/evaluation/pao-accuracy/lock/labels-addendum.json'
NOT_SCORED_DE = 'not scored: reference not precise enough (no labels addendum; labels.scoringCondition not met)'


def outcome(result, reason=None, **extra):
    return {'outcome': result, **({'reason': reason} if reason else {}), **extra}


def not_scored(reason, **extra):
    return outcome('not scored', reason, **extra)


def load(path):
    return json.loads(Path(path).read_bytes())


def percentile(values, p):
    """Nearest rank: sorted[ceil(p n) - 1] (the baseline's convention)."""
    s = sorted(values)
    return s[max(0, math.ceil(p * len(s)) - 1)] if s else None


def upper_bound(errors, n):
    """One-sided 95% exact binomial (Clopper-Pearson) upper bound for errors in n sites."""
    if n == 0:
        return None
    if errors >= n:
        return 1.0
    cdf = lambda p: sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(errors + 1))
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cdf(mid) > ALPHA else (lo, mid)
    return hi


def site_record(site):
    found = [p for p in (site / 'build-receipt.json', site / 'site-stopped.json') if p.is_file()]
    if len(found) != 1:
        raise SystemExit(f'{site} has no single site record (build-receipt.json or site-stopped.json)')
    return load(found[0]), found[0]


def checked_input(path, ground_sha, read_sha, what):
    """Load one input and refuse it when it names another scene's ground.json."""
    doc = load(path)
    named = read_sha(doc, Path(path))
    if ground_sha is None or named != ground_sha:
        raise SystemExit(f'{what} {path} is for ground.json {named}, not this site\'s {ground_sha}')
    return doc


def f_rows_sha(doc, path):
    rows = path.parent / doc['inputs']['rows']['file']
    if sha_file(rows) != doc['inputs']['rows']['sha256']:
        raise SystemExit(f'{rows} differs from the sha256 its metric F results name')
    return load(rows)['groundSha256']


# ---------------------------------------------------------------- per-metric rules

def metric_a(paint_dir, record_sha):
    if paint_dir is None:
        return not_scored('no paint run for this crossing')
    found = [p for p in (paint_dir / 'paint-receipt.json', paint_dir / 'paint-stopped.json') if p.is_file()]
    if len(found) != 1:
        raise SystemExit(f'{paint_dir} has no single paint record')
    receipt = load(found[0])
    if receipt['siteRecord']['sha256'] != record_sha:
        raise SystemExit(f'{found[0]} was made from another site record')
    if receipt['outcome'] != 'measured':
        return not_scored(f"paint step {receipt['stopped']['step']} stopped: {receipt['stopped']['reason']}", paintRecord=rel(found[0]))
    result = paint_dir / 'result.json'
    if sha_file(result) != receipt['outputs']['result.json']:
        raise SystemExit(f'{result} differs from its paint receipt')
    doc = load(result)
    a, b = doc['estimatorA'].get('edgeSAtEval'), doc['estimatorB'].get('edgeSAtEval')
    offsets = {'estimatorAM': a, 'estimatorBM': b, 'paintResult': {'file': rel(result), 'sha256': sha_file(result)}}
    if a is None or b is None:
        return not_scored('an estimator measured no edge (no validated paint method)', **offsets)
    if abs(a - b) > A_AGREE_M + 1e-12:
        return not_scored(f'unresolved: the estimators disagree by {abs(a - b):.3f} m (> {A_AGREE_M} m)', unresolved=True, **offsets)
    if abs(a) >= A_OFFSET_M and abs(b) >= A_OFFSET_M:
        return outcome('fail', f'both offsets are at least {A_OFFSET_M} m and agree within {A_AGREE_M} m', **offsets)
    return not_scored('no validated paint method (A is fail-only); both offsets reported', **offsets)


def metric_b(baseline):
    band = next((x for x in baseline['clickToGround']['bands'] if x['band'] == '12-25 m'), None)
    points = baseline['clickToGround'].get('metricBPoints')
    if band is None or not band.get('placed') or points is None:
        return outcome('fail', 'no placed 12-25 m point (B cannot pass)')
    p95, coverage, shown_max = band['horizontalErrorM']['p95'], band['shownUncertaintyCoverage'], band['shownUncertaintyM']['max']
    parts = {'p95': {'valueM': p95, 'limitM': B_LIMIT_M, 'pass': p95 <= B_LIMIT_M},
             'coverage': {'value': coverage, 'limit': B_COVERAGE, 'pass': coverage >= B_COVERAGE},
             'shownBandMax': {'valueM': shown_max, 'limitM': B_LIMIT_M, 'pass': shown_max <= B_LIMIT_M}}
    failed = [k for k, v in parts.items() if not v['pass']]
    return outcome('fail' if failed else 'pass', f"failed: {', '.join(failed)}" if failed else None, parts=parts,
                   median={'errorM': band['horizontalErrorM']['median'], 'shownM': band['shownUncertaintyM']['median']},
                   placed=band['placed'], points=points['points'])


def metric_c(baseline):
    roles = baseline['suggestionConsistency']['roles']
    rows = {name: (role['metricC']['pointUsedMet'] if 'metricC' in role else None) for name, role in roles.items()}
    missing = [k for k, v in rows.items() if v is None]
    if missing:
        return not_scored(f"no suggestion for {', '.join(missing)}", pointUsedMet=rows)
    return outcome('pass' if all(rows.values()) else 'fail', None, pointUsedMet=rows, label='internal consistency, not accuracy')


def metric_i(baseline):
    if 'honesty' not in baseline:
        return not_scored('the baseline ran without placement samples (honesty counters missing)')
    h = baseline['honesty']
    counts = {'personInsideBoxBlocked': h['sweepRowsWithPersonInsideBox']['blocked'],
              'driverInsideBoxBlocked': h['sweepRowsWithDriverInsideBox']['blocked'],
              'startConflictReportedBlocked': h['startValues']['conflictReportedBlocked'],
              'realObjectOverlapReportedBlocked': h['realObjectOverlap']['whatIfRowsOverlappingRealReportedBlocked'],
              'outOfRangeInputsAccepted': baseline['metricIOutOfRange']['accepted'],
              'badSentencesFailingWrongly': baseline['metricI'].get('failures')}
    parts = {'counts': counts, 'outOfRangeControlAccepted': baseline['metricIOutOfRange']['control']['accepted'],
             'badSentenceValuesDistinct': baseline['metricI'].get('distinctValues')}
    ok = (baseline['metricI']['pass'] and baseline['metricIOutOfRange']['pass']
          and all(counts[k] == 0 for k in counts if k != 'badSentencesFailingWrongly'))
    return outcome('pass' if ok else 'fail', None if ok else 'a count is not 0, or a check did not pass', **parts)


def metric_j(baseline):
    t = baseline['timing']
    return not_scored('J is measured alone in the browser on Sean\'s Mac (first result after a cold page load, Drag/Compare '
                      'p95); the Node values are reported only',
                      node={'firstResultMs': t['firstResultMs']['firstInProcess'], 'createScenarioMs': t['createScenarioMs'],
                            'setbackSweepMs': t['setbackSweepMs'], 'setbackSweepAdultMs': t['setbackSweepAdultMs'],
                            'environment': t['environment']})


def metric_f(results):
    child = results['heights']['child']
    counts = child['counts']
    sweeps = results['shapeV1']['scoringSweeps']
    unresolved = counts['plainBlocked']['unresolved']
    listed = {'plainBlockedRows': counts['plainBlockedRows'], 'byClass': counts['plainBlocked'],
              'unresolvedSetbacksM': counts['plainBlockedSetbacksM']['unresolved'],
              'likelyFalseSetbacksM': counts['plainBlockedSetbacksM']['likelyFalse'],
              'scoringSweeps': {'listed': len(sweeps['listed']), 'dropped': sweeps['dropped']},
              'forcedUnresolved': results['shapeV1'].get('forcedUnresolved')}
    if counts['plainBlocked']['likelyFalse'] > 0:
        return outcome('fail', f"{counts['plainBlocked']['likelyFalse']} likely-false plain-Blocked rows (a bug)", **listed)
    if counts['plainBlockedRows'] == 0:
        return not_scored('no plain-Blocked row to check', **listed)
    if unresolved == counts['plainBlockedRows']:
        return not_scored('every plain-Blocked row is F-unresolved', **listed)
    return outcome('pass', None, **listed)


def metric_f0(results):
    heights = {k: results['heights'][k]['shapeV1']['allowedChanges'] for k in ('child', 'adult')}
    full = results['shapeV1']['allowedChangesFullObjects']
    ok = all(v['pass'] for v in heights.values()) and full['pass']
    return outcome('pass' if ok else 'fail', None if ok else 'an allowed-change violation',
                   violations={k: v['violations'] for k, v in heights.items()}, fullObjectsChecked=full.get('resultsChecked'))


def metric_h(results, role):
    """Recorded-vehicle measured-shape setback rows (never declared or default rows): reference-blocked = metric F
    "supported" (E not scored); reported Blocked = the app's row state. The own-position nearest grid row is information."""
    def tally(rows):
        ref = [r for r in rows if r['class'] == 'supported']
        return {'rows': len(rows), 'referenceBlocked': len(ref), 'reportedBlocked': sum(r['state'] == 'blocked' for r in ref),
                'anyBlocked': any(r['state'] == 'blocked' for r in rows),
                'notReferenceListed': dict(Counter(r['class'] for r in rows if r['class'] in ('ambiguous', 'unresolved'))),
                'stateByClass': {f'{c}/{s}': n for (c, s), n in sorted(Counter((r['class'], r['state']) for r in rows).items())}}
    child, adult = tally(results['heights']['child']['rows']), tally(results['heights']['adult']['rows'])
    own = results['testedBox']['setbackM']
    near = min(results['heights']['child']['rows'], key=lambda r: abs(r['setbackM'] - own))
    info = {'child': child, 'adult': {**adult, 'note': 'reported, no target'},
            'ownPositionNearestRow': {'ownSetbackM': own, 'rowSetbackM': near['setbackM'], 'distanceM': abs(near['setbackM'] - own),
                                      'class': near['class'], 'state': near['state'],
                                      'note': 'the own setback is not on the 0.25 m row grid; information only'},
            'rule': 'recorded vehicle\'s measured-shape setback rows; reference-blocked = metric F "supported"; target >= 50% reported Blocked'}
    if role != BLOCKED_TEST and role not in DEVELOPMENT_ROLES:
        return not_scored('not a Blocked-test site', **info)
    if not child['anyBlocked']:
        return outcome('fail', 'all-Unknown: no child row is reported Blocked', **info)
    if child['referenceBlocked'] == 0:
        return not_scored('no reference-blocked child row (listed by name; leaves the miss count)', **info)
    share = child['reportedBlocked'] / child['referenceBlocked']
    return outcome('pass' if share >= .5 else 'fail', None if share >= .5 else f'{share:.0%} of reference-blocked child rows Blocked',
                   share=share, **info)


def metric_k(doc):
    def brief(report):
        if not report:
            return None
        k1 = report['seeThroughK1']
        return {'insideBox': report['insideBox']['count'], 'k1': k1['count'], 'k1CutDepthM': k1.get('cutDepthM'),
                'k1CutAtLeastResultMarginM': k1.get('cutAtLeastResultMarginM'), 'k2': report['unboxedK2']['count']}
    if doc['outcome'] != 'reported':
        return not_scored(doc.get('reason') or doc['outcome'])
    scoring = doc.get('scoringSweeps') or {}
    return outcome('report', None, sceneSweep=brief(doc['sceneSweep'].get('testedBox')),
                   scoringSweeps={'outcome': scoring.get('outcome'), 'used': scoring.get('used'), 'listed': scoring.get('listed'),
                                  'testedBox': brief(scoring.get('testedBox'))},
                   note='report only; never a reference source; Blocked rows with LiDAR evidence against them are metric F likely-false rows')


# ---------------------------------------------------------------- site

def score_site(args):
    site = Path(args.site).absolute()
    record, record_path = site_record(site)
    role, built = record['site']['role'], record['outcome'] in BUILT
    if LABELS_ADDENDUM.exists():
        raise SystemExit('a labels addendum exists: D and E scoring must be added to this script before scoring')
    ground_path = site / 'grounded/ground.json'
    ground_sha = sha_file(ground_path) if built and ground_path.is_file() else None
    inputs = {'siteRecord': {'file': rel(record_path), 'sha256': sha_file(record_path)}}
    for name, value in (('baseline', args.baseline), ('metricFResults', args.metric_f_results), ('metricK', args.metric_k)):
        if value:
            inputs[name] = {'file': rel(value), 'sha256': sha_file(value)}
    metrics = {}
    reason = (f"no built scene: {record['outcome']}" + (f" ({record['stopped']['step']}: {record['stopped']['reason']})" if record.get('stopped') else ''))
    metrics['A'] = metric_a(Path(args.paint).absolute() if args.paint else None, inputs['siteRecord']['sha256'])
    if not built:
        metrics['B'] = outcome('fail', reason + '; B cannot pass')
        for k in ('C', 'F', 'F0', 'I', 'J', 'K'):
            metrics[k] = not_scored(reason)
        metrics['H'] = (outcome('fail', reason + '; a miss') if role == BLOCKED_TEST or role in DEVELOPMENT_ROLES
                        else not_scored('not a Blocked-test site'))
    else:
        baseline = (checked_input(args.baseline, ground_sha, lambda d, _: d['scene']['ground']['sha256'], 'baseline')
                    if args.baseline else None)
        results = (checked_input(args.metric_f_results, ground_sha, f_rows_sha, 'metric F results')
                   if args.metric_f_results else None)
        k = checked_input(args.metric_k, ground_sha, lambda d, _: d['inputs']['ground.json'], 'metric K') if args.metric_k else None
        if baseline is None:
            for name in ('B', 'C', 'I', 'J'):
                metrics[name] = not_scored('no baseline run')
        else:
            metrics.update(B=metric_b(baseline), C=metric_c(baseline), I=metric_i(baseline), J=metric_j(baseline))
        shape = record['outcome'] == 'built-shape'
        for name, rule in (('F', metric_f), ('F0', metric_f0)):
            metrics[name] = (rule(results) if results else
                             not_scored('no recorded-vehicle LiDAR shape (built-box)' if not shape else 'no metric F results'))
        if results:
            metrics['H'] = metric_h(results, role)
        elif role == BLOCKED_TEST or role in DEVELOPMENT_ROLES:
            metrics['H'] = not_scored('no reference-blocked child row: no metric F results for a recorded vehicle '
                                      '(listed by name; leaves the miss count)')
        else:
            metrics['H'] = not_scored('not a Blocked-test site')
        metrics['K'] = metric_k(k) if k else not_scored('no metric K run')
    metrics['D'] = not_scored(NOT_SCORED_DE)
    metrics['E'] = not_scored(NOT_SCORED_DE)
    doc = {'kind': 'pao-m7-site-score', 'label': LABEL, 'mode': record['mode'],
           'site': record['site'], 'build': {'outcome': record['outcome'], 'stopped': record.get('stopped'), 'fixedBand': record.get('fixedBand')},
           'protocolLockSha256': record.get('protocolLockSha256'), 'inputs': inputs,
           'metrics': dict(sorted(metrics.items())),
           'scorer': {'file': 'scripts/evaluation/m7/score.py', 'sha256': sha_file(__file__)}, 'finishedUtc': now()}
    return doc


# ---------------------------------------------------------------- summary

def summarize(args):
    scores = [(Path(p), load(p)) for p in args.scores]
    modes = {d['mode'] for _, d in scores}
    if len(modes) != 1:
        raise SystemExit(f'site scores from more than one mode: {sorted(modes)}')
    locks = {d['protocolLockSha256'] for _, d in scores}
    if len(locks) != 1:
        raise SystemExit('site scores made under more than one lock')
    names = [f"{d['site']['logId'][:8]}/{d['site']['crossingId']}" for _, d in scores]
    if len(set(names)) != len(names):
        raise SystemExit('a site is scored twice')
    held_out = next(iter(modes)).startswith('held-out')
    if held_out and any(d['site']['role'] not in HELD_OUT_ROLES for _, d in scores):
        raise SystemExit('a held-out summary contains a site that is not held out')
    by_metric = {m: {n: d['metrics'][m]['outcome'] for n, (_, d) in zip(names, scores)}
                 for m in ('A', 'B', 'C', 'D', 'E', 'F', 'F0', 'H', 'I', 'J', 'K')}
    reasons = {m: {n: d['metrics'][m].get('reason') for n, (_, d) in zip(names, scores) if d['metrics'][m].get('reason')}
               for m in by_metric}
    # B pooled over the sites with a built scene.
    pooled = [pt for _, d in scores for pt in d['metrics']['B'].get('points', [])]
    b_pooled = None
    if pooled:
        errors, shown = [e for e, _ in pooled], [s for _, s in pooled]
        b_pooled = {'points': len(pooled), 'p95M': percentile(errors, .95), 'medianM': percentile(errors, .5),
                    'coverage': sum(e <= s for e, s in pooled) / len(pooled), 'shownBandMaxM': max(shown),
                    'shownBandMedianM': percentile(shown, .5), 'note': 'reported; each site must pass on its own'}
    # Misses over the Blocked-test sites (H fail = miss; H not scored = listed and left out).
    tests = [(n, d) for n, (_, d) in zip(names, scores) if d['site']['role'] == BLOCKED_TEST or (not held_out and d['site']['role'] in DEVELOPMENT_ROLES)]
    counted = [(n, d) for n, d in tests if d['metrics']['H']['outcome'] in ('pass', 'fail')]
    misses = [n for n, d in counted if d['metrics']['H']['outcome'] == 'fail']
    miss = {'sites': len(counted), 'misses': len(misses), 'missedSites': misses,
            'leftOut': {n: d['metrics']['H'].get('reason') for n, d in tests if d['metrics']['H']['outcome'] not in ('pass', 'fail')},
            'upperBound95': upper_bound(len(misses), len(counted)), 'method': 'one-sided 95% exact binomial (Clopper-Pearson)'}
    f_likely_false = sum(d['metrics']['F'].get('byClass', {}).get('likelyFalse', 0) for _, d in scores)
    a_unresolved = [n for n, (_, d) in zip(names, scores) if d['metrics']['A'].get('unresolved')]
    return {'kind': 'pao-m7-summary', 'label': LABEL, 'mode': next(iter(modes)), 'protocolLockSha256': next(iter(locks)),
            'sites': {n: {'role': d['site']['role'], 'build': d['build']['outcome'], 'score': {'file': rel(p), 'sha256': sha_file(p)}}
                      for n, (p, d) in zip(names, scores)},
            'outcomes': by_metric, 'reasons': reasons,
            'B': {'eachSitePasses': all(v == 'pass' for v in by_metric['B'].values()), 'pooled': b_pooled},
            'A': {'unresolvedSites': a_unresolved, 'failedSites': [n for n, v in by_metric['A'].items() if v == 'fail']},
            'statistics': {'unit': 'one intersection', 'miss': miss,
                           'falseBlocked': {'outcome': 'not measured independently', 'bound': None,
                                            'reason': 'metric E not scored',
                                            'metricFLikelyFalseRows': f_likely_false,
                                            'note': 'metric F likely-false rows are a same-LiDAR consistency check only'}},
            'gate': 'incomplete (the visibility gate is never recorded as passed)',
            'scorer': {'file': 'scripts/evaluation/m7/score.py', 'sha256': sha_file(__file__)}, 'finishedUtc': now()}


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or '').split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('site')
    s.add_argument('--site', required=True)
    s.add_argument('--baseline')
    s.add_argument('--metric-f-results')
    s.add_argument('--metric-k')
    s.add_argument('--paint')
    s.add_argument('--out', required=True)
    m = sub.add_parser('summary')
    m.add_argument('--scores', nargs='+', required=True)
    m.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out).absolute()
    if not out.is_relative_to(ROOT / '.local') or os.path.lexists(out):
        parser.error('--out must be a new file under .local/')
    doc = score_site(args) if args.command == 'site' else summarize(args)
    digest = write_new(out, doc)
    print(json.dumps({'out': rel(out), 'sha256': digest}))


if __name__ == '__main__':
    main()
