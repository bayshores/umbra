"""M7 scoring: one site's metric outcomes from its run files, then the summary over the scored sites.

  nice -n 10 python3 -I -B scripts/evaluation/m7/score.py site --site <run_heldout.py folder> --out <new file> \
      [--baseline <pao-baseline.mjs --scene output>] [--metric-f-results <metric_f_car_top.py score output>] \
      [--metric-k <metric_k.py output>] [--paint <paint_heldout.py folder>] [--failed-step <evaluate_site.py step> ...]
  nice -n 10 python3 -I -B scripts/evaluation/m7/score.py summary --scores <site score> ... --out <new file>

Every target is written as pass, fail or not scored (with its reason), using the locked rules (protocol-lock.json
metrics, referenceBlocked and statistics). Each input is checked against the site: its ground.json sha256 (baseline,
metric F rows, metric K; ground.json against the build receipt) or its site record sha256 (paint). A failed or missing
step is handled like a site with no feasible frame for its targets: B cannot pass, H is a miss at a Blocked-test site,
the rest are not scored; it never counts as a pass. A held-out summary must hold exactly the held-out sites minus the
committed lost-site list. Nothing here changes a result the app shows.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import DEVELOPMENT_ROLES, HELD_OUT_ROLES, ROOT, now, rel, sha_file, write_new  # noqa: E402

LABEL = 'AV2-referenced, development-grade, not surveyed'  # protocol-lock label
B_LIMIT_M, B_COVERAGE = .76, .95
A_OFFSET_M, A_AGREE_M = .30, .15
ALPHA = .05
BLOCKED_TEST = 'held-out-blocked-test'
BUILT = ('built-shape', 'built-box')
LABELS_ADDENDUM = ROOT / '.local/evaluation/pao-accuracy/lock/labels-addendum.json'
F_PROTOCOL = ROOT / '.local/evaluation/pao-accuracy/metric-f-shape-v2/protocol-v4.json'  # metricProtocols.F
F_STEPS = ('metric-f-rows', 'metric-f-profile', 'metric-f-score')  # evaluate_site.py step names
I_SAMPLES = 3000  # pao-baseline.mjs default placement samples (honesty counters)
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
    """From the full-precision points (the band's own values are rounded): p95 and median nearest rank, coverage e <= s."""
    band = next((x for x in baseline['clickToGround']['bands'] if x['band'] == '12-25 m'), None)
    points = (baseline['clickToGround'].get('metricBPoints') or {}).get('points')
    if band is None or not band.get('placed') or not points:
        return outcome('fail', 'no placed 12-25 m point (B cannot pass)')
    if len(points) != band['placed']:
        raise SystemExit(f"metric B: {len(points)} points but the 12-25 m band placed {band['placed']}")
    errors, shown = [e for e, _ in points], [v for _, v in points]
    p95, coverage, shown_max = percentile(errors, .95), sum(e <= v for e, v in points) / len(points), max(shown)
    parts = {'p95': {'valueM': p95, 'limitM': B_LIMIT_M, 'pass': p95 <= B_LIMIT_M},
             'coverage': {'value': coverage, 'limit': B_COVERAGE, 'pass': coverage >= B_COVERAGE},
             'shownBandMax': {'valueM': shown_max, 'limitM': B_LIMIT_M, 'pass': shown_max <= B_LIMIT_M}}
    failed = [k for k, v in parts.items() if not v['pass']]
    return outcome('fail' if failed else 'pass', f"failed: {', '.join(failed)}" if failed else None, parts=parts,
                   median={'errorM': percentile(errors, .5), 'shownM': percentile(shown, .5)},
                   placed=band['placed'], points=points)


def metric_c(baseline):
    roles = baseline['suggestionConsistency']['roles']
    rows = {name: (role['metricC']['pointUsedMet'] if 'metricC' in role else None) for name, role in roles.items()}
    missing = [k for k, v in rows.items() if v is None]
    if any(v is False for v in rows.values()):
        return outcome('fail', None, pointUsedMet=rows, label='internal consistency, not accuracy')
    if missing:
        return not_scored(f"no suggestion for {', '.join(missing)}", pointUsedMet=rows)
    return outcome('pass', None, pointUsedMet=rows, label='internal consistency, not accuracy')


def metric_i(baseline):
    if 'setupUnavailable' in baseline:
        return not_scored(baseline['setupUnavailable']['reason'])
    if 'honesty' not in baseline:
        return not_scored('the baseline ran without placement samples (honesty counters missing)')
    h = baseline['honesty']
    counts = {'personInsideBoxBlocked': h['sweepRowsWithPersonInsideBox']['blocked'],
              'driverInsideBoxBlocked': h['sweepRowsWithDriverInsideBox']['blocked'],
              'startConflictReportedBlocked': h['startValues']['conflictReportedBlocked'],
              'realObjectOverlapReportedBlocked': h['realObjectOverlap']['whatIfRowsOverlappingRealReportedBlocked'],
              'outOfRangeInputsAccepted': baseline['metricIOutOfRange']['accepted'],
              'badSentencesFailingWrongly': baseline['metricI'].get('failures')}
    out_of_range = baseline['metricIOutOfRange']
    parts = {'counts': counts, 'outOfRangeControlAccepted': out_of_range['control']['accepted'],
             'outOfRangeRefusedByAnotherRule': out_of_range['messageMismatches'],
             'badSentenceValuesDistinct': baseline['metricI'].get('distinctValues'),
             'samples': {'count': h['sample']['count'], 'required': I_SAMPLES, 'sweepErrors': h['sweepErrors']},
             'currentVehicleOverlapsReal': {'count': h['realObjectOverlap']['currentOverlapsReal'],
                                            'note': 'information: the current vehicle is the declaration being tested (decision 12); '
                                                    'the count above is the app\'s own what-if rows reported Blocked'}}
    ok = (baseline['metricI']['pass'] and out_of_range['pass'] and out_of_range['messageMismatches'] == 0
          and all(counts[k] == 0 for k in counts if k != 'badSentencesFailingWrongly'))
    if not ok:
        return outcome('fail', 'a count is not 0, or a check did not pass', **parts)
    if h['sample']['count'] != I_SAMPLES or h['sweepErrors']:
        return not_scored(f"honesty sample incomplete: {h['sample']['count']} of {I_SAMPLES} placements, "
                          f"{h['sweepErrors']} sweep errors", **parts)
    return outcome('pass', None, **parts)


def metric_j(baseline):
    if 'timing' not in baseline:
        return not_scored((baseline.get('setupUnavailable') or {}).get('reason') or 'no Node timing in the baseline')
    t = baseline['timing']
    return not_scored('J is measured alone in the browser on Sean\'s Mac (first result after a cold page load, Drag/Compare '
                      'p95); the Node values are reported only',
                      node={'firstResultMs': t['firstResultMs']['firstInProcess'], 'createScenarioMs': t['createScenarioMs'],
                            'setbackSweepMs': t['setbackSweepMs'], 'setbackSweepAdultMs': t['setbackSweepAdultMs'],
                            'environment': t['environment']})


def height_errors(results):
    return {k: h['error'] for k, h in results['heights'].items() if 'error' in h}


def metric_f(results):
    """0 likely-false rows per site, over both heights (as metric_f_car_top.py summarize counts them)."""
    if height_errors(results):
        return not_scored(f'metric F could not evaluate a height: {height_errors(results)}')
    sweeps = results['shapeV1']['scoringSweeps']
    per = {k: {'plainBlockedRows': h['counts']['plainBlockedRows'], 'byClass': h['counts']['plainBlocked'],
               'unresolvedSetbacksM': h['counts']['plainBlockedSetbacksM']['unresolved'],
               'likelyFalseSetbacksM': h['counts']['plainBlockedSetbacksM']['likelyFalse']}
           for k, h in results['heights'].items()}
    total = lambda key: sum(v['byClass'][key] for v in per.values())
    rows, likely_false, unresolved = sum(v['plainBlockedRows'] for v in per.values()), total('likelyFalse'), total('unresolved')
    listed = {'heights': per, 'plainBlockedRows': rows, 'likelyFalseRows': likely_false, 'unresolvedRows': unresolved,
              'scoringSweeps': {'listed': len(sweeps['listed']), 'dropped': sweeps['dropped']},
              'forcedUnresolved': results['shapeV1'].get('forcedUnresolved')}
    if likely_false > 0:
        return outcome('fail', f'{likely_false} likely-false plain-Blocked rows (a bug)', **listed)
    if rows == 0:
        return not_scored('no plain-Blocked row to check', **listed)
    if unresolved == rows:
        return not_scored('every plain-Blocked row is F-unresolved', **listed)
    return outcome('pass', None, **listed)


def metric_f0(results):
    if height_errors(results):
        return not_scored(f'metric F could not evaluate a height: {height_errors(results)}')
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
    child_rows = results['heights']['child']['rows']
    child = tally(child_rows)
    adult = tally(results['heights']['adult']['rows']) if 'error' not in results['heights']['adult'] else results['heights']['adult']
    own = results['testedBox']['setbackM']
    near = min(child_rows, key=lambda r: abs(r['setbackM'] - own)) if child_rows else None
    info = {'child': child, 'adult': {**adult, 'note': 'reported, no target'},
            'ownPositionNearestRow': near and {'ownSetbackM': own, 'rowSetbackM': near['setbackM'], 'distanceM': abs(near['setbackM'] - own),
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
    test_site = role == BLOCKED_TEST or role in DEVELOPMENT_ROLES
    failed = sorted(set(args.failed_step or []))
    if LABELS_ADDENDUM.exists():
        raise SystemExit('a labels addendum exists: D and E scoring must be added to this script before scoring')
    ground_path = site / 'grounded/ground.json'
    ground_sha = sha_file(ground_path) if built and ground_path.is_file() else None
    if built and record.get('outputs', {}).get(rel(ground_path)) != ground_sha:
        raise SystemExit(f'{rel(ground_path)} differs from the sha256 its build receipt names')
    inputs = {'siteRecord': {'file': rel(record_path), 'sha256': sha_file(record_path)}, 'failedSteps': failed}
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
        metrics['H'] = outcome('fail', reason + '; a miss') if test_site else not_scored('not a Blocked-test site')
    else:
        baseline = (checked_input(args.baseline, ground_sha, lambda d, _: d['scene']['ground']['sha256'], 'baseline')
                    if args.baseline else None)
        results = (checked_input(args.metric_f_results, ground_sha, f_rows_sha, 'metric F results')
                   if args.metric_f_results else None)
        if results and results.get('protocolSha256') != sha_file(F_PROTOCOL):
            raise SystemExit(f'{args.metric_f_results} was made under another protocol than {rel(F_PROTOCOL)}')
        k = checked_input(args.metric_k, ground_sha, lambda d, _: d['inputs']['ground.json'], 'metric K') if args.metric_k else None
        # A failed or missing step is handled like a site with no feasible frame for its targets (lock dataAccess):
        # B cannot pass, H is a miss at a Blocked-test site, the other targets are not scored.
        if baseline is None:
            why = 'the baseline step failed' if 'baseline' in failed else 'no baseline results'
            metrics['B'] = outcome('fail', why + '; B cannot pass')
            for name in ('C', 'I', 'J'):
                metrics[name] = not_scored(why)
        else:
            metrics.update(B=metric_b(baseline), C=metric_c(baseline), I=metric_i(baseline), J=metric_j(baseline))
        shape = record['outcome'] == 'built-shape'
        f_failed = [x for x in failed if x in F_STEPS]
        f_missing = (f"metric F step failed: {', '.join(f_failed)}" if f_failed else 'no metric F results') if shape \
            else 'no recorded-vehicle LiDAR shape (built-box)'
        for name, rule in (('F', metric_f), ('F0', metric_f0)):
            metrics[name] = rule(results) if results else not_scored(f_missing)
        if not test_site:
            metrics['H'] = not_scored('not a Blocked-test site')
        elif results and 'error' not in results['heights']['child']:
            metrics['H'] = metric_h(results, role)
        elif shape:
            why = f"metric F could not evaluate the child height: {results['heights']['child']['error']}" if results else f_missing
            metrics['H'] = outcome('fail', f'{why}; a miss (handled like a site with no feasible frame)')
        else:
            # built-box: no recorded-vehicle LiDAR shape, so no reference-blocked row; the all-Unknown check uses the
            # app's own child setback sweep.
            sweep = (baseline or {}).get('setbackSweep')
            if sweep is None:
                why = (baseline or {}).get('setupUnavailable', {}).get('reason') or ('the baseline step failed' if 'baseline' in failed
                                                                                  else 'no baseline results')
                metrics['H'] = outcome('fail', f'no app setback sweep ({why}); a miss (handled like a site with no feasible frame)')
            elif sweep['child']['blockedRows'] == 0:
                metrics['H'] = outcome('fail', 'all-Unknown: no child row is reported Blocked', childBlockedRows=0)
            else:
                metrics['H'] = not_scored('no reference-blocked child row: no recorded-vehicle LiDAR shape (built-box); '
                                          'listed by name, leaves the miss count', childBlockedRows=sweep['child']['blockedRows'])
        metrics['K'] = metric_k(k) if k else not_scored('the metric K step failed' if 'metric-k' in failed else 'no metric K run')
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
    site_set = None
    if held_out:
        # Exactly the split.json held-out sites minus the committed lost-site list (statistics.lostSiteRule).
        lost, lost_sha = common.lost_sites()
        expected = {f"{x['logId'][:8]}/{x['crossingId']}" for x in common.split_sites() if x['role'] in HELD_OUT_ROLES}
        if not lost <= expected:
            raise SystemExit(f'the lost-site list names sites that are not held out: {sorted(lost - expected)}')
        if set(names) != expected - lost:
            raise SystemExit(f'held-out scores must be exactly the held-out sites minus the lost ones; missing '
                             f'{sorted(expected - lost - set(names))}, extra {sorted(set(names) - (expected - lost))}')
        site_set = {'rule': 'split.json held-out sites minus the committed lost-site list', 'lostSites': sorted(lost),
                    'lostSiteList': {'file': common.LOST_SITES_REL, 'sha256': lost_sha}}
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
    f_likely_false = sum(d['metrics']['F'].get('likelyFalseRows', 0) for _, d in scores)
    a_unresolved = [n for n, (_, d) in zip(names, scores) if d['metrics']['A'].get('unresolved')]
    return {'kind': 'pao-m7-summary', 'label': LABEL, 'mode': next(iter(modes)), 'protocolLockSha256': next(iter(locks)),
            'sites': {n: {'role': d['site']['role'], 'build': d['build']['outcome'], 'score': {'file': rel(p), 'sha256': sha_file(p)}}
                      for n, (p, d) in zip(names, scores)},
            'siteSet': site_set or 'rehearsal: the development sites given (no completeness rule)',
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
    s.add_argument('--failed-step', action='append', help='an evaluate_site.py step that failed (repeatable)')
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
