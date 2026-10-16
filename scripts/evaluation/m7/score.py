"""M7 scoring: one site's metric outcomes from its run files, then the summary over the scored sites.

  nice -n 10 python3 -I -B scripts/evaluation/m7/score.py site --site <run_heldout.py folder> --out <new file> \
      [--baseline <pao-baseline.mjs --scene output>] [--metric-f-results <metric_f_car_top.py score output>] \
      [--metric-k <metric_k.py output>] [--paint <paint_heldout.py folder>] [--failed-step <evaluate_site.py step> ...] \
      [--labels-de <the labels-addendum-pinned D/E scorer's output>] [--e-grid <metric-e-grid.mjs output>] \
      [--approach-split-results <its score output>]
  ... score.py site --site <folder> --paint <folder> --evaluation-stopped <evaluate-stopped.json> --out <new file>
  nice -n 10 python3 -I -B scripts/evaluation/m7/score.py summary --scores <site score> ... --out <new file>
  nice -n 10 python3 -I -B scripts/evaluation/m7/score.py runs-list --protocol-lock-sha256 <sha256>   (held-out, once)

Every target is written as pass, fail or not scored (with its reason), using the locked rules (protocol-lock.json
metrics, referenceBlocked and statistics). Each input is checked against the site: its ground.json sha256 (baseline,
metric F rows, metric E grid, metric K; ground.json against the build receipt) or its site record sha256 (paint). A failed or missing
step is handled like a site with no feasible frame for its targets: B cannot pass, H is a miss at a Blocked-test site,
the rest are not scored; it never counts as a pass. FApproachSplit (the approach-half split check) is report only: "reported" or
"not scored", never a pass or a fail, and no other target reads it; its results must name this site's ground.json (through
their rows) and the metric F results given here as their app results. D and E are scored only from the D/E file of the scorer that the labels
addendum pins (protocol-lock labels.scoringCondition); without the addendum they are "not scored". E's rows must be
exactly the rows of the metric E grid (metric-e-grid.mjs: head points with the recorded vehicle at its recorded pose and
the eye at the recorded camera), with the same app states; without the grid E is "not scored". A held-out site must
have been built under the committed labels addendum anchor. A held-out summary must hold exactly the held-out sites minus
the committed lost-site list, each with its paint record and, when the addendum is anchored, a D/E file for every site
the committed label manifest lists with labels (unless its labels-de step failed or it has no built scene). Held-out,
both steps run preflight (the approved lock, this score.py pinned in it); each site score records the sha256 of the
lost-site list, label manifest and runs list it was made under (its build receipt and paint record must name the same
lost-site list, and both records must be the ones the runs list names) and the summary refuses any other or a site
score of another site record or paint record; and the summary takes each site's failed steps from the receipt of the one
evaluate_site.py run its marker names, on the site and paint folders their markers name (one run each). A marked evaluation that ended without a score (its score step failed, or it left no
receipt) is scored by evaluate_site.py --record-stopped with --evaluation-stopped: handled like a site with no feasible
frame (B cannot pass, H a miss at a Blocked-test site, D/E and the rest not scored), with A from its paint record; the
summary checks it against that folder's evaluate-stopped.json. The summary lists by name every held-out site the label
manifest lists with labels whose D or E is not scored, and every crossing whose paint run stopped after paint v4 wrote
result.json (A not scored there; A is fail-only).

runs-list (held-out, once, after every build and paint run has its record and before any evaluate_site.py run) writes the
tracked anchor scripts/evaluation/m7/anchors-2/runs.json: each held-out site minus the lost-site list with the sha256 of the
one site record and one paint record in the folders its build and paint markers name. Committed and pushed, it is
written once like the other anchors, so a site run again after a marker was removed cannot be scored.

The D/E file (labels.scoringCondition) is one JSON object with exactly the keys kind ("pao-m7-labels-de"), site
({logId, crossingId}: strings, and nothing else), groundSha256, labelsAddendumSha256, labels ({sha256} only: never a path),
scorer ({file, sha256}), D ({labelledVehicle, errorsPx1024: [{feature, errorPx}], unresolvedLabels, reason?}) and E
({labelledVehicle, rows: [{height, xM, zM, appState, e}], assumption, reason?}). Nothing here changes a result the app
shows.
"""
import argparse
from collections import Counter
import json
import math
import os
import statistics
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import DEVELOPMENT_ROLES, HELD_OUT_ROLES, ROOT, now, rel, sha_file, write_new  # noqa: E402

LABEL = 'AV2-referenced, development-grade, not surveyed'  # protocol-lock label
B_LIMIT_M, B_COVERAGE = .76, .95
A_OFFSET_M, A_AGREE_M = .30, .15
A_FAIL_REASON = ('both paint v4 offsets are at least 0.30 m, agree within 0.15 m and paint v4 calls the edge resolved: the true '
                 'offset is very likely at least about 0.20 m (not proven to be 0.30 m); a look-alike line before the crossing '
                 '(bright band, stop line, faint first stripe) can also cause this, and the fail counts against the app either '
                 'way (Sean, Oct 14)')
PAINT_PROTOCOL_SHA = 'f9a52b440180e84c67d78dfb963849da73d7131f7bcaaf9ea6d0e94fafdfe273'  # paint.protocolV4
PAINT_CODE_SHA = 'bd2c6996551e03fc39160cbcb718f314467334979264816dca4f14fd0f92617a'  # paint.codeV4
ALPHA = .05
BLOCKED_TEST = 'held-out-blocked-test'
BUILT = ('built-shape', 'built-box')
LABELS_ADDENDUM = common.LABELS_ADDENDUM
F_PROTOCOL = ROOT / '.local/evaluation/pao-accuracy/metric-f-shape-v2/protocol-v4.json'  # metricProtocols.F
F_STEPS = ('metric-f-rows', 'metric-f-profile', 'metric-f-score')  # evaluate_site.py step names
APPROACH_STEPS = ('approach-split', 'approach-rows', 'approach-profile', 'approach-score')  # metrics.FApproachSplit (report only)
I_SAMPLES = 3000  # pao-baseline.mjs default placement samples (honesty counters)
NOT_SCORED_DE = 'not scored: reference not precise enough (no labels addendum; labels.scoringCondition not met)'
DE_KIND = 'pao-m7-labels-de'  # the addendum-pinned D/E scorer's output (labels.scoringCondition)
DE_KEYS = {'kind', 'site', 'groundSha256', 'labelsAddendumSha256', 'labels', 'scorer', 'D', 'E'}  # exactly these
STOPPED_KIND = 'pao-m7-evaluation-stopped'  # evaluate_site.py --record-stopped
D_MEDIAN_PX, D_MAX_PX = 3.0, 6.0  # metrics.D, 1024 px width
D_FEATURES = ('L', 'R', 'S1', 'S2', 'S3')  # metrics.D: each is compared or listed unresolved, once
E_STATES = ('blocked', 'not blocked', 'unresolved')
E_ROW_KEYS = {'height', 'xM', 'zM', 'appState', 'e'}  # labels.scoringCondition: exactly these
E_GRID_KIND = 'pao-m7-e-grid'  # metric-e-grid.mjs


def outcome(result, reason=None, **extra):
    return {'outcome': result, **({'reason': reason} if reason else {}), **extra}


def not_scored(reason, **extra):
    return outcome('not scored', reason, **extra)


def load(path):
    return json.loads(Path(path).read_bytes())


def no_constant(name):
    """json parse_constant: NaN, Infinity and -Infinity are refused."""
    raise SystemExit(f'the JSON holds {name}')


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

def paint_record(paint_dir):
    found = [p for p in (paint_dir / 'paint-receipt.json', paint_dir / 'paint-stopped.json') if p.is_file()]
    if len(found) != 1:
        raise SystemExit(f'{paint_dir} has no single paint record')
    return found[0]


def metric_a(paint_dir, record_sha):
    if paint_dir is None:
        return not_scored('no paint run for this crossing')
    record = paint_record(paint_dir)
    receipt = load(record)
    if receipt['siteRecord']['sha256'] != record_sha:
        raise SystemExit(f'{record} was made from another site record')
    if receipt['outcome'] != 'measured':
        # A is fail-only: a stop after paint v4 wrote result.json is flagged so the summary lists it by name.
        after = {'stoppedAfterMeasuring': True} if (paint_dir / 'result.json').exists() else {}
        return not_scored(f"paint step {receipt['stopped']['step']} stopped: {receipt['stopped']['reason']}", paintRecord=rel(record), **after)
    result = paint_dir / 'result.json'
    if sha_file(result) != receipt['outputs']['result.json']:
        raise SystemExit(f'{result} differs from its paint receipt')
    doc = load(result)
    if (doc.get('protocolSha256'), doc.get('scriptSha256')) != (PAINT_PROTOCOL_SHA, PAINT_CODE_SHA):
        raise SystemExit(f'{result} was not made by paint protocol v4 and code-v4 (protocolSha256/scriptSha256)')
    a, b = doc['estimatorA'].get('edgeSAtEval'), doc['estimatorB'].get('edgeSAtEval')
    near = doc['nearEdge']
    offsets = {'estimatorAM': a, 'estimatorBM': b, 'evalL': doc['frame']['evalL'],
               'paintV4': {k: near.get(k) for k in ('status', 'label', 'reason')},
               'paintResult': {'file': rel(result), 'sha256': sha_file(result)}}
    if a is None or b is None:
        return not_scored('an estimator measured no edge (no validated paint method)', **offsets)
    if abs(a - b) > A_AGREE_M + 1e-12:
        return not_scored(f'estimators disagree by {abs(a - b):.3f} m (> {A_AGREE_M} m)', estimatorsDisagree=True, **offsets)
    if abs(a) < A_OFFSET_M or abs(b) < A_OFFSET_M:
        return not_scored('no validated paint method (A is fail-only); both offsets reported', **offsets)
    if near['status'] != 'resolved':
        return not_scored(f"both offsets are at least {A_OFFSET_M} m but paint v4 itself is not confident "
                          f"(nearEdge {near['status']}: {near.get('reason')})", notConfirmedByPaintV4=True, **offsets)
    return outcome('fail', A_FAIL_REASON, **offsets)


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


def metric_f_approach(split):
    """metrics.FApproachSplit, report only (the shape-v2 split addendum): an evaluation-only shape from the far half of
    the approach, scored on the near half, each row next to the app shape result of the same row. Never a pass or a fail,
    and no other target reads it."""
    heights = {}
    for k, h in split['heights'].items():
        if 'error' in h:
            heights[k] = {'error': h['error']}
            continue
        rep, rows = h['shapeV1'], h['rows']
        heights[k] = {'rows': rep['counts']['rows'], 'boxBlocked': rep['counts']['boxBlocked'], 'shapeBlocked': rep['counts']['shapeBlocked'],
                      'shapeBlockedClasses': rep['shapeBlockedClasses'],
                      'boxBlockedSupported': rep['lostSupported']['boxBlockedSupported'], 'lostSupported': rep['lostSupported']['count'],
                      'appBlockedApproachHeightNotProven': sum(r['appShape']['state'] == 'blocked' and r['state'] == 'unknown'
                                                               and r.get('reason') == 'height-not-proven' for r in rows),
                      'approachBlockedAppNotBlocked': sum(r['state'] == 'blocked' and r['appShape']['state'] != 'blocked' for r in rows)}
    a = split['approachSplit']
    return outcome('reported', 'report only: never a pass or a fail, and no other target reads it',
                   **{k: a.get(k) for k in ('K', 'h', 'farRangeM', 'nearRangeM', 'droppedNearHalf', 'refusal')},
                   forcedUnresolved=split['shapeV1'].get('forcedUnresolved'), heights=heights)


def approach_split(path, ground_sha, metric_f_results):
    """metrics.FApproachSplit from approach-split results scored next to metric_f_results; refused (SystemExit) when
    they are for another scene, protocol or metric F results. evaluate_site.py runs this same check before it passes the
    file on, so a refused file fails only its approach-score step (FApproachSplit not scored), never the score step."""
    split = checked_input(path, ground_sha, f_rows_sha, 'approach-split results')
    if split.get('protocolSha256') != sha_file(F_PROTOCOL) or 'approachSplit' not in split:
        raise SystemExit(f'{path} is not an approach-split result under {rel(F_PROTOCOL)}')
    if (split.get('appResults') or {}).get('sha256') != sha_file(metric_f_results):
        raise SystemExit(f'{path} was scored next to other metric F results')
    return metric_f_approach(split)


def metric_h(results, role):
    """Recorded-vehicle measured-shape setback rows (never declared or default rows): reference-blocked = metric F
    "supported" (referenceBlocked; E has no setback rows and is not used). Reported Blocked = the app's row state. The
    own-position nearest grid row is information."""
    def tally(rows):
        ref = [r for r in rows if r['class'] == 'supported']
        listed = [r for r in rows if r['class'] in ('ambiguous', 'unresolved')]
        by_class = {c: [r['setbackM'] for r in listed if r['class'] == c] for c in sorted({r['class'] for r in listed})}
        return {'rows': len(rows), 'referenceBlocked': len(ref), 'reportedBlocked': sum(r['state'] == 'blocked' for r in ref),
                'anyBlocked': any(r['state'] == 'blocked' for r in rows),
                'notReferenceListed': dict(Counter(r['class'] for r in listed)), 'notReferenceListedSetbacksM': by_class,
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
            'rule': ('recorded vehicle\'s measured-shape setback rows; reference-blocked = metric F "supported"; '
                     'target >= 50% reported Blocked')}
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

def labels_de(path, addendum, addendum_sha, ground_sha, site, labels_sha=None):
    """The D/E file for this site, made by the scorer the labels addendum pins, for this ground.json and this addendum,
    naming the sha256 of the labels it used (held-out: the one the committed label manifest lists for this site)."""
    doc = json.loads(Path(path).read_bytes(), object_pairs_hook=common.unique_keys)  # a repeated key is refused
    pinned = common.de_code(addendum)  # the D/E code folder is checked against the addendum pins
    named, labels = doc.get('site') or {}, (doc.get('labels') or {}).get('sha256')
    problems = [what for what, ok in (
        ('keys (exactly ' + ', '.join(sorted(DE_KEYS)) + ')', set(doc) == DE_KEYS),
        ('kind', doc.get('kind') == DE_KIND),
        ('site (logId and crossingId strings only)', set(named) == {'logId', 'crossingId'} and isinstance(named.get('crossingId'), str)
         and named.get('logId') == site['logId'] and named.get('crossingId') == str(site['crossingId'])),
        ('ground.json', doc.get('groundSha256') == ground_sha),
        ('labels addendum', doc.get('labelsAddendumSha256') == addendum_sha),
        ('labels sha256', isinstance(labels, str) and common.HEX64.fullmatch(labels) and labels_sha in (None, labels)),
        ('labels (only its sha256: never a path)', set(doc.get('labels') or {}) == {'sha256'}),
        ('D/E scorer', doc.get('scorer') == pinned),
        ('D and E objects', isinstance(doc.get('D'), dict) and isinstance(doc.get('E'), dict))) if not ok]
    if problems:
        raise SystemExit(f'D/E file {path} differs from this site in: {", ".join(problems)}')
    return doc


def evaluation_stopped(path, record, record_path):
    """The evaluate-stopped.json that evaluate_site.py --record-stopped wrote for this site record."""
    doc = load(path)
    if doc.get('kind') != STOPPED_KIND or doc.get('site') != record['site'] \
            or doc.get('siteRecord') != {'file': rel(record_path), 'sha256': sha_file(record_path)}:
        raise SystemExit(f'{path} is not the evaluate-stopped record of this site record')
    return doc


def metric_d(doc):
    """metrics.D: |error| at 1024 px width over L, R and the resolved roofline stations; median <= 3 px, max <= 6 px."""
    d = doc['D']
    if not d.get('labelledVehicle'):
        return not_scored(d.get('reason') or 'no labelled recorded vehicle')
    errors, unresolved = d.get('errorsPx1024'), d.get('unresolvedLabels')
    if not (isinstance(errors, list) and isinstance(unresolved, list)):
        raise SystemExit('D needs errorsPx1024 and unresolvedLabels lists')
    number = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)  # noqa: E731
    if not all(isinstance(e, dict) and number(e.get('errorPx')) for e in errors) \
            or sorted([e.get('feature') for e in errors] + unresolved, key=str) != sorted(D_FEATURES):
        raise SystemExit(f'D must give each of {", ".join(D_FEATURES)} once, as a finite errorPx or as an unresolved label')
    listed = {'errors': errors, 'unresolvedLabels': unresolved, 'unresolvedCount': len(unresolved)}
    values = [abs(e['errorPx']) for e in errors]
    if not values:
        return not_scored('no resolved label to compare', **listed)
    median, largest = statistics.median(values), max(values)
    ok = median <= D_MEDIAN_PX and largest <= D_MAX_PX
    return outcome('pass' if ok else 'fail', None if ok else f'median {median:.2f} px, max {largest:.2f} px (limits 3, 6 at 1024 px)',
                   medianPx=median, maxPx=largest, **listed)


def e_grid(path, ground_sha):
    """metric-e-grid.mjs output for this site's ground.json, under metric F protocol v4 (its target heights): one row per
    grid head point and height, with the app state."""
    # Read as strictly as the D/E scorer reads it (no repeated key, no NaN or Infinity), so a grid accepted here never
    # makes the D/E scorer refuse (which would lose D too).
    doc = json.loads(Path(path).read_bytes(), object_pairs_hook=common.unique_keys, parse_constant=no_constant)
    if not isinstance(doc, dict) or doc.get('groundSha256') != ground_sha or ground_sha is None:
        raise SystemExit(f'metric E grid {path} is for ground.json {doc.get("groundSha256") if isinstance(doc, dict) else None}, '
                         f'not this site\'s {ground_sha}')
    rows = doc.get('rows')
    number = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)  # noqa: E731
    ok = isinstance(rows, list) and all(isinstance(r, dict) and r.get('height') in ('child', 'adult') and number(r.get('xM'))
                                        and number(r.get('zM')) and isinstance(r.get('state'), str)
                                        and isinstance(r.get('target'), list) and len(r['target']) == 3 and all(map(number, r['target']))
                                        and r['target'][0] == r['xM'] and r['target'][2] == r['zM'] for r in rows)
    keys = [(r['height'], r['xM'], r['zM']) for r in rows] if ok else []
    problems = [what for what, good in (
        ('kind', doc.get('kind') == E_GRID_KIND),
        (f'protocol (metric F {rel(F_PROTOCOL)})', doc.get('protocolSha256') == sha_file(F_PROTOCOL)),
        ('rows (each {height child or adult, finite xM and zM, state, target [xM, finite, zM]})', ok),
        ('a head point given twice', len(set(keys)) == len(keys)),
        ('row count', ok and (doc.get('counts') or {}).get('rows') == len(rows))) if not good]
    if problems:
        raise SystemExit(f'metric E grid {path} is not valid: {", ".join(problems)}')
    return doc


def metric_e(doc, grid, missing='no metric E grid'):
    """metrics.E: 0 false Blocked (app plain Blocked where E is resolved not blocked) per site; no resolved point: not scored.
    E's rows are the metric E grid's head points (the recorded vehicle at its recorded pose, the eye at the recorded
    camera): exactly one per grid row, with that row's app state. missing: why E is not scored without the grid."""
    e = doc['E']
    if not e.get('labelledVehicle'):
        return not_scored(e.get('reason') or 'no labelled recorded vehicle')
    if grid is None:
        return not_scored(missing)
    rows = e.get('rows')
    if not (isinstance(rows, list) and all(isinstance(r, dict) for r in rows)) or not str(e.get('assumption') or '').strip():
        raise SystemExit('E needs a rows list of objects and its assumption text')
    bad = [r for r in rows if set(r) != E_ROW_KEYS or r.get('e') not in E_STATES or r.get('height') not in ('child', 'adult')]
    if bad:
        raise SystemExit(f'E rows need exactly {", ".join(sorted(E_ROW_KEYS))}, a known height and state: {bad[:3]}')
    grid_rows = {(r['height'], r['xM'], r['zM']): r['state'] for r in grid['rows']}
    keys = [(r['height'], r['xM'], r['zM']) for r in rows]
    problems = [what for what, ok in (
        ('a row given twice', len(set(keys)) == len(keys)),
        (f'rows missing {sorted(set(grid_rows) - set(keys))[:3]} or extra {sorted(set(keys) - set(grid_rows), key=str)[:3]}',
         set(keys) == set(grid_rows)),
        ('an app state that differs from the grid row', all(r['appState'] == grid_rows.get(k) for r, k in zip(rows, keys)))) if not ok]
    if problems:
        raise SystemExit(f'E rows do not match this site\'s metric E grid: {"; ".join(problems)}')
    resolved = [r for r in rows if r['e'] != 'unresolved']
    false = [{'height': r['height'], 'xM': r['xM'], 'zM': r['zM']} for r in resolved if r['e'] == 'not blocked' and r['appState'] == 'blocked']
    listed = {'rows': len(rows), 'resolved': len(resolved), 'byState': dict(Counter(r['e'] for r in rows)),
              'byAppStateAndE': {f'{a}/{b}': n for (a, b), n in sorted(Counter((r['appState'], r['e']) for r in rows).items())},
              'falseBlocked': false, 'assumption': e['assumption'],
              'grid': {'points': (grid.get('counts') or {}).get('points'), 'limits': grid.get('limits')}}
    if not resolved:
        return not_scored(e.get('reason') or 'no resolved head point', **listed)
    return outcome('fail' if false else 'pass', f'{len(false)} false Blocked' if false else None, **listed)


def score_site(args):
    site = Path(args.site).resolve()
    record, record_path = site_record(site)
    role, built = record['site']['role'], record['outcome'] in BUILT
    held_out = record['mode'].startswith('held-out')
    test_site = role == BLOCKED_TEST or role in DEVELOPMENT_ROLES
    failed = sorted(set(args.failed_step or []))
    name = f"{record['site']['logId'][:8]}/{record['site']['crossingId']}"
    if args.evaluation_stopped and not held_out:
        raise SystemExit('--evaluation-stopped is for held-out sites (evaluate_site.py --record-stopped)')
    if args.evaluation_stopped and (failed or args.baseline or args.metric_f_results or args.metric_k or args.labels_de
                                    or args.e_grid or args.approach_split_results):
        raise SystemExit('--evaluation-stopped takes only --site, --paint and --out: the stopped evaluation\'s outputs are not used')
    stopped = evaluation_stopped(args.evaluation_stopped, record, record_path) if args.evaluation_stopped else None
    if held_out:
        common.preflight(record.get('protocolLockSha256'))  # the approved lock, with this score.py pinned in it
    # Held-out: the committed anchor, the addendum file and the site's build receipt must name the same addendum (or none).
    addendum_sha = common.labels_addendum(held_out, LABELS_ADDENDUM)
    if held_out and record.get('labelsAddendumSha256', 'missing') != addendum_sha:
        raise SystemExit('the site was built under another labels addendum than the committed anchor')
    addendum = load(LABELS_ADDENDUM) if addendum_sha else None
    if args.labels_de and addendum is None:
        raise SystemExit('--labels-de needs the labels addendum (protocol-lock labels.scoringCondition)')
    labels_sha, anchors = None, None
    if held_out:
        lost, lost_sha = common.lost_sites()
        if name in lost:
            raise SystemExit(f'{name} is on the committed lost-site list: it is not scored')
        entries, manifest_sha = common.labels_manifest(addendum_sha) if addendum_sha else ({}, None)
        runs, runs_sha = common.runs_list()
        anchors = {'lostSiteList': lost_sha, 'labelsManifest': manifest_sha, 'runs': runs_sha}  # the summary refuses any other
        if record.get('lostSiteList') != {'file': common.LOST_SITES_REL, 'sha256': lost_sha}:
            raise SystemExit('the site was built under another lost-site list than the committed one')
        listed = runs.get(name, (None, None))
        if listed[0] != sha_file(record_path) or args.paint and listed[1] != sha_file(paint_record(Path(args.paint).resolve())):
            raise SystemExit('the site record or paint record is not the one the committed runs list names (one run per site)')
        if args.labels_de:
            labels_sha = entries[name]
            if labels_sha is None:
                raise SystemExit('--labels-de for a site the committed label manifest lists without labels')
    ground_path = site / 'grounded/ground.json'
    ground_sha = sha_file(ground_path) if built and ground_path.is_file() else None
    if built and record.get('outputs', {}).get(rel(ground_path)) != ground_sha:
        raise SystemExit(f'{rel(ground_path)} differs from the sha256 its build receipt names')
    inputs = {'siteRecord': {'file': rel(record_path), 'sha256': sha_file(record_path)}, 'failedSteps': failed}
    for name, value in (('baseline', args.baseline), ('metricFResults', args.metric_f_results), ('metricK', args.metric_k),
                        ('eGrid', args.e_grid), ('approachSplitResults', args.approach_split_results)):
        if value:
            inputs[name] = {'file': rel(value), 'sha256': sha_file(value)}
    de = None
    if args.labels_de:
        if ground_sha is None:
            raise SystemExit('a D/E file for a site without a built scene')
        de = labels_de(args.labels_de, addendum, addendum_sha, ground_sha, record['site'], labels_sha)
        inputs['labelsDE'] = {'file': rel(args.labels_de), 'sha256': sha_file(args.labels_de), 'labelsSha256': de['labels']['sha256']}
    if args.e_grid and ground_sha is None:
        raise SystemExit('a metric E grid for a site without a built scene')
    if stopped:
        inputs['evaluationStopped'] = {'file': rel(args.evaluation_stopped), 'sha256': sha_file(args.evaluation_stopped)}
    metrics = {}
    reason = '; '.join(x for x in (
        None if built else f"no built scene: {record['outcome']}" + (f" ({record['stopped']['step']}: {record['stopped']['reason']})"
                                                                      if record.get('stopped') else ''),
        stopped and f"the evaluation stopped: {stopped['reason']}") if x)
    paint = Path(args.paint).resolve() if args.paint else None
    if paint:
        inputs['paint'] = {'file': rel(paint_record(paint)), 'sha256': sha_file(paint_record(paint))}
        if held_out and load(paint_record(paint)).get('lostSiteList') != {'file': common.LOST_SITES_REL, 'sha256': anchors['lostSiteList']}:
            raise SystemExit('the paint run was made under another lost-site list than the committed one')
    metrics['A'] = metric_a(paint, inputs['siteRecord']['sha256'])
    results = None
    if not built or stopped:  # handled like a site with no feasible frame
        metrics['B'] = outcome('fail', reason + '; B cannot pass')
        for k in ('C', 'F', 'F0', 'FApproachSplit', 'I', 'J', 'K'):
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
        grid = e_grid(args.e_grid, ground_sha) if args.e_grid else None
        if de:
            metrics['D'] = metric_d(de)
            metrics['E'] = metric_e(de, grid, 'the metric-e-grid step failed' if 'metric-e-grid' in failed else 'no metric E grid')
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
        if args.approach_split_results and not results:
            raise SystemExit('--approach-split-results needs the metric F results it was scored next to')
        if args.approach_split_results:
            metrics['FApproachSplit'] = approach_split(args.approach_split_results, ground_sha, args.metric_f_results)
        else:
            a_failed = [x for x in failed if x in APPROACH_STEPS]
            metrics['FApproachSplit'] = not_scored(f"approach-split step failed: {', '.join(a_failed)}" if a_failed
                                                   else 'no approach-split results' if results else f_missing)
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
    if not de:
        why = (NOT_SCORED_DE if addendum is None else reason if not built or stopped else 'the labels-de step failed' if 'labels-de' in failed
               else 'no D/E file for this site (no held-out labels for it)')
        metrics['D'], metrics['E'] = not_scored(why), not_scored(why)
    doc = {'kind': 'pao-m7-site-score', 'label': LABEL, 'mode': record['mode'],
           'site': record['site'], 'build': {'outcome': record['outcome'], 'stopped': record.get('stopped'), 'fixedBand': record.get('fixedBand')},
           'protocolLockSha256': record.get('protocolLockSha256'), 'labelsAddendumSha256': addendum_sha, 'anchors': anchors,
           'inputs': inputs,
           'metrics': dict(sorted(metrics.items())),
           'scorer': {'file': 'scripts/evaluation/m7/score.py', 'sha256': sha_file(__file__)}, 'finishedUtc': now()}
    return doc


# ---------------------------------------------------------------- summary

def failed_step(step):
    """An evaluate_site.py step that failed: a non-zero exit, or output the driver refused."""
    return bool(step.get('exit') or step.get('refused'))


def one_run(path, doc):
    """Held-out: what is wrong with this site score as the one run of each step at its site (empty: nothing). It must lie
    in the folder the site's evaluate marker names, with its site record and paint record in the folders the build and
    paint markers name. A scored evaluation: the score.json that the receipt (made by this evaluate_site.py, its score step
    passed) names, with exactly the failed steps that receipt lists. A stopped evaluation (evaluate_site.py
    --record-stopped): stopped-score.json, made from that folder's evaluate-stopped.json, where the receipt is missing or
    its score step failed."""
    log, crossing = doc['site']['logId'], doc['site']['crossingId']
    path = Path(path).resolve()
    folder = path.parent
    receipt_path = folder / 'evaluate-receipt.json'
    receipt = load(receipt_path) if receipt_path.is_file() else None
    score_steps = [s for s in (receipt or {}).get('steps', []) if s.get('step') == 'score']
    scored = len(score_steps) == 1 and not failed_step(score_steps[0])
    driver = sha_file(Path(__file__).resolve().parent / 'evaluate_site.py')
    parent = lambda key: str(Path(doc['inputs'][key]['file']).parent) if key in doc['inputs'] else None  # noqa: E731
    checks = [
        ('not the evaluation the evaluate marker names', common.marked_folder('evaluate', log, crossing) == rel(folder)),
        ('not the build the build marker names', common.marked_folder('build', log, crossing) == parent('siteRecord')),
        ('not the paint run the paint marker names', common.marked_folder('paint', log, crossing) == parent('paint'))]
    if 'evaluationStopped' in doc['inputs']:
        stop_path = folder / 'evaluate-stopped.json'
        stop = load(stop_path) if stop_path.is_file() else {}
        checks += [
            ('a stopped score is not stopped-score.json', path.name == 'stopped-score.json'),
            ('the score names another evaluate-stopped.json', stop_path.is_file() and doc['inputs']['evaluationStopped']
             == {'file': rel(stop_path), 'sha256': sha_file(stop_path)}),
            ('evaluate-stopped.json for another site record', stop.get('kind') == STOPPED_KIND and stop.get('site') == doc['site']
             and stop.get('siteRecord') == doc['inputs']['siteRecord']),
            ('evaluate-stopped.json made by another evaluate_site.py', (stop.get('driver') or {}).get('sha256') == driver),
            ('evaluate-stopped.json names another evaluate receipt', stop.get('evaluateReceipt')
             == ({'file': rel(receipt_path), 'sha256': sha_file(receipt_path)} if receipt else None)),
            ('the marked evaluation was scored: it is not a stopped site', not scored),
            ('a stopped site lists failed steps', doc['inputs']['failedSteps'] == [])]
    else:
        failed = sorted(s['step'] for s in (receipt or {}).get('steps', []) if failed_step(s) and s.get('step') != 'score')
        checks += [
            ('no evaluate_site.py receipt next to the score', receipt is not None),
            ('the score step of the evaluate receipt failed', scored),
            ('evaluate receipt for another site', (receipt or {}).get('kind') == 'pao-m7-site-evaluation'
             and (receipt or {}).get('site') == doc['site']),
            ('evaluate receipt does not name this score', (receipt or {}).get('outputs', {}).get(rel(path)) == sha_file(path)),
            ('evaluate receipt for another site record', (receipt or {}).get('siteRecord') == doc['inputs']['siteRecord']),
            ('failed steps differ from the evaluate receipt', failed == doc['inputs']['failedSteps']),
            ('evaluate receipt made by another evaluate_site.py', ((receipt or {}).get('driver') or {}).get('sha256') == driver)]
    return [what for what, ok in checks if not ok]


def summarize(args):
    scores = [(Path(p), load(p)) for p in args.scores]
    modes = {d['mode'] for _, d in scores}
    if len(modes) != 1:
        raise SystemExit(f'site scores from more than one mode: {sorted(modes)}')
    locks = {d['protocolLockSha256'] for _, d in scores}
    if len(locks) != 1:
        raise SystemExit('site scores made under more than one lock')
    for what, values in (('scorer', {d['scorer']['sha256'] for _, d in scores}),
                         ('labels addendum', {d.get('labelsAddendumSha256') for _, d in scores})):
        if len(values) != 1:
            raise SystemExit(f'site scores made with more than one {what}')
    names = [f"{d['site']['logId'][:8]}/{d['site']['crossingId']}" for _, d in scores]
    if len(set(names)) != len(names):
        raise SystemExit('a site is scored twice')
    held_out = next(iter(modes)).startswith('held-out')
    site_set, labels_info = None, None
    if held_out:
        common.preflight(next(iter(locks)))  # the approved lock, with this score.py pinned in it
        if {d['scorer']['sha256'] for _, d in scores} != {sha_file(__file__)}:
            raise SystemExit('held-out site scores made by another score.py than the locked one')
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
        # A is fail-only: a held-out site scored without its paint record could hide a fail.
        no_paint = [n for n, (_, d) in zip(names, scores) if 'paint' not in d['inputs']]
        if no_paint:
            raise SystemExit(f'held-out sites scored without their paint record: {no_paint}')
        addendum_sha = common.labels_addendum(True, LABELS_ADDENDUM)
        if {d.get('labelsAddendumSha256') for _, d in scores} != {addendum_sha}:
            raise SystemExit('held-out site scores made under another labels addendum than the committed anchor')
        if addendum_sha:
            # Every site the committed label manifest lists with labels has its D/E file, unless its labels-de step
            # failed, its evaluation stopped or it has no built scene; no site has a D/E file the manifest does not list.
            wanted, manifest_sha = common.labels_manifest(addendum_sha)
            wrong = [n for n, (_, d) in zip(names, scores)
                     if (d['inputs'].get('labelsDE') or {}).get('labelsSha256', wanted[n]) != wanted[n]
                     or wanted[n] and 'labelsDE' not in d['inputs'] and 'labels-de' not in d['inputs']['failedSteps']
                     and 'evaluationStopped' not in d['inputs'] and d['build']['outcome'] in BUILT]
            if wrong:
                raise SystemExit(f'held-out sites whose D/E input does not match the committed label manifest: {wrong}')
            site_set['labelsManifest'] = {'file': common.LABELS_MANIFEST_REL, 'sha256': manifest_sha}
            labelled = [n for n in names if wanted[n]]
            labels_info = {'sitesWithLabels': labelled,
                           'dOrENotScored': {n: {m: d['metrics'][m].get('reason') for m in ('D', 'E') if d['metrics'][m]['outcome'] == 'not scored'}
                                             for n, (_, d) in zip(names, scores) if wanted[n]
                                             and any(d['metrics'][m]['outcome'] == 'not scored' for m in ('D', 'E'))},
                           'note': 'listed by name: a site the label manifest lists with labels whose D or E is not scored'}
        # The anchor files a site score was made under must be the committed ones now (each is written once).
        listed, runs_sha = common.runs_list()
        anchors = {'lostSiteList': lost_sha, 'labelsManifest': manifest_sha if addendum_sha else None, 'runs': runs_sha}
        stale = [n for n, (_, d) in zip(names, scores) if d.get('anchors') != anchors]
        if stale:
            raise SystemExit(f'held-out sites scored under another lost-site list, label manifest or runs list than the committed ones: {stale}')
        unlisted = [n for n, (_, d) in zip(names, scores) if listed.get(n) != tuple(d['inputs'].get(k, {}).get('sha256') for k in ('siteRecord', 'paint'))]
        if unlisted:
            raise SystemExit(f'held-out site scores of another site record or paint record than the committed runs list names: {unlisted}')
        # One run of each step per site, and the failed steps taken from that run's receipt (never a bare claim).
        runs = {n: problems for n, (p, d) in zip(names, scores) if (problems := one_run(p, d))}
        if runs:
            raise SystemExit(f'held-out site scores that are not the one recorded run of each step: {runs}')
    by_metric = {m: {n: d['metrics'][m]['outcome'] for n, (_, d) in zip(names, scores)}
                 for m in ('A', 'B', 'C', 'D', 'E', 'F', 'F0', 'FApproachSplit', 'H', 'I', 'J', 'K')}
    reasons = {m: {n: d['metrics'][m].get('reason') for n, (_, d) in zip(names, scores)
                   if d['metrics'][m].get('reason') and d['metrics'][m]['outcome'] != 'reported'}
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
    # False Blocked from metric E (statistics.falseBlocked): over the sites with at least one E-resolved result.
    e_counted = [(n, d) for n, (_, d) in zip(names, scores) if d['metrics']['E']['outcome'] in ('pass', 'fail')]
    if e_counted:
        false_sites = [n for n, d in e_counted if d['metrics']['E']['outcome'] == 'fail']
        false_blocked = {'outcome': 'measured by metric E', 'sites': len(e_counted), 'falseBlockedSites': false_sites,
                         'leftOut': {n: d['metrics']['E'].get('reason') for n, (_, d) in zip(names, scores)
                                     if d['metrics']['E']['outcome'] not in ('pass', 'fail')},
                         'upperBound95': upper_bound(len(false_sites), len(e_counted)),
                         'method': 'one-sided 95% exact binomial (Clopper-Pearson)', 'metricFLikelyFalseRows': f_likely_false}
    else:
        false_blocked = {'outcome': 'not measured independently', 'bound': None, 'reason': 'metric E not scored',
                         'metricFLikelyFalseRows': f_likely_false,
                         'note': 'metric F likely-false rows are a same-LiDAR consistency check only'}
    d_scored = {n: v for n, v in by_metric['D'].items() if v in ('pass', 'fail')}
    d_errors = [abs(e['errorPx']) for _, d in scores for e in d['metrics']['D'].get('errors', [])]
    d_pooled = {'errors': len(d_errors), 'medianPx': statistics.median(d_errors), 'maxPx': max(d_errors),
                'note': 'reported; each site must pass on its own'} if d_errors else None
    a_flag = lambda key: [n for n, (_, d) in zip(names, scores) if d['metrics']['A'].get(key)]  # noqa: E731
    return {'kind': 'pao-m7-summary', 'label': LABEL, 'mode': next(iter(modes)), 'protocolLockSha256': next(iter(locks)),
            'sites': {n: {'role': d['site']['role'], 'build': d['build']['outcome'], 'score': {'file': rel(p), 'sha256': sha_file(p)}}
                      for n, (p, d) in zip(names, scores)},
            'siteSet': site_set or 'rehearsal: the development sites given (no completeness rule)',
            'outcomes': by_metric, 'reasons': reasons,
            'B': {'eachSitePasses': all(v == 'pass' for v in by_metric['B'].values()), 'pooled': b_pooled},
            'D': {'scoredSites': len(d_scored), 'eachSitePasses': all(v == 'pass' for v in d_scored.values()) if d_scored else None,
                  'failedSites': [n for n, v in d_scored.items() if v == 'fail'], 'pooled': d_pooled},
            'A': {'scoredSites': sum(v != 'not scored' for v in by_metric['A'].values()), 'sites': len(names),
                  'failedSites': [n for n, v in by_metric['A'].items() if v == 'fail'],
                  'estimatorsDisagreeSites': a_flag('estimatorsDisagree'), 'notConfirmedByPaintV4Sites': a_flag('notConfirmedByPaintV4'),
                  'stoppedAfterMeasuringSites': a_flag('stoppedAfterMeasuring'),
                  'note': 'fail-only: "not scored" says nothing about the map, and A never passes; a fail means the true offset is '
                          'very likely at least about 0.20 m, or a look-alike line fooled both estimators (counted against the app); '
                          'stoppedAfterMeasuringSites: the paint run stopped after paint v4 wrote result.json, so A is not scored there'},
            'labels': labels_info,
            'FApproachSplit': {'note': 'report only (the shape-v2 split addendum): never a pass or a fail, and no other target reads it',
                               'sites': {n: d['metrics']['FApproachSplit'] for n, (_, d) in zip(names, scores)
                                         if d['metrics']['FApproachSplit']['outcome'] == 'reported'}},
            'statistics': {'unit': 'one intersection', 'miss': miss,
                           'falseBlocked': false_blocked},
            'gate': 'incomplete (the visibility gate is never recorded as passed)',
            'scorer': {'file': 'scripts/evaluation/m7/score.py', 'sha256': sha_file(__file__)}, 'finishedUtc': now()}


def runs_list(lock_sha256):
    """Held-out, once, after every build and paint run has its record and before any evaluate_site.py run: the runs
    list (statistics.oneRunRule) of each held-out site minus the lost-site list, with the sha256 of the one site record
    in the folder its build marker names and the one paint record in the folder its paint marker names (made from that
    site record). Written to the tracked anchor path; commit and push it before any evaluation."""
    common.preflight(lock_sha256)
    runs, missing = [], []
    for name, x in sorted(common.held_out_names().items()):
        build, paint = (common.marked_folder(step, x['logId'], x['crossingId']) for step in ('build', 'paint'))
        records = [build and [p for p in (ROOT / build / 'build-receipt.json', ROOT / build / 'site-stopped.json') if p.is_file()],
                   paint and [p for p in (ROOT / paint / 'paint-receipt.json', ROOT / paint / 'paint-stopped.json') if p.is_file()]]
        if not all(r and len(r) == 1 for r in records):
            missing.append(name)
            continue
        site_sha, paint_sha = sha_file(records[0][0]), sha_file(records[1][0])
        if load(records[1][0]).get('siteRecord', {}).get('sha256') != site_sha:
            raise SystemExit(f'{name}: the paint record was made from another site record')
        runs.append({'logId': x['logId'], 'crossingId': str(x['crossingId']), 'siteRecordSha256': site_sha, 'paintRecordSha256': paint_sha})
    if missing:
        raise SystemExit(f'held-out sites without one marked site record and one marked paint record (run or record them first): {missing}')
    return {'runs': runs}


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or '').split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('site')
    s.add_argument('--site', required=True)
    s.add_argument('--baseline')
    s.add_argument('--metric-f-results')
    s.add_argument('--metric-k')
    s.add_argument('--approach-split-results', help='report only: metric_f_car_top.py score output of the approach split')
    s.add_argument('--paint')
    s.add_argument('--failed-step', action='append', help='an evaluate_site.py step that failed (repeatable)')
    s.add_argument('--labels-de', help='the D/E file of the scorer the labels addendum pins')
    s.add_argument('--e-grid', help='metric-e-grid.mjs output (metric E head points)')
    s.add_argument('--evaluation-stopped', help='evaluate_site.py --record-stopped: the evaluate-stopped.json of this site')
    s.add_argument('--out', required=True)
    m = sub.add_parser('summary')
    m.add_argument('--scores', nargs='+', required=True)
    m.add_argument('--out', required=True)
    r = sub.add_parser('runs-list', help=f'held-out: write {common.RUNS_REL} (then commit and push it)')
    r.add_argument('--protocol-lock-sha256', required=True)
    args = parser.parse_args()
    if args.command == 'runs-list':
        digest = write_new(ROOT / common.RUNS_REL, runs_list(args.protocol_lock_sha256), mode=0o644)
        print(json.dumps({'out': common.RUNS_REL, 'sha256': digest, 'next': 'commit and push it before any evaluate_site.py run'}))
        return
    out = Path(args.out).resolve()
    if not out.is_relative_to(ROOT / '.local') or os.path.lexists(out):
        parser.error('--out must be a new file under .local/')
    doc = score_site(args) if args.command == 'site' else summarize(args)
    digest = write_new(out, doc)
    print(json.dumps({'out': rel(out), 'sha256': digest}))


if __name__ == '__main__':
    main()
