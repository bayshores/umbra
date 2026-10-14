"""Deterministic labeling session: rough prompts, yes/no bisection, candidate picks, repeat checks, snapping.

A session is a pure function of (task, protocol, labeler id, recorded answers). Each CLI call replays the
recorded answers through the question generator until it reaches the first unanswered question, which
is rendered and shown. The labeler never types a coordinate: only grid cells (rough prompts), yes/no/
cannot_tell, counts and candidate numbers. Final positions come from the bisection bracket and, for
scanline features, the gradient snap.

Predicates are monotone over an integer position axis. Two feature families:
- line features (extremes L, R, T, B): position i is the line between pixel i-1 and pixel i (coordinate
  i - 0.5). Estimate = coordinate of the final 'outside' line.
- pixel features (profile stations, wheel contacts, paint stations): position i is pixel i on the
  scanline. Estimate = boundary between the final outside and inside pixels; then snapped.

Protocol v2 coverage design (`search.ambiguity`, coverage-design.md section 1): a `cannot_tell` during bisection marks
that position ambiguous instead of ending the feature. The search stops at an ordinary result (b - a = 1) or a range
result (every position between the two definite ends was answered `cannot_tell`). A range is accepted when it has at
most `maxInteriorPx` positions and the feature used at most `maxCannotTell` `cannot_tell` answers; its estimate is the
middle of the range and `rangeHalfWidthPx` states the +/- (0 for an ordinary result, whose estimate equals v1's).
A `cannot_tell` at a bracket end expands the bracket; a pick question accepts `cannot_tell` (it acts like 0).
Without `search.ambiguity` (protocol v1) every behaviour, answer and result is exactly as before.
"""
from __future__ import annotations

import hashlib
import random

import numpy as np

import imaging
import snap


class Question(dict):
    """A question yielded by the generator. Keys: id, kind, feature, text, options, render."""


VALID = {'yesno': ('yes', 'no', 'cannot_tell')}


def _cannot_tell_form(text):
    return text.lower().replace(' ', '_').replace("can't_tell", 'cannot_tell')


def normalize_answer(question, answer):
    """Validate an answer string for a question. Returns the canonical value or raises ValueError."""
    text = str(answer).strip()
    kind = question['kind']
    if kind == 'yesno':
        value = _cannot_tell_form(text)
        if value not in VALID['yesno']:
            raise ValueError('Answer yes, no or cannot_tell')
        return value
    if kind == 'pick' and (question.get('render') or {}).get('allowCannotTell') and _cannot_tell_form(text) == 'cannot_tell':
        return 'cannot_tell'  # protocol v2 (search.ambiguity): an honest cannot_tell is allowed in pick questions
    if kind in ('box', 'cell'):
        if text.lower() == 'none':
            return 'none'
        meta = question['render']['overviewMeta']
        if kind == 'box':
            imaging.parse_box(meta, text)
        else:
            imaging.cell_bounds(meta, text)
        return text.upper().replace(' ', '')
    if kind in ('count', 'pick'):
        if not text.isdigit():
            raise ValueError(f'Answer a whole number from 0 to {question["options"]}')
        value = int(text)
        if not 0 <= value <= question['options']:
            raise ValueError(f'Answer a whole number from 0 to {question["options"]}')
        return str(value)
    raise ValueError(f'Unknown question kind {kind}')


class Context:
    def __init__(self, task, protocol, labeler, rgb):
        self.task, self.p, self.labeler, self.rgb = task, protocol, labeler, rgb
        seed = hashlib.sha256(f"{task['taskId']}|{labeler}|{protocol['version']}".encode()).hexdigest()
        self.rng = random.Random(int(seed[:16], 16))
        self.counter = 0
        self.overview_meta = task['overview']
        self.height, self.width = rgb.shape[:2]

    def question(self, kind, feature, text, options=None, render=None):
        self.counter += 1
        return Question(id=self.counter, kind=kind, feature=feature, text=text, options=options, render=render or {})

    def context_region(self, box):
        x0, x1, y0, y1 = box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half = max(x1 - x0, y1 - y0) / 2 * (1 + self.p['render']['contextMargin']) + 8
        return [int(round(cx - half)), int(round(cy - half)), int(round(2 * half)), int(round(2 * half))]


# ----- prompts -------------------------------------------------------------------------------------

def ask_box(ctx, feature, text):
    answer = yield ctx.question('box', feature, text, render={'type': 'overview', 'overviewMeta': ctx.overview_meta})
    if answer == 'none':
        return None
    return imaging.parse_box(ctx.overview_meta, answer)


def ask_cell(ctx, feature, text):
    answer = yield ctx.question('cell', feature, text, render={'type': 'overview', 'overviewMeta': ctx.overview_meta})
    if answer == 'none':
        return None
    return imaging.cell_bounds(ctx.overview_meta, answer)


# ----- generic monotone search ----------------------------------------------------------------------

def search(ctx, feature, spec):
    """Bracket check -> optional candidate narrowing -> bisection -> repeat check.

    spec keys: family ('line'|'pixel'), axis ('x'|'y'), fixed (scan column/row for pixel features),
    a, b (positions, a < b), qa (expected answer at a: True means 'yes'), text(pos) -> question text,
    window(pos, jitter) -> zoom window, context (region), candidates (optional callable), along (for line
    features: the centre of the zoom window along the line), pick_text.
    Returns dict with status, bracket, estimate and the question trail.
    """
    p = ctx.p['search']
    ambiguity = p.get('ambiguity')  # protocol v2 coverage design; None keeps the v1 search exactly
    a, b, qa = spec['a'], spec['b'], spec['qa']
    trail = {'expansions': 0, 'candidate': None, 'repeat': [], 'restarts': 0}
    if ambiguity:
        trail['cannotTell'] = 0

    def ask(pos, purpose, jitter=(0, 0)):
        render = {'type': 'question', 'context': spec['context'], 'window': spec['window'](pos, jitter),
                  'zoom': spec['zoom'], 'marker': spec['marker'](pos), 'guides': spec.get('guides', [])}
        answer = yield ctx.question('yesno', feature, spec['text'](pos), render=dict(render, purpose=purpose, position=int(pos)))
        if answer == 'cannot_tell' and ambiguity:
            trail['cannotTell'] += 1
        return None if answer == 'cannot_tell' else answer == 'yes'

    def pick_candidate(lo, hi):
        """Offer gradient-peak candidates in (lo, hi); returns the checked narrow bracket or None."""
        found = spec['candidates'](lo, hi)
        if not found:
            trail['candidate'] = {'offered': [], 'picked': None, 'accepted': False}
            return None
        numbered = [(k + 1, spec['axis'], c, spec['fixed']) for k, (c, _strength) in enumerate(found)]
        r = ctx.p['render']
        lo_c, hi_c = min(c for c, _ in found), max(c for c, _ in found)
        along = max(r['pixelWindow'], int(np.ceil(hi_c - lo_c)) + 2 * r['pickMarginPx'] + 1)
        zoom = int(max(r['pickMinZoom'], min(spec['zoom'], r['pickMaxPanelPx'] // along)))
        start_along = int(np.floor((lo_c + hi_c) / 2 - along / 2 + 0.5))
        start_perp = int(spec['fixed']) - r['pixelWindow'] // 2
        window = [start_perp, start_along, r['pixelWindow'], along] if spec['axis'] == 'y' else [start_along, start_perp, along, r['pixelWindow']]
        render = {'type': 'question', 'context': spec['context'], 'window': window,
                  'zoom': zoom, 'marker': {'type': 'column' if spec['axis'] == 'y' else 'row', 'at': spec['fixed']},
                  'candidates': numbered, 'guides': spec.get('guides', []), 'purpose': 'pick'}
        if ambiguity:
            render['allowCannotTell'] = True
        if r.get('tickOnPixelBoundary'):
            render['tickOnPixelBoundary'] = True  # ticks drawn on the pixel boundary nearest each candidate (design section 3)
        answer = yield ctx.question('pick', feature, spec['pick_text'](len(found)), options=len(found), render=render)
        unsure = answer == 'cannot_tell'
        pick = 0 if unsure else int(answer)
        trail['candidate'] = {'offered': [round(c, 3) for c, _ in found], 'picked': pick, 'accepted': False}
        if unsure:
            trail['candidate']['cannotTell'] = True  # recorded; behaves like 0 (no candidate)
            trail['cannotTell'] += 1
        if not pick:
            return None
        mixed = int(round(found[pick - 1][0]))  # pixel whose span contains the candidate boundary
        k_in, k_out = spec['candidateInside'], p['candidateOutsidePixels']
        lo_pos = mixed - (k_in if qa else k_out)
        hi_pos = mixed + (k_out if qa else k_in)
        got_lo = yield from ask(lo_pos, 'candidate-low')
        got_hi = (yield from ask(hi_pos, 'candidate-high')) if got_lo is not None else None
        if got_lo == qa and got_hi == (not qa):
            trail['candidate']['accepted'] = True
            return lo_pos, hi_pos
        return None

    if spec.get('candidateFirst'):
        # thin structures (paint): the predicate is only monotone near the edge, so the labeler first picks
        # the edge among gradient candidates, then a narrow bracket around it is checked.
        narrow = yield from pick_candidate(a, b)
        if narrow is None:
            reason = 'no candidate edge' if not trail['candidate']['offered'] else (
                'labeler rejected all candidates' if not trail['candidate']['picked'] else 'candidate bracket check failed')
            if trail['candidate'].get('cannotTell'):
                reason = 'cannot_tell at candidate pick'
            return {'status': 'unresolved', 'reason': reason, **trail}
        a, b = narrow
        checked = (a, b)
    else:
        # 1. bracket check with bounded expansion
        for end in ('a', 'b'):
            while True:
                pos = a if end == 'a' else b
                got = yield from ask(pos, 'bracket-' + end)
                if got is None and not ambiguity:
                    return {'status': 'unresolved', 'reason': f'cannot_tell at bracket {end}', **trail}
                # protocol v2: a cannot_tell at a bracket end is treated like a wrong answer there (the bracket expands)
                want = qa if end == 'a' else (not qa)
                if got == want:
                    break
                if trail['expansions'] >= p['maxExpansions']:
                    return {'status': 'unresolved', 'reason': f'bracket {end} failed after expansions', **trail}
                trail['expansions'] += 1
                if end == 'a':
                    a -= spec['expandStep']
                else:
                    b += spec['expandStep']
        checked = (a, b)
        # 2. optional candidate narrowing (pixel features)
        if spec.get('candidates') and b - a > p['candidateMinSpan']:
            narrow = yield from pick_candidate(a, b)
            if narrow is not None and a <= narrow[0] and narrow[1] <= b:
                a, b = narrow
    start = (a, b)

    # 3. bisection, 4. repeat check (one restart allowed)
    while True:
        a, b = start
        unsure = set()  # protocol v2: positions answered cannot_tell (the set U of coverage-design.md section 1)
        while b - a > 1:
            if ambiguity:
                interior = [pos for pos in range(a + 1, b) if pos not in unsure]
                if not interior:
                    break  # range result: every position between the definite ends was answered cannot_tell
                m = min(interior, key=lambda pos: (abs(2 * pos - (a + b)), pos))  # nearest the midpoint; ties: lower
            else:
                m = (a + b) // 2
            got = yield from ask(m, 'bisect')
            if got is None:
                if not ambiguity:
                    return {'status': 'unresolved', 'reason': 'cannot_tell during bisection', **trail}
                unsure.add(m)
                continue
            if got == qa:
                a = m
            else:
                b = m
            unsure = {pos for pos in unsure if a < pos < b}
        if ambiguity:
            n = b - a - 1
            span = {'bracket': [int(a), int(b)], 'ambiguous': sorted(int(pos) for pos in unsure), 'rangeHalfWidthPx': n / 2}
            if n > ambiguity['maxInteriorPx']:
                return {'status': 'unresolved', 'reason': 'ambiguous range too wide', **span, **trail}
            if n > 0 and trail['cannotTell'] > ambiguity['maxCannotTell']:
                return {'status': 'unresolved', 'reason': 'too many cannot_tell', **span, **trail}
        ends = ['a', 'b']
        ctx.rng.shuffle(ends)
        consistent = True
        for end in ends:  # re-ask BOTH final bracket ends on a shifted zoom window
            pos = a if end == 'a' else b
            jitter = (ctx.rng.randint(-p['repeatJitter'], p['repeatJitter']), ctx.rng.randint(-p['repeatJitter'], p['repeatJitter']))
            got = yield from ask(pos, 'repeat', jitter)
            want = qa if end == 'a' else (not qa)
            trail['repeat'].append({'end': end, 'position': int(pos), 'consistent': got == want})
            if got != want:
                consistent = False
                break
        if consistent:
            break
        if trail['restarts'] >= p['maxRestarts']:
            return {'status': 'unresolved', 'reason': 'repeat answer contradicted bisection twice', **trail}
        trail['restarts'] += 1
    if ambiguity:
        # coverage-design.md section 1; n = 0 gives exactly the v1 formulas below
        n = b - a - 1
        if spec['family'] == 'line':
            estimate = (a + 0.5 if qa else a - 0.5) + n / 2
        else:
            estimate = (a + b) / 2
        return {'status': 'resolved', 'bracket': [int(a), int(b)], 'checkedBracket': [int(checked[0]), int(checked[1])],
                'estimate': float(estimate), 'ambiguous': sorted(int(pos) for pos in unsure), 'rangeHalfWidthPx': n / 2, **trail}
    outside = a if not qa else b  # the position whose answer is 'no' (outside the target)
    if spec['family'] == 'line':
        estimate = outside - 0.5
    else:
        estimate = a + 0.5
    return {'status': 'resolved', 'bracket': [int(a), int(b)], 'checkedBracket': [int(checked[0]), int(checked[1])],
            'estimate': float(estimate), **trail}


# ----- feature builders -------------------------------------------------------------------------------

EXTREME = {
    'L': {'axis': 'x', 'qa': False, 'word': 'LEFT', 'side': 'left of'},
    'R': {'axis': 'x', 'qa': True, 'word': 'RIGHT', 'side': 'right of'},
    'T': {'axis': 'y', 'qa': False, 'word': 'TOP', 'side': 'above'},
    'B': {'axis': 'y', 'qa': True, 'word': 'BOTTOM', 'side': 'below'},
}
EXTREME_POINT = {'L': 'LEFTMOST', 'R': 'RIGHTMOST', 'T': 'HIGHEST (top-most)', 'B': 'LOWEST (bottom-most, usually a tire touching the road)'}


def extreme_feature(ctx, name, cell, box):
    p = ctx.p
    info = EXTREME[name]
    x_first, x_last, y_first, y_last = cell
    m = p['search']['bracketMargin']
    noun = ctx.task['target']['noun']
    if info['axis'] == 'x':
        a, b = x_first - m, x_last + 1 + m
        along = (y_first + y_last) // 2
    else:
        a, b = y_first - m, y_last + 1 + m
        along = (x_first + x_last) // 2
    r = p['render']
    half_perp, half_along = r['lineWindowPerp'] // 2, r['lineWindowAlong'] // 2

    def window(pos, jitter):
        jx, jy = jitter
        if info['axis'] == 'x':
            return [int(pos - half_perp + jx), int(along - half_along + jy), r['lineWindowPerp'], r['lineWindowAlong']]
        return [int(along - half_along + jx), int(pos - half_perp + jy), r['lineWindowAlong'], r['lineWindowPerp']]

    spec = {'family': 'line', 'axis': info['axis'], 'fixed': None, 'a': a, 'b': b, 'qa': info['qa'],
            'expandStep': x_last - x_first + 1,
            'context': ctx.context_region(box), 'window': window, 'zoom': r['lineZoom'],
            'marker': lambda pos: {'type': 'vline' if info['axis'] == 'x' else 'hline', 'at': pos - 0.5},
            'text': lambda pos: (f"Look at the magenta {'vertical' if info['axis'] == 'x' else 'horizontal'} line in all three panels. "
                                 f"Is ANY part of the target {noun} {info['side']} the magenta line (anywhere along it; body, glass, mirrors and "
                                 f"tires count, antennas do not)? Answer yes, no or cannot_tell.")}
    result = yield from search(ctx, name, spec)
    result['cell'] = list(cell)
    return result


def pixel_text(ctx, mode):
    noun = ctx.task['target']['noun']
    if mode == 'paint':
        return lambda pos: ('Is the pixel inside the magenta square on the white crosswalk paint (not road surface, not shadow, '
                            'not a vehicle)? Answer yes, no or cannot_tell.')
    return lambda pos: (f'Is the pixel inside the magenta square part of the target {noun} (body, glass, mirror or tire), '
                        'not the background behind it or the road below it? Answer yes, no or cannot_tell.')


def pixel_feature(ctx, feature, axis, fixed, a, b, qa, box, mode, sign, expand_step):
    r, p = ctx.p['render'], ctx.p
    half = r['pixelWindow'] // 2

    def window(pos, jitter):
        jx, jy = jitter
        if axis == 'y':
            return [int(fixed - half + jx), int(pos - half + jy), r['pixelWindow'], r['pixelWindow']]
        return [int(pos - half + jx), int(fixed - half + jy), r['pixelWindow'], r['pixelWindow']]

    def marker(pos):
        return {'type': 'pixel', 'at': (int(fixed), int(pos)) if axis == 'y' else (int(pos), int(fixed))}

    inside = 'the white crosswalk paint' if mode == 'paint' else f"the target {ctx.task['target']['noun']}"
    outside = {'paint': 'the road surface nearer the camera', 'top': 'the background above it',
               'contact': 'the road below it'}[mode]

    def candidates(lo, hi):
        return snap.candidates(ctx.rgb, axis, fixed, lo, hi, sign=sign, count=p['snap']['maxCandidates'],
                               sigma=p['snap']['sigma'], min_strength=p['snap']['minStrength'])

    spec = {'family': 'pixel', 'axis': axis, 'fixed': fixed, 'a': a, 'b': b, 'qa': qa, 'expandStep': expand_step,
            'context': ctx.context_region(box), 'window': window, 'zoom': r['pixelZoom'], 'marker': marker,
            'text': pixel_text(ctx, mode), 'candidates': candidates,
            'candidateInside': p['search']['candidateInsidePixels'][mode], 'candidateFirst': mode == 'paint',
            'pick_text': lambda n: (f'Coloured numbered ticks (1 to {n}) mark candidate edges on the scan line through the magenta '
                                    f'column/row. Which number sits exactly on the boundary between {inside} and {outside}? '
                                    'Answer the number, or 0 if none of them is on that boundary.' +
                                    (' Answer cannot_tell if you cannot tell.' if p['search'].get('ambiguity') else ''))}
    result = yield from search(ctx, feature, spec)
    result.update({'axis': axis, 'fixed': int(fixed)})
    if result['status'] == 'resolved':
        # protocol v2: a range result widens the snapping window to its stated +/- (v1 results carry no range: unchanged)
        window = max(p['snap']['window'], result.get('rangeHalfWidthPx', 0))
        snapped = snap.snap(ctx.rgb, axis, fixed, result['estimate'], window=window,
                            sigma=p['snap']['sigma'], min_strength=p['snap']['minStrength'])
        result['snapped'] = None if snapped is None else round(snapped[0], 4)
        result['value'] = result['snapped'] if snapped is not None else result['estimate']
    return result


def station_columns(ctx, extremes, box):
    """Roofline station columns.

    Protocol v1 ('carColumns' absent): fixed columns from the task file. For real vehicles these were 25/50/75 % of the
    projected AV2 box, so a labeler reading the task list could recover the box edges (audit finding; disclosed in
    protocol-v1.errata.md). Earlier v2 draft ('labelerExtremes'): fractions of the span between THIS labeler's own left
    and right answers (bisection estimates; the labeler's own grid-box edge for a side left unresolved). No box,
    LiDAR or map number is involved, and the task file carries no vehicle columns.
    Protocol v2 coverage design ('labelerExtremesOnGrid', coverage-design.md section 2): the same fractions, each moved
    to the nearest grid-cell centre column of the overview the labeler sees (x0 + k * cell + cell // 2), so labelers
    whose answers differ by a pixel or two measure the same column. Duplicates are handled in car_generator.
    """
    st = ctx.p['stations']
    rule = st.get('carColumns', 'task')
    if rule == 'task':
        return [int(c) for c in ctx.task.get('stationColumns', [])]
    if rule not in ('labelerExtremes', 'labelerExtremesOnGrid'):
        raise ValueError(f'unknown stations.carColumns rule {rule}')
    left = extremes['L'].get('value', box[0] - 0.5)
    right = extremes['R'].get('value', box[1] + 0.5)
    if right - left < st.get('carMinSpanPx', 4):
        return []
    if rule == 'labelerExtremes':
        return [int(np.floor(left + f * (right - left) + 0.5)) for f in st['carFractions']]
    meta = ctx.overview_meta
    x0, cell, count = meta['region'][0], meta['cell'], meta['columns']
    columns = []
    for f in st['carFractions']:
        target = left + f * (right - left)
        k = int(np.floor((target - x0 - cell // 2) / cell + 0.5))
        k = min(max(k, 0), count - 1)
        columns.append(int(x0 + k * cell + cell // 2))
    return columns


def car_generator(ctx):
    task, p = ctx.task, ctx.p
    noun = task['target']['noun']
    ring = 'cyan ring'
    out = {'kind': 'car', 'features': {}}
    box = yield from ask_box(ctx, 'box', (
        f"Open the overview image. The target {noun} is marked by the small {ring}. {task['target'].get('hint', '')} "
        f"Give the smallest grid box that contains ALL of the target {noun} (body, glass, mirrors and tires; ignore antennas), "
        "as TOPLEFT:BOTTOMRIGHT cells, for example C4:H9. Use the RAW or ENHANCED copy, whichever is clearer. "
        "Answer none if the target is not visible."))
    out['box'] = None if box is None else list(box)
    if box is None:
        out['status'] = 'unresolved'
        return out
    extremes = {}
    for name in ('L', 'R', 'T', 'B'):
        cell = yield from ask_cell(ctx, name, (
            f'Using the overview grid, which single cell contains the {EXTREME_POINT[name]} point of the target {noun}? '
            'Answer one cell like C4, or none if that part is hidden or cut off.'))
        if cell is None:
            extremes[name] = {'status': 'unresolved', 'reason': 'not visible (labeler)'}
        else:
            extremes[name] = yield from extreme_feature(ctx, name, cell, box)
        if extremes[name]['status'] == 'resolved':
            extremes[name]['value'] = extremes[name]['estimate']
        out['features'][name] = extremes[name]
    top = extremes['T'].get('value', box[2] - 0.5)
    bottom = extremes['B'].get('value', box[3] + 0.5)
    columns = station_columns(ctx, extremes, box)
    if p['stations'].get('carColumns', 'task') != 'task':
        out['stationColumns'] = columns  # v2: the labeler's own columns (v1 results keep their original shape)
    on_grid = p['stations'].get('carColumns') == 'labelerExtremesOnGrid'
    for k, column in enumerate(columns):
        if on_grid and column in columns[:k]:
            out['features'][f'S{k + 1}'] = {'status': 'unresolved', 'reason': 'same grid column', 'fixed': int(column)}
            continue
        a = int(np.floor(top)) - p['search']['bracketMargin']
        b = int(round(top + p['stations']['insideFraction'] * (bottom - top)))
        if b - a < 3:
            out['features'][f'S{k + 1}'] = {'status': 'unresolved', 'reason': 'bracket too small', 'fixed': int(column)}
            continue
        out['features'][f'S{k + 1}'] = yield from pixel_feature(ctx, f'S{k + 1}', 'y', int(column), a, b, False, box, 'top', 0,
                                                                 p['search']['pixelExpandStep'])
    count = yield ctx.question('count', 'wheels', (
        f'Using the overview, how many of the target {noun}\'s tires can you see where they touch the road '
        f'(0 to {p["wheels"]["maxCount"]})? Count a tire only if its lowest point is visible.'),
        options=p['wheels']['maxCount'], render={'type': 'overview', 'overviewMeta': ctx.overview_meta})
    count = int(count)
    for w in range(count):
        cell = yield from ask_cell(ctx, f'W{w + 1}', (
            f'Using the overview grid, which single cell contains the lowest point of visible tire number {w + 1} '
            '(numbering the visible tires from left to right)? Answer one cell like C4, or none.'))
        if cell is None:
            out['features'][f'W{w + 1}'] = {'status': 'unresolved', 'reason': 'not given'}
            continue
        x_first, x_last, y_first, y_last = cell
        column = (x_first + x_last) // 2
        m = p['search']['bracketMargin']
        out['features'][f'W{w + 1}'] = yield from pixel_feature(ctx, f'W{w + 1}', 'y', column, y_first - m - 1, y_last + m + 1, True, box,
                                                                 'contact', 0, y_last - y_first + 1)
        out['features'][f'W{w + 1}']['cell'] = list(cell)
    out['wheelCount'] = count
    out['status'] = 'complete'
    return out


def paint_generator(ctx):
    task, p = ctx.task, ctx.p
    out = {'kind': 'paint', 'features': {}}
    box = yield from ask_box(ctx, 'box', (
        "Open the overview image. " + task['target'].get('hint', '') +
        " Give the smallest grid box that contains the NEAR edge of the crosswalk paint (the painted edge closest to the camera, "
        "i.e. the lowest painted edge in the image) between the dashed cyan guide lines, as TOPLEFT:BOTTOMRIGHT cells, "
        "for example C4:H9. Answer none if no crosswalk paint is visible there."))
    out['box'] = None if box is None else list(box)
    if box is None:
        out['status'] = 'unresolved'
        return out
    x_first, x_last, y_first, y_last = box
    for k, column in enumerate(task.get('stationColumns', [])):
        name = f'P{k + 1}'
        if not x_first <= column <= x_last:
            out['features'][name] = {'status': 'unresolved', 'reason': 'column outside labeler box', 'fixed': int(column)}
            continue
        r = p['render']
        along = (y_first + y_last) // 2
        height = min(r['lineWindowAlong'], y_last - y_first + 1 + 2 * r['pixelWindow'] // 2)
        render = {'type': 'question', 'context': ctx.context_region(box), 'zoom': r['lineZoom'],
                  'window': [int(column - r['lineWindowPerp'] // 2), int(along - height // 2), r['lineWindowPerp'], int(height)],
                  'marker': {'type': 'column', 'at': int(column)}, 'guides': [], 'purpose': 'visible'}
        visible = yield ctx.question('yesno', name, (
            'The two magenta lines enclose one pixel column. Inside that column, is there white crosswalk paint whose NEAR edge '
            '(the paint edge closest to the camera, i.e. the lower edge) is clearly visible and not hidden by a vehicle, person '
            'or shadow? Answer yes, no or cannot_tell.'), render=render)
        if visible != 'yes':
            out['features'][name] = {'status': 'unresolved', 'reason': f'paint edge visible: {visible}', 'fixed': int(column)}
            continue
        m = p['search']['bracketMargin']
        out['features'][name] = yield from pixel_feature(ctx, name, 'y', int(column), y_first - m, y_last + m, True, box, 'paint', -1,
                                                          p['search']['pixelExpandStep'])
    out['status'] = 'complete'
    return out


GENERATORS = {'car': car_generator, 'paint': paint_generator}


def run(task, protocol, labeler, rgb, answers):
    """Replay `answers` (list of canonical strings). Returns (pending_question or None, result or None, asked)."""
    ctx = Context(task, protocol, labeler, rgb)
    gen = GENERATORS[task['kind']](ctx)
    asked = []
    try:
        question = next(gen)
        while True:
            asked.append(question)
            if len(asked) > len(answers):
                return question, None, asked
            question = gen.send(normalize_answer(question, answers[len(asked) - 1]))
    except StopIteration as stop:
        return None, stop.value, asked
