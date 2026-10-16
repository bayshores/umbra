"""Tool-test oracle: answers labelkit questions from known answers, optionally with noise.

Used only to test the tools and to measure the method's floor (quantisation + snapping error with a
perfect labeler). It is never a labeler and its sessions are never written into the labels area.

ambiguous_band (coverage-design.md section 6; synthetic truth only, default off): an "honest" oracle answers
cannot_tell to a yes/no question about a pixel whose true coverage is in [lo, hi), and to a line question when the
most-covered pixel beyond the line is in [lo, hi). The band is checked before any answer flip and draws no random
number, so with the band off every answer (and every flip) is exactly as before. Other question kinds (box, cell,
count, pick, paint "visible") have no band rule in the design and stay definite.
"""
from __future__ import annotations

import random

import numpy as np

import imaging


def cell_label(meta, i, j):
    x0, y0, cell = meta['region'][0], meta['region'][1], meta['cell']
    c = int(np.clip((i - x0) // cell, 0, meta['columns'] - 1))
    r = int(np.clip((j - y0) // cell, 0, meta['rows'] - 1))
    return f'{imaging.column_label(c)}{r + 1}'


class Oracle:
    def __init__(self, truth, task, flip=0.0, seed=0, lidar_pick_tolerance=8.0, ambiguous_band=None):
        self.t, self.task, self.flip = truth, task, flip
        self.band = tuple(ambiguous_band) if ambiguous_band else None
        self.lidar_pick_tolerance = lidar_pick_tolerance
        self.rng = random.Random(seed)
        self.meta = task['overview']
        self.mode = 'paint' if task['kind'] == 'paint' else 'car'
        self._points = None

    def covered(self, i, j):
        return self.t.paint_covered(i, j) if self.mode == 'paint' else self.t.car_covered(i, j)

    def car_pixels(self):
        if self._points is None:
            x0, y0, w, h = self.meta['region']
            pts = [(i, j) for j in range(y0, y0 + h) for i in range(x0, x0 + w) if self.t.car_covered(i, j)]
            self._points = np.array(pts) if pts else np.zeros((0, 2), int)
        return self._points

    def extreme_point(self, name):
        p = self.car_pixels()
        k = {'L': np.argmin(p[:, 0]), 'R': np.argmax(p[:, 0]), 'T': np.argmin(p[:, 1]), 'B': np.argmax(p[:, 1])}[name]
        return p[k]

    def wheels(self):
        """Lowest visible car pixels of separate 'feet' along the bottom profile (left to right)."""
        p = self.car_pixels()
        cols = np.unique(p[:, 0])
        bottom = np.array([p[p[:, 0] == c, 1].max() for c in cols])
        lowest = bottom.max()
        feet = cols[bottom >= lowest - 6]
        groups = np.split(feet, np.where(np.diff(feet) > 3)[0] + 1)
        return [(int(g[len(g) // 2]), int(bottom[np.searchsorted(cols, g[len(g) // 2])])) for g in groups if len(g)]

    def predicate(self, q):
        render = q['render']
        pos = render['position']
        marker = render['marker']
        if marker['type'] == 'pixel':
            i, j = marker['at']
            return self.covered(i, j)
        p = self.car_pixels()
        feature = q['feature']
        if feature == 'L':
            return bool((p[:, 0] <= pos - 1).any())
        if feature == 'R':
            return bool((p[:, 0] >= pos).any())
        if feature == 'T':
            return bool((p[:, 1] <= pos - 1).any())
        if feature == 'B':
            return bool((p[:, 1] >= pos).any())
        raise ValueError(feature)

    def uncertain(self, q):
        """True when the band applies (synthetic truth) and the deciding coverage lies in [lo, hi)."""
        if not self.band or self.t.source != 'synthetic':
            return False
        lo, hi = self.band
        marker = q['render']['marker']
        if marker['type'] == 'pixel':
            i, j = marker['at']
            value = self.t.coverage(i, j, paint=self.mode == 'paint')
        else:
            value = self.t.coverage_beyond(q['feature'], q['render']['position'])
        return lo <= value < hi

    def answer(self, q):
        kind = q['kind']
        if kind == 'box':
            if self.mode == 'car':
                p = self.car_pixels()
                return f"{cell_label(self.meta, p[:, 0].min(), p[:, 1].min())}:{cell_label(self.meta, p[:, 0].max(), p[:, 1].max())}"
            cols = self.task['stationColumns']
            rows = [self.t.paint_near(c) for c in cols]
            rows = [r for r in rows if r is not None]
            return f"{cell_label(self.meta, min(cols), min(rows) - 4)}:{cell_label(self.meta, max(cols), max(rows) + 2)}"
        if kind == 'cell':
            feature = q['feature']
            if feature in ('L', 'R', 'T', 'B'):
                i, j = self.extreme_point(feature)
                return cell_label(self.meta, i, j)
            k = int(feature[1:]) - 1
            feet = self.wheels()
            if k >= len(feet):
                return 'none'
            return cell_label(self.meta, *feet[k])
        if kind == 'count':
            return str(min(len(self.wheels()), q['options']))
        if kind == 'pick':
            render = q['render']
            column = render['marker']['at']
            if self.mode == 'paint':
                target = self.t.paint_near(column)
            else:
                span = self.t.column_span(column)
                target = None if span is None else (span[0] if q['feature'].startswith('S') else span[1])
            if target is None:
                return '0'
            best = min(render['candidates'], key=lambda c: abs(c[2] - target))
            tolerance = 1.5 if self.t.source == 'synthetic' else self.lidar_pick_tolerance
            return str(best[0]) if abs(best[2] - target) <= tolerance else '0'
        if kind == 'yesno':
            purpose = q['render'].get('purpose')
            if purpose == 'visible':
                return 'yes' if self.t.paint_near(q['render']['marker']['at']) is not None else 'no'
            if self.uncertain(q):
                return 'cannot_tell'
            value = self.predicate(q)
            if self.flip and self.rng.random() < self.flip:
                value = not value
            return 'yes' if value else 'no'
        raise ValueError(kind)
