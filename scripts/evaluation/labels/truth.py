"""Known answers for scoring (and for the tool-test oracle). Reads entries of the sealed answer key.

Synthetic entries are exact (union of convex projected polygons). LiDAR entries are inner bounds with a
sampling-gap estimate and a coarse registration caveat; bottom values are one-sided.
"""
from __future__ import annotations

import numpy as np

from lidar_ref import paint_row_at
from synth import column_interval


class Truth:
    def __init__(self, entry):
        self.entry = entry
        self.source = entry['source']
        self.kind = entry['kind']
        if self.source == 'synthetic':
            self.car_polygons = [np.asarray(p, float) for p in entry.get('carPolygons', [])]
            self.paint_polygons = [np.asarray(p, float) for p in entry.get('paintPolygons', [])]

    # ----- car ---------------------------------------------------------------------------------------
    def extreme(self, name):
        if self.source == 'synthetic':
            allv = np.vstack(self.car_polygons)
            return {'L': allv[:, 0].min(), 'R': allv[:, 0].max(), 'T': allv[:, 1].min(), 'B': allv[:, 1].max()}[name]
        return self.entry['reference'][name]

    def column_span(self, column):
        """(top, bottom) of the car silhouette at a pixel-centre column, or None."""
        if self.source == 'synthetic':
            spans = [column_interval(p, float(column)) for p in self.car_polygons]
            spans = [s for s in spans if s is not None]
            return None if not spans else (min(s[0] for s in spans), max(s[1] for s in spans))
        top = self.entry['topProfile'].get(str(int(column)), self.entry['topProfile'].get(int(column)))
        bottom = self.entry['bottomProfile'].get(str(int(column)), self.entry['bottomProfile'].get(int(column)))
        return None if top is None or bottom is None else (top, bottom)

    # ----- paint -------------------------------------------------------------------------------------
    def paint_near(self, column):
        """Row of the near (lowest) paint edge at a pixel-centre column, or None."""
        if self.source == 'synthetic':
            spans = [column_interval(p, float(column)) for p in self.paint_polygons]
            spans = [s for s in spans if s is not None]
            return None if not spans else max(s[1] for s in spans)
        row, _bar = paint_row_at(self.entry['bars'], column)
        return row

    # ----- helpers for the oracle --------------------------------------------------------------------
    def car_covered(self, i, j):
        """Pixel (column i, row j) at least half covered by the car (synthetic: coverage map; LiDAR: profile)."""
        if self.source == 'synthetic':
            cov = self.entry['_coverage']
            x0, y0 = self.entry['coverageOrigin']
            if not (0 <= j - y0 < cov.shape[0] and 0 <= i - x0 < cov.shape[1]):
                return False
            return bool(cov[j - y0, i - x0] >= 0.5)
        span = self.column_span(i)
        return span is not None and span[0] <= j <= span[1]

    def paint_covered(self, i, j):
        if self.source == 'synthetic':
            cov = self.entry['_paintCoverage']
            x0, y0 = self.entry['paintCoverageOrigin']
            if not (0 <= j - y0 < cov.shape[0] and 0 <= i - x0 < cov.shape[1]):
                return False
            return bool(cov[j - y0, i - x0] >= 0.5)
        row = self.paint_near(i)
        return row is not None and row - 3 <= j <= row - 0.5
