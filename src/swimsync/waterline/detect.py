"""Waterline detection v1: color threshold on the water + robust line fit.

Assumes a side view where the water is BELOW the waterline (camera at or above
water level). The waterline is stored as a line y = a*x + b in pixel coordinates.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Waterline:
    a: float  # slope
    b: float  # intercept
    confidence: float  # share of image columns that found a water edge (0..1)

    def y_at(self, x):
        return self.a * np.asarray(x, dtype=float) + self.b

    def is_above(self, x, y, margin: float = 0.0):
        """True where point (x, y) is above the waterline (smaller y = higher in the image)."""
        return np.asarray(y, dtype=float) < self.y_at(x) - margin


def water_mask(frame, hue=(75, 130), min_sat=40, min_val=40):
    """Binary mask of 'pool water' pixels (blue/cyan in HSV; OpenCV hue is 0..180)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lo = np.array([hue[0], min_sat, min_val], dtype=np.uint8)
    hi = np.array([hue[1], 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lo, hi)
    k = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def _longest_run_start(flags):
    """Index where the longest stretch of True values starts, or None."""
    if not flags.any():
        return None
    padded = np.concatenate([[False], flags, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    starts, ends = edges[::2], edges[1::2]
    return int(starts[np.argmax(ends - starts)])


def _robust_line(xs, ys, tol=6.0, iters=200, seed=0):
    """RANSAC line fit: returns (a, b, inlier_mask), ignoring outlier columns."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    rng = np.random.default_rng(seed)
    i, j = rng.integers(0, len(xs), (2, iters))
    keep = xs[i] != xs[j]
    i, j = i[keep], j[keep]
    a = (ys[j] - ys[i]) / (xs[j] - xs[i])
    b = ys[i] - a * xs[i]
    resid = np.abs(ys[None, :] - (a[:, None] * xs[None, :] + b[:, None]))
    inliers = resid[np.argmax((resid < tol).sum(axis=1))] < tol
    a, b = np.polyfit(xs[inliers], ys[inliers], 1)
    return float(a), float(b), inliers


def detect_waterline(frame, exclude_boxes=(), step=8, run=25, fill=0.8, **mask_kw):
    """Find the waterline in one frame.

    For each sampled column, the edge is where the tallest stretch of water
    starts (rows where at least `fill` of the next `run` rows are water). The
    tallest stretch is the pool itself, so blue walls or banners higher up are
    skipped. A RANSAC fit then ignores columns that disagree (e.g. a blue banner
    merging with the water). Columns inside `exclude_boxes` (the swimmer) are
    skipped so the body doesn't bend the line.
    Returns a Waterline, or None if too few columns found an edge.
    """
    mask = water_mask(frame, **mask_kw) > 0
    h, w = mask.shape

    # share of water in the next `run` rows, for every row and column
    csum = np.vstack([np.zeros((1, w)), np.cumsum(mask, axis=0)])
    rows = np.arange(h - run)
    window = (csum[rows + run] - csum[rows]) / run  # shape (h-run, w)

    xs, ys, tried = [], [], 0
    for x in range(0, w, step):
        if any(x1 <= x <= x2 for x1, _, x2, _ in exclude_boxes):
            continue
        tried += 1
        start = _longest_run_start(window[:, x] >= fill)
        if start is not None:
            xs.append(x)
            # the window reaches `fill` a few rows before the true edge; correct for it
            ys.append(start + run * (1 - fill))

    if len(xs) < 5:
        return None
    a, b, inliers = _robust_line(xs, ys)
    return Waterline(a, b, int(inliers.sum()) / max(tried, 1))


class SmoothedWaterline:
    """Exponential smoothing across frames so the line doesn't jitter.

    A new line more than `jump` px away from the smoothed one (at either image
    edge) is not averaged in:
      - confident (>= `sure` of columns agree): a scene cut, start over there now;
      - unsure (e.g. splash): ignored, unless it holds for `confirm` frames in a row.
    """

    def __init__(self, alpha=0.3, jump=40.0, sure=0.5, confirm=10, width=1920):
        self.alpha = alpha
        self.jump = jump
        self.sure = sure
        self.confirm = confirm
        self.edges = np.array([0.0, float(width)])
        self.line = None
        self.far = []  # recent lines far from the smoothed one

    def _close(self, l1, l2):
        return np.abs(l1.y_at(self.edges) - l2.y_at(self.edges)).max() <= self.jump

    def update(self, new):
        if new is None:
            return self.line
        if self.line is None:
            self.line = new
        elif self._close(new, self.line):
            self.far = []
            k = self.alpha
            self.line = Waterline(
                (1 - k) * self.line.a + k * new.a,
                (1 - k) * self.line.b + k * new.b,
                new.confidence,
            )
        else:
            self.far = self.far + [new] if self.far and self._close(new, self.far[-1]) else [new]
            if new.confidence >= self.sure or len(self.far) >= self.confirm:
                self.line, self.far = new, []
        return self.line


def line_from_points(x1, y1, x2, y2):
    a = (y2 - y1) / (x2 - x1) if x2 != x1 else 0.0
    return Waterline(a, y1 - a * x1, 1.0)
