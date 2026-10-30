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


def detect_waterline(frame, exclude_boxes=(), step=8, run=25, fill=0.8, **mask_kw):
    """Find the waterline in one frame.

    For each sampled column, the edge is the first row from the top below which
    at least `fill` of the next `run` rows are water. Columns inside
    `exclude_boxes` (the swimmer) are skipped so the body doesn't bend the line.
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
        hits = np.flatnonzero(window[:, x] >= fill)
        if hits.size:
            xs.append(x)
            # the window reaches `fill` a few rows before the true edge; correct for it
            ys.append(hits[0] + run * (1 - fill))

    if len(xs) < 5:
        return None
    pts = np.array(list(zip(xs, ys)), dtype=np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    if abs(vx) < 1e-6:
        return None
    a = vy / vx
    b = y0 - a * x0
    return Waterline(float(a), float(b), len(xs) / max(tried, 1))


class SmoothedWaterline:
    """Exponential smoothing across frames so the line doesn't jitter."""

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.line = None

    def update(self, new):
        if new is None:
            return self.line
        if self.line is None:
            self.line = new
        else:
            k = self.alpha
            self.line = Waterline(
                (1 - k) * self.line.a + k * new.a,
                (1 - k) * self.line.b + k * new.b,
                new.confidence,
            )
        return self.line


def line_from_points(x1, y1, x2, y2):
    a = (y2 - y1) / (x2 - x1) if x2 != x1 else 0.0
    return Waterline(a, y1 - a * x1, 1.0)
