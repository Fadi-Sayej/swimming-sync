"""Rules from a leg's outline and knee to pass/fail and the share of the leg out of the water.

Score (agreed Nov 24): knee under water = FAIL; knee out = the share of the leg, toes to hip, out
of the water. The hip is usually hidden, so the thigh counts as long as toes-to-knee: water at the
knee = 50%, halfway up the thigh = 75%, at the hip = 100%.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class LegHeight:
    status: str  # "pass" | "fail" (knee under water, or no knee found: knee None) | "no score"
    score: float | None = None
    toes: tuple | None = None
    knee: tuple | None = None
    water: tuple | None = None
    knee_conf: float | None = None
    box: tuple | None = None  # the leg's box and outline (for drawing), set by measure_frame
    mask: object = None


def leg_ends(mask, min_pixels=200, end_share=0.05):
    """(toes, water) of a leg outline, in pixels, or None if the outline is too small.

    The two ends along the outline's main direction, each the mean of its last `end_share` of
    pixels; the water is the end lower in the image (her feet point up).
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < min_pixels:
        return None
    pts = np.column_stack([xs, ys]).astype(float)
    centre = pts.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov((pts - centre).T))
    proj = (pts - centre) @ vecs[:, -1]
    lo, hi = np.quantile(proj, [end_share, 1 - end_share])
    a, b = pts[proj <= lo].mean(axis=0), pts[proj >= hi].mean(axis=0)
    toes, water = (a, b) if a[1] < b[1] else (b, a)
    return tuple(toes), tuple(water)


def toes_cut(mask, toes, water, reach=0.08, margin=3):
    """Does the leg run off the picture at its toes end (outline pixels near the toes touch the edge)?"""
    ys, xs = np.nonzero(mask)
    t = np.asarray(toes, float)
    near = np.hypot(xs - t[0], ys - t[1]) <= reach * np.linalg.norm(np.asarray(water, float) - t) + margin
    h, w = mask.shape
    xs, ys = xs[near], ys[near]
    return bool(((xs <= margin) | (xs >= w - 1 - margin) | (ys <= margin) | (ys >= h - 1 - margin)).any())


def knee_is_out(toes, knee, water, margin=0.05):
    """Is the knee between the toes and the water along the leg (not past the water)?"""
    t, k, w = (np.asarray(p, float) for p in (toes, knee, water))
    v = w - t
    if v @ v == 0:
        return False
    pos = (k - t) @ v / (v @ v)
    return margin < pos < 1 - margin


def height_score(toes, knee, water):
    """Share of the leg out of the water (0.5 at the knee .. 1.0 at the hip); None if no shin length."""
    t, k, w = (np.asarray(p, float) for p in (toes, knee, water))
    shin = np.linalg.norm(k - t)
    if shin < 1:
        return None
    return float(min(1.0, (shin + np.linalg.norm(w - k)) / (2 * shin)))


def judge_leg(toes, water, knee, knee_conf, toes_cut=False):
    """LegHeight for one leg: no score without toes/water; FAIL when no knee is found or it is under water.

    toes_cut: the leg runs off the picture, so pass/fail still counts but there is no percentage.
    """
    if toes is None or water is None:
        return LegHeight("no score", knee=knee, knee_conf=knee_conf)
    if knee is None or not knee_is_out(toes, knee, water):
        return LegHeight("fail", toes=toes, knee=knee, water=water, knee_conf=knee_conf)
    return LegHeight("pass", None if toes_cut else height_score(toes, knee, water), toes, knee, water, knee_conf)
