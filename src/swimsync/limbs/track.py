"""Leg sides by following the legs through the video (milestone 2, step 3).

On a frame with only the legs (a vertical, a split), the pose model can't tell left from right
(problems #25). But a few seconds earlier or later her torso is usually in view, and there the
pose sides are much better. So: find the nearest such "anchor" frame in the same shot, take the
leg sides there, and follow the two leg boxes frame by frame to the target frame.
"""

import cv2
import numpy as np

from swimsync.dataset.facing import FEET_UP, sides_from_facing
from swimsync.limbs.sides import assign_sides

TORSO = (5, 6, 11, 12)  # shoulders and hips


def is_anchor(kp, conf_min=0.5):
    """Torso in view (both shoulders and both hips sure): the pose model's sides can be trusted."""
    return all(kp[i][2] >= conf_min for i in TORSO)


def is_cut(prev, cur, min_corr=0.5):
    """Scene cut between two BGR frames: their colour histograms (hue, saturation) differ."""
    hists = []
    for f in (prev, cur):
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        hists.append(cv2.normalize(h, h).flatten())
    return cv2.compareHist(hists[0], hists[1], cv2.HISTCMP_CORREL) < min_corr


def _center(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def pair(tracks, boxes):
    """{side: box index}: each tracked leg to a different box, smallest total move.

    With one box, only the nearest leg takes it; the other keeps its last place.
    """
    move = {s: [float(np.linalg.norm(_center(t) - _center(b))) for b in boxes] for s, t in tracks.items()}
    if not boxes:
        return {}
    if len(boxes) == 1:
        s = min(move, key=lambda s: move[s][0])
        return {s: 0}
    i, j = min(((i, j) for i in range(len(boxes)) for j in range(len(boxes)) if i != j),
               key=lambda ij: move["left"][ij[0]] + move["right"][ij[1]])
    return {"left": i, "right": j}


def _anchor_tracks(f):
    """{"left": box, "right": box} on an anchor frame, or None."""
    people = [kp for kp in f["people"] if is_anchor(kp)]
    if len(f["legs"]) < 2 or not people:
        return None
    sides, _ = assign_sides([("leg", b) for b in f["legs"]], people)
    tracks = {s: b for s, b in zip(sides, f["legs"]) if s is not None}
    return tracks if len(tracks) == 2 else None


def facing_anchor(f, min_conf=0.8):
    """{"left": box, "right": box} from a sure front/back answer on two legs, or None.

    f["facing"]: (answer, confidence) of the front/back model on the two legs. Assumes her feet
    point up; on frames labeled both ways that matches the hand-drawn sides 88% of the time (#25).
    """
    if len(f["legs"]) != 2 or not f.get("facing"):
        return None
    answer, conf = f["facing"]
    if answer not in ("front", "back") or conf < min_conf:
        return None
    a, b = f["legs"]
    sides = sides_from_facing(answer, FEET_UP, a, b)
    return {sides[0]: a, sides[1]: b} if sides else None


def nearest_anchor(frames, target, anchor, max_gap=None):
    """(index, {"left": box, "right": box}) of the anchor frame nearest `target` in its shot.

    (None, None) when the shot has no anchor within `max_gap` frames (any distance if None).
    """
    start = max([i for i in range(target + 1) if frames[i]["cut"]], default=0)
    end = min([i - 1 for i in range(target + 1, len(frames)) if frames[i]["cut"]], default=len(frames) - 1)
    for a in sorted(range(start, end + 1), key=lambda i: abs(i - target)):
        if max_gap is not None and abs(a - target) > max_gap:
            break
        tracks = anchor(frames[a])
        if tracks is not None:
            return a, tracks
    return None, None


def track_sides(frames, target, target_boxes, anchor=_anchor_tracks, max_gap=None):
    """Sides for the leg boxes `target_boxes` on frame `target`, from the nearest anchor in the shot.

    frames: [{"legs": [box], "cut": True if a cut comes right before this frame, plus what `anchor`
    reads: "people" (keypoints) for the torso anchor, "facing" for facing_anchor}], in time order.
    Returns (sides, frames between the anchor and the target), or ([None, ...], None) when the
    shot has no anchor (within `max_gap` frames, if given: legs that spin or overlap swap boxes, so
    long carries go wrong, #25).
    """
    a, tracks = nearest_anchor(frames, target, anchor, max_gap)
    if a is None:
        return [None] * len(target_boxes), None
    step = 1 if target > a else -1
    for k in range(a + step, target, step):
        for s, i in pair(tracks, frames[k]["legs"]).items():
            tracks[s] = frames[k]["legs"][i]
    found = pair(tracks, target_boxes)
    sides = [None] * len(target_boxes)
    for s, i in found.items():
        sides[i] = s
    return sides, abs(target - a)


def sides_from_masks(masks, boxes):
    """Sides for `boxes` from the followed leg masks {"left": mask, "right": mask} (bool, H x W).

    Each box takes a different leg, the pairing that puts the most mask pixels inside the boxes;
    a box no leg mask reaches gets None.
    """
    def inside(m, b):
        x1, y1, x2, y2 = (max(int(round(v)), 0) for v in b)
        return int(m[y1:y2, x1:x2].sum())

    px = {s: [inside(m, b) for b in boxes] for s, m in masks.items()}
    sides = [None] * len(boxes)
    if len(boxes) == 1:
        s = max(px, key=lambda s: px[s][0])
        sides[0] = s if px[s][0] > 0 else None
        return sides
    i, j = max(((i, j) for i in range(len(boxes)) for j in range(len(boxes)) if i != j),
               key=lambda ij: px["left"][ij[0]] + px["right"][ij[1]], default=(None, None))
    for s, k in (("left", i), ("right", j)):
        if k is not None and px[s][k] > 0:
            sides[k] = s
    return sides
