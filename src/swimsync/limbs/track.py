"""Leg sides by following the legs through the video (milestone 2, step 3).

On a frame with only the legs (a vertical, a split), the pose model can't tell left from right
(problems #25). But a few seconds earlier or later her torso is usually in view, and there the
pose sides are much better. So: find the nearest such "anchor" frame in the same shot, take the
leg sides there, and follow the two leg boxes frame by frame to the target frame.
"""

import cv2
import numpy as np

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


def track_sides(frames, target, target_boxes):
    """Sides for the leg boxes `target_boxes` on frame `target`, from the nearest anchor in the shot.

    frames: [{"legs": [box], "people": [keypoints (17, 3)], "cut": True if a cut comes right before
    this frame}], in time order. Returns (sides, frames between the anchor and the target), or
    ([None, ...], None) when the shot has no anchor.
    """
    start = max([i for i in range(target + 1) if frames[i]["cut"]], default=0)
    end = min([i - 1 for i in range(target + 1, len(frames)) if frames[i]["cut"]], default=len(frames) - 1)
    for a in sorted(range(start, end + 1), key=lambda i: abs(i - target)):
        tracks = _anchor_tracks(frames[a])
        if tracks is None:
            continue
        step = 1 if target > a else -1
        for k in range(a + step, target, step):
            for s, i in pair(tracks, frames[k]["legs"]).items():
                tracks[s] = frames[k]["legs"][i]
        found = pair(tracks, target_boxes)
        sides = [None] * len(target_boxes)
        for s, i in found.items():
            sides[i] = s
        return sides, abs(target - a)
    return [None] * len(target_boxes), None
