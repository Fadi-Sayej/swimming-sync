"""Pose inside the leg / arm boxes (milestone 2, step 3): joints and left/right per box.

The box models find legs and arms but can't tell left from right on new pools (a coin flip).
Here the pose model gives a skeleton; each leg/arm box takes the skeleton limb that overlaps it
most, which gives the box its joints (hip-knee-ankle, shoulder-elbow-wrist) and its side.
Two leg boxes always get one left and one right (the best pairing overall).
"""

import numpy as np

from swimsync.dataset.limbs import LIMBS, limb_boxes


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _match(boxes, kp, conf_min, min_iou):
    """Sides for one person: (list of "left"/"right"/None, total overlap)."""
    pose = limb_boxes(kp, np.inf, np.inf, conf_min=conf_min)
    sides, total = [None] * len(boxes), 0.0
    for kind in ("leg", "arm"):
        idx = [i for i, (k, _) in enumerate(boxes) if k == kind]
        if not idx:
            continue
        cost = np.array([[box_iou(boxes[i][1], pose[f"{s}_{kind}"]) if f"{s}_{kind}" in pose else 0.0
                          for s in ("left", "right")] for i in idx])
        cost[cost < min_iou] = 0.0
        # every way to give "left" and "right" to at most one box each (None = to no box)
        options = [None] + list(range(len(idx)))
        li, ri = max(((a, b) for a in options for b in options if a is None or a != b),
                     key=lambda ab: (0 if ab[0] is None else cost[ab[0], 0])
                     + (0 if ab[1] is None else cost[ab[1], 1]))
        for r, c in ((li, 0), (ri, 1)):
            if r is not None and cost[r, c] > 0:
                sides[idx[r]] = ("left", "right")[c]
                total += cost[r, c]
    return sides, total


def assign_sides(boxes, people, conf_min=0.3, min_iou=0.1):
    """boxes: [("leg" | "arm", (x1, y1, x2, y2))]; people: COCO keypoints (17, 3) per person.

    Returns (sides, keypoints of the chosen person or None). The chosen person is the one whose
    limbs overlap the boxes most. A box with no skeleton limb overlapping it gets None.
    """
    best = ([None] * len(boxes), None, 0.0)
    for kp in people:
        sides, total = _match(boxes, kp, conf_min, min_iou)
        if total > best[2]:
            best = (sides, kp, total)
    return best[0], best[1]


def limb_joints(kp, kind, side, conf_min=0.3):
    """The box's joints: [(x, y, conf)] for hip-knee-ankle or shoulder-elbow-wrist."""
    return [tuple(kp[i]) for i in LIMBS[f"{side}_{kind}"] if kp[i][2] >= conf_min]


def crop_around(boxes, img_w, img_h, pad=0.5):
    """Region around all boxes, padded by `pad` of its longer side, as ints inside the image."""
    b = np.asarray(boxes, float)
    x1, y1 = b[:, :2].min(axis=0)
    x2, y2 = b[:, 2:].max(axis=0)
    p = pad * max(x2 - x1, y2 - y1)
    return (int(max(x1 - p, 0)), int(max(y1 - p, 0)),
            int(min(x2 + p, img_w)), int(min(y2 + p, img_h)))


def to_image(kp, crop):
    """Keypoints found in a crop -> image pixels."""
    kp = np.array(kp, float)
    kp[:, 0] += crop[0]
    kp[:, 1] += crop[1]
    return kp


def pose_people(model, img, boxes=None, device=None, pad=0.5):
    """Keypoints (n, 17, 3) of every person; zoomed on the region around `boxes` if given."""
    crop = (0, 0, img.shape[1], img.shape[0])
    if boxes:
        crop = crop_around(boxes, img.shape[1], img.shape[0], pad)
    r = model.predict(img[crop[1]:crop[3], crop[0]:crop[2]], device=device, verbose=False)[0]
    if r.keypoints is None or r.keypoints.data is None:
        return []
    return [to_image(kp, crop) for kp in r.keypoints.data.cpu().numpy()]
