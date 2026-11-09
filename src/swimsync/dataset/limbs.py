"""Starting boxes for each leg and arm, from pose keypoints (COCO order).

Sides are the swimmer's own left/right, as the pose model names them. The model
often swaps them when she is upside down, so these boxes are only a starting
point for labeling.
"""

import numpy as np

# hip/shoulder, knee/elbow, ankle/wrist; COCO keypoint indices
LIMBS = {
    "left_leg": (11, 13, 15),
    "right_leg": (12, 14, 16),
    "left_arm": (5, 7, 9),
    "right_arm": (6, 8, 10),
}
# how far the foot / hand reaches past the ankle / wrist, as a share of the shin / forearm
REACH = {"left_leg": 0.6, "right_leg": 0.6, "left_arm": 0.7, "right_arm": 0.7}


def limb_boxes(kp, img_w, img_h, conf_min=0.3, pad=0.1, min_pad=8.0):
    """{label: (x1, y1, x2, y2)} for every limb with at least 2 confident keypoints.

    The box covers the confident joints, plus the foot/hand past the last joint
    when the two lower joints are known, plus `pad` of the limb length around it.
    """
    kp = np.asarray(kp, float)
    boxes = {}
    for label, idx in LIMBS.items():
        pts = kp[list(idx)]
        ok = pts[:, 2] >= conf_min
        if ok.sum() < 2:
            continue
        xy = pts[ok, :2]
        if ok[1] and ok[2]:  # knee + ankle (elbow + wrist): add the pointed foot / hand
            mid, end = pts[1, :2], pts[2, :2]
            xy = np.vstack([xy, end + REACH[label] * (end - mid)])
        length = np.linalg.norm(np.diff(pts[ok, :2], axis=0), axis=1).sum()
        p = max(pad * length, min_pad)
        x1, y1 = xy.min(axis=0) - p
        x2, y2 = xy.max(axis=0) + p
        boxes[label] = (
            float(np.clip(x1, 0, img_w)), float(np.clip(y1, 0, img_h)),
            float(np.clip(x2, 0, img_w)), float(np.clip(y2, 0, img_h)),
        )
    return boxes
