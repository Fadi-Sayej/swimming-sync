"""How good are the ready-made pose model's knees? (leg height, Task 1)

Uses labels that already exist: the knee clicks in Label Studio "M2 keypoints - test" and the
hand-drawn leg boxes of the 4-class test set. A leg box "has its knee out" when a hand knee click
lies inside it (box grown 10%). The ready-made yolo11m-pose runs on the full frame and zoomed on
the legs; its knees (COCO 13, 14) with confidence >= 0.3 count. Sides are ignored (nearest knee).
Our knee model (plan Task 8, if trained) runs on a crop around each leg box.

Usage: python experiments/height/knee_check.py   (Label Studio running)
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parents[1] / "limbs"))
from facing_eval import DS  # noqa: E402

from swimsync.dataset.export_pose import labeled_points  # noqa: E402
from swimsync.dataset.export_yolo import labeled_frames  # noqa: E402
from swimsync.dataset.facing import _api  # noqa: E402
from swimsync.limbs.sides import pose_people  # noqa: E402
from swimsync.waterline.above_water import pick_device  # noqa: E402

KNEES = (13, 14)
OURS = {"our knee model (1 point)": "experiments/limbs/output/knees/weights/best.pt",
        "our knee model (17-point layout)": "experiments/limbs/output/knees17/weights/best.pt",
        "our knee model + leg height labels": "experiments/limbs/output/knees17b/weights/best.pt",
        "our knee model + leg height labels + no-leg crops": "experiments/limbs/output/knees17c/weights/best.pt"}


def leg_boxes(img_path, w, h):
    """Hand-drawn leg boxes (classes 0, 1 of the 4-class test labels), pixels."""
    lab = DS / "labels/test" / (Path(img_path).stem + ".txt")
    out = []
    for line in lab.read_text().splitlines():
        if line.strip():
            c, cx, cy, bw, bh = map(float, line.split())
            if int(c) in (0, 1):
                out.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def inside(p, box, grow=0.1):
    x1, y1, x2, y2 = box
    gx, gy = grow * (x2 - x1), grow * (y2 - y1)
    return x1 - gx <= p[0] <= x2 + gx and y1 - gy <= p[1] <= y2 + gy


def main():
    dev = pick_device()
    pose = YOLO("yolo11m-pose.pt")
    lab, _ = labeled_frames(_api(), "M2 keypoints - test")
    images = (DS / "test_newpool.txt").read_text().split() + (DS / "test_budapest.txt").read_text().split()
    from swimsync.height.measure import OurKnees

    ours = {k: OurKnees(w, device=dev) for k, w in OURS.items() if Path(w).exists()}
    n = {v: dict(legs=0, agree=0, hand_out=0, model_out=0, both_out=0, k10=0, k20=0, knees=0)
         for v in ("full frame", "zoomed", *ours)}
    for img_path in images:
        name = Path(img_path).name
        if name not in lab:
            continue
        pts = labeled_points(lab[name]) or {}
        hand_knees = [p for k, p in pts.items() if k.endswith("_knee")]
        img = cv2.imread(img_path)
        legs = leg_boxes(img_path, img.shape[1], img.shape[0])
        if not legs:
            continue
        for variant in n:
            if variant in ours:
                per_leg = [[k[:2]] if k else [] for k in ours[variant].knees(img, [(b, None) for b in legs])]
            else:
                people = pose_people(pose, img, legs if variant == "zoomed" else None, dev)
                pred = [(kp[i, 0], kp[i, 1]) for kp in people for i in KNEES if kp[i, 2] >= 0.3]
                per_leg = [pred] * len(legs)
            c = n[variant]
            for box, pred in zip(legs, per_leg):
                size = max(box[2] - box[0], box[3] - box[1])
                hk = [p for p in hand_knees if inside(p, box)]
                mk = [p for p in pred if inside(p, box)]
                c["legs"] += 1
                c["hand_out"] += bool(hk)
                c["model_out"] += bool(mk)
                c["both_out"] += bool(hk) and bool(mk)
                c["agree"] += bool(hk) == bool(mk)
                for p in hk:
                    c["knees"] += 1
                    d = min((np.hypot(p[0] - q[0], p[1] - q[1]) for q in pred), default=np.inf) / size
                    c["k10"] += d <= 0.1
                    c["k20"] += d <= 0.2
    for v, c in n.items():
        print(f"{v}: {c['legs']} leg boxes; knee out by hand {c['hand_out']}, by model {c['model_out']}, both {c['both_out']}; "
              f"knee out/not agree {c['agree']}/{c['legs']} ({100 * c['agree'] / max(c['legs'], 1):.0f}%); "
              f"hand knees {c['knees']}: model knee within 10% of the leg {c['k10']} "
              f"({100 * c['k10'] / max(c['knees'], 1):.0f}%), within 20% {c['k20']} ({100 * c['k20'] / max(c['knees'], 1):.0f}%)")


if __name__ == "__main__":
    main()
