"""Leg sides from front/back (#25), scored on the test frames.

Truth: the hand-drawn 4-class boxes (data/raw/datasets/m2_4cls, test), frames with both legs.
For each frame the front/back model looks at a square crop around both legs and answers front,
back or unsure; swimsync.dataset.facing turns front/back into her left and right, assuming her
feet point up (legs out of the water). On 59 frames labeled both ways that geometry matches the
hand-drawn sides on 52 (88%); the pose model's feet direction was worse (29 of 36), because it
often puts the feet at the bottom of upside-down legs.

Two inputs, as in sides_eval.py:
- hand boxes: your two leg boxes;
- model boxes: the outline model's legs when it finds exactly two, each matched to a hand box
  (IoU >= 0.5).

Usage: python experiments/limbs/facing_eval.py [--facing .../facing/weights/best.pt] [--min-conf 0.5]
"""
import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from swimsync.dataset.facing import sides_from_facing, square_crop
from swimsync.limbs.sides import box_iou
from swimsync.waterline.above_water import pick_device

DS = Path("data/raw/datasets/m2_4cls")
FACING = "experiments/limbs/output/facing/weights/best.pt"
BOXES = "experiments/limbs/output/seg_b3/weights/best.pt"
FEET_UP = np.array([0.0, -1.0])


def hand_legs(img_path, w, h):
    """{'left': box, 'right': box} from the 4-class labels (classes 0, 1 = left, right leg)."""
    out = {}
    lab = DS / "labels/test" / (Path(img_path).stem + ".txt")
    for line in lab.read_text().splitlines():
        if line.strip():
            c, cx, cy, bw, bh = map(float, line.split())
            if int(c) in (0, 1):
                out["left" if int(c) == 0 else "right"] = ((cx - bw / 2) * w, (cy - bh / 2) * h,
                                                           (cx + bw / 2) * w, (cy + bh / 2) * h)
    return out


def front_or_back(model, img, legs, device):
    """('front' / 'back' / 'unsure', probability) for the square crop around the legs."""
    x1, y1, x2, y2 = square_crop(legs, img.shape[1], img.shape[0])
    r = model.predict(img[y1:y2, x1:x2], imgsz=384, verbose=False, device=device)[0]
    return model.names[int(r.probs.top1)], float(r.probs.top1conf)


def model_pair(boxer, img, truth, device):
    """The outline model's two leg boxes as (her left, her right), or None if not both found."""
    r = boxer.predict(img, imgsz=960, conf=0.25, verbose=False, device=device)[0]
    found = [b for c, b in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist()) if boxer.names[int(c)] == "leg"]
    if len(found) != 2:
        return None
    a, b = found
    if box_iou(a, truth["left"]) + box_iou(b, truth["right"]) < box_iou(a, truth["right"]) + box_iou(b, truth["left"]):
        a, b = b, a
    return (a, b) if box_iou(a, truth["left"]) >= 0.5 and box_iou(b, truth["right"]) >= 0.5 else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--facing", default=FACING)
    p.add_argument("--min-conf", type=float, default=0.0, help="decide only when the model is this sure")
    a = p.parse_args()
    dev = pick_device()
    clf, boxer = YOLO(a.facing), YOLO(BOXES)
    groups = {"newpool": (DS / "test_newpool.txt").read_text().split(),
              "budapest": (DS / "test_budapest.txt").read_text().split()}
    for group, images in groups.items():
        n = {"hand boxes": Counter(), "model boxes": Counter()}
        for img_path in images:
            img = cv2.imread(img_path)
            truth = hand_legs(img_path, img.shape[1], img.shape[0])
            if len(truth) < 2:
                continue
            inputs = {"hand boxes": (truth["left"], truth["right"]), "model boxes": model_pair(boxer, img, truth, dev)}
            for inp, legs in inputs.items():
                c = n[inp]
                c["frames"] += 1
                if legs is None:
                    continue
                c["found"] += 1
                answer, conf = front_or_back(clf, img, legs, dev)
                if answer == "unsure" or conf < a.min_conf:
                    continue
                sides = sides_from_facing(answer, FEET_UP, legs[0], legs[1])
                if sides is None:  # crossed legs
                    continue
                c["decided"] += 1
                c["right"] += sides == ("left", "right")
        print(f"\n{group} ({len(images)} frames)")
        for inp, c in n.items():
            print(f"  {inp:11s}: both legs {c['frames']}, found {c['found']}, front/back given {c['decided']}, "
                  f"sides right {c['right']} ({100 * c['right'] / max(c['decided'], 1):.0f}%)")


if __name__ == "__main__":
    main()
