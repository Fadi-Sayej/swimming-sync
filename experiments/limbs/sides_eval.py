"""Left/right from the pose model inside the leg/arm boxes (step 3), scored on the test frames.

Truth: the hand-drawn 4-class boxes (data/raw/datasets/m2_4cls, test). Two inputs:
- hand boxes: the side logic alone, on the boxes you drew;
- model boxes: the batch-3 outline model's boxes (confidence >= 0.25). A hand box counts as
  found when a model box of the same kind (leg/arm) overlaps it (IoU >= 0.5); the side is
  scored on found limbs, the same way as the 4-class detector.
Pose runs on the full frame, or zoomed on the region around the boxes ("zoom").

Usage: python experiments/limbs/sides_eval.py [--pose yolo11m-pose.pt]
"""
import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from swimsync.limbs.sides import assign_sides, box_iou, pose_people
from swimsync.waterline.above_water import pick_device

DS = Path("data/raw/datasets/m2_4cls")
NAMES = ["left_leg", "right_leg", "left_arm", "right_arm"]
BOXES = "experiments/limbs/output/seg_b3/weights/best.pt"


def hand_limbs(img_path, w, h):
    lab = DS / "labels/test" / (Path(img_path).stem + ".txt")
    out = []
    for line in lab.read_text().splitlines():
        if line.strip():
            c, cx, cy, bw, bh = map(float, line.split())
            side, kind = NAMES[int(c)].split("_")
            out.append((kind, side, ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pose", default="yolo11m-pose.pt")
    a = p.parse_args()
    dev = pick_device()
    pose, boxer = YOLO(a.pose), YOLO(BOXES)
    groups = {"newpool": (DS / "test_newpool.txt").read_text().split(),
              "budapest": (DS / "test_budapest.txt").read_text().split()}
    for group, images in groups.items():
        # (input, pose mode, kind) -> Counter(limbs, found, decided, right)
        n = {}
        for img_path in images:
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            truth = hand_limbs(img_path, w, h)
            if not truth:
                continue
            r = boxer.predict(img, imgsz=960, conf=0.25, verbose=False, device=dev)[0]
            model_boxes = [(boxer.names[int(c)], tuple(b)) for c, b in
                           zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist())]
            for inp, boxes in (("hand boxes", [(k, b) for k, _, b in truth]), ("model boxes", model_boxes)):
                for mode in ("full frame", "zoom"):
                    people = pose_people(pose, img, [b for _, b in boxes] if mode == "zoom" and boxes else None, dev)
                    sides, _ = assign_sides(boxes, people) if boxes else ([], None)
                    for t, (kind, side, tb) in enumerate(truth):
                        c = n.setdefault((inp, mode, kind), Counter())
                        c["limbs"] += 1
                        if inp == "hand boxes":
                            j = t
                        else:
                            cand = [(box_iou(tb, b), i) for i, (k, b) in enumerate(boxes) if k == kind]
                            iou, j = max(cand, default=(0, None))
                            if iou < 0.5:
                                continue
                        c["found"] += 1
                        if sides[j] is not None:
                            c["decided"] += 1
                            c["right"] += sides[j] == side
        print(f"\n{group} ({len(images)} frames)")
        for (inp, mode, kind), c in sorted(n.items()):
            print(f"  {inp:11s} {mode:10s} {kind}: found {c['found']}/{c['limbs']}, side decided "
                  f"{c['decided']}, right {c['right']} ({100 * c['right'] / max(c['decided'], 1):.0f}% of decided)")


if __name__ == "__main__":
    main()
