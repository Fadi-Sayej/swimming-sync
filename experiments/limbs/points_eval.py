"""How close the pose model's points are to the hand-labeled ones (step 3), on the keypoint test set.

Test = data/raw/datasets/m2_pose/images/test: 87 frames cropped around her leg/arm boxes (the zoom
the pose model gets at use time), 12 labeled points each at most. Per frame, the predicted
person whose box overlaps the labeled box most is scored. "Her size" = the longer side of the
labeled box.

Usage: python experiments/limbs/points_eval.py [--pose yolo11m-pose.pt yolo11x-pose.pt ...]
"""
import argparse
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from swimsync.dataset.export_pose import FLIP_IDX
from swimsync.limbs.sides import box_iou
from swimsync.waterline.above_water import pick_device

DS = Path("data/raw/datasets/m2_pose")


def labels(img, w, h):
    v = np.array((DS / "labels/test" / (img.stem + ".txt")).read_text().split(), float)
    cx, cy, bw, bh = v[1:5] * [w, h, w, h]
    kp = v[5:].reshape(17, 3) * [w, h, 1]
    return [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], kp


def score(model, dev, imgsz=640):
    n = {"points": 0, "10%": 0, "20%": 0, "pairs": 0, "swapped": 0}
    for img in sorted((DS / "images/test").glob("*.jpg")):
        r = model.predict(str(img), imgsz=imgsz, conf=0.1, verbose=False, device=dev)[0]
        h, w = r.orig_shape
        box, truth = labels(img, w, h)
        size = max(box[2] - box[0], box[3] - box[1])
        seen = truth[:, 2] > 0
        n["points"] += int(seen.sum())
        if r.keypoints is None or len(r.boxes) == 0:
            continue
        best = max(range(len(r.boxes)), key=lambda i: box_iou(box, r.boxes.xyxy[i].tolist()))
        pred = r.keypoints.data[best].cpu().numpy()
        d = np.linalg.norm(pred[:, :2] - truth[:, :2], axis=1)
        n["10%"] += int((seen & (d <= 0.1 * size)).sum())
        n["20%"] += int((seen & (d <= 0.2 * size)).sum())
        for k in range(17):
            o = FLIP_IDX[k]
            if o != k and seen[k] and seen[o]:
                n["pairs"] += 1
                n["swapped"] += int(np.linalg.norm(pred[k, :2] - truth[o, :2]) < d[k])
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pose", nargs="+", default=["yolo11m-pose.pt"])
    a = p.parse_args()
    dev = pick_device()
    for m in a.pose:
        n = score(YOLO(m), dev)
        print(f"{m}: {n['points']} points; within 10% {n['10%']} ({100 * n['10%'] / n['points']:.0f}%), "
              f"within 20% {n['20%']} ({100 * n['20%'] / n['points']:.0f}%); "
              f"nearer the other side {n['swapped']}/{n['pairs']} ({100 * n['swapped'] / max(n['pairs'], 1):.0f}%)",
              flush=True)


if __name__ == "__main__":
    main()
