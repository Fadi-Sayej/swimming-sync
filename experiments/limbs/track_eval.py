"""Leg sides by following the legs through the video (step 3), scored on the test frames.

For every test frame: read the video 3 s before and after it, and on each video frame keep the
leg boxes (batch-3 outline model), the pose skeletons and whether a camera cut comes right
before it. Cached in data/raw/frames/m2/track/ (~20 min the first time). Then the leg sides
come from the nearest frame with the torso in view (swimsync.limbs.track), and are scored
against the hand-drawn 4-class labels, next to the pose model on the frame alone.

Usage: python experiments/limbs/track_eval.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from sides_eval import DS, hand_limbs  # noqa: E402

from swimsync.limbs.sides import assign_sides, box_iou, pose_people  # noqa: E402
from swimsync.limbs.track import is_cut, track_sides  # noqa: E402
from swimsync.waterline.above_water import pick_device  # noqa: E402

WINDOW = 3.0  # seconds before and after the test frame
CACHE = Path("data/raw/frames/m2/track")
BOXES = "experiments/limbs/output/seg_b3/weights/best.pt"


def leg_boxes(boxer, img, dev):
    r = boxer.predict(img, imgsz=960, conf=0.25, verbose=False, device=dev)[0]
    return [b for c, b in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist()) if boxer.names[int(c)] == "leg"]


def build(img_path, boxer, pose, dev):
    stem = Path(img_path).stem
    video, t = stem.rsplit("_", 1)
    t = float(t[:-1])
    cap = cv2.VideoCapture(f"data/raw/test/{video}.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_MSEC, max(t - WINDOW, 0) * 1000)
    ref = cv2.resize(cv2.imread(img_path), (160, 90)).astype(float)
    frames, diffs, prev = [], [], None
    for _ in range(int(2 * WINDOW * fps) + 1):
        ok, f = cap.read()
        if not ok:
            break
        diffs.append(float(np.abs(cv2.resize(f, (160, 90)).astype(float) - ref).mean()))
        frames.append({"legs": leg_boxes(boxer, f, dev),
                       "people": [kp.tolist() for kp in pose_people(pose, f, None, dev)],
                       "cut": bool(prev is not None and is_cut(prev, f))})
        prev = f
    target = int(np.argmin(diffs))
    return {"fps": fps, "target": target, "target_diff": diffs[target], "frames": frames}


def main():
    dev = pick_device()
    boxer, pose = YOLO(BOXES), YOLO("yolo11m-pose.pt")
    CACHE.mkdir(parents=True, exist_ok=True)
    groups = {"newpool": (DS / "test_newpool.txt").read_text().split(),
              "budapest": (DS / "test_budapest.txt").read_text().split()}
    for group, images in groups.items():
        n, gaps, far = {}, [], 0
        for img_path in images:
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            legs = [(side, b) for kind, side, b in hand_limbs(img_path, w, h) if kind == "leg"]
            if not legs:
                continue
            cache = CACHE / (Path(img_path).stem + ".json")
            if not cache.exists():
                cache.write_text(json.dumps(build(img_path, boxer, pose, dev)))
                print("cached", cache.name, flush=True)
            c = json.loads(cache.read_text())
            frames = [{**f, "people": [np.array(p) for p in f["people"]]} for f in c["frames"]]
            far += c["target_diff"] > 10  # the video frame doesn't look like the test image
            model_legs = leg_boxes(boxer, img, dev)
            people = pose_people(pose, img, None, dev)
            for inp, boxes in (("hand boxes", [b for _, b in legs]), ("model boxes", model_legs)):
                pose_sides = assign_sides([("leg", b) for b in boxes], people)[0] if boxes else []
                trk_sides, gap = track_sides(frames, c["target"], boxes) if boxes else ([], None)
                if inp == "hand boxes" and gap is not None:
                    gaps.append(gap / c["fps"])
                both = [t if t is not None else p for t, p in zip(trk_sides, pose_sides)]
                for side, tb in legs:
                    if inp == "hand boxes":
                        j = [b for _, b in legs].index(tb)
                    else:
                        cand = [(box_iou(tb, b), i) for i, b in enumerate(boxes)]
                        iou, j = max(cand, default=(0, None))
                        if iou < 0.5:
                            for m in ("pose on the frame", "tracking", "tracking, else pose"):
                                n.setdefault((inp, m), Counter())["limbs"] += 1
                            continue
                    for m, s in (("pose on the frame", pose_sides), ("tracking", trk_sides),
                                 ("tracking, else pose", both)):
                        k = n.setdefault((inp, m), Counter())
                        k["limbs"] += 1
                        k["found"] += 1
                        if s[j] is not None:
                            k["decided"] += 1
                            k["right"] += s[j] == side
        print(f"\n{group} ({len(images)} frames; {far} video frames not matching the image; "
              f"anchor {np.median(gaps) if gaps else float('nan'):.1f} s away (median))")
        for (inp, m), k in sorted(n.items()):
            print(f"  {inp:11s} {m:20s} legs: found {k['found']}/{k['limbs']}, decided {k['decided']}, "
                  f"right {k['right']} ({100 * k['right'] / max(k['decided'], 1):.0f}% of decided)")


if __name__ == "__main__":
    main()
