"""Leg sides carried from a sure front/back frame by SAM 2 following each leg (#25), on the test frames.

track_facing_eval.py carries the sides by moving box centers frame to frame; that swaps the legs
when she spins or they cross, so it is only trusted up to 0.1 s. Here SAM 2's video mode follows
each leg instead: on the nearest sure front/back frame in the shot (swimsync.limbs.track), the two
leg boxes become two objects ("left", "right"), SAM 2 follows them to the test frame (backwards in
time if the sure frame comes after), and the test frame's leg boxes take the side of the mask
inside them (sides_from_masks). Masks cached in data/raw/frames/m2/sam_track/.

Usage: python experiments/limbs/sam_track_eval.py [--min-conf 0.8]
"""
import argparse
import json
import sys
import tempfile
from collections import Counter
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.models.sam import SAM2VideoPredictor

sys.path.insert(0, str(Path(__file__).parent))
from facing_eval import FACING, hand_legs, model_pair  # noqa: E402
from track_eval import BOXES, CACHE, WINDOW  # noqa: E402
from track_facing_eval import add_facing  # noqa: E402

from swimsync.limbs.track import facing_anchor, nearest_anchor, sides_from_masks, track_sides  # noqa: E402
from swimsync.waterline.above_water import pick_device  # noqa: E402

DS = Path("data/raw/datasets/m2_4cls")
OUT = CACHE.parent / "sam_track"
BINS = [(0, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, 99)]


def bin_label(lo, hi):
    return f"{lo}-{hi} s" if hi < 99 else f">{lo} s"


def read_frames(stem, first, last):
    """Video frames first..last of the cached clip around the test image (as track_eval reads them)."""
    video, t = stem.rsplit("_", 1)
    cap = cv2.VideoCapture(f"data/raw/test/{video}.mp4")
    cap.set(cv2.CAP_PROP_POS_MSEC, max(float(t[:-1]) - WINDOW, 0) * 1000)
    out = []
    for i in range(last + 1):
        ok, f = cap.read()
        if not ok:
            break
        if i >= first:
            out.append(f)
    return out


def follow(frames, boxes, dev):
    """SAM 2 masks of each box's object on the last frame, starting from `boxes` on frame 0."""
    h, w = frames[0].shape[:2]
    with tempfile.TemporaryDirectory() as tmp:
        clip = str(Path(tmp) / "clip.mp4")
        out = cv2.VideoWriter(clip, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
        for f in frames:
            out.write(f)
        out.release()
        pred = SAM2VideoPredictor(overrides=dict(model="sam2.1_b.pt", imgsz=1024, device=dev, verbose=False,
                                                 save=False))
        pred.args.conf = -1.0  # keep lost (empty) objects so mask i stays box i
        last = None
        for r in pred(source=clip, bboxes=[list(map(float, b)) for b in boxes], stream=True):
            last = r
    if last is None or last.masks is None:
        return [np.zeros((h, w), bool) for _ in boxes]
    return [m.cpu().numpy().astype(bool) for m in last.masks.data]


def sam_masks(stem, c, a, tracks, dev):
    """{"left": mask, "right": mask} on the test frame, followed from anchor frame `a` (cached)."""
    f = OUT / f"{stem}.npz"
    if not f.exists():
        t = c["target"]
        frames = read_frames(stem, min(a, t), max(a, t))
        if a > t:
            frames = frames[::-1]  # follow backwards in time
        left, right = follow(frames, [tracks["left"], tracks["right"]], dev)
        np.savez_compressed(f, left=left, right=right, anchor=a)
    z = np.load(f)
    return {"left": z["left"], "right": z["right"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--facing", default=FACING)
    p.add_argument("--min-conf", type=float, default=0.8)
    a = p.parse_args()
    dev = pick_device()
    OUT.mkdir(parents=True, exist_ok=True)
    clf, boxer = YOLO(a.facing), YOLO(BOXES)
    anchor = partial(facing_anchor, min_conf=a.min_conf)
    groups = {"newpool": (DS / "test_newpool.txt").read_text().split(),
              "budapest": (DS / "test_budapest.txt").read_text().split()}
    for group, images in groups.items():
        n = Counter()  # (input, method, gap bin, "decided"/"right"), plus frame counts
        for img_path in images:
            img = cv2.imread(img_path)
            truth = hand_legs(img_path, img.shape[1], img.shape[0])
            stem = Path(img_path).stem
            cache = CACHE / f"{stem}.json"
            if len(truth) < 2 or not cache.exists():
                continue
            c = add_facing(cache, clf, dev)
            n["frames"] += 1
            idx, tracks = nearest_anchor(c["frames"], c["target"], anchor)
            if idx is None:
                continue
            n["with a sure frame"] += 1
            gap = abs(idx - c["target"]) / c["fps"]
            b = next(bin_label(lo, hi) for lo, hi in BINS if lo <= gap <= hi)
            masks = sam_masks(stem, c, idx, tracks, dev)
            inputs = {"hand boxes": (truth["left"], truth["right"]), "model boxes": model_pair(boxer, img, truth, dev)}
            for inp, legs in inputs.items():
                if legs is None:
                    continue
                for m, sides in (("SAM 2 follows", sides_from_masks(masks, list(legs))),
                                 ("box centers", track_sides(c["frames"], c["target"], list(legs), anchor=anchor)[0])):
                    if None in sides:
                        continue
                    n[inp, m, b, "decided"] += 1
                    n[inp, m, b, "right"] += sides == ["left", "right"]
            print("done", stem, f"{gap:.2f} s", flush=True)
        print(f"\n{group}: {n['frames']} frames with both legs, {n['with a sure frame']} with a sure frame in the shot")
        for inp in ("hand boxes", "model boxes"):
            for m in ("SAM 2 follows", "box centers"):
                labels = [bin_label(lo, hi) for lo, hi in BINS]
                row = [f"{lab}: {n[inp, m, lab, 'right']}/{n[inp, m, lab, 'decided']}" for lab in labels]
                tot_r = sum(n[inp, m, lab, "right"] for lab in labels)
                tot_d = sum(n[inp, m, lab, "decided"] for lab in labels)
                print(f"  {inp:11s} {m:13s} right/decided by distance to the sure frame: {'  '.join(row)}"
                      f"   all: {tot_r}/{tot_d}")


if __name__ == "__main__":
    main()
