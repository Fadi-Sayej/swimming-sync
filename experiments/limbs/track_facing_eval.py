"""Leg sides from sure front/back frames, carried through the video (#25), scored on the test frames.

The front/back model is right about her sides when it is sure, but it is sure on few frames
(facing_eval.py). Here every video frame within 3 s of a test frame gets a front/back answer;
the nearest sure frame in the same shot gives the sides, and the two leg boxes are followed
from there to the test frame (swimsync.limbs.track). Leg boxes per video frame come from
track_eval.py's cache (data/raw/frames/m2/track/); the front/back answers are added to it
(~10 min the first time).

Carrying is only reliable for a few frames (legs that spin or overlap swap boxes): sides from a
sure frame up to 0.1 s away were right 11 of 12 times, up to 0.5 s 20 of 28.

Usage: python experiments/limbs/track_facing_eval.py [--min-conf 0.8] [--max-carry 0.1]
"""
import argparse
import json
import sys
from collections import Counter
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from facing_eval import FACING, front_or_back, hand_legs, model_pair  # noqa: E402
from track_eval import BOXES, CACHE, WINDOW  # noqa: E402

from swimsync.limbs.track import facing_anchor, track_sides  # noqa: E402
from swimsync.waterline.above_water import pick_device  # noqa: E402

DS = Path("data/raw/datasets/m2_4cls")


def add_facing(cache, clf, dev):
    """Front/back answer on every cached video frame with two leg boxes (None otherwise)."""
    c = json.loads(cache.read_text())
    if all("facing" in f for f in c["frames"]):
        return c
    video, t = cache.stem.rsplit("_", 1)
    cap = cv2.VideoCapture(f"data/raw/test/{video}.mp4")
    cap.set(cv2.CAP_PROP_POS_MSEC, max(float(t[:-1]) - WINDOW, 0) * 1000)  # same frames as track_eval
    for f in c["frames"]:
        ok, img = cap.read()
        f["facing"] = front_or_back(clf, img, f["legs"], dev) if ok and len(f["legs"]) == 2 else None
    cache.write_text(json.dumps(c))
    return c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--facing", default=FACING)
    p.add_argument("--min-conf", type=float, default=0.8)
    p.add_argument("--max-carry", type=float, default=0.1, help="seconds; carry sides at most this far")
    a = p.parse_args()
    dev = pick_device()
    clf, boxer = YOLO(a.facing), YOLO(BOXES)
    anchor = partial(facing_anchor, min_conf=a.min_conf)
    groups = {"newpool": (DS / "test_newpool.txt").read_text().split(),
              "budapest": (DS / "test_budapest.txt").read_text().split()}
    for group, images in groups.items():
        n, gaps = {"hand boxes": Counter(), "model boxes": Counter()}, []
        for img_path in images:
            img = cv2.imread(img_path)
            truth = hand_legs(img_path, img.shape[1], img.shape[0])
            cache = CACHE / (Path(img_path).stem + ".json")
            if len(truth) < 2 or not cache.exists():
                continue
            c = add_facing(cache, clf, dev)
            inputs = {"hand boxes": (truth["left"], truth["right"]), "model boxes": model_pair(boxer, img, truth, dev)}
            for inp, legs in inputs.items():
                k = n[inp]
                k["frames"] += 1
                if legs is None:
                    continue
                k["found"] += 1
                sides, gap = track_sides(c["frames"], c["target"], list(legs), anchor=anchor,
                                         max_gap=round(a.max_carry * c["fps"]))
                if sides[0] is None or sides[1] is None:
                    continue
                k["decided"] += 1
                k["right"] += sides == ["left", "right"]
                if inp == "hand boxes":
                    gaps.append(gap / c["fps"])
        print(f"\n{group} ({len(images)} frames; sure frame {np.median(gaps) if gaps else float('nan'):.1f} s "
              f"away (median))")
        for inp, k in n.items():
            print(f"  {inp:11s}: both legs {k['frames']}, found {k['found']}, sides given {k['decided']}, "
                  f"right {k['right']} ({100 * k['right'] / max(k['decided'], 1):.0f}%)")


if __name__ == "__main__":
    main()
