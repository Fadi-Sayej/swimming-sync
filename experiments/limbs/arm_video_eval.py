"""Arm boxes helped by the frames just before and after (step 1), scored on the test frames.

The sync score runs on video, so an arm missed on one frame but found on its neighbours can be
filled in. For every test frame: the outline model's arm boxes on the test image and on the
video frames up to `K` before and after it (the video frame that looks most like the image is
the middle one). Arm AP50 against the hand-drawn arm boxes (m2_2cls), as in the arm
box tests (results.md, 2026-11-21).

Usage: python experiments/limbs/arm_video_eval.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class

sys.path.insert(0, str(Path(__file__).parent))
from track_eval import BOXES, CACHE  # noqa: E402

from swimsync.limbs.sides import box_iou  # noqa: E402
from swimsync.waterline.above_water import pick_device  # noqa: E402

DS = Path("data/raw/datasets/m2_2cls")
K = 2
OUT = CACHE.parent / "arm_video.json"


def arms(model, img, dev, conf=0.001):
    r = model.predict(img, imgsz=960, conf=conf, iou=0.7, verbose=False, device=dev)[0]
    return [(b, c) for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())
            if model.names[int(k)] == "arm"]


def clip_around(img_path, k, look=1.0):
    """Video frames around a test image: {offset: frame} for offsets -k..k (0 = best match)."""
    stem = Path(img_path).stem
    video, t = stem.rsplit("_", 1)
    cap = cv2.VideoCapture(f"data/raw/test/{video}.mp4")
    cap.set(cv2.CAP_PROP_POS_MSEC, max(float(t[:-1]) - look, 0) * 1000)
    ref = cv2.resize(cv2.imread(img_path), (160, 90)).astype(float)
    frames, diffs = [], []
    for _ in range(int(2 * look * cap.get(cv2.CAP_PROP_FPS)) + 1):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
        diffs.append(float(np.abs(cv2.resize(f, (160, 90)).astype(float) - ref).mean()))
    target = int(np.argmin(diffs))
    return {i - target: frames[i] for i in range(max(target - k, 0), min(target + k + 1, len(frames)))}


def build(model, dev):
    data = {}
    for group in ("newpool", "budapest"):
        for img_path in (DS / f"test_{group}.txt").read_text().split():
            near = {o: arms(model, f, dev, conf=0.1) for o, f in clip_around(img_path, K).items() if o != 0}
            data[img_path] = {"group": group, "frame": arms(model, cv2.imread(img_path), dev), "near": near}
        print("built", group, flush=True)
    OUT.write_text(json.dumps(data))


def truth(img_path):
    h, w = cv2.imread(img_path).shape[:2]
    lab = DS / "labels" / Path(img_path).parent.name / (Path(img_path).stem + ".txt")
    g = np.array([list(map(float, x.split())) for x in lab.read_text().splitlines() if x.strip()]).reshape(-1, 5)
    g = g[g[:, 0] == 1]
    return [[(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h] for _, x, y, bw, bh in g]


def ap50(data, group, get):
    tp, conf, n_gt = [], [], 0
    for img_path, d in data.items():
        if d["group"] != group:
            continue
        gt = truth(img_path)
        n_gt += len(gt)
        pred = sorted(get(d), key=lambda p: -p[1])
        used, t = set(), []
        for b, _ in pred:
            ious = [box_iou(g, b) for g in gt]
            i = int(np.argmax(ious)) if ious else None
            hit = i is not None and ious[i] >= 0.5 and i not in used
            used |= {i} if hit else set()
            t.append(hit)
        tp += t
        conf += [c for _, c in pred]
    tp, conf = np.array(tp, bool), np.array(conf)
    res = ap_per_class(tp[:, None], conf, np.ones(len(tp)), np.ones(n_gt))
    return float(res[5][0, 0]), int(tp.sum()), n_gt


def with_neighbours(d, k, need_both, match=0.3, keep=0.25):
    """Add arms seen on nearby frames (within k) but not on this one.

    need_both: only when seen both before and after (a gap, not an arm leaving the view).
    The added box is the nearest frame's, with that frame's confidence halved.
    """
    have = [b for b, c in d["frame"] if c >= keep]
    extra = []
    for off in sorted((int(o) for o in d["near"]), key=abs):
        if abs(off) > k:
            continue
        for b, c in d["near"][str(off)]:
            if c < keep or any(box_iou(b, h) >= match for h in have + [e for e, _ in extra]):
                continue
            other = [x for o in d["near"] if abs(int(o)) <= k and np.sign(int(o)) == -np.sign(off)
                     for x, cx in d["near"][o] if cx >= keep and box_iou(b, x) >= match]
            if need_both and not other:
                continue
            extra.append((b, c / 2))
    return d["frame"] + extra


def smoothed(d, k, match=0.3):
    """Each box's confidence averaged with its best match on every nearby frame (0 if none)."""
    out = []
    for b, c in d["frame"]:
        cs = [c]
        for o, boxes in d["near"].items():
            if abs(int(o)) <= k:
                cs.append(max((cx for x, cx in boxes if box_iou(b, x) >= match), default=0.0))
        out.append((b, float(np.mean(cs))))
    return out


def main():
    if not OUT.exists():
        build(YOLO(BOXES), pick_device())
    data = json.loads(OUT.read_text())
    variants = {"this frame only": lambda d: d["frame"]}
    for k in (1, 2):
        variants[f"+ arms seen within {k}, either side"] = lambda d, k=k: with_neighbours(d, k, False)
        variants[f"+ arms seen within {k}, both sides"] = lambda d, k=k: with_neighbours(d, k, True)
        variants[f"confidence averaged over {k}"] = lambda d, k=k: smoothed(d, k)
    for group in ("newpool", "budapest"):
        for name, get in variants.items():
            a, found, n = ap50(data, group, get)
            print(f"{group:9s} {name:34s} arm AP50 {a:.3f}  (found {found} of {n})")


if __name__ == "__main__":
    main()
