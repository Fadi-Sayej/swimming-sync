"""Pick frames for labeling leg / arm boxes, with pre-filled boxes from the pose model.

Usage:
    python -m swimsync.dataset.pick_frames \
        --train data/raw/jr2026_*.mp4 --n-train 400 \
        --test data/raw/test/*.mp4 --n-test-each 25

Each video is checked every `--step` seconds. A frame is a candidate when the
pose model finds a person surrounded by pool water (the swimmer) and CLIP says it
is a pool shot (not a stage, sofa, empty arena or TV graphic; see scene.py).
About 10% of the picks are
pool shots with no swimmer found (fully underwater, or too small), labeled with
no boxes or whatever is visible. Picks are spread over each whole video, at
least `--min-gap` seconds apart.

Writes to --out (default data/raw/frames/m2):
    images/train/*.jpg, images/test/*.jpg
    tasks_train.json, tasks_test.json   Label Studio import files (with pre-filled boxes)
    manifest.csv                        file, split, video, time, kind, pre-filled boxes
    scan/<video>.json                   scan results, reused on the next run
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .labelstudio import make_task
from .limbs import limb_boxes

EMPTY_SHARE = 0.1  # share of picks with no swimmer found
SCAN_VERSION = 2  # bump when the scan output changes, so old caches are redone


def pool_mask(frame, hue=(80, 102), min_sat=50, min_val=90):
    """Pool water: cyan, azure or teal. Navy blues (sofa backdrop, graphics, hue ~106) don't count.

    Tuned on Junior Worlds 2026 (Budapest), Doha 2024, Paris 2025/26 and Budapest 2024.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (hue[0], min_sat, min_val), (hue[1], 255, 255)) > 0


def ring_share(mask, box, grow=0.5):
    """Share of `mask` pixels in the ring around `box` (box grown by `grow` of its size)."""
    h, w = mask.shape
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    X1, Y1 = int(max(0, x1 - grow * bw)), int(max(0, y1 - grow * bh))
    X2, Y2 = int(min(w, x2 + grow * bw)), int(min(h, y2 + grow * bh))
    i1, j1, i2, j2 = int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))
    outer = mask[Y1:Y2, X1:X2]
    inner = mask[j1:j2, i1:i2]
    n = outer.size - inner.size
    return float(outer.sum() - inner.sum()) / n if n > 0 else 0.0


def swimmer_index(xyxy, mask, min_ring=0.3):
    """Index of the biggest person surrounded by pool water, or None."""
    xyxy = np.asarray(xyxy, float).reshape(-1, 4)
    best, best_area = None, -1.0
    for i, b in enumerate(xyxy):
        area = (b[2] - b[0]) * (b[3] - b[1])
        if area > best_area and ring_share(mask, b) >= min_ring:
            best, best_area = i, area
    return best


def spread_pick(times, n, min_gap):
    """Indices of up to `n` times spread evenly over the whole span, `min_gap` apart."""
    times = np.asarray(times, float)
    if n <= 0 or times.size == 0:
        return []
    order = np.argsort(times)
    t = times[order]
    blocked = np.zeros(len(t), bool)  # too close to a frame already picked
    picked = []
    for target in np.linspace(t[0], t[-1], n):
        if blocked.all():
            break
        dist = np.where(blocked, np.inf, np.abs(t - target))
        k = int(np.argmin(dist))
        picked.append(k)
        blocked |= np.abs(t - t[k]) < min_gap
    return sorted(order[picked].tolist())


def scan(video, model, scene, step, device, cache_dir):
    """Candidates every `step` s: {"t", "kind": swimmer|empty, "kp", "pool"} (cached as JSON).

    "pool" is CLIP's probability that the frame is a pool shot; `choose` filters on it.
    """
    cache = cache_dir / f"{video.stem}.json"
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved["step"] == step and saved.get("version") == SCAN_VERSION:
            return saved
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    found = []
    for t in tqdm(np.arange(0, duration - 0.5, step), desc=video.name):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        mask = pool_mask(frame)
        r = model.predict(frame, device=device, verbose=False)[0]
        xyxy = r.boxes.xyxy.cpu().numpy() if r.boxes is not None and len(r.boxes) else np.zeros((0, 4))
        i = swimmer_index(xyxy, mask)
        if i is not None:
            kp = r.keypoints.data[i].cpu().numpy().round(2).tolist()
            c = {"t": round(float(t), 2), "kind": "swimmer", "kp": kp}
        elif mask.mean() >= 0.6:
            c = {"t": round(float(t), 2), "kind": "empty", "kp": None}
        else:
            continue
        c["pool"] = round(scene.score(frame), 3)
        found.append(c)
    cap.release()
    saved = {"video": str(video), "step": step, "version": SCAN_VERSION,
             "width": w, "height": h, "candidates": found}
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(saved))
    return saved


def choose(scans, n_total, min_gap, min_pool=0.3):
    """Split `n_total` over videos by their number of swimmer candidates, then spread picks.

    Frames CLIP doesn't see as a pool shot (`pool` < `min_pool`) are left out.
    """
    scans = [{**s, "candidates": [c for c in s["candidates"] if c.get("pool", 1.0) >= min_pool]}
             for s in scans]
    counts = np.array([sum(c["kind"] == "swimmer" for c in s["candidates"]) for s in scans], float)
    share = counts / counts.sum() if counts.sum() else np.full(len(scans), 1 / max(len(scans), 1))
    n_v = np.round(share * n_total).astype(int)
    picks = []
    for s, n in zip(scans, n_v):
        cands = s["candidates"]
        swim = [c for c in cands if c["kind"] == "swimmer"]
        n_empty = int(round(EMPTY_SHARE * n))
        chosen = [swim[k] for k in spread_pick([c["t"] for c in swim], n - n_empty, min_gap)]
        taken = [c["t"] for c in chosen]
        empty = [c for c in cands if c["kind"] == "empty" and all(abs(c["t"] - t) >= min_gap for t in taken)]
        chosen += [empty[k] for k in spread_pick([c["t"] for c in empty], n_empty, min_gap)]
        picks += [(s, c) for c in sorted(chosen, key=lambda c: c["t"])]
    return picks


def export(picks, split, out, url_prefix):
    """Save the picked frames as JPGs and write the Label Studio tasks file."""
    img_dir = out / "images" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    tasks, rows, caps = [], [], {}
    for s, c in tqdm(picks, desc=f"saving {split}"):
        video = Path(s["video"])
        cap = caps.setdefault(video, cv2.VideoCapture(str(video)))
        cap.set(cv2.CAP_PROP_POS_MSEC, c["t"] * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        name = f"{video.stem}_{c['t']:07.1f}s.jpg"
        cv2.imwrite(str(img_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        h, w = frame.shape[:2]
        boxes = limb_boxes(c["kp"], w, h) if c["kp"] is not None else {}
        rel = f"images/{split}/{name}"
        tasks.append(make_task(url_prefix + rel, boxes, w, h, source=f"{video.name} @ {c['t']:.1f}s"))
        rows.append({"file": rel, "split": split, "video": video.name, "time_s": c["t"],
                     "kind": c["kind"], "prefilled_boxes": len(boxes)})
    for cap in caps.values():
        cap.release()
    (out / f"tasks_{split}.json").write_text(json.dumps(tasks, indent=1))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", type=Path, nargs="+", required=True)
    p.add_argument("--test", type=Path, nargs="*", default=[])
    p.add_argument("--n-train", type=int, default=400)
    p.add_argument("--n-test-each", type=int, default=25)
    p.add_argument("--step", type=float, default=2.0, help="seconds between checked frames (train)")
    p.add_argument("--test-step", type=float, default=1.0, help="seconds between checked frames (test)")
    p.add_argument("--min-gap", type=float, default=5.0, help="minimum seconds between picks")
    p.add_argument("--min-pool", type=float, default=0.3,
                   help="minimum CLIP 'pool shot' probability for a frame to be picked")
    p.add_argument("--model", default="yolo11m-pose.pt")
    p.add_argument("--out", type=Path, default=Path("data/raw/frames/m2"))
    p.add_argument("--url-prefix", default="/data/local-files/?d=",
                   help="prefix for image paths in the Label Studio tasks")
    a = p.parse_args()

    from ultralytics import YOLO

    from swimsync.waterline.above_water import pick_device

    from .scene import PoolShotClassifier

    model, device = YOLO(a.model), pick_device()
    scene = PoolShotClassifier(device=device)
    cache = a.out / "scan"
    train = [scan(v, model, scene, a.step, device, cache) for v in a.train]
    test = [scan(v, model, scene, a.test_step, device, cache) for v in a.test]

    rows = export(choose(train, a.n_train, a.min_gap, a.min_pool), "train", a.out, a.url_prefix)
    test_picks = [pk for s in test for pk in choose([s], a.n_test_each, a.min_gap, a.min_pool)]
    rows += export(test_picks, "test", a.out, a.url_prefix)

    with open(a.out / "manifest.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    for split in ("train", "test"):
        r = [x for x in rows if x["split"] == split]
        print(f"{split}: {len(r)} frames, {sum(x['kind'] == 'empty' for x in r)} with no swimmer found, "
              f"{sum(x['prefilled_boxes'] > 0 for x in r)} with pre-filled boxes")
    print(f"Saved to {a.out}")


if __name__ == "__main__":
    main()
