"""Knee keypoint dataset (leg height, plan Task 8): one crop per hand-drawn leg box, one point.

From the keypoint labels ("M2 keypoints - train" / "- test"): each leg box (the guide boxes) gets
a square crop padded 30%; the object is the leg (its box) with one keypoint, the knee clicked
inside that box, or an invisible point when there is none (knee under water or hidden). Sides
don't matter here, so mirrored training copies are safe (flip_idx [0]).

    python -m swimsync.dataset.export_knees   # -> data/raw/datasets/m3_knees  (Label Studio running)
"""
import shutil
from pathlib import Path

OUT = Path("data/raw/datasets/m3_knees")
OUT17 = Path("data/raw/datasets/m3_knees17")
LEG_BOXES = ("left_leg", "right_leg")
JOINTS17 = {"hip": (11, 12), "knee": (13, 14), "ankle": (15, 16)}  # COCO slots, both sides


def leg_crop(box, w, h, pad=0.3):
    """Square around the leg box (longer side + 2 x pad), moved inside the image, as ints."""
    x1, y1, x2, y2 = box
    side = min(round(max(x2 - x1, y2 - y1) * (1 + 2 * pad)), w, h)
    out = []
    for lo, hi, size in ((x1, x2, w), (y1, y2, h)):
        start = min(max(round((lo + hi) / 2 - side / 2), 0), size - side)
        out += [start, start + side]
    return out[0], out[2], out[1], out[3]


def knee_line(knee, leg_box, crop):
    """YOLO pose line: class 0 (leg), its box and one keypoint, normalised to the crop."""
    cx1, cy1, cx2, cy2 = crop
    cw, ch = cx2 - cx1, cy2 - cy1
    bx1, by1 = max(leg_box[0], cx1) - cx1, max(leg_box[1], cy1) - cy1
    bx2, by2 = min(leg_box[2], cx2) - cx1, min(leg_box[3], cy2) - cy1
    vals = [(bx1 + bx2) / 2 / cw, (by1 + by2) / 2 / ch, (bx2 - bx1) / cw, (by2 - by1) / ch]
    vals += [(knee[0] - cx1) / cw, (knee[1] - cy1) / ch, 2] if knee else [0, 0, 0]
    return "0 " + " ".join(f"{v:.6g}" for v in vals)


def leg_line17(joints, leg_box, crop):
    """YOLO pose line in the ready-made model's 17-point layout (keeps its pretrained point head).

    joints: {"knee"|"ankle"|"hip": (x, y) or None}; sides don't matter, so each joint goes in both
    its left and right slot (identical, so mirroring with the usual flip_idx is safe).
    """
    cx1, cy1, cx2, cy2 = crop
    cw, ch = cx2 - cx1, cy2 - cy1
    head = knee_line(None, leg_box, crop).split()[1:5]
    kps = [[0.0, 0.0, 0.0] for _ in range(17)]
    for name, slots in JOINTS17.items():
        p = joints.get(name)
        if p is not None:
            for i in slots:
                kps[i] = [(p[0] - cx1) / cw, (p[1] - cy1) / ch, 2.0]
    return "0 " + " ".join(head) + " " + " ".join(f"{v:.6g}" for k in kps for v in k)


def _knee_in(box, knees, grow=0.15):
    x1, y1, x2, y2 = box
    gx, gy = grow * (x2 - x1), grow * (y2 - y1)
    inside = [p for p in knees if x1 - gx <= p[0] <= x2 + gx and y1 - gy <= p[1] <= y2 + gy]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return min(inside, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2, default=None)


def write_dataset(splits, frames_root, out=OUT, points17=False):
    """splits: {split: {name: (folder, Label Studio result)}} -> crops + labels + data.yaml."""
    import cv2

    from swimsync.dataset.export_pose import labeled_points
    from swimsync.dataset.keypoints import hand_boxes

    if out.exists():
        shutil.rmtree(out)
    counts = {}
    for split, items in splits.items():
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        n_legs = n_knees = 0
        for name, (folder, result) in sorted(items.items()):
            points = labeled_points(result) or {}
            knees = [p for k, p in points.items() if k.endswith("_knee")]
            joints = {j: [p for k, p in points.items() if k.endswith("_" + j)] for j in JOINTS17}
            boxes = {k: b for k, b in hand_boxes(result).items() if k in LEG_BOXES}
            if not boxes:
                continue
            img = cv2.imread(str(frames_root / "images" / folder / name))
            h, w = img.shape[:2]
            for leg, box in boxes.items():
                crop = leg_crop(box, w, h)
                knee = _knee_in(box, knees)
                stem = f"{Path(name).stem}_{leg}"
                cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), img[crop[1]:crop[3], crop[0]:crop[2]])
                line = (leg_line17({j: _knee_in(box, ps) for j, ps in joints.items()}, box, crop) if points17
                        else knee_line(knee, box, crop))
                (out / "labels" / split / f"{stem}.txt").write_text(line)
                n_legs += 1
                n_knees += knee is not None
        counts[split] = (n_legs, n_knees)
    from swimsync.dataset.export_pose import FLIP_IDX

    shape = f"kpt_shape: [17, 3]\nflip_idx: {FLIP_IDX}\n" if points17 else "kpt_shape: [1, 3]\nflip_idx: [0]\n"
    (out / "data.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                                   + shape + "names:\n  0: leg\n")
    return counts


def main():
    from swimsync.dataset.export_yolo import labeled_frames, project_titles, split_train_val
    from swimsync.dataset.facing import _api

    api = _api()
    data = {}
    for split in ("train", "test"):
        title = f"M2 keypoints - {split}"
        pid = project_titles(api)[title]
        tasks = api.get(f"/api/tasks?project={pid}&page_size=10000&fields=all")["tasks"]
        folders = {t["data"]["image"].split("/")[-1]: t["data"]["image"].split("d=images/")[1].split("/")[0]
                   for t in tasks}
        lab, _ = labeled_frames(api, title)
        data[split] = {n: (folders[n], r) for n, r in lab.items()}
    train, val = split_train_val(list(data["train"]))
    splits = {"train": {n: data["train"][n] for n in train}, "val": {n: data["train"][n] for n in val},
              "test": data["test"]}
    for out, p17 in ((OUT, False), (OUT17, True)):
        for split, (legs, knees) in write_dataset(splits, Path("data/raw/frames/m2"), out, p17).items():
            print(f"{out.name} {split}: {legs} leg crops, {knees} with a knee")
    print(f"Saved to {OUT} and {OUT17}")


if __name__ == "__main__":
    main()
