"""Pose training set from the keypoint labels (milestone 2, step 3).

    python -m swimsync.dataset.export_pose    # Label Studio must be running

Writes data/raw/datasets/m2_pose: each labeled frame cropped around her leg/arm boxes (the
same zoom the pose model gets at use time, `swimsync.limbs.sides.pose_people`), with one
swimmer per crop in YOLO pose format: class, box, then 17 COCO points (x y visible). Face
points are never labeled (visible 0). Frames with a point given twice are left out and listed.
val = the last 20% (by time) of each training video; test = "M2 keypoints - test".
"""

import shutil
from pathlib import Path

import numpy as np

from swimsync.dataset.keypoints import POINTS, hand_boxes

# COCO: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles (left/right pairs)
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def labeled_points(result):
    """{point name: (x, y) in pixels}, or None if a point was given twice."""
    out = {}
    for r in result:
        if r.get("type") != "keypointlabels":
            continue
        name = r["value"]["keypointlabels"][0]
        if name in out:
            return None
        out[name] = (r["value"]["x"] * r["original_width"] / 100, r["value"]["y"] * r["original_height"] / 100)
    return out


def pose_line(points, limb_boxes, crop):
    """YOLO pose line for one swimmer in `crop` (x1, y1, x2, y2 in the frame).

    Her box covers her limb boxes and every labeled point.
    """
    cx1, cy1, cx2, cy2 = crop
    cw, ch = cx2 - cx1, cy2 - cy1
    xy = [(b[0], b[1]) for b in limb_boxes] + [(b[2], b[3]) for b in limb_boxes] + list(points.values())
    xy = np.clip(np.array(xy, float) - (cx1, cy1), 0, (cw, ch))
    (x1, y1), (x2, y2) = xy.min(axis=0), xy.max(axis=0)
    kps = np.zeros((17, 3))
    for name, (x, y) in points.items():
        kps[POINTS[name]] = ((x - cx1) / cw, (y - cy1) / ch, 2)
    vals = [(x1 + x2) / 2 / cw, (y1 + y2) / 2 / ch, (x2 - x1) / cw, (y2 - y1) / ch] + [round(v, 6) for v in kps.ravel()]
    return "0 " + " ".join(f"{v:.6g}" for v in vals)


def write_dataset(out, splits, frames_root, pad=0.5):
    """splits: {split: {file name: (image folder, Label Studio result)}}. Returns skipped names."""
    import cv2

    from swimsync.limbs.sides import crop_around

    if out.exists():
        shutil.rmtree(out)
    skipped = []
    for split, items in splits.items():
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        for name, (folder, result) in sorted(items.items()):
            points = labeled_points(result)
            boxes = list(hand_boxes(result).values())
            if points is None or not (boxes or points):
                skipped.append(name)
                continue
            img = cv2.imread(str(frames_root / "images" / folder / name))
            h, w = img.shape[:2]
            region = boxes + [(x, y, x, y) for x, y in points.values()]
            crop = crop_around(region, w, h, pad)
            cv2.imwrite(str(out / "images" / split / name), img[crop[1]:crop[3], crop[0]:crop[2]])
            (out / "labels" / split / name).with_suffix(".txt").write_text(pose_line(points, boxes, crop))
    has_test = any((out / "images" / "test").glob("*.jpg")) if "test" in splits else False
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n" + ("test: images/test\n" if has_test else "")
        + f"kpt_shape: [17, 3]\nflip_idx: {FLIP_IDX}\nnames:\n  0: swimmer\n")
    return skipped


def main():
    from swimsync.dataset.export_yolo import labeled_frames, project_titles, split_train_val
    from swimsync.dataset.labelstudio import LabelStudioAPI

    cred = dict(line.split(": ", 1) for line in
                Path("data/raw/labelstudio/credentials.txt").read_text().splitlines() if ": " in line)
    api = LabelStudioAPI(cred["url"], cred["api_token"])
    folder_of = lambda r: next((x for x in ("train3", "train2", "train", "test")  # noqa: E731
                                if f"d=images/{x}/" in r), "train")
    data = {}
    for split in ("train", "test"):
        title = f"M2 keypoints - {split}"
        pid = project_titles(api)[title]
        tasks = api.get(f"/api/tasks?project={pid}&page_size=10000&fields=all")["tasks"]
        folders = {t["data"]["image"].split("/")[-1]: folder_of(t["data"]["image"]) for t in tasks}
        lab, n = labeled_frames(api, title)
        data[split] = {name: (folders[name], r) for name, r in lab.items()}
        print(f"{title}: {len(lab)} labeled of {n}")
    train, val = split_train_val(list(data["train"]))
    splits = {"train": {n: data["train"][n] for n in train}, "val": {n: data["train"][n] for n in val},
              "test": data["test"]}
    out = Path("data/raw/datasets/m2_pose")
    skipped = write_dataset(out, splits, Path("data/raw/frames/m2"))
    for s in splits:
        print(f"{s}: {len(list((out / 'images' / s).glob('*.jpg')))} crops")
    if skipped:
        print(f"Left out ({len(skipped)}, a point given twice or nothing labeled): " + ", ".join(skipped))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
