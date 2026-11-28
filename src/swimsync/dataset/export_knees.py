"""Knee keypoint dataset (leg height, plan Task 8): one crop per hand-drawn leg box, one point.

From the keypoint labels ("M2 keypoints - train" / "- test"): each leg box (the guide boxes) gets
a square crop padded 30%; the object is the leg (its box) with one keypoint, the knee clicked
inside that box, or an invisible point when there is none (knee under water or hidden). Sides
don't matter here, so mirrored training copies are safe (flip_idx [0]).

    python -m swimsync.dataset.export_knees          # -> data/raw/datasets/m3_knees  (Label Studio running)
    python -m swimsync.dataset.export_knees height   # -> m3_knees17b: the same + "M3 leg height - train"
    python -m swimsync.dataset.export_knees height noleg   # -> m3_knees17c: + "no leg here" crops

"M3 leg height - train" adds legs at the boundary: each "knee out" leg with its knee clicked, and each
"knee under water" leg as an invisible knee, cropped around the outline model's leg box (as when the
knee model runs) or, if the model missed that leg, around the clicked points.
"""
import shutil
import sys
from pathlib import Path

OUT = Path("data/raw/datasets/m3_knees")
OUT17 = Path("data/raw/datasets/m3_knees17")
OUT17B = Path("data/raw/datasets/m3_knees17b")
OUT17C = Path("data/raw/datasets/m3_knees17c")
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


def height_legs(result):
    """[(toes, knee or None, water)] for the "M3 leg height" legs a knee crop can use: "knee out"
    with its knee clicked, or "knee under water" (no knee); both need the toes and water points."""
    from swimsync.dataset.height_labels import parse_height

    out = []
    for leg in parse_height(result).values():
        if leg["toes"] is None or leg["water"] is None:
            continue
        if leg["status"] == "knee out" and leg["knee"] is not None:
            out.append((leg["toes"], leg["knee"], leg["water"]))
        elif leg["status"] == "knee under water":
            out.append((leg["toes"], None, leg["water"]))
    return out


def height_leg_box(toes, knee, water, boxes, grow=0.15, pad=0.12):
    """The outline model's leg box holding the toes and water points (box grown by `grow`; the
    nearest if several), so the crop matches what the knee model sees when it runs; otherwise a
    box around the clicked points, padded by `pad` of the toes -> water length (at least 10 px)."""
    pts = [p for p in (toes, knee, water) if p is not None]
    mid = ((toes[0] + water[0]) / 2, (toes[1] + water[1]) / 2)
    holding = []
    for b in boxes:
        gx, gy = grow * (b[2] - b[0]), grow * (b[3] - b[1])
        if all(b[0] - gx <= p[0] <= b[2] + gx and b[1] - gy <= p[1] <= b[3] + gy for p in (toes, water)):
            holding.append(b)
    if holding:
        return min(holding, key=lambda b: ((b[0] + b[2]) / 2 - mid[0]) ** 2 + ((b[1] + b[3]) / 2 - mid[1]) ** 2)
    d = max(10, pad * ((water[0] - toes[0]) ** 2 + (water[1] - toes[1]) ** 2) ** 0.5)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return round(min(xs) - d), round(min(ys) - d), round(max(xs) + d), round(max(ys) + d)


def negatives_allowed(result):
    """Can the outline model's other boxes on this frame be "no leg here" crops? Only when every leg
    is accounted for: used as a knee crop, or not out of the water (no points). Not when a leg is
    pressed to the other (its box is a leg) or a leg was clicked but can't be used."""
    from swimsync.dataset.height_labels import parse_height

    for leg in parse_height(result).values():
        st, pts = leg["status"], [leg[k] is not None for k in ("toes", "knee", "water")]
        if st == "knee out" and all(pts):
            continue
        if st == "knee under water" and (pts[0] == pts[2]):  # toes + water (used) or neither (not out)
            continue
        if st == "not out of the water":
            continue
        return False
    return True


def unmatched_boxes(boxes, used, overlap_max=0.5):
    """Boxes that are not a used leg's box and don't mostly overlap one (the same leg found twice)."""
    def overlap(a, b):
        iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
        return iw * ih / small if small > 0 else 0.0
    return [b for b in boxes if b not in used and all(overlap(b, u) <= overlap_max for u in used)]


def add_height_crops(splits, img_dir, out, find_boxes, no_leg=False):
    """Add one 17-point-layout crop per usable "M3 leg height" leg to the dataset in `out`.

    splits: {split: {image name: Label Studio result}}; find_boxes(img) -> the outline model's leg boxes.
    no_leg: also add a crop with no leg (empty label) for each other outline-model box on a frame
    where every leg is accounted for (arms, bodies, legs under water), so those give no knee.
    """
    import cv2

    counts = {}
    for split, items in splits.items():
        n_legs = n_knees = n_none = 0
        for name, result in sorted(items.items()):
            legs = height_legs(result)
            if not legs and not no_leg:
                continue
            img = cv2.imread(str(img_dir / name))
            h, w = img.shape[:2]
            boxes = find_boxes(img)
            used = []
            for i, (toes, knee, water) in enumerate(legs, 1):
                box = height_leg_box(toes, knee, water, boxes)
                used.append(box)  # the model's box or one around the clicks: either way not a "no leg"
                box = (max(box[0], 0), max(box[1], 0), min(box[2], w), min(box[3], h))
                crop = leg_crop(box, w, h)
                if knee is not None and not (crop[0] <= knee[0] < crop[2] and crop[1] <= knee[1] < crop[3]):
                    continue
                stem = f"{Path(name).stem}_height{i}"
                cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), img[crop[1]:crop[3], crop[0]:crop[2]])
                (out / "labels" / split / f"{stem}.txt").write_text(leg_line17({"knee": knee}, box, crop))
                n_legs += 1
                n_knees += knee is not None
            if no_leg and negatives_allowed(result):
                for j, box in enumerate(unmatched_boxes(boxes, used), 1):
                    crop = leg_crop(box, w, h)
                    stem = f"{Path(name).stem}_noleg{j}"
                    cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), img[crop[1]:crop[3], crop[0]:crop[2]])
                    (out / "labels" / split / f"{stem}.txt").write_text("")
                    n_none += 1
        counts[split] = (n_legs, n_knees, n_none)
    return counts


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


def export_with_height(api, splits, no_leg=False):
    """m3_knees17b: the M2 keypoint crops (17-point layout) + the "M3 leg height - train" legs;
    m3_knees17c (no_leg): + a "no leg here" crop for each other outline-model box on those frames."""
    from ultralytics import YOLO

    from swimsync.dataset.export_yolo import labeled_frames, split_train_val
    from swimsync.dataset.height_labels import ROOT, TRAIN_TITLE
    from swimsync.height.measure import SEG, leg_outlines
    from swimsync.waterline.above_water import pick_device

    out = OUT17C if no_leg else OUT17B
    for split, (legs, knees) in write_dataset(splits, Path("data/raw/frames/m2"), out, True).items():
        print(f"{out.name} {split} (M2 keypoints): {legs} leg crops, {knees} with a knee")
    lab, _ = labeled_frames(api, TRAIN_TITLE)
    train, val = split_train_val(list(lab))
    dev, seg = pick_device(), YOLO(SEG)
    counts = add_height_crops({"train": {n: lab[n] for n in train}, "val": {n: lab[n] for n in val}},
                              ROOT / "images" / "height_train", out,
                              lambda img: [b for b, _ in leg_outlines(img, seg, dev)], no_leg)
    for split, (legs, knees, none) in counts.items():
        print(f"{out.name} {split} (M3 leg height): {legs} leg crops, {knees} with a knee, {none} with no leg")
    print(f"Saved to {out}")


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
    if sys.argv[1:2] == ["height"]:
        export_with_height(api, splits, no_leg=sys.argv[2:3] == ["noleg"])
        return
    for out, p17 in ((OUT, False), (OUT17, True)):
        for split, (legs, knees) in write_dataset(splits, Path("data/raw/frames/m2"), out, p17).items():
            print(f"{out.name} {split}: {legs} leg crops, {knees} with a knee")
    print(f"Saved to {OUT} and {OUT17}")


if __name__ == "__main__":
    main()
