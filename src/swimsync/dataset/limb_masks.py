"""Leg / arm outlines from the labeled boxes, with SAM 2 (milestone 2, step 4).

Usage (Label Studio must be running, see scripts/labelstudio.sh):
    python -m swimsync.dataset.limb_masks

1. For every labeled frame (all "M2 legs+arms - train*" projects and "- test"), SAM 2.1
   outlines the limb inside each box. Cached in data/raw/frames/m2/outlines.json.
2. Each outline is checked (not empty, not the whole box, not spilling far outside).
   A frame with a failed outline is left out, so no limb is ever treated as "nothing".
3. The test outlines go to the Label Studio project "M2 limb outlines - test" for review:
   outline right -> Submit, any outline wrong -> Skip.
4. Writes the YOLO segmentation set data/raw/datasets/m2_seg (classes leg, arm):
   train/val from the outlines (same split as the detector), test from the reviewed frames.
"""

import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["leg", "arm"]
TO_2 = {"left_leg": "leg", "right_leg": "leg", "left_arm": "arm", "right_arm": "arm"}
FRAMES = Path("data/raw/frames/m2")
OUT = Path("data/raw/datasets/m2_seg")
REVIEW_TITLE = "M2 limb outlines - test"


# --- checks and formats ---------------------------------------------------------------

def mask_ok(mask, box, min_fill=0.03, max_fill=0.9, max_spill=0.25, grow=0.1):
    """Is a SAM outline believable for its box?

    min_fill / max_fill: share of the box covered (a limb never fills all of it; all of it means
    SAM took the water). max_spill: share of the mask outside the box grown by `grow`.
    """
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    area = mask.sum()
    if area == 0:
        return False
    inside = mask[max(y1, 0):y2, max(x1, 0):x2].sum()
    fill = inside / max((x2 - x1) * (y2 - y1), 1)
    gx, gy = grow * (x2 - x1), grow * (y2 - y1)
    X1, Y1, X2, Y2 = int(max(x1 - gx, 0)), int(max(y1 - gy, 0)), int(x2 + gx), int(y2 + gy)
    spill = 1 - mask[Y1:Y2, X1:X2].sum() / area
    return bool(min_fill <= fill <= max_fill and spill <= max_spill)


def mask_polygon(mask, epsilon=0.002):
    """Outline of the biggest piece of `mask` as an (n, 2) array of x, y in 0..1, or None."""
    h, w = mask.shape
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, epsilon * cv2.arcLength(c, True), True).reshape(-1, 2).astype(float)
    if len(c) < 3:
        return None
    c[:, 0] /= w
    c[:, 1] /= h
    return np.clip(c, 0, 1)


def yolo_seg_line(cls, poly):
    return f"{cls} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)


def ls_polygon(poly, label, img_w, img_h):
    return {
        "from_name": "label", "to_name": "image", "type": "polygonlabels",
        "original_width": img_w, "original_height": img_h,
        "value": {"points": [[100 * x, 100 * y] for x, y in poly], "polygonlabels": [label]},
    }


def review_config():
    return """<View>
  <Header value="Outlines right? Submit. Any outline wrong or missing? Skip the frame."/>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <PolygonLabels name="label" toName="image">
    <Label value="leg" background="#2a78d6"/>
    <Label value="arm" background="#c43c8a"/>
  </PolygonLabels>
</View>"""


# --- pipeline -----------------------------------------------------------------------------

def boxes_of(result, w, h):
    """Label Studio box results -> [(class name, (x1, y1, x2, y2) in pixels)]."""
    out = []
    for r in result:
        if r.get("type") == "rectanglelabels":
            v = r["value"]
            x1, y1 = v["x"] * w / 100, v["y"] * h / 100
            out.append((TO_2[v["rectanglelabels"][0]], (x1, y1, x1 + v["width"] * w / 100, y1 + v["height"] * h / 100)))
    return out


def outline_frames(frames, cache_file):
    """frames: {file name: (image folder, box results)} -> cached outlines per frame."""
    from ultralytics import SAM

    from swimsync.waterline.above_water import pick_device

    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    model, device = None, pick_device()
    for name, (folder, result) in frames.items():
        img_path = FRAMES / "images" / folder / name
        im = cv2.imread(str(img_path))
        h, w = im.shape[:2]
        boxes = boxes_of(result, w, h)
        key = json.dumps([[c, [round(v, 1) for v in b]] for c, b in boxes])
        if name in cache and cache[name]["key"] == key:
            continue
        limbs = []
        if boxes:
            if model is None:
                model = SAM("sam2.1_b.pt")
            r = model.predict(im, bboxes=[list(b) for _, b in boxes], device=device, verbose=False)[0]
            masks = r.masks.data.cpu().numpy() > 0.5
            for (cls, box), m in zip(boxes, masks):
                poly = mask_polygon(m)
                limbs.append({"cls": cls, "box": list(box), "ok": mask_ok(m, box) and poly is not None,
                              "poly": poly.round(5).tolist() if poly is not None else None})
        cache[name] = {"folder": folder, "w": w, "h": h, "key": key, "limbs": limbs}
    cache_file.write_text(json.dumps(cache))
    return cache


def _link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_seg_dataset(splits):
    """splits: {split: {file name: (image folder, [yolo seg lines])}}."""
    if OUT.exists():
        shutil.rmtree(OUT)
    lists = defaultdict(list)
    for split, items in splits.items():
        (OUT / "images" / split).mkdir(parents=True)
        (OUT / "labels" / split).mkdir(parents=True)
        for name, (folder, lines) in items.items():
            img = OUT / "images" / split / name
            _link(FRAMES / "images" / folder / name, img)
            (OUT / "labels" / split / name).with_suffix(".txt").write_text("\n".join(lines))
            if split == "test":
                lists["budapest" if name.startswith("budapest") else "newpool"].append(str(img.resolve()))
    for group, paths in lists.items():
        (OUT / f"test_{group}.txt").write_text("\n".join(sorted(paths)) + "\n")
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    base = f"path: {OUT.resolve()}\ntrain: images/train\nval: images/val\n"
    for yaml, group in (("data.yaml", "newpool"), ("data_budapest.yaml", "budapest")):
        test = f"test: test_{group}.txt\n" if lists[group] else ""
        (OUT / yaml).write_text(base + test + f"names:\n{names}\n")


def main():
    from .export_yolo import labeled_frames, project_titles, split_train_val
    from .labelstudio import LabelStudioAPI, setup_project

    cred = dict(line.split(": ", 1) for line in
                Path("data/raw/labelstudio/credentials.txt").read_text().splitlines() if ": " in line)
    api = LabelStudioAPI(cred["url"], cred["api_token"])
    prefix = "M2 legs+arms - "
    train_lab = {}
    for title in sorted(project_titles(api)):
        if title.startswith(prefix + "train"):
            lab, _ = labeled_frames(api, title)
            train_lab.update({n: (title[len(prefix):], r) for n, r in lab.items()})
    test_lab, _ = labeled_frames(api, prefix + "test")

    outlines = outline_frames({**train_lab, **{n: ("test", r) for n, r in test_lab.items()}},
                              FRAMES / "outlines.json")

    def lines_of(o):
        return [yolo_seg_line(CLASSES.index(lm["cls"]), lm["poly"]) for lm in o["limbs"]]

    good = {n for n, o in outlines.items() if all(lm["ok"] for lm in o["limbs"])}
    n_limbs = sum(len(o["limbs"]) for o in outlines.values())
    n_bad = sum(not lm["ok"] for o in outlines.values() for lm in o["limbs"])

    # review project: the test frames whose outlines all passed the check
    tasks = []
    for n in sorted(test_lab):
        if n in good:
            o = outlines[n]
            result = [ls_polygon(lm["poly"], lm["cls"], o["w"], o["h"]) for lm in o["limbs"]]
            tasks.append({"data": {"image": f"/data/local-files/?d=images/test/{n}"},
                          "predictions": [{"model_version": "sam2.1_b", "result": result}] if result else []})
    if REVIEW_TITLE not in project_titles(api):
        pid = api.post("/api/projects", {"title": REVIEW_TITLE, "label_config": review_config()})["id"]
        api.post("/api/storages/localfiles", {"project": pid, "title": "frames",
                                              "path": str((FRAMES / "images" / "test").resolve()),
                                              "use_blob_urls": False})
        api.post(f"/api/projects/{pid}/import", tasks)
        print(f"'{REVIEW_TITLE}': imported {len(tasks)} test frames for review")

    # reviewed test frames: submitted polygons (possibly edited), skipped frames left out
    test_seg = {}
    reviewed, _ = labeled_frames(api, REVIEW_TITLE)
    for n, result in reviewed.items():
        lines = []
        for r in result:
            if r.get("type") == "polygonlabels":
                pts = np.array(r["value"]["points"], float) / 100
                lines.append(yolo_seg_line(CLASSES.index(r["value"]["polygonlabels"][0]), pts))
        test_seg[n] = ("test", lines)

    train_names = [n for n in train_lab if n in good]
    train, val = split_train_val(train_names)
    write_seg_dataset({
        "train": {n: (outlines[n]["folder"], lines_of(outlines[n])) for n in train},
        "val": {n: (outlines[n]["folder"], lines_of(outlines[n])) for n in val},
        "test": test_seg,
    })
    dropped = len(train_lab) - len(train_names)
    print(f"outlines: {n_limbs} limbs, {n_bad} failed the check")
    print(f"train {len(train)} + val {len(val)} frames ({dropped} left out: a failed outline); "
          f"test {len(test_seg)} reviewed of {len(tasks)} sent for review")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
