"""Keypoint labels for training the pose model on swimmers (milestone 2, step 3).

12 points per swimmer: shoulders, elbows, wrists, hips, knees, ankles, each with her own
left/right. The frames already have hand-drawn left/right leg and arm boxes (97% consistent on
leg sides), so the pre-filled points from the pose model take their sides from those boxes, and
points outside the boxes (limbs the model invented) are left out.

    python -m swimsync.dataset.keypoints    # Label Studio projects "M2 keypoints - train" / "- test"
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from swimsync.dataset.limbs import LIMBS, limb_boxes
from swimsync.limbs.sides import box_iou

# point name -> COCO keypoint index (the pose model's order)
POINTS = {
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16,
}
HOTKEYS = "qwertyasdfgh"
BOX_COLORS = {"left_leg": "#2a78d6", "right_leg": "#eb6834", "left_arm": "#2ca58d", "right_arm": "#c43c8a"}


def point_limb(name):
    """'left_knee' -> 'left_leg'."""
    side, joint = name.split("_")
    return f"{side}_{'arm' if joint in ('shoulder', 'elbow', 'wrist') else 'leg'}"


def kp_config():
    pts = "\n".join(
        f'    <Label value="{n}" background="{"#2a78d6" if n.startswith("left") else "#eb6834"}" hotkey="{HOTKEYS[i]}"/>'
        for i, n in enumerate(POINTS))
    boxes = "\n".join(f'    <Label value="{n}" background="{c}"/>' for n, c in BOX_COLORS.items())
    return f"""<View>
  <Header value="Her OWN left (blue) / right (orange). Drag wrong points, delete invented ones, add missing visible ones. The thin boxes are your earlier labels, only a guide."/>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <KeyPointLabels name="kp" toName="image" strokeWidth="2" opacity="0.9">
{pts}
  </KeyPointLabels>
  <RectangleLabels name="guide" toName="image" strokeWidth="1" opacity="0.15">
{boxes}
  </RectangleLabels>
</View>"""


def _swap(kp, kind):
    kp = kp.copy()
    left, right = LIMBS[f"left_{kind}"], LIMBS[f"right_{kind}"]
    kp[list(left)], kp[list(right)] = kp[list(right)].copy(), kp[list(left)].copy()
    return kp


def fix_sides(kp, boxes, conf_min=0.3):
    """Swap the pose model's left/right legs (and arms) when the hand-drawn boxes say so.

    boxes: {"left_leg" | "right_leg" | "left_arm" | "right_arm": xyxy}.
    """
    kp = np.asarray(kp, float)
    pose = limb_boxes(kp, np.inf, np.inf, conf_min=conf_min)
    for kind in ("leg", "arm"):
        def fit(a, b):
            pa, hb = pose.get(f"{a}_{kind}"), boxes.get(f"{b}_{kind}")
            return box_iou(pa, hb) if pa is not None and hb is not None else 0.0
        keep = fit("left", "left") + fit("right", "right")
        swap = fit("left", "right") + fit("right", "left")
        if swap > keep:
            kp = _swap(kp, kind)
    return kp


def prefill_points(kp, boxes, conf_min=0.3, pad=0.15):
    """{point name: (x, y)} for confident points inside their own limb's box (padded)."""
    out = {}
    for name, i in POINTS.items():
        b = boxes.get(point_limb(name))
        x, y, c = kp[i]
        if b is None or c < conf_min:
            continue
        px, py = pad * (b[2] - b[0]), pad * (b[3] - b[1])
        if b[0] - px <= x <= b[2] + px and b[1] - py <= y <= b[3] + py:
            out[name] = (float(x), float(y))
    return out


def keypoint_task(image_url, points, boxes, img_w, img_h, model_version="yolo11m-pose + your boxes"):
    """Label Studio task: pre-filled points (pixels) and the hand-drawn boxes as a guide."""
    common = {"to_name": "image", "original_width": img_w, "original_height": img_h}
    result = [{**common, "from_name": "kp", "type": "keypointlabels",
               "value": {"x": 100 * x / img_w, "y": 100 * y / img_h, "width": 0.6, "keypointlabels": [n]}}
              for n, (x, y) in points.items()]
    result += [{**common, "from_name": "guide", "type": "rectanglelabels",
                "value": {"x": 100 * x1 / img_w, "y": 100 * y1 / img_h, "width": 100 * (x2 - x1) / img_w,
                          "height": 100 * (y2 - y1) / img_h, "rotation": 0, "rectanglelabels": [label]}}
               for label, (x1, y1, x2, y2) in boxes.items()]
    return {"data": {"image": image_url}, "predictions": [{"model_version": model_version, "result": result}]}


def hand_boxes(result):
    """Label Studio box results -> {label: xyxy in pixels} (the last box of each label)."""
    out = {}
    for r in result:
        if r.get("type") != "rectanglelabels":
            continue
        v, w, h = r["value"], r["original_width"], r["original_height"]
        out[v["rectanglelabels"][0]] = (v["x"] * w / 100, v["y"] * h / 100,
                                        (v["x"] + v["width"]) * w / 100, (v["y"] + v["height"]) * h / 100)
    return out


def pick_train(labeled, n, seed=0):
    """n frames with at least one limb box, round-robin over the videos (legs first)."""
    by_video = defaultdict(list)
    for name, (folder, result) in sorted(labeled.items()):
        boxes = hand_boxes(result)
        if boxes:
            by_video[name.rsplit("_", 1)[0]].append((("left_leg" in boxes) + ("right_leg" in boxes), name))
    rng = random.Random(seed)
    for v in by_video.values():
        rng.shuffle(v)
        v.sort(key=lambda x: -x[0])  # frames with both legs first
    picks = []
    while len(picks) < n and any(by_video.values()):
        for v in sorted(by_video):
            if by_video[v] and len(picks) < n:
                picks.append(by_video[v].pop(0)[1])
    return picks


def build_tasks(names, labeled, frames_root, model, device):
    import cv2

    from swimsync.limbs.sides import assign_sides, pose_people

    tasks, n_pts = [], 0
    for name in names:
        folder, result = labeled[name]
        img = cv2.imread(str(frames_root / "images" / folder / name))
        h, w = img.shape[:2]
        boxes = hand_boxes(result)
        people = pose_people(model, img, None, device)
        _, kp = assign_sides([(lab.split("_")[1], b) for lab, b in boxes.items()], people)
        points = prefill_points(fix_sides(kp, boxes), boxes) if kp is not None else {}
        n_pts += len(points)
        tasks.append(keypoint_task(f"/data/local-files/?d=images/{folder}/{name}", points, boxes, w, h))
    return tasks, n_pts


def main():
    from ultralytics import YOLO

    from swimsync.dataset.export_yolo import labeled_frames, project_titles
    from swimsync.dataset.labelstudio import LabelStudioAPI
    from swimsync.waterline.above_water import pick_device

    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    root = Path("data/raw/frames/m2").resolve()
    cred = dict(line.split(": ", 1) for line in
                Path("data/raw/labelstudio/credentials.txt").read_text().splitlines() if ": " in line)
    api = LabelStudioAPI(cred["url"], cred["api_token"])
    titles = project_titles(api)
    prefix = "M2 legs+arms - "
    train_lab = {}
    for t in sorted(titles):
        if t.startswith(prefix + "train"):
            lab, _ = labeled_frames(api, t)
            train_lab.update({n: (t[len(prefix):], r) for n, r in lab.items()})
    test_lab = {n: ("test", r) for n, r in labeled_frames(api, prefix + "test")[0].items() if hand_boxes(r)}

    model, dev = YOLO("yolo11m-pose.pt"), pick_device()
    for split, labeled, names in (("train", train_lab, pick_train(train_lab, n_train)),
                                  ("test", test_lab, sorted(test_lab))):
        title = f"M2 keypoints - {split}"
        if title in titles:
            print(f"'{title}' already exists (id {titles[title]}); not importing again.")
            continue
        tasks, n_pts = build_tasks(names, labeled, root, model, dev)
        random.Random(1).shuffle(tasks)
        pid = api.post("/api/projects", {"title": title, "label_config": kp_config()})["id"]
        for folder in sorted({labeled[n][0] for n in names}):
            api.post("/api/storages/localfiles", {"project": pid, "title": folder,
                                                  "path": str(root / "images" / folder), "use_blob_urls": False})
        api.post(f"/api/projects/{pid}/import", tasks)
        print(f"'{title}': project {pid}, {len(tasks)} frames, {n_pts} pre-filled points "
              f"({n_pts / max(len(tasks), 1):.1f} per frame)")
    (root / "keypoint_frames.json").write_text(json.dumps({"train": pick_train(train_lab, n_train),
                                                           "test": sorted(test_lab)}, indent=1))


if __name__ == "__main__":
    main()
